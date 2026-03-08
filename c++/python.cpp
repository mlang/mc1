#include <print>
#include <format>
#include <span>
#include <chrono>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "audio.hpp"
#include "compiler.hpp"
#include "dag.hpp"

#include <pybind11/pybind11.h>

namespace mc1 {

double perft(pybind11::bytes b)
{
  auto bytes = std::as_bytes(std::span(std::string_view(b)));
  auto dag = DAG::parse(bytes);
  constexpr unsigned int SR = 44100;
  constexpr size_t BS = 32;

  auto r = compile(*dag, SR, BS);
  auto s = r[dag->name];

  // Two-channel audiobus: channel-major layout [ch0 block][ch1 block]
  std::vector<float> abus(2 * BS, 0.0f);

  for (int iter = 0; iter < 10; ++iter) {
    s.process(abus.data());

    std::println("process call {}", iter);
    std::println("i\tch0\tch1");
    for (size_t i = 0; i < BS; ++i) {
      float ch0 = abus[i];
      float ch1 = abus[BS + i];
      std::println("{}\t{:.6g}\t{:.6g}", i, ch0, ch1);
    }
    std::println("");
  }

  const size_t nblocks = SR / BS;

  auto t0 = std::chrono::steady_clock::now();
  for (size_t i = 0; i < nblocks; ++i) {
    s.process(abus.data());
  }
  auto t1 = std::chrono::steady_clock::now();

  std::chrono::duration<double> elapsed = t1 - t0;
  double seconds = elapsed.count();
  double ratio = 1.0 / seconds;

  return seconds;
}

class DSP
{
  uint32_t sample_rate_;
  size_t block_size_;
  uint32_t input_channels_;
  uint32_t output_channels_;
  std::unordered_map<std::string, Result::Synth> synths_by_name_;
  std::vector<Result::Synth> active_synths_;
  std::vector<float> abus_;
  std::unique_ptr<audio_device> audio_device_;

  static uint32_t validate_positive(long value, const char* name)
  {
    if (value <= 0) throw pybind11::value_error(std::string(name) + " must be > 0");
    if (value > static_cast<long>(std::numeric_limits<uint32_t>::max())) {
      throw pybind11::value_error(std::string(name) + " must fit in uint32");
    }
    return static_cast<uint32_t>(value);
  }

  static uint32_t validate_non_negative(long value, const char* name)
  {
    if (value < 0) throw pybind11::value_error(std::string(name) + " must be >= 0");
    if (value > static_cast<long>(std::numeric_limits<uint32_t>::max())) {
      throw pybind11::value_error(std::string(name) + " must fit in uint32");
    }
    return static_cast<uint32_t>(value);
  }

  static float cast_float_control(
      pybind11::handle value,
      std::string_view synth_name,
      std::string_view control_name)
  {
    try {
      return pybind11::cast<float>(value);
    } catch (const pybind11::cast_error&) {
      throw pybind11::value_error(std::format(
          "control '{}' for synth '{}' must be a number",
          control_name,
          synth_name));
    }
  }

  static std::vector<float> parse_control_values(
      pybind11::handle value,
      std::string_view synth_name,
      std::string_view control_name,
      size_t width)
  {
    if (width == 1) {
      return {cast_float_control(value, synth_name, control_name)};
    }

    if (!pybind11::isinstance<pybind11::sequence>(value) ||
        pybind11::isinstance<pybind11::str>(value) ||
        pybind11::isinstance<pybind11::bytes>(value)) {
      throw pybind11::value_error(std::format(
          "control '{}' for synth '{}' expects a sequence of {} numbers",
          control_name,
          synth_name,
          width));
    }

    auto seq = pybind11::reinterpret_borrow<pybind11::sequence>(value);
    if (static_cast<size_t>(pybind11::len(seq)) != width) {
      throw pybind11::value_error(std::format(
          "control '{}' for synth '{}' expects {} values",
          control_name,
          synth_name,
          width));
    }

    std::vector<float> values;
    values.reserve(width);
    for (auto item : seq) {
      values.push_back(cast_float_control(item, synth_name, control_name));
    }
    return values;
  }

public:
  DSP(
      long sample_rate = 44100,
      long block_size = 32,
      long input_channels = 0,
      long output_channels = 2)
  {
    sample_rate_ = validate_positive(sample_rate, "sample_rate");
    block_size_ = static_cast<size_t>(validate_positive(block_size, "block_size"));
    input_channels_ = validate_non_negative(input_channels, "input_channels");
    output_channels_ = validate_non_negative(output_channels, "output_channels");
    if (input_channels_ == 0 && output_channels_ == 0) {
      throw pybind11::value_error("input_channels and output_channels cannot both be 0");
    }
    abus_.resize(static_cast<size_t>(input_channels_ + output_channels_) * block_size_, 0.0f);

    audio_device_ = std::make_unique<audio_device>(
        input_channels_,
        output_channels_,
        &DSP::ma_trampoline,
        this,
        sample_rate_,
        static_cast<uint32_t>(block_size_));
  }

  static void ma_trampoline(
      ma_device* device,
      void* output,
      const void* input,
      ma_uint32 frame_count)
  { static_cast<DSP*>(device->pUserData)->process(output, input, frame_count); }

  void process(void* output, const void* input, ma_uint32 frame_count)
  {
    assert(frame_count % block_size_ == 0);

    const auto* in = static_cast<const float*>(input);
    auto* out = static_cast<float*>(output);
    const size_t nframes = static_cast<size_t>(frame_count);
    const size_t nchunks = nframes / block_size_;
    const size_t in_ch = static_cast<size_t>(input_channels_);
    const size_t out_ch = static_cast<size_t>(output_channels_);

    for (size_t chunk = 0; chunk < nchunks; ++chunk) {
      std::fill(abus_.begin(), abus_.end(), 0.0f);

      const size_t frame_offset = chunk * block_size_;
      for (size_t f = 0; f < block_size_; ++f) {
        const size_t src_frame = frame_offset + f;

        for (size_t ch = 0; ch < out_ch; ++ch) {
          float value = 0.0f;
          if (out) value = out[src_frame * out_ch + ch];
          abus_[ch * block_size_ + f] = value;
        }

        for (size_t ch = 0; ch < in_ch; ++ch) {
          float value = 0.0f;
          if (in) value = in[src_frame * in_ch + ch];
          abus_[(out_ch + ch) * block_size_ + f] = value;
        }
      }

      for (auto& synth : active_synths_) {
        synth.process(abus_.data());
      }

      if (out) {
        for (size_t f = 0; f < block_size_; ++f) {
          const size_t dst_frame = frame_offset + f;
          for (size_t ch = 0; ch < out_ch; ++ch) {
            out[dst_frame * out_ch + ch] = abus_[ch * block_size_ + f];
          }
        }
      }
    }
  }

  void compile_graph(pybind11::bytes b)
  {
    auto bytes = std::as_bytes(std::span(std::string_view(b)));
    auto dag = DAG::parse(bytes);
    if (!dag) {
      throw pybind11::value_error("invalid DAG bytes");
    }

    try {
      auto result = compile(*dag, sample_rate_, block_size_);
      auto synth = result[dag->name];
      synths_by_name_.insert_or_assign(dag->name, std::move(synth));
    } catch (const std::exception& ex) {
      throw pybind11::value_error(std::string("compile failed: ") + ex.what());
    }
  }

  void play(std::string_view synth_name, pybind11::kwargs controls)
  {
    auto it = synths_by_name_.find(std::string(synth_name));
    if (it == synths_by_name_.end()) {
      throw pybind11::value_error(std::format("unknown synth '{}'", synth_name));
    }

    auto synth = it->second;
    for (auto item : controls) {
      std::string control_name;
      try {
        control_name = pybind11::cast<std::string>(item.first);
      } catch (const pybind11::cast_error&) {
        throw pybind11::value_error(std::format(
            "control names for synth '{}' must be strings",
            synth_name));
      }

      auto slot = synth.control_slot(control_name);
      if (!slot) {
        throw pybind11::value_error(std::format(
            "unknown control '{}' for synth '{}'",
            control_name,
            synth_name));
      }

      auto values = parse_control_values(item.second, synth_name, control_name, slot->width);
      synth.set_control_values(control_name, values);
    }

    active_synths_.push_back(std::move(synth));
  }

  uint32_t sample_rate() const { return sample_rate_; }
  size_t block_size() const { return block_size_; }
  uint32_t input_channels() const { return input_channels_; }
  uint32_t output_channels() const { return output_channels_; }
  std::string repr() const
  {
    return std::format(
        "DSP(sample_rate={}, block_size={}, input_channels={}, output_channels={})",
        sample_rate_,
        block_size_,
        input_channels_,
        output_channels_);
  }
};

}

namespace py = pybind11;

PYBIND11_MODULE(_core, m, py::mod_gil_not_used())
{
  using namespace mc1;

  m.doc() = "MiniCollider";
  m.def("perft", &perft);
  py::class_<DSP>(m, "DSP")
  .def(py::init<long, long, long, long>(),
      py::arg("sample_rate") = 44100,
      py::arg("block_size") = 32,
      py::arg("input_channels") = 0,
      py::arg("output_channels") = 2)
  .def_property_readonly("sample_rate", &DSP::sample_rate)
  .def_property_readonly("block_size", &DSP::block_size)
  .def_property_readonly("input_channels", &DSP::input_channels)
  .def_property_readonly("output_channels", &DSP::output_channels)
  .def("compile", &DSP::compile_graph, py::arg("dag_bytes"))
  .def("play", &DSP::play, py::arg("synth_name"))
  .def("__repr__", &DSP::repr);
}
