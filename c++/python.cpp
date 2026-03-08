#include <print>
#include <format>
#include <span>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

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

  std::vector<float> controls = dag->controls;

  // Two-channel audiobus: channel-major layout [ch0 block][ch1 block]
  std::vector<float> abus(2 * BS, 0.0f);

  for (int iter = 0; iter < 10; ++iter) {
    s.process(controls.data(), abus.data());

    std::println("process call {}", iter);
    std::println("i\tch0\tch1");
    for (size_t i = 0; i < BS; ++i) {
      float ch0 = abus[i];
      float ch1 = abus[BS + i];
      std::println("{}\t{}\t{}", i, ch0, ch1);
    }
    std::println("");
  }

  const size_t nblocks = SR / BS;

  auto t0 = std::chrono::steady_clock::now();
  for (size_t i = 0; i < nblocks; ++i) {
    s.process(controls.data(), abus.data());
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
    (void)output;
    (void)input;
    (void)frame_count;
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
  .def("__repr__", &DSP::repr);
}
