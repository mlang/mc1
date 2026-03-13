#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "compiler.hpp"
#include "dag.hpp"
#include "runtime.hpp"

#include <pybind11/pybind11.h>

namespace mc1 {

double perft(pybind11::bytes b)
{
  auto bytes = std::as_bytes(std::span(std::string_view(b)));
  auto dag = DAG::parse(bytes);
  constexpr unsigned int SR = 44100;
  constexpr size_t BS = 32;

  auto r = compile(*dag, SR, BS);
  auto compiled_synth = r[dag->name];
  auto module = compiled_synth.instantiate();

  // Two-channel audiobus: channel-major layout [ch0 block][ch1 block]
  std::vector<float> abus(2 * BS, 0.0f);

  for (int iter = 0; iter < 10; ++iter) {
    module.process(abus.data());

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
    module.process(abus.data());
  }
  auto t1 = std::chrono::steady_clock::now();

  std::chrono::duration<double> elapsed = t1 - t0;
  return elapsed.count();
}

class DSP
{
  uint32_t sample_rate_;
  size_t block_size_;
  uint32_t input_channels_;
  uint32_t output_channels_;
  runtime runtime_;
  std::unordered_map<std::string, Result::CompiledSynth> compiled_synths_by_name_;
  std::unordered_map<uint32_t, std::unique_ptr<module_instance>> modules_by_id_;
  uint32_t next_module_id_ = 1;

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

  static uint32_t validate_output_channels(
      uint32_t input_channels,
      long output_channels)
  {
    auto value = validate_non_negative(output_channels, "output_channels");
    if (input_channels == 0 && value == 0) {
      throw pybind11::value_error("input_channels and output_channels cannot both be 0");
    }
    return value;
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

  template<typename Payload>
  void enqueue_command(Payload payload, const char* action)
  {
    rt_command command{
      .sample_offset = 0,
      .payload = rt_payload{payload},
    };
    if (!runtime_.try_enqueue(command)) {
      throw std::runtime_error(std::string(action) + ": rt command queue overflow");
    }
  }

  module_instance& get_module_instance(uint32_t module_id)
  {
    auto it = modules_by_id_.find(module_id);
    if (it == modules_by_id_.end()) {
      throw pybind11::value_error(std::format("unknown module_id {}", module_id));
    }
    return *it->second;
  }

  void reap_retired_modules()
  {
    retire_token token;
    while (runtime_.try_pop_retired(token)) {
      modules_by_id_.erase(token.module->module_id);
    }
  }

public:
  DSP(
      long sample_rate = 44100,
      long block_size = 32,
      long input_channels = 0,
      long output_channels = 2)
  : sample_rate_{validate_positive(sample_rate, "sample_rate")}
  , block_size_{static_cast<size_t>(validate_positive(block_size, "block_size"))}
  , input_channels_{validate_non_negative(input_channels, "input_channels")}
  , output_channels_{validate_output_channels(input_channels_, output_channels)}
  , runtime_{sample_rate_, block_size_, input_channels_, output_channels_}
  {}

  void compile_graph(pybind11::bytes b)
  {
    reap_retired_modules();

    auto bytes = std::as_bytes(std::span(std::string_view(b)));
    auto dag = DAG::parse(bytes);
    if (!dag) {
      throw pybind11::value_error("invalid DAG bytes");
    }

    try {
      auto result = compile(*dag, sample_rate_, block_size_);
      auto compiled_synth = result[dag->name];
      compiled_synths_by_name_.insert_or_assign(dag->name, std::move(compiled_synth));
    } catch (const pybind11::error_already_set&) {
      throw;
    } catch (const std::exception& ex) {
      throw pybind11::value_error(std::string("compile failed: ") + ex.what());
    }
  }

  uint32_t add(std::string_view synth_name, pybind11::kwargs controls)
  {
    reap_retired_modules();

    auto compiled_synth_it = compiled_synths_by_name_.find(std::string(synth_name));
    if (compiled_synth_it == compiled_synths_by_name_.end()) {
      throw pybind11::value_error(std::format("unknown synth '{}'", synth_name));
    }

    auto& compiled_synth = compiled_synth_it->second;

    auto module = compiled_synth.instantiate();
    for (auto item : controls) {
      std::string control_name;
      try {
        control_name = pybind11::cast<std::string>(item.first);
      } catch (const pybind11::cast_error&) {
        throw pybind11::value_error(std::format(
            "control names for synth '{}' must be strings",
            synth_name));
      }

      auto slot = compiled_synth.control_slot(control_name);
      if (!slot) {
        throw pybind11::value_error(std::format(
            "unknown control '{}' for synth '{}'",
            control_name,
            synth_name));
      }

      auto values = parse_control_values(item.second, synth_name, control_name, slot->width);
      module.set_control_values(slot->index, values);
    }

    auto module_id = next_module_id_++;
    auto instance = std::make_unique<module_instance>(module_instance{
      .module_id = module_id,
      .synth_name = std::string(synth_name),
      .compiled_synth = compiled_synth,
      .module = std::move(module),
    });

    auto* instance_ptr = instance.get();
    modules_by_id_.insert_or_assign(module_id, std::move(instance));
    try {
      enqueue_command(start_module{module_id, instance_ptr}, "add failed");
      return module_id;
    } catch (...) {
      modules_by_id_.erase(module_id);
      throw;
    }
  }

  void set(long module_id, pybind11::kwargs controls)
  {
    reap_retired_modules();

    auto validated_module_id = validate_non_negative(module_id, "module_id");
    auto& instance = get_module_instance(validated_module_id);

    for (auto item : controls) {
      std::string control_name;
      try {
        control_name = pybind11::cast<std::string>(item.first);
      } catch (const pybind11::cast_error&) {
        throw pybind11::value_error(std::format(
            "control names for synth '{}' must be strings",
            instance.synth_name));
      }

      auto slot = instance.compiled_synth.control_slot(control_name);
      if (!slot) {
        throw pybind11::value_error(std::format(
            "unknown control '{}' for synth '{}'",
            control_name,
            instance.synth_name));
      }

      auto values = parse_control_values(
          item.second,
          instance.synth_name,
          control_name,
          slot->width);

      for (size_t offset = 0; offset < values.size(); ++offset) {
        enqueue_command(
            set_control_value{
              .module_id = validated_module_id,
              .control_index = static_cast<uint32_t>(slot->index + offset),
              .value = values[offset],
            },
            "set failed");
      }
    }
  }

  void remove(long module_id)
  {
    reap_retired_modules();

    auto validated_module_id = validate_non_negative(module_id, "module_id");
    get_module_instance(validated_module_id);

    enqueue_command(stop_module{validated_module_id}, "remove failed");
  }

  void start()
  {
    runtime_.start();
  }

  void stop() noexcept
  {
    runtime_.stop();
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

} // namespace mc1

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
  .def("add", &DSP::add, py::arg("synth_name"))
  .def("set", &DSP::set, py::arg("module_id"))
  .def("remove", &DSP::remove, py::arg("module_id"))
  .def("start", &DSP::start)
  .def("stop", &DSP::stop)
  .def("__repr__", &DSP::repr);
}
