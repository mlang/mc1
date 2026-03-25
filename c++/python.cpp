#include "audio_buffer.hpp"
#include "compiler.hpp"
#include "dag.hpp"
#include "runtime.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/chrono.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mc1 {

namespace {

template<typename... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};

template<typename... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

DAG parse_dag_or_throw(pybind11::bytes bytes_object)
{
  auto bytes = std::as_bytes(std::span(std::string_view(bytes_object)));
  auto dag = DAG::parse(bytes);
  if (!dag) {
    throw pybind11::value_error(
      std::string("invalid DAG bytes: ") + std::string(mlang::to_string(dag.error()))
    );
  }
  return std::move(*dag);
}

uint32_t validate_positive_arg(long value, const char* name)
{
  if (value <= 0) throw pybind11::value_error(std::string(name) + " must be > 0");
  if (value > static_cast<long>(std::numeric_limits<uint32_t>::max())) {
    throw pybind11::value_error(std::string(name) + " must fit in uint32");
  }
  return static_cast<uint32_t>(value);
}

uint32_t validate_non_negative_arg(long value, const char* name)
{
  if (value < 0) throw pybind11::value_error(std::string(name) + " must be >= 0");
  if (value > static_cast<long>(std::numeric_limits<uint32_t>::max())) {
    throw pybind11::value_error(std::string(name) + " must fit in uint32");
  }
  return static_cast<uint32_t>(value);
}

std::optional<double> validate_timeout_arg(pybind11::handle value, const char* name)
{
  if (value.is_none()) return std::nullopt;

  if (pybind11::isinstance<pybind11::bool_>(value)) {
    throw pybind11::value_error(std::string(name) + " must be None or a non-negative number");
  }

  try {
    auto timeout = pybind11::cast<double>(value);
    if (timeout < 0.0) {
      throw pybind11::value_error(std::string(name) + " must be >= 0");
    }
    if (!std::isfinite(timeout)) {
      throw pybind11::value_error(std::string(name) + " must be finite");
    }
    return timeout;
  } catch (const pybind11::cast_error&) {
    throw pybind11::value_error(std::string(name) + " must be None or a non-negative number");
  }
}

} // namespace

double perft(pybind11::bytes b)
{
  auto dag = parse_dag_or_throw(b);
  constexpr unsigned int SR = 44100;
  constexpr size_t BS = 32;

  auto r = compile(dag, SR, BS);
  auto compiled_synth = r[dag.name];
  auto synth = compiled_synth.instantiate();

  // Two-channel audiobus: channel-major layout [ch0 block][ch1 block]
  aligned_float_buffer abus(2 * BS);

  for (int iter = 0; iter < 10; ++iter) {
    abus.fill(0.0f);
    synth.process(abus.data());

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
    abus.fill(0.0f);
    synth.process(abus.data());
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
  std::unordered_map<uint32_t, std::unique_ptr<synth_instance>> synths_by_id_;
  std::unordered_map<uint32_t, bool> pending_idle_status_by_request_id_;
  uint32_t next_synth_id_ = 1;
  uint32_t next_request_id_ = 1;

  static uint32_t validate_positive(long value, const char* name)
  {
    return validate_positive_arg(value, name);
  }

  static uint32_t validate_non_negative(long value, const char* name)
  {
    return validate_non_negative_arg(value, name);
  }

  static std::optional<double> validate_timeout(pybind11::handle value, const char* name)
  {
    return validate_timeout_arg(value, name);
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

  void enqueue_command(time_point when, rt_payload payload, const char* action)
  {
    rt_command command{.when = when, .payload = payload};
    if (!runtime_.try_enqueue(command)) {
      throw std::runtime_error(std::string(action) + ": rt command queue overflow");
    }
  }

  void enqueue_idle_query(uint32_t request_id)
  {
    rt_command command{
      .when = std::chrono::utc_clock::now(),
      .payload = query_idle_status{.request_id = request_id},
    };
    if (!runtime_.try_enqueue(command)) {
      throw std::runtime_error("wait_until_idle failed: rt command queue overflow");
    }
  }

  synth_instance& get_synth_instance(uint32_t synth_id)
  {
    auto it = synths_by_id_.find(synth_id);
    if (it == synths_by_id_.end()) {
      throw pybind11::value_error(std::format("unknown synth_id {}", synth_id));
    }
    return *it->second;
  }

  void drain_runtime_events()
  {
    rt_event event;
    while (runtime_.try_pop_event(event)) {
      std::visit(overloaded{
        [this](const synth_retired_event& retired) {
          synths_by_id_.erase(retired.synth_id);
        },
        [this](const idle_status_event& idle_status) {
          pending_idle_status_by_request_id_.insert_or_assign(idle_status.request_id, idle_status.idle);
        },
      }, event.payload);
    }
  }

  std::optional<bool> consume_idle_status(uint32_t request_id)
  {
    auto it = pending_idle_status_by_request_id_.find(request_id);
    if (it == pending_idle_status_by_request_id_.end()) return std::nullopt;

    auto idle = it->second;
    pending_idle_status_by_request_id_.erase(it);
    return idle;
  }

  uint32_t create_synth(time_point when,
    std::string_view synth_name,
    synth_insert_mode insert_mode,
    uint32_t anchor_synth_id,
    const char* action,
    pybind11::kwargs controls
  )
  {
    auto compiled_synth_it = compiled_synths_by_name_.find(std::string(synth_name));
    if (compiled_synth_it == compiled_synths_by_name_.end()) {
      throw pybind11::value_error(std::format("unknown synth '{}'", synth_name));
    }

    auto& compiled_synth = compiled_synth_it->second;

    auto synth = compiled_synth.instantiate();
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
      synth.set_control_values(slot->index, values);
    }

    auto synth_id = next_synth_id_++;
    auto instance = std::make_unique<synth_instance>(synth_instance{
      .synth_id = synth_id,
      .synth_name = std::string(synth_name),
      .compiled_synth = compiled_synth,
      .synth = std::move(synth),
    });

    auto* instance_ptr = instance.get();
    synths_by_id_.insert_or_assign(synth_id, std::move(instance));
    try {
      enqueue_command(when,
        start_synth{
          .synth_id = synth_id,
          .synth = instance_ptr,
          .insert_mode = insert_mode,
          .anchor_synth_id = anchor_synth_id,
        },
        action
      );
      return synth_id;
    } catch (...) {
      synths_by_id_.erase(synth_id);
      throw;
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
    drain_runtime_events();

    auto dag = parse_dag_or_throw(b);

    try {
      auto result = compile(dag, sample_rate_, block_size_);
      auto compiled_synth = result[dag.name];
      compiled_synths_by_name_.insert_or_assign(dag.name, std::move(compiled_synth));
    } catch (const pybind11::error_already_set&) {
      throw;
    } catch (const std::exception& ex) {
      throw pybind11::value_error(std::string("compile failed: ") + ex.what());
    }
  }

  uint32_t append(time_point when, std::string_view synth_name, pybind11::kwargs controls)
  {
    drain_runtime_events();
    return create_synth(when,
      synth_name, synth_insert_mode::append, 0, "append failed", controls
    );
  }

  uint32_t prepend(time_point when, std::string_view synth_name, pybind11::kwargs controls)
  {
    drain_runtime_events();
    return create_synth(when,
      synth_name, synth_insert_mode::prepend, 0, "prepend failed", controls
    );
  }

  uint32_t insert_before(time_point when,
    long before_synth_id, std::string_view synth_name,
    pybind11::kwargs controls
  )
  {
    drain_runtime_events();

    auto validated_synth_id = validate_non_negative(before_synth_id, "before_synth_id");
    get_synth_instance(validated_synth_id);

    return create_synth(when,
      synth_name, synth_insert_mode::before, validated_synth_id,
      "insert_before failed",
      controls
    );
  }

  uint32_t insert_after(time_point when,
    long after_synth_id, std::string_view synth_name,
    pybind11::kwargs controls
  )
  {
    drain_runtime_events();

    auto validated_synth_id = validate_non_negative(after_synth_id, "after_synth_id");
    get_synth_instance(validated_synth_id);

    return create_synth(when,
      synth_name, synth_insert_mode::after, validated_synth_id,
      "insert_after failed",
      controls
    );
  }

  void set(time_point when, long synth_id, pybind11::kwargs controls)
  {
    drain_runtime_events();

    auto validated_synth_id = validate_non_negative(synth_id, "synth_id");
    auto& instance = get_synth_instance(validated_synth_id);

    for (auto item : controls) {
      auto control_name = pybind11::cast<std::string>(item.first);
      auto slot = instance.compiled_synth.control_slot(control_name);
      if (!slot) {
        throw pybind11::value_error(std::format(
            "unknown control '{}' for synth '{}'",
            control_name,
            instance.synth_name));
      }

      auto values = parse_control_values(
        item.second, instance.synth_name, control_name, slot->width
      );

      for (size_t offset = 0; offset < values.size(); ++offset) {
        enqueue_command(when,
          set_control_value{
            .synth_id = validated_synth_id,
            .control_index = static_cast<uint32_t>(slot->index + offset),
            .value = values[offset],
          },
          "set failed"
        );
      }
    }
  }

  void remove(time_point when, long synth_id)
  {
    drain_runtime_events();

    auto validated_synth_id = validate_non_negative(synth_id, "synth_id");
    get_synth_instance(validated_synth_id);

    enqueue_command(when, stop_synth{validated_synth_id}, "remove failed");
  }

  void start()
  { runtime_.start(); }

  void stop() noexcept
  { runtime_.stop(); }

  bool wait_until_idle(pybind11::handle timeout)
  {
    drain_runtime_events();

    if (!runtime_.started()) {
      throw pybind11::value_error("wait_until_idle is only available while DSP is started");
    }

    auto validated_timeout = validate_timeout(timeout, "timeout");
    auto deadline = validated_timeout
      ? std::optional<std::chrono::steady_clock::time_point>{
          std::chrono::steady_clock::now() +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(*validated_timeout))}
      : std::nullopt;

    auto request_id = next_request_id_++;
    auto query_pending = false;

    {
      pybind11::gil_scoped_release release;
      while (true) {
        drain_runtime_events();
        if (query_pending) {
          if (auto idle = consume_idle_status(request_id)) {
            if (*idle) break;
            request_id = next_request_id_++;
            query_pending = false;
          }
        }
        if (deadline && std::chrono::steady_clock::now() >= *deadline) {
          return false;
        }
        if (!query_pending) {
          try {
            enqueue_idle_query(request_id);
            query_pending = true;
          } catch (const std::runtime_error&) {
          }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }

    drain_runtime_events();
    return true;
  }

  std::vector<uint32_t> synth_ids()
  {
    drain_runtime_events();

    if (runtime_.started()) {
      throw pybind11::value_error("synth_ids is only available while DSP is stopped");
    }

    auto ids = runtime_.synth_ids();
    drain_runtime_events();
    return ids;
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
  .def_property_readonly("synth_ids", &DSP::synth_ids)
  .def("compile", &DSP::compile_graph, py::arg("dag_bytes"))
  .def("append", &DSP::append, py::arg("when"), py::arg("synth_name"))
  .def("prepend", &DSP::prepend, py::arg("when"), py::arg("synth_name"))
  .def("insert_before", &DSP::insert_before, py::arg("when"), py::arg("before_synth_id"), py::arg("synth_name"))
  .def("insert_after", &DSP::insert_after, py::arg("when"), py::arg("after_synth_id"), py::arg("synth_name"))
  .def("set", &DSP::set, py::arg("when"), py::arg("synth_id"))
  .def("remove", &DSP::remove, py::arg("when"), py::arg("synth_id"))
  .def("start", &DSP::start)
  .def("stop", &DSP::stop)
  .def("wait_until_idle", &DSP::wait_until_idle, py::arg("timeout") = py::none())
  .def("__repr__", &DSP::repr);
}
