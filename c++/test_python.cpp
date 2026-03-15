#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "audio_buffer.hpp"
#include "compiler.hpp"
#include "dag.hpp"
#include "runtime.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace mc1 {

namespace {

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

const char* control_kind_name(control_kind kind)
{
  switch (kind) {
  case control_kind::value:
    return "value";
  case control_kind::trigger:
    return "trigger";
  }
  return "unknown";
}

float cast_float_control(pybind11::handle value, std::string_view control_name)
{
  try {
    return pybind11::cast<float>(value);
  } catch (const pybind11::cast_error&) {
    throw pybind11::value_error("control '" + std::string(control_name) + "' must be a number");
  }
}

std::vector<float> parse_control_values(
    pybind11::handle value,
    std::string_view control_name,
    size_t width)
{
  if (width == 1) {
    return {cast_float_control(value, control_name)};
  }

  if (!pybind11::isinstance<pybind11::sequence>(value) ||
      pybind11::isinstance<pybind11::str>(value) ||
      pybind11::isinstance<pybind11::bytes>(value)) {
    throw pybind11::value_error(
        "control '" + std::string(control_name) + "' expects a sequence of " +
        std::to_string(width) + " numbers");
  }

  auto seq = pybind11::reinterpret_borrow<pybind11::sequence>(value);
  if (static_cast<size_t>(pybind11::len(seq)) != width) {
    throw pybind11::value_error(
        "control '" + std::string(control_name) + "' expects " +
        std::to_string(width) + " values");
  }

  std::vector<float> values;
  values.reserve(width);
  for (auto item : seq) {
    values.push_back(cast_float_control(item, control_name));
  }
  return values;
}

pybind11::dict validate_control_step(pybind11::handle step)
{
  if (step.is_none()) {
    return pybind11::dict();
  }
  if (!pybind11::isinstance<pybind11::dict>(step)) {
    throw pybind11::value_error("control steps must be dicts or None");
  }
  return pybind11::reinterpret_borrow<pybind11::dict>(step);
}

void apply_controls(
    const Result::CompiledSynth& compiled_synth,
    Result::Module& module,
    pybind11::handle step)
{
  auto controls = validate_control_step(step);
  for (auto item : controls) {
    auto control_name = pybind11::cast<std::string>(item.first);
    auto slot = compiled_synth.control_slot(control_name);
    if (!slot) {
      throw pybind11::value_error("unknown control '" + control_name + "'");
    }
    module.set_control_values(slot->index, parse_control_values(item.second, control_name, slot->width));
  }
}

void enqueue_controls(
    runtime& runtime,
    const Result::CompiledSynth& compiled_synth,
    uint32_t module_id,
    pybind11::handle step)
{
  auto controls = validate_control_step(step);
  for (auto item : controls) {
    auto control_name = pybind11::cast<std::string>(item.first);
    auto slot = compiled_synth.control_slot(control_name);
    if (!slot) {
      throw pybind11::value_error("unknown control '" + control_name + "'");
    }
    auto values = parse_control_values(item.second, control_name, slot->width);
    for (size_t offset = 0; offset < values.size(); ++offset) {
      rt_command command{
        .sample_offset = 0,
        .payload = rt_payload{set_control_value{
          .module_id = module_id,
          .control_index = static_cast<uint32_t>(slot->index + offset),
          .value = values[offset],
        }},
      };
      if (!runtime.try_enqueue(command)) {
        throw pybind11::value_error("rt command queue overflow");
      }
    }
  }
}

} // namespace

pybind11::list compiled_controls(
    pybind11::bytes b,
    long sample_rate = 44100,
    long block_size = 32)
{
  auto dag = parse_dag_or_throw(b);

  auto result = compile(
      dag,
      validate_positive_arg(sample_rate, "sample_rate"),
      static_cast<size_t>(validate_positive_arg(block_size, "block_size")));
  auto compiled_synth = result[dag.name];

  pybind11::list descriptors;
  for (auto const& control : compiled_synth.control_descs()) {
    pybind11::dict descriptor;
    descriptor["name"] = control.name;
    descriptor["index"] = control.index;
    descriptor["width"] = control.width;
    descriptor["kind"] = control_kind_name(control.kind);
    descriptors.append(std::move(descriptor));
  }
  return descriptors;
}

pybind11::dict render_control_blocks(
    pybind11::bytes b,
    pybind11::list control_steps,
    long sample_rate = 44100,
    long block_size = 32,
    long output_channels = 2)
{
  auto dag = parse_dag_or_throw(b);

  auto validated_block_size = static_cast<size_t>(validate_positive_arg(block_size, "block_size"));
  auto validated_output_channels = static_cast<size_t>(validate_positive_arg(output_channels, "output_channels"));
  auto result = compile(
      dag,
      validate_positive_arg(sample_rate, "sample_rate"),
      validated_block_size);
  auto compiled_synth = result[dag.name];
  auto module = compiled_synth.instantiate();

  aligned_float_buffer abus(validated_output_channels * validated_block_size);
  pybind11::list rendered_blocks;
  pybind11::list done_actions;
  for (auto step : control_steps) {
    apply_controls(compiled_synth, module, step);
    abus.fill(0.0f);
    auto done_action = module.process(abus.data());

    pybind11::list rendered_block;
    for (float sample : abus) {
      rendered_block.append(sample);
    }
    rendered_blocks.append(std::move(rendered_block));
    done_actions.append(done_action);
  }

  pybind11::dict result_dict;
  result_dict["blocks"] = std::move(rendered_blocks);
  result_dict["done_actions"] = std::move(done_actions);
  return result_dict;
}

pybind11::list render_blocks(
    pybind11::bytes b,
    long blocks,
    long sample_rate = 44100,
    long block_size = 32,
    long output_channels = 2)
{
  auto dag = parse_dag_or_throw(b);

  auto validated_blocks = static_cast<size_t>(validate_positive_arg(blocks, "blocks"));
  auto validated_block_size = static_cast<size_t>(validate_positive_arg(block_size, "block_size"));
  auto validated_output_channels = static_cast<size_t>(validate_positive_arg(output_channels, "output_channels"));
  auto result = compile(
      dag,
      validate_positive_arg(sample_rate, "sample_rate"),
      validated_block_size);
  auto compiled_synth = result[dag.name];
  auto module = compiled_synth.instantiate();

  aligned_float_buffer abus(validated_output_channels * validated_block_size);
  pybind11::list rendered_blocks;
  for (size_t i = 0; i < validated_blocks; ++i) {
    abus.fill(0.0f);
    module.process(abus.data());

    pybind11::list rendered_block;
    for (float sample : abus) {
      rendered_block.append(sample);
    }
    rendered_blocks.append(std::move(rendered_block));
  }
  return rendered_blocks;
}

std::uintptr_t aligned_abus_modulo(
    long block_size = 32,
    long channels = 2)
{
  auto validated_block_size = static_cast<size_t>(validate_positive_arg(block_size, "block_size"));
  auto validated_channels = static_cast<size_t>(validate_positive_arg(channels, "channels"));
  aligned_float_buffer abus(validated_channels * validated_block_size);
  return reinterpret_cast<std::uintptr_t>(abus.data()) % audio_buffer_alignment;
}

pybind11::list runtime_module_ids_per_block(
    pybind11::bytes b,
    pybind11::list control_steps,
    long sample_rate = 44100,
    long block_size = 32,
    long output_channels = 2)
{
  auto dag = parse_dag_or_throw(b);

  auto validated_sample_rate = validate_positive_arg(sample_rate, "sample_rate");
  auto validated_block_size = static_cast<size_t>(validate_positive_arg(block_size, "block_size"));
  auto validated_output_channels = validate_positive_arg(output_channels, "output_channels");
  auto result = compile(dag, validated_sample_rate, validated_block_size);
  auto compiled_synth = result[dag.name];

  runtime rt(validated_sample_rate, validated_block_size, 0, validated_output_channels);

  auto instance = std::make_unique<module_instance>(module_instance{
    .module_id = 1,
    .synth_name = dag.name,
    .compiled_synth = compiled_synth,
    .module = compiled_synth.instantiate(),
  });
  auto* instance_ptr = instance.get();

  rt_command start{
    .sample_offset = 0,
    .payload = rt_payload{start_module{
      .module_id = instance_ptr->module_id,
      .module = instance_ptr,
      .insert_mode = module_insert_mode::append,
      .anchor_module_id = 0,
    }},
  };
  if (!rt.try_enqueue(start)) {
    throw pybind11::value_error("rt command queue overflow");
  }

  std::vector<float> output(static_cast<size_t>(validated_output_channels) * validated_block_size, 0.0f);
  pybind11::list blocks_module_ids;
  for (auto step : control_steps) {
    enqueue_controls(rt, compiled_synth, instance_ptr->module_id, step);
    std::fill(output.begin(), output.end(), 0.0f);
    rt.process(output.data(), nullptr, static_cast<uint32_t>(validated_block_size));

    pybind11::list ids;
    for (auto module_id : rt.module_ids()) {
      ids.append(module_id);
    }
    blocks_module_ids.append(std::move(ids));

    retire_token token;
    while (rt.try_pop_retired(token)) {}
  }

  return blocks_module_ids;
}

} // namespace mc1

namespace py = pybind11;

PYBIND11_MODULE(_test, m, py::mod_gil_not_used())
{
  using namespace mc1;

  m.doc() = "MiniCollider test helpers";
  m.def("_compiled_controls", &compiled_controls,
      py::arg("dag_bytes"),
      py::arg("sample_rate") = 44100,
      py::arg("block_size") = 32);
  m.def("_render_blocks", &render_blocks,
      py::arg("dag_bytes"),
      py::arg("blocks"),
      py::arg("sample_rate") = 44100,
      py::arg("block_size") = 32,
      py::arg("output_channels") = 2);
  m.def("_render_control_blocks", &render_control_blocks,
      py::arg("dag_bytes"),
      py::arg("control_steps"),
      py::arg("sample_rate") = 44100,
      py::arg("block_size") = 32,
      py::arg("output_channels") = 2);
  m.def("_runtime_module_ids_per_block", &runtime_module_ids_per_block,
      py::arg("dag_bytes"),
      py::arg("control_steps"),
      py::arg("sample_rate") = 44100,
      py::arg("block_size") = 32,
      py::arg("output_channels") = 2);
  m.def("_aligned_abus_modulo", &aligned_abus_modulo,
      py::arg("block_size") = 32,
      py::arg("channels") = 2);
}
