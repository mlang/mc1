#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "compiler.hpp"
#include "dag.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace mc1 {

namespace {

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

} // namespace

pybind11::list compiled_controls(
    pybind11::bytes b,
    long sample_rate = 44100,
    long block_size = 32)
{
  auto bytes = std::as_bytes(std::span(std::string_view(b)));
  auto dag = DAG::parse(bytes);
  if (!dag) {
    throw pybind11::value_error("invalid DAG bytes");
  }

  auto result = compile(
      *dag,
      validate_positive_arg(sample_rate, "sample_rate"),
      static_cast<size_t>(validate_positive_arg(block_size, "block_size")));
  auto compiled_synth = result[dag->name];

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

pybind11::list render_blocks(
    pybind11::bytes b,
    long blocks,
    long sample_rate = 44100,
    long block_size = 32,
    long output_channels = 2)
{
  auto bytes = std::as_bytes(std::span(std::string_view(b)));
  auto dag = DAG::parse(bytes);
  if (!dag) {
    throw pybind11::value_error("invalid DAG bytes");
  }

  auto validated_blocks = static_cast<size_t>(validate_positive_arg(blocks, "blocks"));
  auto validated_block_size = static_cast<size_t>(validate_positive_arg(block_size, "block_size"));
  auto validated_output_channels = static_cast<size_t>(validate_positive_arg(output_channels, "output_channels"));
  auto result = compile(
      *dag,
      validate_positive_arg(sample_rate, "sample_rate"),
      validated_block_size);
  auto compiled_synth = result[dag->name];
  auto module = compiled_synth.instantiate();

  std::vector<float> abus(validated_output_channels * validated_block_size, 0.0f);
  pybind11::list rendered_blocks;
  for (size_t i = 0; i < validated_blocks; ++i) {
    std::fill(abus.begin(), abus.end(), 0.0f);
    module.process(abus.data());

    pybind11::list rendered_block;
    for (float sample : abus) {
      rendered_block.append(sample);
    }
    rendered_blocks.append(std::move(rendered_block));
  }
  return rendered_blocks;
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
}
