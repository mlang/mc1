#include "dag.hpp"

#include "mlang/bytes.hpp"

namespace mc1 {

using mlang::get_value;
using mlang::get_values;
using mlang::get_pstring;

namespace {

std::expected<control_kind, mlang::parse_error>
parse_control_kind(std::span<const std::byte> &bytes)
{
  auto kind = get_value<uint8_t>(bytes);
  if (!kind) {
    return std::unexpected(kind.error());
  }

  switch (*kind) {
  case static_cast<uint8_t>(control_kind::value):
    return control_kind::value;
  case static_cast<uint8_t>(control_kind::trigger):
    return control_kind::trigger;
  default:
    return std::unexpected(mlang::parse_error::invalid_data);
  }
}

template<class T, class Parser>
std::expected<std::vector<T>, mlang::parse_error>
parse_n(std::span<const std::byte> &bytes, size_t count, Parser &&parser)
{
  auto values = std::vector<T>{};
  values.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    auto value = parser(bytes);
    if (!value) {
      return std::unexpected(value.error());
    }
    values.push_back(std::move(*value));
  }
  return values;
}

} // namespace

std::expected<DAG::op, mlang::parse_error> DAG::op::parse(std::span<const std::byte> &bytes)
{
  auto name = get_pstring(bytes);
  if (!name) {
    return std::unexpected(name.error());
  }

  auto rate = get_value<char>(bytes);
  if (!rate) {
    return std::unexpected(rate.error());
  }

  auto num_out = get_value<size_t>(bytes);
  if (!num_out) {
    return std::unexpected(num_out.error());
  }

  auto nargs = get_value<size_t>(bytes);
  if (!nargs) {
    return std::unexpected(nargs.error());
  }

  auto args = get_values<size_t>(bytes, *nargs);
  if (!args) {
    return std::unexpected(args.error());
  }

  return op{
    std::move(*name),
    *rate,
    *num_out,
    std::move(*args),
  };
}

std::expected<DAG, mlang::parse_error> DAG::parse(std::span<const std::byte> &bytes)
{
  auto name = get_pstring(bytes);
  if (!name) {
    return std::unexpected(name.error());
  }

  auto nconsts = get_value<size_t>(bytes);
  if (!nconsts) {
    return std::unexpected(nconsts.error());
  }
  auto consts = get_values<float>(bytes, *nconsts);
  if (!consts) {
    return std::unexpected(consts.error());
  }

  auto nctrlvals = get_value<size_t>(bytes);
  if (!nctrlvals) {
    return std::unexpected(nctrlvals.error());
  }
  auto ctrlvals = get_values<float>(bytes, *nctrlvals);
  if (!ctrlvals) {
    return std::unexpected(ctrlvals.error());
  }

  auto nctrlnames = get_value<size_t>(bytes);
  if (!nctrlnames) {
    return std::unexpected(nctrlnames.error());
  }
  auto control_names = parse_n<ControlName>(bytes, *nctrlnames,
    [](std::span<const std::byte> &bytes) -> std::expected<ControlName, mlang::parse_error>
    {
      auto control_name = get_pstring(bytes);
      if (!control_name) {
        return std::unexpected(control_name.error());
      }

      auto control_index = get_value<size_t>(bytes);
      if (!control_index) {
        return std::unexpected(control_index.error());
      }

      auto control_kind = parse_control_kind(bytes);
      if (!control_kind) {
        return std::unexpected(control_kind.error());
      }

      return ControlName{
        std::move(*control_name),
        *control_index,
        *control_kind,
      };
    });
  if (!control_names) {
    return std::unexpected(control_names.error());
  }

  auto nops = get_value<size_t>(bytes);
  if (!nops) {
    return std::unexpected(nops.error());
  }
  auto ops = parse_n<op>(bytes, *nops, &op::parse);
  if (!ops) {
    return std::unexpected(ops.error());
  }

  return DAG{
    std::move(*name),
    std::move(*consts),
    std::move(*ctrlvals),
    std::move(*control_names),
    std::move(*ops),
  };
}

std::ostream& operator<<(std::ostream& os, const DAG& d)
{
  os << "Name: " << d.name << "\n";
  os << "Constants: ";
  for (const auto& constant : d.constants) {
    os << constant << " ";
  }
  os << "\nControls: ";
  for (const auto& control : d.controls) {
    os << control << " ";
  }
  os << "\nControl Names: ";
  for (const auto& controlName : d.controlNames) {
    os << "(" << controlName.name << ", " << controlName.index << ", "
       << static_cast<int>(controlName.kind) << ") ";
  }
  os << "\nOperations:\n";
  for (const auto& operation : d.ops) {
    os << "  Name: " << operation.name << "\n";
    os << "  Rate: " << operation.rate << "\n";
    os << "  Args: ";
    for (const auto& arg : operation.args) {
      os << arg << " ";
    }
    os << "\n";
  }
  return os;
}

}
