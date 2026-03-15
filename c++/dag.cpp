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
  return get_pstring(bytes).and_then([&](std::string name) {
    return get_value<char>(bytes).and_then([&](char rate) {
      return get_value<size_t>(bytes).and_then([&](size_t num_out) {
        return get_value<size_t>(bytes).and_then([&](size_t nargs) {
          return get_values<size_t>(bytes, nargs);
	}).and_then([&](std::vector<size_t> args) -> std::expected<op, mlang::parse_error> {
          return op{ std::move(name), rate, num_out, std::move(args) };
        });
      });
    });
  });
}

std::expected<DAG, mlang::parse_error> DAG::parse(std::span<const std::byte> &bytes)
{
  return get_pstring(bytes).and_then([&](std::string name) {
    return get_value<size_t>(bytes).and_then([&](size_t nconsts) {
      return get_values<float>(bytes, nconsts);
    }).and_then([&](std::vector<float> consts) {
      return get_value<size_t>(bytes).and_then([&](size_t nctrlvals) {
        return get_values<float>(bytes, nctrlvals);
      }).and_then([&](std::vector<float> ctrlvals) {
        return get_value<size_t>(bytes).and_then([&](size_t nctrlnames) {
          return parse_n<ControlName>(bytes, nctrlnames,
            [](std::span<const std::byte> &bytes) -> std::expected<ControlName, mlang::parse_error>
            {
              return get_pstring(bytes).and_then([&](std::string name) {
                return get_value<size_t>(bytes).and_then([&](size_t index) {
                  return parse_control_kind(bytes).and_then([&](control_kind kind) -> std::expected<ControlName, mlang::parse_error> {
                    return ControlName{ std::move(name), index, kind };
                  });
                });
              });
            }
          );
        }).and_then([&](std::vector<ControlName> control_names) {
          return get_value<size_t>(bytes).and_then([&](size_t nops) {
            return parse_n<op>(bytes, nops, &op::parse);
          }).and_then([&](std::vector<op> ops) -> std::expected<DAG, mlang::parse_error> {
            return DAG{
              std::move(name), std::move(consts), std::move(ctrlvals),
              std::move(control_names), std::move(ops)
            };
          });
        });
      });
    });
  });
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
