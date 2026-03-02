#include "dag.hpp"

#include "mlang/bytes.hpp"

namespace mc1 {

using mlang::get_value;
using mlang::get_values;
using mlang::get_pstring;

std::optional<DAG::op> DAG::op::parse(std::span<const std::byte> &bytes)
{
  if (auto name = get_pstring(bytes)) {
    if (auto rate = get_value<char>(bytes)) {
      if (auto num_out = get_value<size_t>(bytes)) {
        if (auto nargs = get_value<size_t>(bytes)) {
          if (auto args = get_values<size_t>(bytes, nargs.value())) {
            return op{
              std::move(name.value()),
              rate.value(), num_out.value(), std::move(args.value())
            };
          }
        }
      }
    }
  }
  return std::nullopt;
}

std::optional<DAG> DAG::parse(std::span<const std::byte> &bytes)
{
  if (auto n = get_value<const size_t>(bytes)) {
    if (auto consts = get_values<float>(bytes, n.value())) {
      if (auto nctrlvals = get_value<const size_t>(bytes)) {
        if (auto ctrlvals = get_values<float>(bytes, nctrlvals.value())) {
          if (auto nops = get_value<const size_t>(bytes)) {
            std::vector<op> ops;
            ops.reserve(nops.value());
            for (int i = 0; i != nops.value(); i++) {
              auto op = op::parse(bytes);
              if (!op) return std::nullopt;
              ops.emplace_back(std::move(op.value()));
            }

            return DAG{
              std::move(consts.value()),
              std::move(ctrlvals.value()),
              std::move(ops)
            };
          }
        }
      }
    }
  }
  return std::nullopt;
}

std::ostream& operator<<(std::ostream& os, const DAG& d)
{
  os << "Constants: ";
  for (const auto& constant : d.constants) {
    os << constant << " ";
  }
  os << "\nControls: ";
  for (const auto& control : d.controls) {
    os << control << " ";
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
