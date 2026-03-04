#pragma once

#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>


#include <nlohmann/json.hpp>

namespace mc1 {

struct DAG final {
  std::string name;
  std::vector<float> constants;
  std::vector<float> controls;
  struct op {
    std::string name;
    char rate;
    size_t num_out;
    std::vector<size_t> args;

    static std::optional<op> parse(std::span<const std::byte>&);
  };
  std::vector<op> ops;

  static std::optional<DAG> parse(std::span<const std::byte>&);
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DAG::op, name, rate, num_out, args)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DAG, name, constants, controls, ops)

std::ostream& operator<<(std::ostream&, const DAG&);

}
