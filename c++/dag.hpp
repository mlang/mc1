#pragma once

#include "mlang/bytes.hpp"

#include <iostream>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>


namespace mc1 {

enum class control_kind : uint8_t {
  value = 0,
  trigger = 1,
};

struct DAG final {
  struct ControlName {
    std::string name;
    size_t index;
    control_kind kind{control_kind::value};
  };

  std::string name;
  std::vector<float> constants;
  std::vector<float> controls;
  std::vector<ControlName> controlNames;
  struct op {
    std::string name;
    char rate;
    size_t num_out;
    std::vector<size_t> args;

    static std::expected<op, mlang::parse_error> parse(std::span<const std::byte>&);
  };
  std::vector<op> ops;

  static std::expected<DAG, mlang::parse_error> parse(std::span<const std::byte>&);
};

std::ostream& operator<<(std::ostream&, const DAG&);

}
