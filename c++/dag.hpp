#pragma once

#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>


namespace mc1 {

struct DAG final {
  struct ControlName {
    std::string name;
    size_t index;
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

    static std::optional<op> parse(std::span<const std::byte>&);
  };
  std::vector<op> ops;

  static std::optional<DAG> parse(std::span<const std::byte>&);
};

std::ostream& operator<<(std::ostream&, const DAG&);

}
