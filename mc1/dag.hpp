#pragma once

#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace MiniCollider {

struct dag final {
  std::vector<float> constants;
  std::vector<float> controls;
  struct op {
    std::string name;
    char rate;
    std::vector<unsigned short> args;

    static std::optional<op> parse(std::span<const std::byte>&);
  };
  std::vector<op> ops;

  static std::optional<dag> parse(std::span<const std::byte>&);
};

std::ostream& operator<<(std::ostream&, const dag&);

}
