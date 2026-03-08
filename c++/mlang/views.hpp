#pragma once

#include <ranges>

namespace mlang::views {

template<typename T>
constexpr auto sampled_interval(T period, size_t n)
{
  return
    std::views::iota(size_t{}, n)
  | std::views::transform([delta = period / n](size_t i) { return delta * i; })
  ;
}

}
