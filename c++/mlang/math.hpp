#pragma once

#include <numbers>

namespace mlang::numbers {

template<typename T> inline constexpr T tau_v = T{2} * std::numbers::pi_v<T>;
inline constexpr double tau = tau_v<double>;

}
