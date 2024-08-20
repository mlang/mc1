#include <numbers>

namespace mlang {

template<typename T> inline constexpr T tau = T{2} * std::numbers::pi_v<T>;

}
