// A minimalistic "library" to parse data from bytes without UB.

#pragma once

#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace mlang {

template<typename T>
concept TriviallyCopyable = std::is_trivially_copyable_v<T>;

template<TriviallyCopyable T>
std::optional<T> get_value(std::span<const std::byte> &span)
{
  if (sizeof(T) > span.size()) return std::nullopt;

  std::optional<std::remove_const_t<T>> result{std::in_place};
  std::memcpy(std::addressof(result.value()), span.data(), sizeof(T));
  span = span.subspan(sizeof(T));

  return result;
}

template<TriviallyCopyable T>
std::optional<std::vector<T>> get_values(std::span<const std::byte> &span, size_t n)
{
  const auto size = sizeof(T) * n;
  if (size > span.size()) return std::nullopt;

  std::optional<std::vector<std::remove_const_t<T>>> result(std::in_place, n);
  std::memcpy(result->data(), span.data(), size);
  span = span.subspan(size);

  return result;
}

inline std::optional<std::string>
get_pstring(std::span<const std::byte> &span)
{
  auto const size = get_value<const unsigned char>(span);
  if (!size || size.value() > span.size()) return std::nullopt;

  std::optional<std::string> result(std::in_place,
    std::string::size_type(size.value()), std::string::value_type(0)
  );
  std::memcpy(result->data(), span.data(), size.value());
  span = span.subspan(size.value());

  return result;
}

}
