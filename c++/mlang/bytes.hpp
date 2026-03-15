// A minimalistic "library" to parse data from bytes without UB.

#pragma once

#include <cstring>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace mlang {

enum class parse_error {
  truncated,
  invalid_data,
};

inline constexpr std::string_view to_string(parse_error error) noexcept
{
  switch (error) {
  case parse_error::truncated:
    return "truncated";
  case parse_error::invalid_data:
    return "invalid data";
  }
  return "unknown";
}

template<typename T>
concept TriviallyCopyable = std::is_trivially_copyable_v<T>;

template<TriviallyCopyable T>
std::expected<std::remove_const_t<T>, parse_error>
get_value(std::span<const std::byte> &span)
{
  if (sizeof(T) > span.size()) {
    return std::unexpected(parse_error::truncated);
  }

  auto result = std::remove_const_t<T>{};
  std::memcpy(std::addressof(result), span.data(), sizeof(T));
  span = span.subspan(sizeof(T));

  return result;
}

template<TriviallyCopyable T>
std::expected<std::vector<std::remove_const_t<T>>, parse_error>
get_values(std::span<const std::byte> &span, size_t n)
{
  const auto size = sizeof(T) * n;
  if (size > span.size()) {
    return std::unexpected(parse_error::truncated);
  }

  auto result = std::vector<std::remove_const_t<T>>(n);
  std::memcpy(result.data(), span.data(), size);
  span = span.subspan(size);

  return result;
}

inline std::expected<std::string, parse_error>
get_pstring(std::span<const std::byte> &span)
{
  auto size = get_value<unsigned char>(span);
  if (!size) {
    return std::unexpected(size.error());
  }
  if (*size > span.size()) {
    return std::unexpected(parse_error::truncated);
  }

  auto result = std::string(
    std::string::size_type(*size), std::string::value_type(0)
  );
  std::memcpy(result.data(), span.data(), *size);
  span = span.subspan(*size);

  return result;
}

}
