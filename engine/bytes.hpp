#pragma once

#include <optional>
#include <span>

namespace mlang {

template<typename T>
concept TriviallyCopyable = std::is_trivially_copyable_v<T>;

template<TriviallyCopyable T>
std::optional<T> get_value(std::span<const std::byte> &span)
{
  if (sizeof(T) > span.size()) return std::nullopt;

  std::optional<T> result{std::in_place};
  std::memcpy(std::addressof(result.value()), span.data(), sizeof(T));
  span = span.subspan(sizeof(T));

  return result;
}

template<TriviallyCopyable T>
std::optional<std::vector<T>> get_values(std::span<const std::byte> &span, size_t n)
{
  const auto size = sizeof(T) * n;
  if (size > span.size()) return std::nullopt;

  std::optional<std::vector<T>> result(std::in_place, n);
  std::memcpy(result->data(), span.data(), size);
  span = span.subspan(size);

  return result;
}

std::optional<std::string> get_pstring(std::span<const std::byte> &span)
{
  if (auto n = get_value<unsigned char>(span)) {
    if (auto chars = get_values<char>(span, *n)) {
      return std::string{chars->begin(), chars->end()};
    }
  }
  return std::nullopt;
}

}
