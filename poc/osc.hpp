#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace mc1::osc {

struct blob final {
  std::vector<std::byte> data;
};

using argument = std::variant<int32_t, float, std::string, blob>;

struct message final {
  std::string address;
  std::vector<argument> args;
};

struct bundle final {
  uint64_t timetag;
  std::vector<std::vector<std::byte>> elements;
};

using packet = std::variant<message, bundle>;

inline std::optional<packet> decode_packet(std::span<const std::byte> bytes);

namespace detail {

inline size_t pad4(size_t n)
{
  return (n + 3) & ~size_t(3);
}

template<typename T>
inline void append_be(std::vector<std::byte> &out, T value)
{
  static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
  if constexpr (std::is_floating_point_v<T>) {
    using U = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
    U bits = 0;
    std::memcpy(&bits, &value, sizeof(T));
    if constexpr (std::endian::native == std::endian::little) bits = std::byteswap(bits);
    auto p = reinterpret_cast<const std::byte*>(&bits);
    out.insert(out.end(), p, p + sizeof(bits));
  } else {
    using U = std::make_unsigned_t<T>;
    U u = static_cast<U>(value);
    if constexpr (std::endian::native == std::endian::little) u = std::byteswap(u);
    auto p = reinterpret_cast<const std::byte*>(&u);
    out.insert(out.end(), p, p + sizeof(u));
  }
}

template<typename T>
inline std::optional<T> read_be(std::span<const std::byte> bytes, size_t &offset)
{
  static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
  if (offset + sizeof(T) > bytes.size()) return std::nullopt;

  if constexpr (std::is_floating_point_v<T>) {
    using U = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
    U bits = 0;
    std::memcpy(&bits, bytes.data() + offset, sizeof(bits));
    if constexpr (std::endian::native == std::endian::little) bits = std::byteswap(bits);
    T value = 0;
    std::memcpy(&value, &bits, sizeof(T));
    offset += sizeof(T);
    return value;
  } else {
    using U = std::make_unsigned_t<T>;
    U u = 0;
    std::memcpy(&u, bytes.data() + offset, sizeof(u));
    if constexpr (std::endian::native == std::endian::little) u = std::byteswap(u);
    offset += sizeof(T);
    return static_cast<T>(u);
  }
}

inline void append_padded_string(std::vector<std::byte> &out, std::string_view text)
{
  out.insert(out.end(),
    reinterpret_cast<const std::byte*>(text.data()),
    reinterpret_cast<const std::byte*>(text.data() + text.size())
  );
  out.push_back(std::byte(0));
  while (out.size() % 4 != 0) out.push_back(std::byte(0));
}

inline std::optional<std::string> read_padded_string(std::span<const std::byte> bytes, size_t &offset)
{
  if (offset >= bytes.size()) return std::nullopt;

  size_t nul = offset;
  while (nul < bytes.size() && bytes[nul] != std::byte(0)) ++nul;
  if (nul == bytes.size()) return std::nullopt;

  const size_t end = pad4(nul + 1);
  if (end > bytes.size()) return std::nullopt;
  for (size_t i = nul + 1; i < end; ++i) {
    if (bytes[i] != std::byte(0)) return std::nullopt;
  }

  std::string result(nul - offset, '\0');
  std::memcpy(result.data(), bytes.data() + offset, result.size());
  offset = end;
  return result;
}

inline void append_blob(std::vector<std::byte> &out, std::span<const std::byte> data)
{
  append_be<int32_t>(out, static_cast<int32_t>(data.size()));
  out.insert(out.end(), data.begin(), data.end());
  while (out.size() % 4 != 0) out.push_back(std::byte(0));
}

inline std::optional<std::vector<std::byte>>
read_blob(std::span<const std::byte> bytes, size_t &offset)
{
  auto size = read_be<int32_t>(bytes, offset);
  if (!size || *size < 0) return std::nullopt;
  const size_t n = static_cast<size_t>(*size);
  if (offset + n > bytes.size()) return std::nullopt;

  const size_t end_raw = offset + n;
  const size_t end = pad4(end_raw);
  if (end > bytes.size()) return std::nullopt;
  for (size_t i = end_raw; i < end; ++i) {
    if (bytes[i] != std::byte(0)) return std::nullopt;
  }

  std::vector<std::byte> result(n);
  std::memcpy(result.data(), bytes.data() + offset, n);
  offset = end;
  return result;
}

} // namespace detail

inline std::optional<message> decode_message(std::span<const std::byte> bytes)
{
  size_t offset = 0;
  auto address = detail::read_padded_string(bytes, offset);
  if (!address || address->empty() || address->front() != '/') return std::nullopt;

  auto typetags = detail::read_padded_string(bytes, offset);
  if (!typetags || typetags->empty() || typetags->front() != ',') return std::nullopt;

  std::vector<argument> args;
  args.reserve(typetags->size() > 0 ? typetags->size() - 1 : 0);
  for (size_t i = 1; i < typetags->size(); ++i) {
    switch ((*typetags)[i]) {
    case 'i': {
      auto v = detail::read_be<int32_t>(bytes, offset);
      if (!v) return std::nullopt;
      args.emplace_back(*v);
      break;
    }
    case 'f': {
      auto v = detail::read_be<float>(bytes, offset);
      if (!v) return std::nullopt;
      args.emplace_back(*v);
      break;
    }
    case 's': {
      auto v = detail::read_padded_string(bytes, offset);
      if (!v) return std::nullopt;
      args.emplace_back(std::move(*v));
      break;
    }
    case 'b': {
      auto v = detail::read_blob(bytes, offset);
      if (!v) return std::nullopt;
      args.emplace_back(blob{std::move(*v)});
      break;
    }
    default:
      return std::nullopt;
    }
  }

  if (offset != bytes.size()) return std::nullopt;
  return message{std::move(*address), std::move(args)};
}

inline std::optional<bundle> decode_bundle(std::span<const std::byte> bytes)
{
  size_t offset = 0;
  auto marker = detail::read_padded_string(bytes, offset);
  if (!marker || *marker != "#bundle") return std::nullopt;

  auto timetag = detail::read_be<uint64_t>(bytes, offset);
  if (!timetag) return std::nullopt;

  std::vector<std::vector<std::byte>> elements;
  while (offset < bytes.size()) {
    auto size = detail::read_be<int32_t>(bytes, offset);
    if (!size || *size <= 0) return std::nullopt;
    const size_t n = static_cast<size_t>(*size);
    if (offset + n > bytes.size()) return std::nullopt;

    std::vector<std::byte> element(n);
    std::memcpy(element.data(), bytes.data() + offset, n);
    offset += n;

    if (!decode_packet(element)) return std::nullopt;
    elements.emplace_back(std::move(element));
  }

  return bundle{*timetag, std::move(elements)};
}

inline std::optional<packet> decode_packet(std::span<const std::byte> bytes)
{
  size_t offset = 0;
  auto marker = detail::read_padded_string(bytes, offset);
  if (!marker) return std::nullopt;
  if (*marker == "#bundle") {
    auto b = decode_bundle(bytes);
    if (!b) return std::nullopt;
    return packet{std::move(*b)};
  }

  auto m = decode_message(bytes);
  if (!m) return std::nullopt;
  return packet{std::move(*m)};
}

inline std::vector<std::byte> encode_message(std::string_view address, std::span<const argument> args)
{
  if (address.empty() || address.front() != '/') return {};

  std::vector<std::byte> out;
  detail::append_padded_string(out, address);

  std::string typetags = ",";
  typetags.reserve(args.size() + 1);
  for (const auto& arg : args) {
    if (std::holds_alternative<int32_t>(arg)) typetags.push_back('i');
    else if (std::holds_alternative<float>(arg)) typetags.push_back('f');
    else if (std::holds_alternative<std::string>(arg)) typetags.push_back('s');
    else if (std::holds_alternative<blob>(arg)) typetags.push_back('b');
  }
  detail::append_padded_string(out, typetags);

  for (const auto& arg : args) {
    if (auto v = std::get_if<int32_t>(&arg)) detail::append_be<int32_t>(out, *v);
    else if (auto v = std::get_if<float>(&arg)) detail::append_be<float>(out, *v);
    else if (auto v = std::get_if<std::string>(&arg)) detail::append_padded_string(out, *v);
    else if (auto v = std::get_if<blob>(&arg)) detail::append_blob(out, v->data);
  }

  return out;
}

inline std::vector<std::byte> encode_bundle(uint64_t timetag, std::span<const std::vector<std::byte>> elements)
{
  std::vector<std::byte> out;
  detail::append_padded_string(out, "#bundle");
  detail::append_be<uint64_t>(out, timetag);

  for (const auto& element : elements) {
    detail::append_be<int32_t>(out, static_cast<int32_t>(element.size()));
    out.insert(out.end(), element.begin(), element.end());
  }
  return out;
}

inline std::vector<std::byte> encode_packet(const packet &p)
{
  if (auto m = std::get_if<message>(&p)) {
    return encode_message(m->address, std::span<const argument>(m->args));
  }
  const auto &b = std::get<bundle>(p);
  return encode_bundle(b.timetag, std::span<const std::vector<std::byte>>(b.elements));
}

} // namespace mc1::osc
