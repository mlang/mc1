#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <span>
#include <string_view>
#include <type_traits>
#include <variant>

namespace mc1::osc {

enum class decode_error {
  truncated,
  invalid_padding,
  invalid_address,
  invalid_typetags,
  invalid_bundle,
  unsupported_typetag,
  trailing_bytes,
};

struct blob_view final {
  std::span<const std::byte> bytes;
};

using atom_view = std::variant<int32_t, float, std::string_view, blob_view>;

namespace detail {

inline size_t pad4(size_t n)
{
  return (n + 3) & ~size_t(3);
}

template<typename T>
inline std::expected<T, decode_error> read_be(std::span<const std::byte> bytes, size_t& offset)
{
  static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
  if (offset + sizeof(T) > bytes.size()) {
    return std::unexpected(decode_error::truncated);
  }

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
    U value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    if constexpr (std::endian::native == std::endian::little) value = std::byteswap(value);
    offset += sizeof(T);
    return static_cast<T>(value);
  }
}

inline std::expected<std::string_view, decode_error> read_padded_string(
    std::span<const std::byte> bytes,
    size_t& offset)
{
  if (offset >= bytes.size()) {
    return std::unexpected(decode_error::truncated);
  }

  size_t nul = offset;
  while (nul < bytes.size() && bytes[nul] != std::byte(0)) ++nul;
  if (nul == bytes.size()) {
    return std::unexpected(decode_error::truncated);
  }

  const size_t end = pad4(nul + 1);
  if (end > bytes.size()) {
    return std::unexpected(decode_error::truncated);
  }
  for (size_t i = nul + 1; i < end; ++i) {
    if (bytes[i] != std::byte(0)) {
      return std::unexpected(decode_error::invalid_padding);
    }
  }

  auto* begin = reinterpret_cast<const char*>(bytes.data() + offset);
  auto text = std::string_view(begin, nul - offset);
  offset = end;
  return text;
}

inline std::expected<std::span<const std::byte>, decode_error> read_blob(
    std::span<const std::byte> bytes,
    size_t& offset)
{
  auto size = read_be<int32_t>(bytes, offset);
  if (!size) return std::unexpected(size.error());
  if (*size < 0) return std::unexpected(decode_error::truncated);

  const size_t n = static_cast<size_t>(*size);
  if (offset + n > bytes.size()) return std::unexpected(decode_error::truncated);

  const size_t end_raw = offset + n;
  const size_t end = pad4(end_raw);
  if (end > bytes.size()) return std::unexpected(decode_error::truncated);
  for (size_t i = end_raw; i < end; ++i) {
    if (bytes[i] != std::byte(0)) {
      return std::unexpected(decode_error::invalid_padding);
    }
  }

  auto result = bytes.subspan(offset, n);
  offset = end;
  return result;
}

} // namespace detail

class arg_cursor final
{
public:
  arg_cursor(std::string_view typetags, std::span<const std::byte> bytes)
  : typetags_{typetags}
  , bytes_{bytes}
  {}

  bool empty() const noexcept
  {
    return tag_index_ >= typetags_.size();
  }

  std::expected<atom_view, decode_error> next()
  {
    if (empty()) {
      return std::unexpected(decode_error::trailing_bytes);
    }

    switch (typetags_[tag_index_++]) {
    case 'i': {
      auto value = detail::read_be<int32_t>(bytes_, offset_);
      if (!value) return std::unexpected(value.error());
      return atom_view{*value};
    }
    case 'f': {
      auto value = detail::read_be<float>(bytes_, offset_);
      if (!value) return std::unexpected(value.error());
      return atom_view{*value};
    }
    case 's': {
      auto value = detail::read_padded_string(bytes_, offset_);
      if (!value) return std::unexpected(value.error());
      return atom_view{*value};
    }
    case 'b': {
      auto value = detail::read_blob(bytes_, offset_);
      if (!value) return std::unexpected(value.error());
      return atom_view{blob_view{*value}};
    }
    default:
      return std::unexpected(decode_error::unsupported_typetag);
    }
  }

  size_t bytes_consumed() const noexcept
  {
    return offset_;
  }

private:
  std::string_view typetags_;
  std::span<const std::byte> bytes_;
  size_t tag_index_ = 0;
  size_t offset_ = 0;
};

struct message_view final {
  std::string_view address;
  std::string_view typetags;
  std::span<const std::byte> argument_bytes;

  arg_cursor arguments() const
  {
    return arg_cursor{typetags, argument_bytes};
  }
};

struct bundle_element_view final {
  std::span<const std::byte> packet_bytes;
};

class bundle_cursor final
{
public:
  explicit bundle_cursor(std::span<const std::byte> bytes)
  : bytes_{bytes}
  {}

  bool empty() const noexcept
  {
    return offset_ >= bytes_.size();
  }

  std::expected<bundle_element_view, decode_error> next()
  {
    auto size = detail::read_be<int32_t>(bytes_, offset_);
    if (!size) return std::unexpected(size.error());
    if (*size <= 0) return std::unexpected(decode_error::invalid_bundle);

    const size_t n = static_cast<size_t>(*size);
    if (offset_ + n > bytes_.size()) {
      return std::unexpected(decode_error::truncated);
    }

    auto packet_bytes = bytes_.subspan(offset_, n);
    offset_ += n;
    return bundle_element_view{packet_bytes};
  }

private:
  std::span<const std::byte> bytes_;
  size_t offset_ = 0;
};

struct bundle_view final {
  uint64_t timetag;
  std::span<const std::byte> element_bytes;

  bundle_cursor elements() const
  {
    return bundle_cursor{element_bytes};
  }
};

using packet_view = std::variant<message_view, bundle_view>;

inline std::expected<packet_view, decode_error> decode_packet_view(std::span<const std::byte> bytes);

inline std::expected<message_view, decode_error> decode_message_view(std::span<const std::byte> bytes)
{
  size_t offset = 0;

  auto address = detail::read_padded_string(bytes, offset);
  if (!address) return std::unexpected(address.error());
  if (address->empty() || address->front() != '/') {
    return std::unexpected(decode_error::invalid_address);
  }

  auto typetags = detail::read_padded_string(bytes, offset);
  if (!typetags) return std::unexpected(typetags.error());
  if (typetags->empty() || typetags->front() != ',') {
    return std::unexpected(decode_error::invalid_typetags);
  }

  auto args = bytes.subspan(offset);
  auto cursor = arg_cursor{typetags->substr(1), args};
  while (!cursor.empty()) {
    auto atom = cursor.next();
    if (!atom) return std::unexpected(atom.error());
  }
  if (cursor.bytes_consumed() != args.size()) {
    return std::unexpected(decode_error::trailing_bytes);
  }

  return message_view{
    *address,
    typetags->substr(1),
    args,
  };
}

inline std::expected<bundle_view, decode_error> decode_bundle_view(std::span<const std::byte> bytes)
{
  size_t offset = 0;

  auto marker = detail::read_padded_string(bytes, offset);
  if (!marker) return std::unexpected(marker.error());
  if (*marker != "#bundle") {
    return std::unexpected(decode_error::invalid_bundle);
  }

  auto timetag = detail::read_be<uint64_t>(bytes, offset);
  if (!timetag) return std::unexpected(timetag.error());

  auto elements = bytes.subspan(offset);
  auto cursor = bundle_cursor{elements};
  while (!cursor.empty()) {
    auto element = cursor.next();
    if (!element) return std::unexpected(element.error());
    auto packet = decode_packet_view(element->packet_bytes);
    if (!packet) return std::unexpected(packet.error());
  }

  return bundle_view{
    *timetag,
    elements,
  };
}

inline std::expected<packet_view, decode_error> decode_packet_view(std::span<const std::byte> bytes)
{
  size_t offset = 0;
  auto marker = detail::read_padded_string(bytes, offset);
  if (!marker) return std::unexpected(marker.error());
  if (*marker == "#bundle") {
    auto bundle = decode_bundle_view(bytes);
    if (!bundle) return std::unexpected(bundle.error());
    return packet_view{*bundle};
  }

  auto message = decode_message_view(bytes);
  if (!message) return std::unexpected(message.error());
  return packet_view{*message};
}

} // namespace mc1::osc
