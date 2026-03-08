#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "osc.hpp"

int main()
{
  using namespace mc1::osc;

  {
    std::vector<argument> args{
      int32_t(42), 0.25f, std::string("abc"), blob{std::vector<std::byte>{std::byte(1), std::byte(2)}}
    };
    auto bytes = encode_message("/mc1/test", std::span<const argument>(args));
    auto parsed = decode_packet(bytes);
    if (!parsed) return 1;
    auto msg = std::get_if<message>(&parsed.value());
    if (!msg) return 2;
    if (msg->address != "/mc1/test") return 3;
    if (msg->args.size() != 4) return 4;
  }

  {
    auto elem = encode_message("/mc1/test", std::span<const argument>());
    std::vector<std::vector<std::byte>> elements{elem};
    auto bytes = encode_bundle(1, std::span<const std::vector<std::byte>>(elements));
    auto parsed = decode_packet(bytes);
    if (!parsed) return 5;
    auto b = std::get_if<bundle>(&parsed.value());
    if (!b) return 6;
    if (b->timetag != 1) return 7;
    if (b->elements.size() != 1) return 8;
  }

  {
    std::vector<std::byte> invalid{
      std::byte('/'), std::byte('b'), std::byte('a'), std::byte('d'),
      std::byte(0), std::byte(0), std::byte(0), std::byte(0)
    };
    if (decode_packet(invalid).has_value()) return 9;
  }

  return 0;
}
