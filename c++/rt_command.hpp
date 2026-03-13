#pragma once

#include <cstdint>
#include <type_traits>
#include <variant>

namespace mc1 {

struct module_instance;

struct start_module final {
  uint32_t module_id{};
  module_instance* module{};
};

struct stop_module final {
  uint32_t module_id{};
};

struct set_control_value final {
  uint32_t module_id{};
  uint32_t control_index{};
  float value{};
};

using rt_payload = std::variant<
  start_module,
  stop_module,
  set_control_value
>;

struct rt_command final {
  uint32_t sample_offset{};
  rt_payload payload{};
};

static_assert(std::is_trivially_copyable_v<start_module>);
static_assert(std::is_trivially_copyable_v<stop_module>);
static_assert(std::is_trivially_copyable_v<set_control_value>);
static_assert(std::is_trivially_copyable_v<rt_payload>);
static_assert(std::is_trivially_destructible_v<rt_payload>);
static_assert(std::is_standard_layout_v<rt_payload>);
static_assert(std::is_trivially_copyable_v<rt_command>);
static_assert(std::is_trivially_destructible_v<rt_command>);

} // namespace mc1
