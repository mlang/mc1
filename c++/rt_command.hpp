#pragma once

#include <cstdint>
#include <type_traits>
#include <variant>

namespace mc1 {

struct module_instance;

enum class module_insert_mode : uint8_t {
  append,
  prepend,
  before,
  after,
};

struct start_module final {
  uint32_t module_id{};
  module_instance* module{};
  module_insert_mode insert_mode{module_insert_mode::append};
  uint32_t anchor_module_id{};
};

struct stop_module final {
  uint32_t module_id{};
};

struct set_control_value final {
  uint32_t module_id{};
  uint32_t control_index{};
  float value{};
};

struct query_idle_status final {
  uint32_t request_id{};
};

using rt_payload = std::variant<
  start_module,
  stop_module,
  set_control_value,
  query_idle_status
>;

struct rt_command final {
  uint64_t time_tag{};
  rt_payload payload{};
};

struct module_retired_event final {
  uint32_t module_id{};
};

struct idle_status_event final {
  uint32_t request_id{};
  bool idle{};
};

using rt_event_payload = std::variant<
  module_retired_event,
  idle_status_event
>;

struct rt_event final {
  rt_event_payload payload{};
};

static_assert(std::is_trivially_copyable_v<start_module>);
static_assert(std::is_trivially_copyable_v<stop_module>);
static_assert(std::is_trivially_copyable_v<set_control_value>);
static_assert(std::is_trivially_copyable_v<query_idle_status>);
static_assert(std::is_trivially_copyable_v<rt_payload>);
static_assert(std::is_trivially_destructible_v<rt_payload>);
static_assert(std::is_standard_layout_v<rt_payload>);
static_assert(std::is_trivially_copyable_v<rt_command>);
static_assert(std::is_trivially_destructible_v<rt_command>);
static_assert(std::is_trivially_copyable_v<module_retired_event>);
static_assert(std::is_trivially_copyable_v<idle_status_event>);
static_assert(std::is_trivially_copyable_v<rt_event_payload>);
static_assert(std::is_trivially_destructible_v<rt_event_payload>);
static_assert(std::is_standard_layout_v<rt_event_payload>);
static_assert(std::is_trivially_copyable_v<rt_event>);
static_assert(std::is_trivially_destructible_v<rt_event>);

} // namespace mc1
