#pragma once

#include <chrono>
#include <cstdint>
#include <type_traits>
#include <variant>

namespace mc1 {

using duration = std::chrono::duration<double>;
using time_point = std::chrono::utc_time<duration>;

struct synth_instance;

enum class synth_insert_mode : uint8_t { append, prepend, before, after };

struct start_synth final {
  uint32_t synth_id{};
  synth_instance* synth{};
  synth_insert_mode insert_mode{synth_insert_mode::append};
  uint32_t anchor_synth_id{};
};

struct stop_synth final {
  uint32_t synth_id{};
};

struct set_control_value final {
  uint32_t synth_id{};
  uint32_t control_index{};
  float value{};
};

struct query_idle_status final {
  uint32_t request_id{};
};

using rt_payload = std::variant<
  start_synth,
  stop_synth,
  set_control_value,
  query_idle_status
>;

struct rt_command final {
  time_point when{};
  rt_payload payload{};
};

struct synth_retired_event final {
  uint32_t synth_id{};
};

struct idle_status_event final {
  uint32_t request_id{};
  bool idle{};
};

using rt_event_payload = std::variant<
  synth_retired_event,
  idle_status_event
>;

struct rt_event final {
  rt_event_payload payload{};
};

static_assert(std::is_trivially_copyable_v<start_synth>);
static_assert(std::is_trivially_copyable_v<stop_synth>);
static_assert(std::is_trivially_copyable_v<set_control_value>);
static_assert(std::is_trivially_copyable_v<query_idle_status>);
static_assert(std::is_trivially_copyable_v<rt_payload>);
static_assert(std::is_trivially_destructible_v<rt_payload>);
static_assert(std::is_standard_layout_v<rt_payload>);
static_assert(std::is_trivially_copyable_v<rt_command>);
static_assert(std::is_trivially_destructible_v<rt_command>);
static_assert(std::is_trivially_copyable_v<synth_retired_event>);
static_assert(std::is_trivially_copyable_v<idle_status_event>);
static_assert(std::is_trivially_copyable_v<rt_event_payload>);
static_assert(std::is_trivially_destructible_v<rt_event_payload>);
static_assert(std::is_standard_layout_v<rt_event_payload>);
static_assert(std::is_trivially_copyable_v<rt_event>);
static_assert(std::is_trivially_destructible_v<rt_event>);

} // namespace mc1
