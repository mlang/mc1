#include "runtime.hpp"

#include "audio.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iterator>
#include <variant>

namespace mc1 {

namespace {

template<typename... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};

template<typename... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

constexpr uint64_t osc_immediate_time_tag = 1;
constexpr uint64_t ntp_epoch_offset_seconds = 2'208'988'800ULL;
constexpr uint64_t nanoseconds_per_second = 1'000'000'000ULL;
constexpr uint64_t ntp_fraction_scale = 1ULL << 32;

uint64_t unix_nanoseconds_to_ntp_time_tag(uint64_t unix_nanoseconds) noexcept
{
  const uint64_t unix_seconds = unix_nanoseconds / nanoseconds_per_second;
  const uint64_t remainder_nanoseconds = unix_nanoseconds % nanoseconds_per_second;
  const uint64_t ntp_seconds = unix_seconds + ntp_epoch_offset_seconds;
  const uint64_t ntp_fraction = (remainder_nanoseconds * ntp_fraction_scale) / nanoseconds_per_second;
  return (ntp_seconds << 32) | ntp_fraction;
}

uint64_t current_time_tag() noexcept
{
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const auto unix_nanoseconds = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
  return unix_nanoseconds_to_ntp_time_tag(unix_nanoseconds);
}

bool is_due_time_tag(uint64_t command_time_tag, uint64_t now_time_tag) noexcept
{
  return command_time_tag == osc_immediate_time_tag || command_time_tag <= now_time_tag;
}

void ma_trampoline(
    ma_device* device,
    void* output,
    const void* input,
    ma_uint32 frame_count)
{
  static_cast<runtime*>(device->pUserData)->process(static_cast<float *>(output), static_cast<const float *>(input), frame_count);
}

} // namespace

runtime::runtime(
    uint32_t sample_rate,
    size_t block_size,
    uint32_t input_channels,
    uint32_t output_channels)
: block_size_{block_size}
, input_channels_{input_channels}
, output_channels_{output_channels}
, abus_(static_cast<size_t>(input_channels + output_channels) * block_size)
, audio_device_{std::make_unique<audio_device>(
      input_channels_,
      output_channels_,
      &ma_trampoline,
      this,
      sample_rate,
      static_cast<uint32_t>(block_size_))}
{}

runtime::~runtime() = default;

bool runtime::try_enqueue(const rt_command& command) noexcept
{ return commands_.push(command); }

bool runtime::try_pop_event(rt_event& event) noexcept
{ return events_.pop(event); }

bool runtime::push_event(const rt_event& event) noexcept
{ return events_.push(event); }

bool runtime::retire_module(uint32_t module_id) noexcept
{
  return push_event(rt_event{
    .payload = rt_event_payload{module_retired_event{.module_id = module_id}},
  });
}

void runtime::collect_commands() noexcept
{
  rt_command cmd;
  while (scheduled_commands_.size() < scheduled_commands_.capacity() && commands_.pop(cmd)) {
    auto insert_at = std::upper_bound(
        scheduled_commands_.begin(),
        scheduled_commands_.end(),
        cmd.time_tag,
        [](uint64_t time_tag, const scheduled_command& scheduled) {
          return time_tag < scheduled.command.time_tag;
        });
    scheduled_commands_.insert(insert_at, scheduled_command{
      .sequence = next_sequence_++,
      .command = cmd,
    });
  }
}

void runtime::consume_due_commands(uint64_t now_time_tag) noexcept
{
  collect_commands();

  while (!scheduled_commands_.empty()) {
    auto& scheduled = scheduled_commands_.front();
    if (!is_due_time_tag(scheduled.command.time_tag, now_time_tag)) break;

    auto command = scheduled.command;
    scheduled_commands_.erase(scheduled_commands_.begin());
    apply_command(command);
  }
}

void runtime::start()
{ audio_device_->start(); }

void runtime::stop() noexcept
{ audio_device_->stop(); }

bool runtime::started() const noexcept
{ return audio_device_->started(); }

bool runtime::idle() noexcept
{
  collect_commands();
  return commands_.empty() && scheduled_commands_.empty() && modules_.empty();
}

std::vector<uint32_t> runtime::module_ids()
{
  return module_ids(current_time_tag());
}

std::vector<uint32_t> runtime::module_ids(uint64_t now_time_tag)
{
  consume_due_commands(now_time_tag);

  return modules_
    | std::views::transform([](module_instance* module) { return module->module_id; })
    | std::ranges::to<std::vector>();
}

module_instance* runtime::find_module(uint32_t module_id) noexcept
{
  auto it = std::find_if(modules_.begin(), modules_.end(), [module_id](module_instance* module) {
    return module != nullptr && module->module_id == module_id;
  });
  return it == modules_.end() ? nullptr : *it;
}

void runtime::apply_command(const rt_command& command) noexcept
{
  std::visit(overloaded{
    [this](const start_module& payload) noexcept {
      if (find_module(payload.module_id) != nullptr) {
        retire_module(payload.module_id);
        return;
      }
      if (modules_.size() >= modules_.capacity()) {
        retire_module(payload.module_id);
        return;
      }

      auto insert_at = modules_.end();
      switch (payload.insert_mode) {
      case module_insert_mode::append:
        insert_at = modules_.end();
        break;
      case module_insert_mode::prepend:
        insert_at = modules_.begin();
        break;
      case module_insert_mode::before:
      case module_insert_mode::after: {
        auto anchor = std::find_if(
            modules_.begin(),
            modules_.end(),
            [&payload](module_instance* module) {
              return module != nullptr && module->module_id == payload.anchor_module_id;
            });
        if (anchor == modules_.end()) {
          retire_module(payload.module_id);
          return;
        }
        insert_at = payload.insert_mode == module_insert_mode::before ? anchor : std::next(anchor);
        break;
      }
      }

      modules_.insert(insert_at, payload.module);
    },
    [this](const stop_module& payload) noexcept {
      auto it = std::find_if(modules_.begin(), modules_.end(), [&payload](module_instance* module) {
        return module != nullptr && module->module_id == payload.module_id;
      });
      if (it == modules_.end()) return;

      auto* module = *it;
      if (!retire_module(module->module_id)) return;

      modules_.erase(it);
    },
    [this](const set_control_value& payload) noexcept {
      auto* module = find_module(payload.module_id);
      if (module == nullptr) return;
      module->module.set_control(payload.control_index, payload.value);
    },
    [this](const query_idle_status& payload) noexcept {
      push_event(rt_event{
        .payload = rt_event_payload{idle_status_event{
          .request_id = payload.request_id,
          .idle = idle(),
        }},
      });
    },
  }, command.payload);
}

void runtime::render_block(float* output, const float* input, size_t frame_offset) noexcept
{
  const size_t input_channels = static_cast<size_t>(input_channels_);
  const size_t output_channels = static_cast<size_t>(output_channels_);

  assert(is_aligned(abus_.data(), audio_buffer_alignment));
  abus_.fill(0.0f);

  if (input != nullptr) {
    for (size_t frame = 0; frame < block_size_; ++frame) {
      const size_t src_frame = frame_offset + frame;

      for (size_t channel = 0; channel < input_channels; ++channel) {
        abus_[(output_channels + channel) * block_size_ + frame] = input[src_frame * input_channels + channel];
      }
    }
  }

  for (size_t index = 0; index < modules_.size();) {
    auto* module = modules_[index];
    module->module.process(abus_.data());

    if (module->module.should_remove() && retire_module(module->module_id)) {
      modules_.erase(modules_.begin() + static_cast<std::ptrdiff_t>(index));
      continue;
    }

    ++index;
  }

  if (output == nullptr) return;

  for (size_t frame = 0; frame < block_size_; ++frame) {
    const size_t dst_frame = frame_offset + frame;
    for (size_t channel = 0; channel < output_channels; ++channel) {
      output[dst_frame * output_channels + channel] = abus_[channel * block_size_ + frame];
    }
  }
}

void runtime::process(float* output, const float* input, uint32_t frame_count)
{
  process(output, input, frame_count, 0);
}

void runtime::process(float* output, const float* input, uint32_t frame_count, uint64_t now_time_tag)
{
  assert(frame_count % block_size_ == 0);

  const size_t total_frames = static_cast<size_t>(frame_count);

  for (size_t frame_offset = 0; frame_offset < total_frames; frame_offset += block_size_) {
    consume_due_commands(now_time_tag == 0 ? current_time_tag() : now_time_tag);
    render_block(output, input, frame_offset);
  }
}

} // namespace mc1
