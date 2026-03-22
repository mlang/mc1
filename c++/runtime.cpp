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

bool runtime::retire_synth(uint32_t synth_id) noexcept
{
  return push_event(rt_event{
    .payload = rt_event_payload{synth_retired_event{.synth_id = synth_id}},
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
  return commands_.empty() && scheduled_commands_.empty() && synths_.empty();
}

std::vector<uint32_t> runtime::synth_ids()
{
  return synth_ids(current_time_tag());
}

std::vector<uint32_t> runtime::synth_ids(uint64_t now_time_tag)
{
  consume_due_commands(now_time_tag);

  return synths_
    | std::views::transform([](synth_instance* synth) { return synth->synth_id; })
    | std::ranges::to<std::vector>();
}

synth_instance* runtime::find_synth(uint32_t synth_id) noexcept
{
  auto it = std::find_if(synths_.begin(), synths_.end(), [synth_id](synth_instance* synth) {
    return synth != nullptr && synth->synth_id == synth_id;
  });
  return it == synths_.end() ? nullptr : *it;
}

void runtime::apply_command(const rt_command& command) noexcept
{
  std::visit(overloaded{
    [this](const start_synth& payload) noexcept {
      if (find_synth(payload.synth_id) != nullptr) {
        retire_synth(payload.synth_id);
        return;
      }
      if (synths_.size() >= synths_.capacity()) {
        retire_synth(payload.synth_id);
        return;
      }

      auto insert_at = synths_.end();
      switch (payload.insert_mode) {
      case synth_insert_mode::append:
        insert_at = synths_.end();
        break;
      case synth_insert_mode::prepend:
        insert_at = synths_.begin();
        break;
      case synth_insert_mode::before:
      case synth_insert_mode::after: {
        auto anchor = std::find_if(
            synths_.begin(),
            synths_.end(),
            [&payload](synth_instance* synth) {
              return synth != nullptr && synth->synth_id == payload.anchor_synth_id;
            });
        if (anchor == synths_.end()) {
          retire_synth(payload.synth_id);
          return;
        }
        insert_at = payload.insert_mode == synth_insert_mode::before ? anchor : std::next(anchor);
        break;
      }
      }

      synths_.insert(insert_at, payload.synth);
    },
    [this](const stop_synth& payload) noexcept {
      auto it = std::find_if(synths_.begin(), synths_.end(), [&payload](synth_instance* synth) {
        return synth != nullptr && synth->synth_id == payload.synth_id;
      });
      if (it == synths_.end()) return;

      auto* synth = *it;
      if (!retire_synth(synth->synth_id)) return;

      synths_.erase(it);
    },
    [this](const set_control_value& payload) noexcept {
      auto* synth = find_synth(payload.synth_id);
      if (synth == nullptr) return;
      synth->synth.set_control(payload.control_index, payload.value);
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

  for (size_t index = 0; index < synths_.size();) {
    auto* synth = synths_[index];
    synth->synth.process(abus_.data());

    if (synth->synth.should_remove() && retire_synth(synth->synth_id)) {
      synths_.erase(synths_.begin() + static_cast<std::ptrdiff_t>(index));
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
