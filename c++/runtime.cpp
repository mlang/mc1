#include "runtime.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iterator>
#include <ranges>
#include <variant>

namespace mc1 {

namespace {

template<typename... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};

void trampoline(
  float *output, const float * input,
  uint32_t frame_count, void *data
)
{ static_cast<runtime*>(data)->process(output, input, frame_count); }

} // namespace

runtime::runtime(
  uint32_t sample_rate,
  size_t block_size,
  uint32_t input_channels,
  uint32_t output_channels)
: block_size_{block_size}
, input_channels_{input_channels}, output_channels_{output_channels}
, abus_(block_size * (input_channels + output_channels))
, audio_device_{input_channels_, output_channels_,
    &trampoline, this, sample_rate, block_size_
  }
{}

runtime::~runtime() = default;

bool runtime::try_enqueue(const rt_command& command) noexcept
{ return commands_.push(command); }

bool runtime::try_pop_event(rt_event& event) noexcept
{ return events_.pop(event); }

bool runtime::push_event(const rt_event& event) noexcept
{ return events_.push(event); }

bool runtime::retire_synth(uint32_t synth_id) noexcept
{ return push_event(synth_retired_event{.synth_id = synth_id}); }

void runtime::collect_commands() noexcept
{
  rt_command cmd;
  auto schedule_not_full = [this]{
    return scheduled_commands_.size() < scheduled_commands_.capacity();
  };
  while (schedule_not_full() && commands_.pop(cmd)) {
    scheduled_commands_.insert(std::ranges::upper_bound(
      scheduled_commands_, cmd.when, std::ranges::less{}, &rt_command::when
    ), cmd);
  }
}

void runtime::consume_due_commands(time_point now) noexcept
{
  collect_commands();

  while (!scheduled_commands_.empty()) {
    auto scheduled = scheduled_commands_.front();
    if (scheduled.when > now) break;

    scheduled_commands_.erase(scheduled_commands_.begin());
    apply_command(scheduled);
  }
}

void runtime::start()
{ audio_device_.start(); }

void runtime::stop() noexcept
{ audio_device_.stop(); }

bool runtime::started() const noexcept
{ return audio_device_.started(); }

bool runtime::idle() noexcept
{
  collect_commands();
  return commands_.empty() && scheduled_commands_.empty() && synths_.empty();
}

std::vector<uint32_t> runtime::synth_ids()
{ return synth_ids(std::chrono::utc_clock::now()); }

std::vector<uint32_t> runtime::synth_ids(time_point now)
{
  consume_due_commands(now);

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
      push_event(idle_status_event{
        .request_id = payload.request_id,
        .idle = idle(),
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
  process(output, input, frame_count, std::chrono::utc_clock::now());
}

void runtime::process(float* output, const float* input, uint32_t frame_count, time_point now)
{
  assert(frame_count % block_size_ == 0);

  const size_t total_frames = static_cast<size_t>(frame_count);

  const auto block_duration = duration(block_size_) / audio_device_.sample_rate();
  for (size_t frame_offset = 0; frame_offset < total_frames; frame_offset += block_size_) {
    now += block_duration;
    consume_due_commands(now);
    render_block(output, input, frame_offset);
  }
}

} // namespace mc1
