#include "runtime.hpp"

#include "audio.hpp"

#include <algorithm>
#include <cassert>
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
, abus_(static_cast<size_t>(input_channels + output_channels) * block_size, 0.0f)
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
{
  return commands_.try_push(command);
}

bool runtime::try_pop_retired(retire_token& token) noexcept
{
  return retired_modules_.try_pop(token);
}

void runtime::retire_module(module_instance* module) noexcept
{
  if (module == nullptr) return;
  retired_modules_.try_push(retire_token{module});
}

void runtime::drain_commands() noexcept
{
  rt_command command;
  while (commands_.try_pop(command)) {
    apply_command(command);
  }
}

void runtime::start()
{
  audio_device_->start();
}

void runtime::stop() noexcept
{
  audio_device_->stop();
}

bool runtime::started() const noexcept
{
  return audio_device_->started();
}

std::vector<uint32_t> runtime::module_ids()
{
  drain_commands();

  std::vector<uint32_t> ids;
  ids.reserve(modules_.size());
  for (auto* module : modules_) {
    ids.push_back(module->module_id);
  }
  return ids;
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
  assert(command.sample_offset == 0);

  std::visit(overloaded{
    [this](const start_module& payload) noexcept {
      if (find_module(payload.module_id) != nullptr) {
        retire_module(payload.module);
        return;
      }
      if (modules_.size() >= modules_.capacity()) {
        retire_module(payload.module);
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
          retire_module(payload.module);
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
      retire_token token{module};
      if (!retired_modules_.try_push(token)) return;

      modules_.erase(it);
    },
    [this](const set_control_value& payload) noexcept {
      auto* module = find_module(payload.module_id);
      if (module == nullptr) return;
      module->module.set_control(payload.control_index, payload.value);
    },
  }, command.payload);
}

void runtime::render_block(float* output, const float* input, size_t frame_offset) noexcept
{
  const size_t input_channels = static_cast<size_t>(input_channels_);
  const size_t output_channels = static_cast<size_t>(output_channels_);

  std::fill(abus_.begin(), abus_.end(), 0.0f);

  if (input != nullptr) {
    for (size_t frame = 0; frame < block_size_; ++frame) {
      const size_t src_frame = frame_offset + frame;

      for (size_t channel = 0; channel < input_channels; ++channel) {
        abus_[(output_channels + channel) * block_size_ + frame] = input[src_frame * input_channels + channel];
      }
    }
  }

  for (auto* module : modules_) {
    module->module.process(abus_.data());
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
  assert(frame_count % block_size_ == 0);

  drain_commands();

  const size_t total_frames = static_cast<size_t>(frame_count);

  for (size_t frame_offset = 0; frame_offset < total_frames; frame_offset += block_size_) {
    render_block(output, input, frame_offset);
  }
}

} // namespace mc1
