#include "audio_device.hpp"

#include <format>
#include <stdexcept>
#include <string>

#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

namespace mc1 {

namespace {

struct callback_info
{
  audio_device::callback_type *callback;
  void                        *data;
};

}

struct audio_device::implementation final {
  callback_info callback;
  ma_device     device{};
  bool          started = false;
};

namespace {

std::string error_message(const char* action, ma_result result)
{ return std::format("{} failed: {}", action, ma_result_description(result)); }

ma_device_type get_device_type(uint32_t input_channels, uint32_t output_channels)
{
  if (input_channels > 0 && output_channels > 0) {
    return ma_device_type_duplex;
  }
  if (output_channels > 0) {
    return ma_device_type_playback;
  }
  if (input_channels > 0) {
    return ma_device_type_capture;
  }

  throw std::invalid_argument("audio_device requires at least one input or output channel");
}

void ma_trampoline(
  ma_device *device, void *output, const void *input, ma_uint32 frame_count
)
{
  auto &info = *static_cast<const callback_info *>(device->pUserData);
  info.callback(
    static_cast<float *>(output), static_cast<const float *>(input),
    frame_count, info.data
  );
}

} // namespace

audio_device::audio_device(
  uint32_t input_channels, uint32_t output_channels,
  callback_type *callback, void* callback_data,
  uint32_t sample_rate, uint32_t period_frames
)
: impl{std::make_unique<implementation>(
    implementation{
      .callback=callback_info{.callback = callback, .data = callback_data}
    }
  )}
{
  if (callback == nullptr) {
    throw std::invalid_argument("audio_device callback must not be null");
  }
  if (sample_rate == 0) {
    throw std::invalid_argument("audio_device sample_rate must be > 0");
  }
  if (period_frames == 0) {
    throw std::invalid_argument("audio_device period_frames must be > 0");
  }

  auto config = ma_device_config_init(get_device_type(input_channels, output_channels));

  if (input_channels > 0) {
    config.capture.format = ma_format_f32;
    config.capture.channels = input_channels;
  }

  if (output_channels > 0) {
    config.playback.format = ma_format_f32;
    config.playback.channels = output_channels;
  }

  config.sampleRate = sample_rate;
  config.periodSizeInFrames = period_frames;
  config.dataCallback = ma_trampoline;
  config.pUserData = &impl->callback;

  ma_result result = ma_device_init(nullptr, &config, &impl->device);
  if (result != MA_SUCCESS) {
    throw std::runtime_error(error_message("ma_device_init", result));
  }
}

audio_device::~audio_device()
{
  stop();
  ma_device_uninit(&impl->device);
}

void audio_device::start()
{
  if (impl->started) return;

  ma_result result = ma_device_start(&impl->device);
  if (result != MA_SUCCESS) {
    throw std::runtime_error(error_message("ma_device_start", result));
  }

  impl->started = true;
}

void audio_device::stop() noexcept
{
  if (!started()) return;

  ma_result result = ma_device_stop(&impl->device);
  if (result == MA_SUCCESS) {
    impl->started = false;
  }
}

bool audio_device::started() const noexcept
{ return impl->started; }

uint32_t audio_device::sample_rate() const
{ return impl->device.sampleRate; }

} // namespace mc1
