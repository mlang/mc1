#define MINIAUDIO_IMPLEMENTATION
#include "audio.hpp"

#include <format>
#include <stdexcept>
#include <string>

namespace mc1 {

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

} // namespace

audio_device::audio_device(
    uint32_t input_channels,
    uint32_t output_channels,
    ma_device_data_proc callback,
    void* callback_data,
    uint32_t sample_rate,
    uint32_t period_frames)
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

  ma_device_config config = ma_device_config_init(get_device_type(input_channels, output_channels));

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
  config.dataCallback = callback;
  config.pUserData = callback_data;

  ma_result result = ma_device_init(nullptr, &config, &device_);
  if (result != MA_SUCCESS) {
    throw std::runtime_error(error_message("ma_device_init", result));
  }
}

audio_device::~audio_device()
{
  stop();
  ma_device_uninit(&device_);
}

void audio_device::start()
{
  if (started_) {
    return;
  }

  ma_result result = ma_device_start(&device_);
  if (result != MA_SUCCESS) {
    throw std::runtime_error(error_message("ma_device_start", result));
  }

  started_ = true;
}

void audio_device::stop() noexcept
{
  if (!started_) return;

  ma_result result = ma_device_stop(&device_);
  if (result == MA_SUCCESS) {
    started_ = false;
  }
}

bool audio_device::started() const noexcept
{ return started_; }

} // namespace mc1
