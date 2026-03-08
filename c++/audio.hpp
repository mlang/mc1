#pragma once

#include <cstdint>

#include <miniaudio.h>

namespace mc1 {

class audio_device final
{
public:
  audio_device(
      uint32_t input_channels,
      uint32_t output_channels,
      ma_device_data_proc callback,
      void* callback_data,
      uint32_t sample_rate = 44100,
      uint32_t period_frames = 32);

  ~audio_device();

  audio_device(const audio_device&) = delete;
  audio_device& operator=(const audio_device&) = delete;
  audio_device(audio_device&&) = delete;
  audio_device& operator=(audio_device&&) = delete;

  void start();
  void stop() noexcept;

private:
  ma_device device_{};
  bool started_ = false;
};

} // namespace mc1
