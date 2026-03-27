#pragma once

#include <cstdint>
#include <memory>


namespace mc1 {

class audio_device final
{
  struct implementation;
  std::unique_ptr<implementation> impl;

public:
  audio_device(
    uint32_t input_channels,
    uint32_t output_channels,
    void(*callback)(float *, const float *, uint32_t, void*),
    void* callback_data,
    uint32_t sample_rate = 44100,
    uint32_t period_frames = 32
  );

  ~audio_device();

  audio_device(const audio_device&) = delete;
  audio_device& operator=(const audio_device&) = delete;
  audio_device(audio_device&&) = delete;
  audio_device& operator=(audio_device&&) = delete;

  void start();
  void stop() noexcept;
  bool started() const noexcept;

  uint32_t sample_rate() const;
};

} // namespace mc1
