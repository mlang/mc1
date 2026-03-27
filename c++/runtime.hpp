#pragma once

#include "audio_device.hpp"
#include "audio_buffer.hpp"
#include "compiler.hpp"
#include "rt_command.hpp"

#include <boost/container/static_vector.hpp>
#include <boost/lockfree/spsc_queue.hpp>

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace mc1 {

template<typename T, size_t Capacity>
using fixed_spsc_queue = boost::lockfree::spsc_queue<T, boost::lockfree::capacity<Capacity>>;

using boost::container::static_vector;

struct synth_instance final {
  uint32_t synth_id{};
  std::string synth_name;
  Result::CompiledSynth compiled_synth;
  Result::Synth synth;
};

class runtime final
{
  size_t block_size_;
  uint32_t input_channels_;
  uint32_t output_channels_;
  aligned_float_buffer abus_;
  fixed_spsc_queue<rt_command, 1024> commands_;
  fixed_spsc_queue<rt_event, 1024> events_;
  static_vector<rt_command, 8192> scheduled_commands_{};
  static_vector<synth_instance*, 1024> synths_{};
  audio_device audio_device_;

public:
  runtime(
    uint32_t sample_rate,
    size_t block_size,
    uint32_t input_channels,
    uint32_t output_channels
  );
  ~runtime();

  bool try_enqueue(const rt_command& command) noexcept;
  bool try_pop_event(rt_event& event) noexcept;

  void start();
  void stop() noexcept;
  bool started() const noexcept;
  std::vector<uint32_t> synth_ids();
  std::vector<uint32_t> synth_ids(time_point now);

  void process(float* output, const float* input, uint32_t frame_count);
  void process(float* output, const float* input, uint32_t frame_count, time_point now);

private:
  bool push_event(const rt_event& event) noexcept;
  bool retire_synth(uint32_t synth_id) noexcept;
  void collect_commands() noexcept;
  void consume_due_commands(time_point now) noexcept;
  bool idle() noexcept;
  synth_instance* find_synth(uint32_t synth_id) noexcept;
  void apply_command(const rt_command& command) noexcept;
  void render_block(float* output, const float* input, size_t frame_offset) noexcept;
};

} // namespace mc1
