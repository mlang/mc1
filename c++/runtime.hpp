#pragma once

#include "audio_buffer.hpp"
#include "compiler.hpp"
#include "rt_command.hpp"

#include <boost/container/static_vector.hpp>
#include <boost/lockfree/spsc_queue.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace mc1 {

template<typename T, size_t Capacity>
using fixed_spsc_queue = boost::lockfree::spsc_queue<T, boost::lockfree::capacity<Capacity>>;

class audio_device;

struct module_instance final {
  uint32_t module_id{};
  std::string synth_name;
  Result::CompiledSynth compiled_synth;
  Result::Module module;
};

struct retire_token final {
  module_instance* module{};
};

static_assert(std::is_trivially_copyable_v<retire_token>);
static_assert(std::is_trivially_destructible_v<retire_token>);

class runtime final
{
public:
  runtime(
      uint32_t sample_rate,
      size_t block_size,
      uint32_t input_channels,
      uint32_t output_channels);
  ~runtime();

  bool try_enqueue(const rt_command& command) noexcept;
  bool try_pop_retired(retire_token& token) noexcept;

  void start();
  void stop() noexcept;
  bool started() const noexcept;
  std::vector<uint32_t> module_ids();

  void process(float* output, const float* input, uint32_t frame_count);

private:
  size_t block_size_;
  uint32_t input_channels_;
  uint32_t output_channels_;
  aligned_float_buffer abus_;
  fixed_spsc_queue<rt_command, 1024> commands_;
  fixed_spsc_queue<retire_token, 1024> retired_modules_;
  boost::container::static_vector<module_instance*, 1024> modules_{};
  std::unique_ptr<audio_device> audio_device_;

  bool retire_module(module_instance* module) noexcept;
  void drain_commands() noexcept;
  module_instance* find_module(uint32_t module_id) noexcept;
  void apply_command(const rt_command& command) noexcept;
  void render_block(float* output, const float* input, size_t frame_offset) noexcept;
};

} // namespace mc1
