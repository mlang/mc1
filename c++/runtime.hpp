#pragma once

#include "compiler.hpp"
#include "rt_command.hpp"
#include "rt_queue.hpp"

#include <boost/container/static_vector.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace mc1 {

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

using rt_command_queue = fixed_spsc_queue<rt_command, 1024>;
using rt_retire_queue = fixed_spsc_queue<retire_token, 1024>;

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

  void process(float* output, const float* input, uint32_t frame_count);

private:
  static constexpr size_t max_modules = 1024;

  size_t block_size_;
  uint32_t input_channels_;
  uint32_t output_channels_;
  std::vector<float> abus_;
  rt_command_queue commands_;
  rt_retire_queue retired_modules_;
  boost::container::static_vector<module_instance*, max_modules> modules_{};
  std::unique_ptr<audio_device> audio_device_;

  module_instance* find_module(uint32_t module_id) noexcept;
  void apply_command(const rt_command& command) noexcept;
  void render_block(float* output, const float* input, size_t frame_offset) noexcept;
};

} // namespace mc1
