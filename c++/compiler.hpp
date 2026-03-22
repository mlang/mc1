#pragma once

#include "audio_buffer.hpp"
#include "dag.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <libgccjit++.h>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mc1 {

struct transparent_string_hash {
  using is_transparent = void;

  size_t operator()(std::string_view value) const noexcept
  {
    return std::hash<std::string_view>{}(value);
  }

  size_t operator()(const std::string& value) const noexcept
  {
    return operator()(std::string_view{value});
  }

  size_t operator()(const char* value) const noexcept
  {
    return operator()(std::string_view{value});
  }
};

class Result
{
  using init_fn_t = void (*)(void *);
  using process_fn_t = uint32_t (*)(void *, const float *, float *);

public:
  struct ControlDesc {
    std::string name;
    size_t index;
    size_t width;
    control_kind kind{control_kind::value};
  };

  struct SynthDescriptor {
    std::string name;
    std::vector<float> controls;
    std::vector<ControlDesc> control_descs;
  };

  class Synth;

  class CompiledSynth {
  public:
    struct ControlSlot {
      size_t index;
      size_t width;
      control_kind kind{control_kind::value};
    };

    using controls_by_name_map = std::unordered_map<
      std::string,
      ControlSlot,
      transparent_string_hash,
      std::equal_to<>
    >;

  private:
    std::shared_ptr<gcc_jit_result> owner_;
    init_fn_t init_{};
    process_fn_t process_{};
    size_t state_size_{};
    std::vector<float> default_controls_;
    std::vector<ControlDesc> control_descs_;
    controls_by_name_map controls_by_name_;

  public:
    CompiledSynth(
      std::shared_ptr<gcc_jit_result> owner,
      init_fn_t init,
      process_fn_t process,
      size_t state_size,
      std::vector<float> default_controls,
      std::vector<ControlDesc> control_descs,
      controls_by_name_map controls_by_name
    )
    : owner_{std::move(owner)}
    , init_{init}
    , process_{process}
    , state_size_{state_size}
    , default_controls_{std::move(default_controls)}
    , control_descs_{std::move(control_descs)}
    , controls_by_name_{std::move(controls_by_name)}
    {}

    CompiledSynth(CompiledSynth const&) = default;
    CompiledSynth& operator=(CompiledSynth const&) = default;
    CompiledSynth(CompiledSynth&&) noexcept = default;
    CompiledSynth& operator=(CompiledSynth&&) noexcept = default;

    bool has_control(std::string_view name) const
    { return controls_by_name_.contains(name); }

    std::optional<ControlSlot> control_slot(std::string_view name) const
    {
      if (auto it = controls_by_name_.find(name); it != controls_by_name_.end()) {
        return it->second;
      }
      return std::nullopt;
    }

    const std::vector<ControlDesc>& control_descs() const
    { return control_descs_; }

    Synth instantiate() const;
  };

  class Synth {
  private:
    std::shared_ptr<gcc_jit_result> owner_;
    process_fn_t process_{};
    std::vector<std::byte> state_;
    std::vector<float> controls_;
    std::vector<size_t> trigger_control_indices_;
    uint32_t last_done_action_{};
    uint32_t pending_done_action_{};

  public:
    Synth(
      std::shared_ptr<gcc_jit_result> owner,
      init_fn_t init,
      process_fn_t process,
      size_t state_size,
      std::vector<float> controls,
      std::vector<size_t> trigger_control_indices
    )
    : owner_{std::move(owner)}
    , process_{process}
    , state_{state_size}
    , controls_{std::move(controls)}
    , trigger_control_indices_{std::move(trigger_control_indices)}
    {
      if (init) init(state_.data());
    }

    Synth(Synth const&) = default;
    Synth& operator=(Synth const&) = default;
    Synth(Synth&&) noexcept = default;
    Synth& operator=(Synth&&) noexcept = default;

    float get_control(size_t index) const
    {
      if (index >= controls_.size()) {
        assert(false);
        return 0.0f;
      }
      return controls_[index];
    }

    void set_control(size_t index, float value)
    {
      if (index >= controls_.size()) {
        assert(false);
        return;
      }
      controls_[index] = value;
    }

    void set_control_values(size_t index, std::span<const float> values)
    {
      if (index + values.size() > controls_.size()) {
        assert(false);
        return;
      }
      std::copy_n(values.begin(), values.size(), controls_.begin() + index);
    }

    uint32_t process(float *abus)
    {
      assert(process_ != nullptr);
      assert(is_aligned(abus, audio_buffer_alignment));
      last_done_action_ = process_(state_.data(), controls_.data(), abus);
      if (last_done_action_ != 0 && pending_done_action_ == 0) {
        pending_done_action_ = last_done_action_;
      }
      for (auto index : trigger_control_indices_) {
        controls_[index] = 0.0f;
      }
      return last_done_action_;
    }

    uint32_t done_action() const noexcept
    { return last_done_action_; }

    uint32_t pending_done_action() const noexcept
    { return pending_done_action_; }

    bool should_remove() const noexcept
    { return pending_done_action_ == 1; }
  };

private:
  std::shared_ptr<gcc_jit_result> r_;
  std::unordered_map<std::string, CompiledSynth, transparent_string_hash, std::equal_to<>> compiled_synths_by_name_;

public:
  Result(gcc_jit_result *r, std::vector<SynthDescriptor> synths)
  : r_{r, &gcc_jit_result_release}
  {
    for (auto &synth : synths) {
      std::string init_name{synth.name};
      init_name += "_init";
      std::string process_name{synth.name};
      process_name += "_process";
      std::string state_size_name{synth.name};
      state_size_name += "_state_size";
      auto state_size_ptr = gcc_jit_result_get_global(r_.get(), state_size_name.c_str());
      auto state_size = state_size_ptr ? *reinterpret_cast<size_t const*>(state_size_ptr) : 0;

      auto controls_by_name = CompiledSynth::controls_by_name_map{};
      for (auto const &control : synth.control_descs) {
        controls_by_name.insert_or_assign(
          control.name,
          CompiledSynth::ControlSlot{control.index, control.width, control.kind}
        );
      }

      auto init = reinterpret_cast<init_fn_t>(gcc_jit_result_get_code(r_.get(), init_name.c_str()));
      auto process = reinterpret_cast<process_fn_t>(gcc_jit_result_get_code(r_.get(), process_name.c_str()));

      compiled_synths_by_name_.insert_or_assign(
        synth.name,
        CompiledSynth{
          r_,
          init,
          process,
          state_size,
          synth.controls,
          synth.control_descs,
          std::move(controls_by_name),
        }
      );
    }
  }

  Result(Result const&) = delete;
  Result& operator=(Result const&) = delete;

  Result(Result &&) noexcept = default;

  Result& operator=(Result &&) = delete;

  CompiledSynth operator[](std::string_view name) const
  {
    if (auto it = compiled_synths_by_name_.find(name); it != compiled_synths_by_name_.end()) {
      return it->second;
    }
    throw std::runtime_error("Synth not found");
  }
};

inline Result::Synth Result::CompiledSynth::instantiate() const
{
  return Synth{
    owner_,
    init_,
    process_,
    state_size_,
    default_controls_,
    control_descs_
      | std::views::filter([](const ControlDesc& control) {
          return control.kind == control_kind::trigger;
        })
      | std::views::transform(&ControlDesc::index)
      | std::ranges::to<std::vector>(),
  };
}

Result compile(const DAG &g, unsigned int sample_rate, size_t block_size);

}
