#pragma once

#include "dag.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <libgccjit++.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mc1 {

class Result
{
  using init_fn_t = void (*)(void *);
  using process_fn_t = void (*)(void *, const float *, float *);

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

  class Module;

  class CompiledSynth {
  public:
    struct ControlSlot {
      size_t index;
      size_t width;
      control_kind kind{control_kind::value};
    };

  private:
    std::shared_ptr<gcc_jit_result> owner_;
    init_fn_t init_{};
    process_fn_t process_{};
    size_t state_size_{};
    std::vector<float> default_controls_;
    std::vector<ControlDesc> control_descs_;
    std::unordered_map<std::string, ControlSlot> controls_by_name_;

  public:
    CompiledSynth(
      std::shared_ptr<gcc_jit_result> owner,
      init_fn_t init,
      process_fn_t process,
      size_t state_size,
      std::vector<float> default_controls,
      std::vector<ControlDesc> control_descs,
      std::unordered_map<std::string, ControlSlot> controls_by_name
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
    { return controls_by_name_.contains(std::string(name)); }

    std::optional<ControlSlot> control_slot(std::string_view name) const
    {
      auto it = controls_by_name_.find(std::string(name));
      if (it == controls_by_name_.end()) return std::nullopt;
      return it->second;
    }

    const std::vector<ControlDesc>& control_descs() const
    { return control_descs_; }

    Module instantiate() const;
  };

  class Module {
  private:
    std::shared_ptr<gcc_jit_result> owner_;
    process_fn_t process_{};
    std::vector<std::byte> state_;
    std::vector<float> controls_;
    std::vector<size_t> trigger_control_indices_;

  public:
    Module(
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

    Module(Module const&) = default;
    Module& operator=(Module const&) = default;
    Module(Module&&) noexcept = default;
    Module& operator=(Module&&) noexcept = default;

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

    void process(float *abus)
    {
      assert(process_ != nullptr);
      process_(state_.data(), controls_.data(), abus);
      for (auto index : trigger_control_indices_) {
        controls_[index] = 0.0f;
      }
    }
  };

private:
  std::shared_ptr<gcc_jit_result> r_;
  std::unordered_map<std::string, CompiledSynth> compiled_synths_by_name_;

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

      auto controls_by_name = std::unordered_map<std::string, CompiledSynth::ControlSlot>{};
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
    auto it = compiled_synths_by_name_.find(std::string(name));
    if (it == compiled_synths_by_name_.end()) {
      throw std::runtime_error("Synth not found");
    }
    return it->second;
  }
};

inline Result::Module Result::CompiledSynth::instantiate() const
{
  return Module{
    owner_,
    init_,
    process_,
    state_size_,
    default_controls_,
    [&]() {
      auto trigger_control_indices = std::vector<size_t>{};
      for (auto const& control : control_descs_) {
        if (control.kind == control_kind::trigger) {
          trigger_control_indices.push_back(control.index);
        }
      }
      return trigger_control_indices;
    }(),
  };
}

Result compile(const DAG &g, unsigned int sample_rate, size_t block_size);

}
