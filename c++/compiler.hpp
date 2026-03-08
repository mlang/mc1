#pragma once

#include "dag.hpp"

#include <cassert>
#include <cstddef>
#include <functional>
#include <libgccjit++.h>
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
  };

  struct SynthDescriptor {
    std::string name;
    std::vector<float> controls;
    std::vector<ControlDesc> control_descs;
  };

  class Synth {
  public:
    struct ControlSlot {
      size_t index;
      size_t width;
    };

  private:
    std::shared_ptr<gcc_jit_result> owner_;
    process_fn_t process_{};
    std::vector<std::byte> state_;
    std::vector<float> controls_;
    std::unordered_map<std::string, ControlSlot> controls_by_name_;

  public:
    Synth() = default;

    Synth(
      std::shared_ptr<gcc_jit_result> owner,
      init_fn_t init,
      process_fn_t process,
      size_t state_size,
      std::vector<float> controls,
      std::unordered_map<std::string, ControlSlot> controls_by_name
    )
    : owner_{std::move(owner)}
    , process_{process}
    , state_{state_size}
    , controls_{std::move(controls)}
    , controls_by_name_{std::move(controls_by_name)}
    {
      if (init) init(state_.data());
    }

    Synth(Synth const&) = delete;
    Synth& operator=(Synth const&) = delete;
    Synth(Synth&&) noexcept = default;
    Synth& operator=(Synth&&) noexcept = default;

    bool has_control(std::string_view name) const
    { return controls_by_name_.contains(std::string(name)); }

    float get_control(std::string_view name) const
    {
      auto it = controls_by_name_.find(std::string(name));
      if (it == controls_by_name_.end()) {
        assert(false);
        return 0.0f;
      }
      if (it->second.width != 1) {
        assert(false);
        return 0.0f;
      }
      return controls_[it->second.index];
    }

    void set_control(std::string_view name, float value)
    {
      auto it = controls_by_name_.find(std::string(name));
      if (it == controls_by_name_.end()) {
        assert(false);
        return;
      }
      if (it->second.width != 1) {
        assert(false);
        return;
      }
      controls_[it->second.index] = value;
    }

    void process(float *abus)
    {
      assert(process_ != nullptr);
      process_(state_.data(), controls_.data(), abus);
    }
  };

private:
  std::shared_ptr<gcc_jit_result> r_;
  std::unordered_map<std::string, std::function<Synth()>> creators_;

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

      auto controls_by_name = std::unordered_map<std::string, Synth::ControlSlot>{};
      for (auto const &control : synth.control_descs) {
        controls_by_name.insert_or_assign(
          control.name,
          Synth::ControlSlot{control.index, control.width}
        );
      }

      auto init = reinterpret_cast<init_fn_t>(gcc_jit_result_get_code(r_.get(), init_name.c_str()));
      auto process = reinterpret_cast<process_fn_t>(gcc_jit_result_get_code(r_.get(), process_name.c_str()));
      auto controls = std::move(synth.controls);

      creators_.insert_or_assign(
        std::move(synth.name),
        [owner = r_, init, process, state_size, controls = std::move(controls), controls_by_name = std::move(controls_by_name)]()
        {
          return Synth{
            owner,
            init,
            process,
            state_size,
            controls,
            controls_by_name,
          };
        }
      );
    }
  }

  Result(Result const&) = delete;
  Result& operator=(Result const&) = delete;

  Result(Result &&) noexcept = default;

  Result& operator=(Result &&) = delete;

  Synth operator[](std::string_view name) const
  {
    auto it = creators_.find(std::string(name));
    if (it == creators_.end()) {
      assert(false);
      return {};
    }
    return it->second();
  }
};

Result compile(const DAG &g, unsigned int sample_rate, size_t block_size);

}
