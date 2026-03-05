#pragma once

#include "dag.hpp"

#include <cstddef>
#include <libgccjit++.h>
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
  class Synth {
    std::shared_ptr<gcc_jit_result> owner_;
    process_fn_t process_{};
    std::vector<std::byte> state_;

  public:
    Synth() = default;

    Synth(std::shared_ptr<gcc_jit_result> owner, init_fn_t init, process_fn_t process, size_t state_size)
    : owner_{std::move(owner)}, process_{process}, state_(state_size) { init(state_.data()); }

    Synth(Synth const&) = delete;
    Synth& operator=(Synth const&) = delete;
    Synth(Synth&&) noexcept = default;
    Synth& operator=(Synth&&) noexcept = default;

    void process(const float *controls, float *abus)
    { process_(state_.data(), controls, abus); }
  };

private:
  std::shared_ptr<gcc_jit_result> r_;

public:
  explicit Result(gcc_jit_result *r)
  : r_{r, &gcc_jit_result_release}
  {}

  Result(Result const&) = delete;
  Result& operator=(Result const&) = delete;

  Result(Result &&) noexcept = default;

  Result& operator=(Result &&) = delete;

  Synth operator[](std::string_view name) const
  {
    std::string init_name{name};
    init_name += "_init";
    std::string process_name{name};
    process_name += "_process";
    std::string state_size_name{name};
    state_size_name += "_state_size";
    auto state_size_ptr = gcc_jit_result_get_global(r_.get(), state_size_name.c_str());
    auto state_size = state_size_ptr ? *reinterpret_cast<size_t const*>(state_size_ptr) : 0;
    return Synth{
      r_,
      reinterpret_cast<init_fn_t>(gcc_jit_result_get_code(r_.get(), init_name.c_str())),
      reinterpret_cast<process_fn_t>(gcc_jit_result_get_code(r_.get(), process_name.c_str())),
      state_size
    };
  }
};

Result compile(const DAG &g, unsigned int sample_rate, size_t block_size);

}
