#pragma once

#include "dag.hpp"

#include <cstddef>
#include <libgccjit++.h>
#include <string>
#include <string_view>

namespace mc1 {

class Result
{
  using init_fn_t = void (*)(void *);
  using process_fn_t = void (*)(void *, const float *, float *);

public:
  struct Synth {
    init_fn_t init{};
    process_fn_t process{};
    size_t state_size{};
  };

private:
  gcc_jit_result *r{};

public:
  explicit Result(gcc_jit_result *r) : r{r} {}

  ~Result() { if (r) gcc_jit_result_release(r); }

  Result(Result const&) = delete;
  Result& operator=(Result const&) = delete;

  Result(Result &&o) noexcept : r{o.r} { o.r = nullptr; }

  Result& operator=(Result &&) = delete;

  Synth operator[](std::string_view name) const
  {
    std::string init_name{name};
    init_name += "_init";
    std::string process_name{name};
    process_name += "_process";
    std::string state_size_name{name};
    state_size_name += "_state_size";
    auto state_size_ptr = gcc_jit_result_get_global(r, state_size_name.c_str());
    auto state_size = state_size_ptr ? *reinterpret_cast<size_t const*>(state_size_ptr) : 0;
    return Synth{
      reinterpret_cast<init_fn_t>(gcc_jit_result_get_code(r, init_name.c_str())),
      reinterpret_cast<process_fn_t>(gcc_jit_result_get_code(r, process_name.c_str())),
      state_size
    };
  }
};

Result compile(const DAG &g, unsigned int sample_rate, size_t block_size);

}
