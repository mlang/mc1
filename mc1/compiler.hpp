#pragma once

#include "dag.hpp"

#include <libgccjit++.h>

namespace mc1 {

class Result
{
  using init_fn_t = void (*)();
  using process_fn_t = void (*)(const float *, float *);

  gcc_jit_result *r{};

public:
  const init_fn_t init{};
  const process_fn_t process{};

  explicit Result(gcc_jit_result *r)
  : r{r}
  , init{reinterpret_cast<init_fn_t>(gcc_jit_result_get_code(r, "init"))}
  , process{reinterpret_cast<process_fn_t>(gcc_jit_result_get_code(r, "process"))}
  {}

  ~Result() { if (r) gcc_jit_result_release(r); }

  Result(Result const&) = delete;
  Result& operator=(Result const&) = delete;

  Result(Result &&o) noexcept
  : r{o.r}, init{o.init}, process{o.process}
  { o.r = nullptr; }

  Result& operator=(Result &&) = delete;
};

Result compile(const DAG &g, unsigned int sample_rate, size_t block_size);

}
