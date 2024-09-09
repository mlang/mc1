#pragma once

#include <memory>
#include <libgccjit++.h>

#include "views.hpp"

namespace gccjit {

template<typename T>
inline constexpr std::optional<gcc_jit_types> type_v = std::nullopt;
#define TYPE_V(type, TYPE) template<>         \
inline constexpr std::optional<gcc_jit_types> \
type_v<type> = GCC_JIT_TYPE_##TYPE;

TYPE_V(void, VOID)
TYPE_V(void *, VOID_PTR)
TYPE_V(bool, BOOL)
TYPE_V(short, SHORT)
TYPE_V(unsigned short, UNSIGNED_SHORT)
TYPE_V(int, INT)
TYPE_V(unsigned int, UNSIGNED_INT)
TYPE_V(float, FLOAT)
TYPE_V(double, DOUBLE)
TYPE_V(long double, LONG_DOUBLE)
TYPE_V(const char *, CONST_CHAR_PTR)
TYPE_V(size_t, SIZE_T)

#undef TYPE_V

template<typename T>
type get_type(context gcc)
{
  static_assert(type_v<T>, "No type defined");
  return gcc.get_type(type_v<T>.value());
}

std::shared_ptr<gcc_jit_result> compile_shared(context gcc)
{ return { gcc.compile(), &gcc_jit_result_release }; }

template<typename Signature>
std::shared_ptr<Signature>
get_code(std::shared_ptr<gcc_jit_result> result, const char *name)
{
  return {
    std::move(result),
    reinterpret_cast<Signature *>(gcc_jit_result_get_code(result.get(), name))
  };
}

template<typename T>
function
make_tabled_function(context gcc, std::string name, T period, size_t n, T(*f)(T))
{
  auto fp_type = get_type<T>(gcc);
  assert(n > 0);
  auto array_type = gcc.new_array_type(fp_type.get_const(), n + 1);
  auto table = gcc.new_global(GCC_JIT_GLOBAL_INTERNAL, array_type, name + "__table");
  { // Initialize the table
    std::vector<gccjit::rvalue> init;
    init.reserve(n + 1);
    std::ranges::copy(
      mlang::views::sampled_interval(period, n) | std::views::transform(f) |
      std::views::transform([&](T v) { return gcc.new_rvalue(fp_type, v); }),
      std::back_inserter(init)
    );
    init.push_back(init.front());
    table.set_initializer_rvalue(gcc.new_array_ctor(array_type, init));
  }

  auto const_fp_ptr_type = fp_type.get_const().get_pointer().get_const();
  auto table_a = gcc.new_global(GCC_JIT_GLOBAL_INTERNAL, const_fp_ptr_type, name + "__a");
  table_a.set_initializer_rvalue(table[0].get_address());
  auto table_b = gcc.new_global(GCC_JIT_GLOBAL_INTERNAL, const_fp_ptr_type, name + "__b");
  table_b.set_initializer_rvalue(table[1].get_address());

  auto param = std::vector{gcc.new_param(fp_type, "x")};
  auto func = gcc.new_function(GCC_JIT_FUNCTION_EXPORTED,
    fp_type, name, param, 0
  );
  auto x = func.get_param(0);
  auto index_type = gcc.get_type(GCC_JIT_TYPE_SIZE_T);
  auto i = func.new_local(index_type, "i");
  auto a = func.new_local(fp_type, "a");
  auto b = func.new_local(fp_type, "b");

  auto block = func.new_block("entry");
  block.add_assignment_op(x, GCC_JIT_BINARY_OP_MULT,
    gcc.new_rvalue(fp_type, static_cast<T>(n) / period)
  );
  block.add_assignment(i, gcc.new_cast(x, i.get_type())); 
  block.add_assignment(a, table_a[i]);
  block.add_assignment(b, table_b[i]);
  block.end_with_return(
    gcc.get_builtin_function("__builtin_fma")(
      x - gcc.new_cast(i, fp_type), b - a, a
    )
  );

  return func;
}

}
