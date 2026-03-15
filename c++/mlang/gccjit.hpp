#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <libgccjit++.h>

#include "views.hpp"

namespace mlang::gccjit {

inline ::gccjit::rvalue assume_aligned(::gccjit::rvalue ptr, int alignment)
{
  ::gccjit::context gcc = ptr.get_context();
  return gcc.new_cast(
    gcc.get_builtin_function("__builtin_assume_aligned")(
      gcc.new_cast(ptr, gcc.get_type(GCC_JIT_TYPE_VOID).get_pointer()),
      gcc.new_rvalue(gcc.get_int_type<size_t>(), alignment)
    ),
    ptr.get_type()
  );
}

inline ::gccjit::type new_function_ptr_type(
  ::gccjit::type return_type,
  std::vector<::gccjit::type> &args,
  int is_variadic = 0,
  ::gccjit::location loc = ::gccjit::location()
)
{
  ::gccjit::context gcc = return_type.get_context();
  return gcc_jit_context_new_function_ptr_type(
    gcc.get_inner_context(),
    loc.get_inner_location(),
    return_type.get_inner_type(),
    args.size(), reinterpret_cast<gcc_jit_type **>(args.data()),
    is_variadic
  );
}

inline ::gccjit::rvalue new_call_through_ptr(
  ::gccjit::rvalue fn_ptr,
  std::vector<::gccjit::rvalue> &args,
  ::gccjit::location loc = ::gccjit::location()
)
{
  ::gccjit::context gcc = fn_ptr.get_context();
  return gcc_jit_context_new_call_through_ptr(
    gcc.get_inner_context(),
    loc.get_inner_location(),
    fn_ptr.get_inner_rvalue(),
    args.size(), reinterpret_cast<gcc_jit_rvalue **>(args.data())
  );
}

template<typename T>
inline constexpr std::optional<gcc_jit_types> builtin_type_v = std::nullopt;

#define BUILTIN_TYPE(type, TYPE) template<>   \
inline constexpr std::optional<gcc_jit_types> \
builtin_type_v<type> = GCC_JIT_TYPE_##TYPE;

BUILTIN_TYPE(void, VOID)
BUILTIN_TYPE(void *, VOID_PTR)
BUILTIN_TYPE(bool, BOOL)
BUILTIN_TYPE(char, CHAR)
BUILTIN_TYPE(short, SHORT)
BUILTIN_TYPE(unsigned short, UNSIGNED_SHORT)
BUILTIN_TYPE(int, INT)
BUILTIN_TYPE(unsigned int, UNSIGNED_INT)
BUILTIN_TYPE(float, FLOAT)
BUILTIN_TYPE(double, DOUBLE)
BUILTIN_TYPE(long double, LONG_DOUBLE)
BUILTIN_TYPE(const char *, CONST_CHAR_PTR)
BUILTIN_TYPE(size_t, SIZE_T)

#undef BUILTIN_TYPE

template<typename T>
::gccjit::type get_type(::gccjit::context gcc)
{
  if constexpr (builtin_type_v<T>.has_value()) {
    return gcc.get_type(*builtin_type_v<T>);
  } else {
    using no_ptr_t = std::remove_pointer_t<T>;
    using base_t = std::remove_const_t<no_ptr_t>;

    static_assert(!std::is_same_v<T, base_t>, "No type defined");

    auto type = get_type<base_t>(gcc);
    if constexpr (std::is_const_v<no_ptr_t>) type = type.get_const();
    if constexpr (std::is_pointer_v<T>) type = type.get_pointer();
    return type;
  }
}

inline ::gccjit::rvalue new_sizeof(::gccjit::type type)
{
  auto context = type.get_context();
  auto rv = ::gccjit::rvalue{
    gcc_jit_context_new_sizeof(context.get_inner_context(), type.get_inner_type())
  };
  // libgccjit reports sizeof as int; normalize it to size_t for callers.
  return context.new_cast(rv, get_type<size_t>(context));
}

inline std::shared_ptr<gcc_jit_result> compile_shared(::gccjit::context gcc)
{
  return {gcc.compile(), &gcc_jit_result_release};
}

template<typename Signature>
std::shared_ptr<Signature>
get_code(std::shared_ptr<gcc_jit_result> result, const char *name)
{
  auto const ptr = reinterpret_cast<Signature *>(
    gcc_jit_result_get_code(result.get(), name)
  );
  return {std::move(result), ptr};
}

template<typename T>
::gccjit::function
make_tabled_function(::gccjit::context gcc, std::string name, T period, size_t n, T(*f)(T))
{
  auto fp_type = get_type<T>(gcc);
  assert(n > 0);
  auto array_type = gcc.new_array_type(fp_type.get_const(), n + 1);
  auto table = gcc.new_global(GCC_JIT_GLOBAL_INTERNAL, array_type, name + "__table");
  { // Initialize the table
    std::vector<::gccjit::rvalue> init;
    init.reserve(n + 1);
    std::ranges::copy
    ( mlang::views::sampled_interval(period, n)
    | std::views::transform(f)
    | std::views::transform([&](T v) { return gcc.new_rvalue(fp_type, v); })
    , std::back_inserter(init)
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
  auto func = gcc.new_function(GCC_JIT_FUNCTION_ALWAYS_INLINE,
    fp_type, name, param, 0
  );
  auto x = func.get_param(0);
  auto i = func.new_local(gcc.get_type(GCC_JIT_TYPE_SIZE_T), "i");
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
    gcc.get_builtin_function("__builtin_fmaf")(
      x - gcc.new_cast(i, fp_type), b - a, a
    )
  );

  return func;
}

}
