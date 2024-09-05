#pragma once

#include <memory>
#include <libgccjit++.h>

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

template<typename T> type get_type(context gcc)
{
  static_assert(type_v<T>, "No type defined");
  return gcc.get_type(type_v<T>.value());
}

std::shared_ptr<gcc_jit_result> compile(context gcc)
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

}
