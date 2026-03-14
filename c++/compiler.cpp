#include "compiler.hpp"

#include <numbers>

#include "mlang/gccjit.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <memory>
#include <numbers>
#include <optional>
#include <print>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <libgccjit++.h>

namespace mc1 {

namespace {

constexpr int adsr_stage_idle = 0;
constexpr int adsr_stage_attack = 1;
constexpr int adsr_stage_decay = 2;
constexpr int adsr_stage_sustain = 3;
constexpr int adsr_stage_release = 4;

} // namespace

struct StateTypeBase
{
  gccjit::type st;
  explicit StateTypeBase(gccjit::type st) : st{st} {}
  virtual ~StateTypeBase() = default;
};

template<class> inline constexpr enum gcc_jit_types jit_type_v;
template<> inline constexpr enum gcc_jit_types jit_type_v<void>   = GCC_JIT_TYPE_VOID;
template<> inline constexpr enum gcc_jit_types jit_type_v<float>  = GCC_JIT_TYPE_FLOAT;
template<> inline constexpr enum gcc_jit_types jit_type_v<size_t> = GCC_JIT_TYPE_SIZE_T;
template<> inline constexpr enum gcc_jit_types jit_type_v<uint32_t> = GCC_JIT_TYPE_UNSIGNED_INT;

gccjit::rvalue new_sizeof(gccjit::type type)
{
  auto context = type.get_context();
  auto rv = gccjit::rvalue{
    gcc_jit_context_new_sizeof(context.get_inner_context(), type.get_inner_type())
  };
  // sizeof(type) is of type int for some reason
  return context.new_cast(rv, context.get_type(GCC_JIT_TYPE_SIZE_T));
}

struct Context
{
  gccjit::context gcc;
  unsigned int sample_rate;
  size_t block_size;
  DAG const& graph;

  gccjit::function sinf;

  std::unordered_map<std::string, gccjit::function> kernelCache;
  std::unordered_map<std::string, std::unique_ptr<StateTypeBase>> stateCache;

  Context(unsigned int sample_rate, size_t block_size, DAG const& graph)
  : gcc{gccjit::context::acquire()}
  , sample_rate{sample_rate}, block_size{block_size}, graph{graph}
  , sinf{gccjit::make_tabled_function(gcc, "sinf_lookup", std::numbers::pi_v<float> * 2.0f, 256, std::sinf)}
  , kernelCache{}, stateCache{}
  {}

  ~Context() { gcc.release(); }

  Context(Context const&) = delete;
  Context& operator=(Context const&) = delete;

  // Fancy helpers
  template<class T> gccjit::type type()
  {
    using NoPtr = std::remove_pointer_t<T>;
    using Base = std::remove_const_t<NoPtr>;

    gccjit::type t = gcc.get_type(jit_type_v<Base>);

    if constexpr (std::is_const_v<NoPtr>) t = t.get_const();
    if constexpr (std::is_pointer_v<T>) t = t.get_pointer();

    return t;
  }

  template<class T> gccjit::type type(int n)
  { return gcc.new_array_type(type<T>(), n); }

  template<class Sig>
  gccjit::function import_function(const char* name)
  {
    using traits = fn_traits<Sig>;
    return import_function<Sig>(name, std::make_index_sequence<traits::nargs>{});
  }

private:
  template<class> struct fn_traits;

  template<class R, class... Args>
  struct fn_traits<R(Args...)>
  {
    using return_t = R;
    using args_t = std::tuple<Args...>;
    static constexpr std::size_t nargs = sizeof...(Args);
  };

  template<class R, class... Args>
  struct fn_traits<R(*)(Args...)> : fn_traits<R(Args...)> {};

  template<class Sig, std::size_t... I>
  gccjit::function import_function(const char* name, std::index_sequence<I...>)
  {
    using traits = fn_traits<Sig>;
    using R = typename traits::return_t;
    using Tup = typename traits::args_t;

    auto params = std::vector{
      gcc.new_param(type<std::tuple_element_t<I, Tup>>(), std::format("a{}", I))...
    };

    return gcc.new_function(GCC_JIT_FUNCTION_IMPORTED,
      type<R>(), name, params, 0
    );
  }

public:
  template<class T, class MakeFn>
  T& get_or_make_state(std::string const& key, MakeFn &&make_fn)
  {
    if (auto it = stateCache.find(key); it != stateCache.end())
      return *static_cast<T*>(it->second.get());

    auto created = std::make_unique<T>(make_fn());
    auto &ref = *created;
    stateCache.emplace(key, std::move(created));
    return ref;
  }

  std::string graph_symbol(std::string_view base) const
  {
    return std::format("{}_{}", graph.name, base);
  }

  gccjit::rvalue new_float(float value)
  {
    return gcc.new_rvalue(type<float>(), static_cast<double>(value));
  }

  gccjit::rvalue wrap_tau(gccjit::rvalue phase)
  {
    auto t_int = gcc.get_type(GCC_JIT_TYPE_INT);
    auto tau = new_float(static_cast<float>(2.0 * std::numbers::pi_v<double>));

    auto wrapped_hi = gcc.new_cast(
      gcc.new_comparison(GCC_JIT_COMPARISON_GE, phase, tau), t_int
    );
    auto wrapped_hi_f = gcc.new_cast(wrapped_hi, type<float>());
    auto out = phase - (tau * wrapped_hi_f);

    auto wrapped_lo = gcc.new_cast(
      gcc.new_comparison(GCC_JIT_COMPARISON_LT, out, gcc.zero(type<float>())), t_int
    );
    auto wrapped_lo_f = gcc.new_cast(wrapped_lo, type<float>());
    return out + (tau * wrapped_lo_f);
  }

  gccjit::block loop(gccjit::function fn, gccjit::block entry, auto &&body_fn)
  {
    auto c_BS_size = gcc.new_rvalue(type<size_t>(), long(block_size));

    auto cond  = fn.new_block("cond");
    auto body  = fn.new_block("body");
    auto cont  = fn.new_block("cont");
    auto done  = fn.new_block("done");

    gccjit::lvalue i = fn.new_local(type<size_t>(), "i");
    entry.add_assignment(i, gcc.zero(type<size_t>()));
    entry.end_with_jump(cond);

    cond.end_with_conditional(i < c_BS_size, body, done);

    body_fn(body, cont, i);

    cont.add_assignment(i, i + gcc.one(type<size_t>()));
    cont.end_with_jump(cond);

    return done;
  }
};

class CodegenNode
{
protected:
  Context &ctx;
  std::string name, sig;
  size_t vertex_index;
  size_t num_out;
  gccjit::rvalue rvalue;

  std::string kernel_name() const { return std::format("{}_{}", name, sig); }

  virtual std::optional<gccjit::function> make_kernel() const { return std::nullopt; }

  std::optional<gccjit::function> get_or_make_kernel() const
  {
    auto fn_name = kernel_name();
    if (auto it = ctx.kernelCache.find(fn_name); it != ctx.kernelCache.end())
      return it->second;

    auto created = make_kernel();
    if (created) ctx.kernelCache.emplace(fn_name, *created);
    return created;
  }

  gccjit::function new_kernel(
    std::vector<gccjit::param> params,
    std::optional<gccjit::type> return_type = std::nullopt
  ) const
  {
    return ctx.gcc.new_function(GCC_JIT_FUNCTION_INTERNAL,
      return_type.value_or(
        ctx.gcc.get_type(rate() == 'a' ? GCC_JIT_TYPE_VOID : GCC_JIT_TYPE_FLOAT)
      ),
      kernel_name(), params, 0
    );
  }

public:
  CodegenNode(Context &ctx, std::string name, std::string sig, size_t vertex_index, size_t num_out)
  : ctx{ctx}
  , name{std::move(name)}, sig{std::move(sig)}, vertex_index{vertex_index}
  , num_out{num_out}
  , rvalue{}
  {}

  virtual ~CodegenNode() = default;

  char rate() const { return sig.back(); }

  size_t output_count() const { return num_out; }

  gccjit::rvalue get_rvalue() const
  {
    assert(rvalue.get_inner_rvalue() != nullptr);
    return rvalue;
  }

  gccjit::rvalue get_rvalue(size_t channel) const
  {
    assert(rvalue.get_inner_rvalue() != nullptr);
    assert(rate() == 'a');

    auto c_i = ctx.gcc.new_rvalue(ctx.type<size_t>(), long(ctx.block_size * channel));

    return ctx.gcc.new_array_access(rvalue, c_i).get_address();
  }

  gccjit::param new_param(std::string param_name)
  {
    return ctx.gcc.new_param(
      rate() == 'a' ? ctx.type<const float*>() : ctx.type<float>(),
      param_name
    );
  }

  virtual std::optional<gccjit::type> state_type() const { return std::nullopt; }
  virtual void emit_init(gccjit::block, std::optional<gccjit::lvalue>) {}
  virtual void emit_proc(
    gccjit::function,
    gccjit::block,
    std::optional<gccjit::lvalue>,
    std::optional<gccjit::lvalue>
  ) = 0;
};

class NonGraphArgs : public CodegenNode
{
protected:
  std::vector<size_t> args;

public:
  NonGraphArgs(
    Context &ctx, std::string name, std::string sig, size_t vertex_index, size_t num_out,
    std::vector<size_t> args
  )
  : CodegenNode{ctx, std::move(name), std::move(sig), vertex_index, num_out}
  , args{std::move(args)}
  {}
};

template<class T> concept SpecialIndices = std::is_base_of_v<NonGraphArgs, T>;

template<typename... Ts> struct overload : Ts... { using Ts::operator()...; };

class GraphArgs : public CodegenNode
{
protected:
  std::vector<CodegenNode*> args;

  gccjit::rvalue new_proc_local(
    gccjit::function f,
    gccjit::block b,
    std::variant<gccjit::function, gccjit::rvalue> v,
    std::optional<gccjit::rvalue> state_ptr = std::nullopt
  )
  {
    auto var = f.new_local(
      rate() == 'a'
        ? ctx.type<float>(int(ctx.block_size * output_count()))
        : ctx.type<float>(),
      std::format("e{}", vertex_index)
    );
    auto maybe_call = overload{
      [&](gccjit::function kernel)
      {
        std::vector<gccjit::rvalue> call_args;
        call_args.reserve(args.size() + (rate() == 'a') + (state_ptr ? 1 : 0));

        if (state_ptr) call_args.push_back(*state_ptr);
        if (rate() == 'a') call_args.push_back(var[0].get_address());
        for (auto *op : args) call_args.push_back(op->get_rvalue());

        return ctx.gcc.new_call(kernel, call_args);            // kernel writes to out_ptr
      },
      [](gccjit::rvalue rv) { return rv; }
    };

    if (rate() == 'a') {
      b.add_eval(std::visit(maybe_call, v));

      return var[0].get_address();
    }

    // assign scalar rvalue
    assert(rate() == 'b');
    b.add_assignment(var, std::visit(maybe_call, v));

    return var;
  }

public:
  GraphArgs(
    Context &ctx, std::string name, std::string sig, size_t vertex_index, size_t num_out,
    std::vector<CodegenNode*> args
  )
  : CodegenNode{ctx, std::move(name), std::move(sig), vertex_index, num_out}
  , args{std::move(args)}
  {}
};

// ------------------------------------------------------------------------- //

struct Const final : NonGraphArgs
{
  using NonGraphArgs::NonGraphArgs;

  void emit_proc(
    gccjit::function,
    gccjit::block,
    std::optional<gccjit::lvalue>,
    std::optional<gccjit::lvalue>
  ) override
  {
    assert(args.size() == 1);
    rvalue = ctx.new_float(ctx.graph.constants[args.front()]);
  }
};

struct Control final : NonGraphArgs
{
  using NonGraphArgs::NonGraphArgs;

  void emit_proc(
    gccjit::function f,
    gccjit::block,
    std::optional<gccjit::lvalue>,
    std::optional<gccjit::lvalue>
  ) override
  {
    assert(args.size() == 1);
    rvalue = f.get_param(1)[args.front()];
  }
};

struct In final : GraphArgs
{
  In(
    Context &ctx, std::string name, std::string sig, size_t vertex_index, size_t num_out,
    std::vector<CodegenNode*> args
  )
  : GraphArgs{ctx, std::move(name), std::move(sig), vertex_index, num_out, std::move(args)}
  {
    assert(output_count() == 1);
    assert(this->args.size() == 1);
    assert(this->args[0]->output_count() == 1);
  }

  void emit_proc(
    gccjit::function f,
    gccjit::block,
    std::optional<gccjit::lvalue>,
    std::optional<gccjit::lvalue>
  ) override
  {
    assert(sig == "ba");
    assert(args.size() == 1);

    auto abus = f.get_param(2);
    auto idx = ctx.gcc.new_cast(args[0]->get_rvalue(), ctx.type<size_t>());
    auto c_bs = ctx.gcc.new_rvalue(ctx.type<size_t>(), long(ctx.block_size));

    rvalue = abus[idx * c_bs].get_address();
  }
};

class Out final : public GraphArgs
{
  std::optional<gccjit::function> kernel;

  std::optional<gccjit::function> make_kernel() const override
  {
    // Only support: Out baa
    if (sig != "baa") return std::nullopt;

    auto t_float_ptr = ctx.type<float*>();
    auto t_const_float_ptr = ctx.type<const float*>();

    // void Out_baa(float *dst, const float *src)
    auto p_dst = ctx.gcc.new_param(t_float_ptr, "dst");
    auto p_a = args[1]->new_param("a");
    auto kernel = new_kernel({p_dst, p_a});
    {
      auto entry = kernel.new_block("entry");
      ctx.loop(kernel, entry, [&](gccjit::block body, gccjit::block cont, gccjit::lvalue lv_i) {
        body.add_assignment(p_dst[lv_i], p_a[lv_i]);
        body.end_with_jump(cont);
      }).end_with_return();
    }

    return kernel;
  }

public:
  Out(
    Context &ctx, std::string name, std::string sig, size_t vertex_index, size_t num_out,
    std::vector<CodegenNode*> args
  )
  : GraphArgs{ctx, std::move(name), std::move(sig), vertex_index, num_out, std::move(args)}
  , kernel{get_or_make_kernel()}
  {
    assert(this->args.size() == 2);
    assert(this->args[0]->output_count() == 1);
  }

  void emit_proc(
    gccjit::function f,
    gccjit::block b,
    std::optional<gccjit::lvalue>,
    std::optional<gccjit::lvalue>
  ) override
  {
    assert(sig == "baa");
    assert(args.size() == 2);
    assert(kernel);

    auto abus = f.get_param(2);

    auto idx = ctx.gcc.new_cast(args[0]->get_rvalue(), ctx.type<size_t>());
    auto c_bs = ctx.gcc.new_rvalue(ctx.type<size_t>(), long(ctx.block_size));

    for (size_t ch = 0; ch < args[1]->output_count(); ch++) {
      auto c_ch = ctx.gcc.new_rvalue(ctx.type<size_t>(), long(ch));
      auto dst = abus[(idx + c_ch) * c_bs].get_address();
      b.add_eval((*kernel)(dst, args[1]->get_rvalue(ch)));
    }

    rvalue = abus[idx * c_bs].get_address();
  }
};

class BinOp final : public GraphArgs
{
  enum gcc_jit_binary_op op_kind;
  std::optional<gccjit::function> kernel;

  static enum gcc_jit_binary_op kind_from_name(std::string const& opcode_name)
  {
    if (opcode_name.rfind("Mul", 0) == 0) return GCC_JIT_BINARY_OP_MULT;
    if (opcode_name.rfind("Add", 0) == 0) return GCC_JIT_BINARY_OP_PLUS;
    if (opcode_name.rfind("Sub", 0) == 0) return GCC_JIT_BINARY_OP_MINUS;
    if (opcode_name.rfind("Div", 0) == 0) return GCC_JIT_BINARY_OP_DIVIDE;
    throw std::runtime_error("unknown binop opcode: " + opcode_name);
  }

  static bool is_sig_supported(std::string const& s)
  {
    return s == "aaa" || s == "aba" || s == "baa" || s == "bbb";
  }

  std::optional<gccjit::function> make_kernel() const override
  {
    if (sig == "bbb") return std::nullopt;

    // void f(float *r, <a>, <b>) where <a>/<b> are float or float* depending
    gccjit::param p_r = ctx.gcc.new_param(ctx.type<float *>(), "r");
    gccjit::param p_a = args[0]->new_param("a");
    gccjit::param p_b = args[1]->new_param("b");

    auto kernel = new_kernel({p_r, p_a, p_b});
    {
      auto entry = kernel.new_block("entry");
      ctx.loop(kernel, entry, [&](gccjit::block body, gccjit::block cont, gccjit::lvalue lv_i) {
        auto lv_r_i = p_r[lv_i];

        gccjit::rvalue ra = (args[0]->rate() == 'a') ? p_a[lv_i] : p_a;
        gccjit::rvalue rb = (args[1]->rate() == 'a') ? p_b[lv_i] : p_b;

        auto expr = ctx.gcc.new_binary_op(op_kind, ctx.type<float>(), ra, rb);
        body.add_assignment(lv_r_i, expr);
        body.end_with_jump(cont);
      }).end_with_return();
    }

    return kernel;
  }

public:
  BinOp(
    Context &ctx, std::string name, std::string sig, size_t vertex_index, size_t num_out,
    std::vector<CodegenNode*> args
  )
  : GraphArgs{ctx, std::move(name), std::move(sig), vertex_index, num_out, std::move(args)}
  , op_kind{kind_from_name(this->name)}
  , kernel{get_or_make_kernel()}
  {
    assert(output_count() == 1);
    assert(this->args.size() == 2);
    assert(this->args[0]->output_count() == 1);
    assert(this->args[1]->output_count() == 1);
  }

  void emit_proc(
    gccjit::function f,
    gccjit::block b,
    std::optional<gccjit::lvalue>,
    std::optional<gccjit::lvalue>
  ) override
  {
    assert(args.size() == 2);
    assert(sig.size() == 3);
    assert(is_sig_supported(sig));

    if (sig == "bbb") {
      assert(rate() == 'b');
      auto expr = ctx.gcc.new_binary_op(op_kind,
        ctx.type<float>(), args[0]->get_rvalue(), args[1]->get_rvalue()
      );
      rvalue = new_proc_local(f, b, expr);
      return;
    }

    assert(rate() == 'a');
    assert(kernel);

    rvalue = new_proc_local(f, b, *kernel);
  }
};

class SinOsc final : public GraphArgs
{
  struct StateType final : StateTypeBase
  {
    gccjit::field phase;
    StateType(gccjit::type st, gccjit::field phase) : StateTypeBase{st}, phase{phase} {}
  };

  static constexpr const char *state_cache_key = "SinOsc";

  static StateType make_state_type(Context &ctx)
  {
    auto fld_phase = ctx.gcc.new_field(ctx.type<float>(), "phase");
    auto fields = std::vector{ fld_phase };
    auto st = ctx.gcc.new_struct_type("sinosc_state", fields);
    return StateType(st, fld_phase);
  }

  StateType &ST;
  std::optional<gccjit::function> kernel;
  std::optional<gccjit::type> state_type() const override { return ST.st; }

  static bool is_sig_supported(std::string const& s)
  { return s == "bba" || s == "aba" || s == "baa" || s == "aaa"; }

  std::optional<gccjit::function> make_kernel() const override
  {
    if (!is_sig_supported(sig)) return std::nullopt;

    auto t_state_ptr = ST.st.get_pointer();
    auto t_float_ptr = ctx.type<float*>();

    // void SinOsc_<sig>(State *st, float *out, <freq>, <phase>)
    auto p_st = ctx.gcc.new_param(t_state_ptr, "st");
    auto p_out = ctx.gcc.new_param(t_float_ptr, "out");
    auto p_freq = args[0]->new_param("freq");
    auto p_phase = args[1]->new_param("phase");

    auto k = new_kernel({p_st, p_out, p_freq, p_phase});
    {
      auto entry = k.new_block("entry");

      auto lv_phase = k.new_local(ctx.type<float>(), "phase");
      entry.add_assignment(lv_phase, p_st.dereference_field(ST.phase));

      auto tau_over_sr = ctx.gcc.new_rvalue(
        ctx.type<float>(), (2.0 * std::numbers::pi_v<double>) / double(ctx.sample_rate)
      );

      auto after_loop = ctx.loop(k, entry, [&](gccjit::block body, gccjit::block cont, gccjit::lvalue lv_i) {
        gccjit::rvalue freq  = (args[0]->rate() == 'a') ? p_freq[lv_i] : p_freq;
        gccjit::rvalue phofs = (args[1]->rate() == 'a') ? p_phase[lv_i] : p_phase;

        auto outv = ctx.sinf(ctx.wrap_tau(lv_phase + phofs));
        body.add_assignment(p_out[lv_i], outv);

        // phase += tau*freq/sample_rate
        auto delta = freq * tau_over_sr;
        body.add_assignment(lv_phase, ctx.wrap_tau(lv_phase + delta));
        body.end_with_jump(cont);
      });
      after_loop.add_assignment(p_st.dereference_field(ST.phase), lv_phase);
      after_loop.end_with_return();
    }

    return k;
  }

public:
  SinOsc(
    Context &ctx, std::string name, std::string sig, size_t vertex_index, size_t num_out,
    std::vector<CodegenNode*> args
  )
  : GraphArgs{ctx, std::move(name), std::move(sig), vertex_index, num_out, std::move(args)}
  , ST{ctx.get_or_make_state<StateType>(state_cache_key, [&]{ return make_state_type(ctx); })}
  , kernel{get_or_make_kernel()}
  {
    assert(output_count() == 1);
    assert(this->args.size() == 2);
    assert(this->args[0]->output_count() == 1);
    assert(this->args[1]->output_count() == 1);
    assert(is_sig_supported(this->sig));
  }

  void emit_init(gccjit::block b, std::optional<gccjit::lvalue> state_field) override
  {
    assert(state_field);
    b.add_assignment(state_field->access_field(ST.phase), ctx.gcc.zero(ctx.type<float>()));
  }

  void emit_proc(
    gccjit::function f,
    gccjit::block b,
    std::optional<gccjit::lvalue> state_field,
    std::optional<gccjit::lvalue>
  ) override
  {
    assert(rate() == 'a');
    assert(kernel);
    assert(state_field);
    rvalue = new_proc_local(f, b, *kernel, state_field->get_address());
  }
};

class ADSR final : public GraphArgs
{
  struct StateType final : StateTypeBase
  {
    gccjit::field level;
    gccjit::field stage;
    gccjit::field last_gate;
    gccjit::field release_start;

    StateType(
      gccjit::type st,
      gccjit::field level,
      gccjit::field stage,
      gccjit::field last_gate,
      gccjit::field release_start
    )
    : StateTypeBase{st}
    , level{level}
    , stage{stage}
    , last_gate{last_gate}
    , release_start{release_start}
    {}
  };

  static constexpr const char *state_cache_key = "ADSR";

  static StateType make_state_type(Context &ctx)
  {
    auto t_int = ctx.gcc.get_type(GCC_JIT_TYPE_INT);
    auto fld_level = ctx.gcc.new_field(ctx.type<float>(), "level");
    auto fld_stage = ctx.gcc.new_field(t_int, "stage");
    auto fld_last_gate = ctx.gcc.new_field(t_int, "last_gate");
    auto fld_release_start = ctx.gcc.new_field(ctx.type<float>(), "release_start");
    auto fields = std::vector{fld_level, fld_stage, fld_last_gate, fld_release_start};
    auto st = ctx.gcc.new_struct_type("adsr_state", fields);
    return StateType(st, fld_level, fld_stage, fld_last_gate, fld_release_start);
  }

  static bool is_sig_supported(std::string const& s)
  {
    if (s.size() != 7 || s.back() != 'a') return false;
    return std::ranges::all_of(s | std::views::take(6), [](char rate) {
      return rate == 'a' || rate == 'b';
    });
  }

  StateType &ST;
  std::optional<gccjit::function> kernel;
  std::optional<gccjit::type> state_type() const override { return ST.st; }

  std::optional<gccjit::function> make_kernel() const override
  {
    if (!is_sig_supported(sig)) return std::nullopt;

    auto t_state_ptr = ST.st.get_pointer();
    auto t_float = ctx.type<float>();
    auto t_float_ptr = ctx.type<float*>();
    auto t_int = ctx.gcc.get_type(GCC_JIT_TYPE_INT);
    auto t_u32 = ctx.type<uint32_t>();

    auto p_st = ctx.gcc.new_param(t_state_ptr, "st");
    auto p_out = ctx.gcc.new_param(t_float_ptr, "out");
    auto p_gate = args[0]->new_param("gate");
    auto p_attack = args[1]->new_param("attack");
    auto p_decay = args[2]->new_param("decay");
    auto p_sustain = args[3]->new_param("sustain");
    auto p_release = args[4]->new_param("release");
    auto p_done_value = args[5]->new_param("done_action");
    auto p_done_action_accum = ctx.gcc.new_param(ctx.type<uint32_t*>(), "done_action_accum");

    auto k = new_kernel(
      {p_st, p_out, p_gate, p_attack, p_decay, p_sustain, p_release, p_done_value, p_done_action_accum}
    );
    {
      auto entry = k.new_block("entry");

      auto lv_level = k.new_local(t_float, "level");
      auto lv_stage = k.new_local(t_int, "stage");
      auto lv_last_gate = k.new_local(t_int, "last_gate");
      auto lv_release_start = k.new_local(t_float, "release_start");
      auto zero_size_t = ctx.gcc.zero(ctx.type<size_t>());
      auto lv_done_action_accum = p_done_action_accum[zero_size_t];

      auto zero_f = ctx.gcc.zero(t_float);
      auto one_f = ctx.gcc.one(t_float);
      auto zero_i = ctx.gcc.zero(t_int);
      auto sample_rate = ctx.new_float(static_cast<float>(ctx.sample_rate));
      auto c_stage_idle = ctx.gcc.new_rvalue(t_int, adsr_stage_idle);
      auto c_stage_attack = ctx.gcc.new_rvalue(t_int, adsr_stage_attack);
      auto c_stage_decay = ctx.gcc.new_rvalue(t_int, adsr_stage_decay);
      auto c_stage_sustain = ctx.gcc.new_rvalue(t_int, adsr_stage_sustain);
      auto c_stage_release = ctx.gcc.new_rvalue(t_int, adsr_stage_release);

      auto sample_arg = [&](size_t index, gccjit::param p_arg, gccjit::lvalue lv_i) -> gccjit::rvalue
      {
        return args[index]->rate() == 'a' ? p_arg[lv_i] : p_arg;
      };

      auto non_negative = [&](gccjit::rvalue value) -> gccjit::rvalue
      {
        auto is_negative = ctx.gcc.new_cast(
          ctx.gcc.new_comparison(GCC_JIT_COMPARISON_LT, value, zero_f),
          t_int
        );
        auto is_negative_f = ctx.gcc.new_cast(is_negative, t_float);
        return value * (one_f - is_negative_f);
      };

      auto clamp_unit = [&](gccjit::rvalue value) -> gccjit::rvalue
      {
        auto clamped_low = non_negative(value);
        auto is_high = ctx.gcc.new_cast(
          ctx.gcc.new_comparison(GCC_JIT_COMPARISON_GT, clamped_low, one_f),
          t_int
        );
        auto is_high_f = ctx.gcc.new_cast(is_high, t_float);
        return clamped_low + ((one_f - clamped_low) * is_high_f);
      };

      auto set_action_if_done = [&](gccjit::block block, gccjit::rvalue done_value)
      {
        auto done_is_one = ctx.gcc.new_cast(
          ctx.gcc.new_comparison(GCC_JIT_COMPARISON_EQ, done_value, one_f),
          t_int
        );
        auto done_is_one_u32 = ctx.gcc.new_cast(done_is_one, t_u32);
        block.add_assignment(lv_done_action_accum, ctx.gcc.new_binary_op(
          GCC_JIT_BINARY_OP_BITWISE_OR,
          t_u32,
          lv_done_action_accum,
          done_is_one_u32
        ));
      };

      entry.add_assignment(lv_level, p_st.dereference_field(ST.level));
      entry.add_assignment(lv_stage, p_st.dereference_field(ST.stage));
      entry.add_assignment(lv_last_gate, p_st.dereference_field(ST.last_gate));
      entry.add_assignment(lv_release_start, p_st.dereference_field(ST.release_start));

      auto after_loop = ctx.loop(k, entry, [&](gccjit::block body, gccjit::block cont, gccjit::lvalue lv_i)
      {
        auto gate = sample_arg(0, p_gate, lv_i);
        auto attack = non_negative(sample_arg(1, p_attack, lv_i));
        auto decay = non_negative(sample_arg(2, p_decay, lv_i));
        auto sustain = clamp_unit(sample_arg(3, p_sustain, lv_i));
        auto release = non_negative(sample_arg(4, p_release, lv_i));
        auto done_value = sample_arg(5, p_done_value, lv_i);
        auto gate_on = ctx.gcc.new_cast(
          ctx.gcc.new_comparison(GCC_JIT_COMPARISON_GT, gate, zero_f),
          t_int
        );

        auto gate_on_check = k.new_block("adsr_gate_on_check");
        auto gate_rise_apply = k.new_block("adsr_gate_rise_apply");
        auto gate_fall_check = k.new_block("adsr_gate_fall_check");
        auto gate_stage_check = k.new_block("adsr_gate_stage_check");
        auto gate_fall_apply = k.new_block("adsr_gate_fall_apply");
        auto after_gate = k.new_block("adsr_after_gate");
        auto attack_zero_time_check = k.new_block("adsr_attack_zero_time_check");
        auto attack_zero_apply = k.new_block("adsr_attack_zero_apply");
        auto after_attack_zero = k.new_block("adsr_after_attack_zero");
        auto decay_zero_time_check = k.new_block("adsr_decay_zero_time_check");
        auto decay_zero_apply = k.new_block("adsr_decay_zero_apply");
        auto decay_zero_sustain_apply = k.new_block("adsr_decay_zero_sustain_apply");
        auto decay_zero_release_apply = k.new_block("adsr_decay_zero_release_apply");
        auto after_decay_zero = k.new_block("adsr_after_decay_zero");
        auto sustain_gate_check = k.new_block("adsr_sustain_gate_check");
        auto sustain_release_apply = k.new_block("adsr_sustain_release_apply");
        auto after_sustain_release = k.new_block("adsr_after_sustain_release");
        auto release_zero_time_check = k.new_block("adsr_release_zero_time_check");
        auto release_zero_apply = k.new_block("adsr_release_zero_apply");
        auto after_release_zero = k.new_block("adsr_after_release_zero");
        auto stage_attack_apply = k.new_block("adsr_stage_attack_apply");
        auto stage_attack_finish = k.new_block("adsr_stage_attack_finish");
        auto stage_decay_check = k.new_block("adsr_stage_decay_check");
        auto stage_decay_apply = k.new_block("adsr_stage_decay_apply");
        auto stage_decay_finish = k.new_block("adsr_stage_decay_finish");
        auto stage_decay_finish_sustain = k.new_block("adsr_stage_decay_finish_sustain");
        auto stage_decay_finish_release = k.new_block("adsr_stage_decay_finish_release");
        auto stage_sustain_check = k.new_block("adsr_stage_sustain_check");
        auto stage_sustain_apply = k.new_block("adsr_stage_sustain_apply");
        auto stage_release_check = k.new_block("adsr_stage_release_check");
        auto stage_release_apply = k.new_block("adsr_stage_release_apply");
        auto stage_release_finish = k.new_block("adsr_stage_release_finish");
        auto stage_idle_apply = k.new_block("adsr_stage_idle_apply");
        auto sample_done = k.new_block("adsr_sample_done");

        body.end_with_conditional(
          ctx.gcc.new_comparison(GCC_JIT_COMPARISON_NE, gate_on, zero_i),
          gate_on_check,
          gate_fall_check
        );

        gate_on_check.end_with_conditional(
          ctx.gcc.new_comparison(GCC_JIT_COMPARISON_EQ, lv_last_gate, zero_i),
          gate_rise_apply,
          after_gate
        );
        gate_rise_apply.add_assignment(lv_stage, c_stage_attack);
        gate_rise_apply.end_with_jump(after_gate);

        gate_fall_check.end_with_conditional(
          ctx.gcc.new_comparison(GCC_JIT_COMPARISON_NE, lv_last_gate, zero_i),
          gate_stage_check,
          after_gate
        );
        gate_stage_check.end_with_conditional(
          ctx.gcc.new_comparison(GCC_JIT_COMPARISON_NE, lv_stage, c_stage_idle),
          gate_fall_apply,
          after_gate
        );
        gate_fall_apply.add_assignment(lv_stage, c_stage_release);
        gate_fall_apply.add_assignment(lv_release_start, lv_level);
        gate_fall_apply.end_with_jump(after_gate);

        after_gate.end_with_conditional(
          ctx.gcc.new_comparison(GCC_JIT_COMPARISON_EQ, lv_stage, c_stage_attack),
          attack_zero_time_check,
          after_attack_zero
        );
        attack_zero_time_check.end_with_conditional(
          ctx.gcc.new_comparison(GCC_JIT_COMPARISON_LE, attack, zero_f),
          attack_zero_apply,
          after_attack_zero
        );
        attack_zero_apply.add_assignment(lv_level, one_f);
        attack_zero_apply.add_assignment(lv_stage, c_stage_decay);
        attack_zero_apply.end_with_jump(after_attack_zero);

        after_attack_zero.end_with_conditional(
          ctx.gcc.new_comparison(GCC_JIT_COMPARISON_EQ, lv_stage, c_stage_decay),
          decay_zero_time_check,
          after_decay_zero
        );
        decay_zero_time_check.end_with_conditional(
          ctx.gcc.new_comparison(GCC_JIT_COMPARISON_LE, decay, zero_f),
          decay_zero_apply,
          after_decay_zero
        );
        decay_zero_apply.add_assignment(lv_level, sustain);
        decay_zero_apply.end_with_conditional(
          ctx.gcc.new_comparison(GCC_JIT_COMPARISON_NE, gate_on, zero_i),
          decay_zero_sustain_apply,
          decay_zero_release_apply
        );
        decay_zero_sustain_apply.add_assignment(lv_stage, c_stage_sustain);
        decay_zero_sustain_apply.end_with_jump(after_decay_zero);
        decay_zero_release_apply.add_assignment(lv_stage, c_stage_release);
        decay_zero_release_apply.add_assignment(lv_release_start, lv_level);
        decay_zero_release_apply.end_with_jump(after_decay_zero);

        after_decay_zero.end_with_conditional(
          ctx.gcc.new_comparison(GCC_JIT_COMPARISON_EQ, lv_stage, c_stage_sustain),
          sustain_gate_check,
          after_sustain_release
        );
        sustain_gate_check.end_with_conditional(
          ctx.gcc.new_comparison(GCC_JIT_COMPARISON_EQ, gate_on, zero_i),
          sustain_release_apply,
          after_sustain_release
        );
        sustain_release_apply.add_assignment(lv_stage, c_stage_release);
        sustain_release_apply.add_assignment(lv_release_start, lv_level);
        sustain_release_apply.end_with_jump(after_sustain_release);

        after_sustain_release.end_with_conditional(
          ctx.gcc.new_comparison(GCC_JIT_COMPARISON_EQ, lv_stage, c_stage_release),
          release_zero_time_check,
          after_release_zero
        );
        release_zero_time_check.end_with_conditional(
          ctx.gcc.new_comparison(GCC_JIT_COMPARISON_LE, release, zero_f),
          release_zero_apply,
          after_release_zero
        );
        release_zero_apply.add_assignment(lv_level, zero_f);
        release_zero_apply.add_assignment(lv_stage, c_stage_idle);
        set_action_if_done(release_zero_apply, done_value);
        release_zero_apply.end_with_jump(after_release_zero);

        after_release_zero.end_with_conditional(
          ctx.gcc.new_comparison(GCC_JIT_COMPARISON_EQ, lv_stage, c_stage_attack),
          stage_attack_apply,
          stage_decay_check
        );
        stage_attack_apply.add_assignment(lv_level, lv_level + (one_f / (attack * sample_rate)));
        stage_attack_apply.end_with_conditional(
          ctx.gcc.new_comparison(GCC_JIT_COMPARISON_GE, lv_level, one_f),
          stage_attack_finish,
          sample_done
        );
        stage_attack_finish.add_assignment(lv_level, one_f);
        stage_attack_finish.add_assignment(lv_stage, c_stage_decay);
        stage_attack_finish.end_with_jump(sample_done);

        stage_decay_check.end_with_conditional(
          ctx.gcc.new_comparison(GCC_JIT_COMPARISON_EQ, lv_stage, c_stage_decay),
          stage_decay_apply,
          stage_sustain_check
        );
        stage_decay_apply.add_assignment(
          lv_level,
          lv_level - ((one_f - sustain) / (decay * sample_rate))
        );
        stage_decay_apply.end_with_conditional(
          ctx.gcc.new_comparison(GCC_JIT_COMPARISON_LE, lv_level, sustain),
          stage_decay_finish,
          sample_done
        );
        stage_decay_finish.add_assignment(lv_level, sustain);
        stage_decay_finish.end_with_conditional(
          ctx.gcc.new_comparison(GCC_JIT_COMPARISON_NE, gate_on, zero_i),
          stage_decay_finish_sustain,
          stage_decay_finish_release
        );
        stage_decay_finish_sustain.add_assignment(lv_stage, c_stage_sustain);
        stage_decay_finish_sustain.end_with_jump(sample_done);
        stage_decay_finish_release.add_assignment(lv_stage, c_stage_release);
        stage_decay_finish_release.add_assignment(lv_release_start, lv_level);
        stage_decay_finish_release.end_with_jump(sample_done);

        stage_sustain_check.end_with_conditional(
          ctx.gcc.new_comparison(GCC_JIT_COMPARISON_EQ, lv_stage, c_stage_sustain),
          stage_sustain_apply,
          stage_release_check
        );
        stage_sustain_apply.add_assignment(lv_level, sustain);
        stage_sustain_apply.end_with_jump(sample_done);

        stage_release_check.end_with_conditional(
          ctx.gcc.new_comparison(GCC_JIT_COMPARISON_EQ, lv_stage, c_stage_release),
          stage_release_apply,
          stage_idle_apply
        );
        stage_release_apply.add_assignment(
          lv_level,
          lv_level - (lv_release_start / (release * sample_rate))
        );
        stage_release_apply.end_with_conditional(
          ctx.gcc.new_comparison(GCC_JIT_COMPARISON_LE, lv_level, zero_f),
          stage_release_finish,
          sample_done
        );
        stage_release_finish.add_assignment(lv_level, zero_f);
        stage_release_finish.add_assignment(lv_stage, c_stage_idle);
        set_action_if_done(stage_release_finish, done_value);
        stage_release_finish.end_with_jump(sample_done);

        stage_idle_apply.add_assignment(lv_level, zero_f);
        stage_idle_apply.end_with_jump(sample_done);

        sample_done.add_assignment(p_out[lv_i], lv_level);
        sample_done.add_assignment(lv_last_gate, gate_on);
        sample_done.end_with_jump(cont);
      });

      after_loop.add_assignment(p_st.dereference_field(ST.level), lv_level);
      after_loop.add_assignment(p_st.dereference_field(ST.stage), lv_stage);
      after_loop.add_assignment(p_st.dereference_field(ST.last_gate), lv_last_gate);
      after_loop.add_assignment(p_st.dereference_field(ST.release_start), lv_release_start);
      after_loop.end_with_return();
    }

    return k;
  }

public:
  ADSR(
    Context &ctx, std::string name, std::string sig, size_t vertex_index, size_t num_out,
    std::vector<CodegenNode*> args
  )
  : GraphArgs{ctx, std::move(name), std::move(sig), vertex_index, num_out, std::move(args)}
  , ST{ctx.get_or_make_state<StateType>(state_cache_key, [&]{ return make_state_type(ctx); })}
  , kernel{get_or_make_kernel()}
  {
    assert(output_count() == 1);
    assert(this->args.size() == 6);
    assert(std::ranges::all_of(this->args, [](CodegenNode *arg) { return arg->output_count() == 1; }));
    assert(is_sig_supported(this->sig));
  }

  void emit_init(gccjit::block b, std::optional<gccjit::lvalue> state_field) override
  {
    assert(state_field);
    auto t_int = ctx.gcc.get_type(GCC_JIT_TYPE_INT);
    b.add_assignment(state_field->access_field(ST.level), ctx.gcc.zero(ctx.type<float>()));
    b.add_assignment(state_field->access_field(ST.stage), ctx.gcc.zero(t_int));
    b.add_assignment(state_field->access_field(ST.last_gate), ctx.gcc.zero(t_int));
    b.add_assignment(state_field->access_field(ST.release_start), ctx.gcc.zero(ctx.type<float>()));
  }

  void emit_proc(
    gccjit::function f,
    gccjit::block b,
    std::optional<gccjit::lvalue> state_field,
    std::optional<gccjit::lvalue> done_action
  ) override
  {
    assert(rate() == 'a');
    assert(state_field);
    assert(done_action);
    assert(kernel);

    auto out = f.new_local(
      ctx.type<float>(int(ctx.block_size * output_count())),
      std::format("e{}", vertex_index)
    );
    auto call_args = std::vector<gccjit::rvalue>{
      state_field->get_address(),
      out[0].get_address(),
      args[0]->get_rvalue(),
      args[1]->get_rvalue(),
      args[2]->get_rvalue(),
      args[3]->get_rvalue(),
      args[4]->get_rvalue(),
      args[5]->get_rvalue(),
      done_action->get_address(),
    };
    b.add_eval(ctx.gcc.new_call(*kernel, call_args));

    rvalue = out[0].get_address();
  }
};

class Pan final : public GraphArgs
{
  std::optional<gccjit::function> kernel;

  static bool is_sig_supported(std::string const& s)
  { return s == "aba" || s == "aaa"; }

  std::optional<gccjit::function> make_kernel() const override
  {
    if (!is_sig_supported(sig)) return std::nullopt;

    // void Pan_<sig>(float *out, const float *in, <pan>)
    auto p_out = ctx.gcc.new_param(ctx.type<float *>(), "out");
    auto p_in  = args[0]->new_param("in");
    auto p_pan = args[1]->new_param("pan");

    auto k = new_kernel({p_out, p_in, p_pan});
    {
      auto entry = k.new_block("entry");
      ctx.loop(k, entry, [&](gccjit::block body, gccjit::block cont, gccjit::lvalue lv_i) {
        gccjit::rvalue in  = (args[0]->rate() == 'a') ? p_in[lv_i] : p_in;
        gccjit::rvalue pan = (args[1]->rate() == 'a') ? p_pan[lv_i] : p_pan;

        auto one  = ctx.gcc.one(ctx.type<float>());
        auto half = ctx.gcc.new_rvalue(ctx.type<float>(), 0.5);

        auto l = in * ((one - pan) * half);
        auto r = in * ((one + pan) * half);

        auto c_bs = ctx.gcc.new_rvalue(ctx.type<size_t>(), long(ctx.block_size));
        auto c_0 = ctx.gcc.zero(ctx.type<size_t>());
        auto c_1 = ctx.gcc.one(ctx.type<size_t>());

        body.add_assignment(p_out[lv_i + c_0 * c_bs], l);
        body.add_assignment(p_out[lv_i + c_1 * c_bs], r);
        body.end_with_jump(cont);
      }).end_with_return();
    }

    return k;
  }

public:
  Pan(
    Context &ctx, std::string name, std::string sig, size_t vertex_index, size_t num_out,
    std::vector<CodegenNode*> args
  )
  : GraphArgs{ctx, std::move(name), std::move(sig), vertex_index, num_out, std::move(args)}
  , kernel{get_or_make_kernel()}
  {
    assert(output_count() == 2);
    assert(this->args.size() == 2);
    assert(this->args[0]->output_count() == 1);
    assert(this->args[1]->output_count() == 1);
    assert(is_sig_supported(this->sig));
  }

  void emit_proc(
    gccjit::function f,
    gccjit::block b,
    std::optional<gccjit::lvalue>,
    std::optional<gccjit::lvalue>
  ) override
  {
    assert(args.size() == 2);
    assert(sig.size() == 3);
    assert(is_sig_supported(sig));
    assert(rate() == 'a');
    assert(kernel);

    rvalue = new_proc_local(f, b, *kernel);
  }
};

// ------------------------------------------------------------------------- //

class Registry
{
  using nongraph_args_t = std::vector<size_t>;
  using graph_args_t = std::vector<CodegenNode*>;
  using make_nongraph_fn = std::unique_ptr<CodegenNode>(Context &,std::string,std::string,size_t,size_t,nongraph_args_t);
  using make_graph_fn = std::unique_ptr<CodegenNode>(Context &,std::string,std::string,size_t,size_t,graph_args_t);

  std::unordered_map<std::string, std::variant<make_nongraph_fn*, make_graph_fn*>> maker;

  template<class T> static std::unique_ptr<CodegenNode> make_nongraph(
    Context &ctx, std::string name, std::string sig, size_t vertex_index, size_t num_out,
    nongraph_args_t args
  )
  { return std::make_unique<T>(ctx, name, sig, vertex_index, num_out, std::move(args)); }

  template<class T> static std::unique_ptr<CodegenNode> make_graph(
    Context &ctx, std::string name, std::string sig, size_t vertex_index, size_t num_out,
    graph_args_t args
  )
  { return std::make_unique<T>(ctx, name, sig, vertex_index, num_out, std::move(args)); }

public:
  Registry()
  {
    emplace<Const>("Const");
    emplace<Control>("Control");
    emplace<In>("In");
    emplace<Out>("Out");

    emplace<ADSR>("ADSR");
    emplace<SinOsc>("SinOsc");

    emplace<BinOp>("Mul");
    emplace<BinOp>("Add");
    emplace<BinOp>("Sub");
    emplace<BinOp>("Div");

    emplace<Pan>("Pan");
  }

  std::vector<std::unique_ptr<CodegenNode>>
  instantiate_topologically_ordered_nodes(Context &ctx, DAG const& g) const
  {
    std::vector<std::unique_ptr<CodegenNode>> nodes;
    nodes.reserve(g.ops.size());

    for (auto const &[i, vertex]: g.ops | std::views::enumerate) {
      auto sig = compute_signature(g, i);

      if (is_nongraph_args(vertex.name)) {
        nodes.push_back(create(ctx, vertex.name, sig, i, vertex.num_out, vertex.args));
        continue;
      }

      std::vector<CodegenNode*> args;
      for (auto arg : vertex.args) {
        assert(arg < i);
        args.push_back(nodes[arg].get());
      }

      nodes.push_back(create(ctx, vertex.name, sig, i, vertex.num_out, std::move(args)));
    }

    return nodes;
  }

  bool is_nongraph_args(std::string const& name) const
  {
    auto it = maker.find(name);
    if (it == maker.end())
      throw std::runtime_error(std::format("Opcode '{}' not found", name));
    return std::holds_alternative<make_nongraph_fn*>(it->second);
  }

  std::string compute_signature(const DAG &g, size_t i) const
  {
    auto const &v = g.ops.at(i);

    size_t n_in = is_nongraph_args(v.name) ? 0 : v.args.size();

    std::string sig;
    sig.reserve(n_in + 1);

    for (size_t k = 0; k < n_in; k++) {
      size_t arg_i = v.args.at(k);
      sig.push_back(g.ops.at(arg_i).rate);
    }

    sig.push_back(v.rate);
    return sig;
  }

  std::unique_ptr<CodegenNode> create(
    Context &ctx, std::string name, std::string sig, size_t vertex_index, size_t num_out,
    nongraph_args_t args
  ) const
  {
    auto iter = maker.find(name);
    if (iter == maker.end()) throw std::runtime_error("Not found");

    auto *fn = std::get<make_nongraph_fn *>(iter->second);
    return fn(ctx, std::move(name), std::move(sig), vertex_index, num_out, std::move(args));
  }

  std::unique_ptr<CodegenNode> create(
    Context &ctx, std::string name, std::string sig, size_t vertex_index, size_t num_out,
    graph_args_t args
  ) const
  {
    auto iter = maker.find(name);
    if (iter == maker.end()) throw std::runtime_error("Not found");

    auto *fn = std::get<make_graph_fn *>(iter->second);
    return fn(ctx, std::move(name), std::move(sig), vertex_index, num_out, std::move(args));
  }

  template<SpecialIndices T> void emplace(std::string name)
  {
    auto iter = maker.find(name);
    if (iter != maker.end()) throw std::runtime_error("name already used");
    maker.emplace(std::move(name), &make_nongraph<T>);
  }

  template<class T> void emplace(std::string name)
  {
    auto iter = maker.find(name);
    if (iter != maker.end()) throw std::runtime_error("name already used");
    maker.emplace(std::move(name), &make_graph<T>);
  }
};

Result compile(const DAG &g, unsigned int sample_rate, size_t block_size)
{
  Context ctx{sample_rate, block_size, g};
  ctx.gcc.set_int_option(GCC_JIT_INT_OPTION_OPTIMIZATION_LEVEL, 3);
  ctx.gcc.set_bool_option(GCC_JIT_BOOL_OPTION_DUMP_INITIAL_GIMPLE, true);
  ctx.gcc.set_bool_option(GCC_JIT_BOOL_OPTION_DUMP_GENERATED_CODE, true);
  Registry opcodes;
  auto nodes = opcodes.instantiate_topologically_ordered_nodes(ctx, g);

  auto t_void  = ctx.gcc.get_type(GCC_JIT_TYPE_VOID);
  auto t_size_t = ctx.type<size_t>();
  auto t_u32 = ctx.type<uint32_t>();
  auto t_void_ptr = ctx.type<void*>();

  std::vector<std::optional<gccjit::field>> state_fields(nodes.size());
  std::vector<gccjit::field> state_field_list;
  state_field_list.reserve(nodes.size());

  for (size_t i = 0; i < nodes.size(); i++) {
    if (auto st = nodes[i]->state_type()) {
      auto field = ctx.gcc.new_field(*st, std::format("s{}", i));
      state_fields[i] = field;
      state_field_list.push_back(field);
    }
  }

  auto state_struct = state_field_list.empty()
    ? std::optional<gccjit::type>{}
    : std::optional<gccjit::type>{ctx.gcc.new_struct_type(
        ctx.graph_symbol("state"), state_field_list
      )};

  auto make_node_state = [&](gccjit::param p_state, size_t i) -> std::optional<gccjit::lvalue>
  {
    if (!state_struct || !state_fields[i]) return std::nullopt;

    auto p_graph_state = ctx.gcc.new_cast(p_state, state_struct->get_pointer());
    auto lv_graph_state = p_graph_state.dereference();
    return lv_graph_state.access_field(*state_fields[i]);
  };

  auto init_args = std::vector{
    ctx.gcc.new_param(t_void_ptr, "state"),
  };
  auto init = ctx.gcc.new_function(GCC_JIT_FUNCTION_EXPORTED,
    t_void, ctx.graph_symbol("init"), init_args, 0
  );
  {
    auto entry = init.new_block("entry");
    auto p_state = init.get_param(0);
    for (size_t i = 0; i < nodes.size(); i++) {
      nodes[i]->emit_init(entry, make_node_state(p_state, i));
    }
    entry.end_with_return();
  }

  // uint32_t process(void *state, const float *controls, float *abus)
  auto process_args = std::vector{
    ctx.gcc.new_param(t_void_ptr, "state"),
    ctx.gcc.new_param(ctx.type<const float*>(), "controls"),
    ctx.gcc.new_param(ctx.type<float*>(), "abus"),
  };
  auto process = ctx.gcc.new_function(GCC_JIT_FUNCTION_EXPORTED,
    t_u32, ctx.graph_symbol("process"), process_args, 0
  );
  {
    auto entry = process.new_block("entry");
    auto p_state = process.get_param(0);
    auto done_action = process.new_local(t_u32, "done_action");
    entry.add_assignment(done_action, ctx.gcc.zero(t_u32));
    for (size_t i = 0; i < nodes.size(); i++) {
      nodes[i]->emit_proc(process, entry, make_node_state(p_state, i), done_action);
    }
    entry.end_with_return(done_action);
  }

  auto state_size = ctx.gcc.new_global(
    GCC_JIT_GLOBAL_EXPORTED, t_size_t, ctx.graph_symbol("state_size")
  );
  if (!state_struct) {
    state_size.set_initializer_rvalue(ctx.gcc.zero(t_size_t));
  } else {
    state_size.set_initializer_rvalue(new_sizeof(*state_struct));
  }

  auto control_descs = std::vector<Result::ControlDesc>{};
  if (!g.controlNames.empty()) {
    auto sorted = g.controlNames;
    std::sort(sorted.begin(), sorted.end(), [](auto const &left, auto const &right) {
      return left.index < right.index;
    });

    control_descs.reserve(sorted.size());
    for (size_t i = 0; i < sorted.size(); ++i) {
      auto const begin = sorted[i].index;
      auto const end = (i + 1 < sorted.size()) ? sorted[i + 1].index : g.controls.size();
      if (begin >= g.controls.size()) continue;
      if (end < begin) continue;

      control_descs.push_back(Result::ControlDesc{
        sorted[i].name,
        begin,
        end - begin,
        sorted[i].kind,
      });
    }
  }

  auto synths = std::vector<Result::SynthDescriptor>{};
  synths.push_back(Result::SynthDescriptor{
    g.name,
    g.controls,
    std::move(control_descs),
  });

  return Result{ctx.gcc.compile(), std::move(synths)};
}

}
