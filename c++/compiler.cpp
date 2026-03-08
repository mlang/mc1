#include "compiler.hpp"

#include <numbers>

#include "mlang/gccjit.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
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

  gccjit::function new_kernel(std::vector<gccjit::param> params) const
  {
    return ctx.gcc.new_function(GCC_JIT_FUNCTION_INTERNAL,
      ctx.gcc.get_type(rate() == 'a' ? GCC_JIT_TYPE_VOID : GCC_JIT_TYPE_FLOAT),
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
  virtual void emit_proc(gccjit::function, gccjit::block, std::optional<gccjit::lvalue>) = 0;
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

  void emit_proc(gccjit::function, gccjit::block, std::optional<gccjit::lvalue>) override
  {
    assert(args.size() == 1);
    rvalue = ctx.new_float(ctx.graph.constants[args.front()]);
  }
};

struct Control final : NonGraphArgs
{
  using NonGraphArgs::NonGraphArgs;

  void emit_proc(gccjit::function f, gccjit::block, std::optional<gccjit::lvalue>) override
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

  void emit_proc(gccjit::function f, gccjit::block, std::optional<gccjit::lvalue>) override
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

  void emit_proc(gccjit::function f, gccjit::block b, std::optional<gccjit::lvalue>) override
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

  void emit_proc(gccjit::function f, gccjit::block b, std::optional<gccjit::lvalue>) override
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

  void emit_proc(gccjit::function f, gccjit::block b, std::optional<gccjit::lvalue> state_field) override
  {
    assert(rate() == 'a');
    assert(kernel);
    assert(state_field);
    rvalue = new_proc_local(f, b, *kernel, state_field->get_address());
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

  void emit_proc(gccjit::function f, gccjit::block b, std::optional<gccjit::lvalue>) override
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

  // process(void *state, const float *controls, float *abus)
  auto process_args = std::vector{
    ctx.gcc.new_param(t_void_ptr, "state"),
    ctx.gcc.new_param(ctx.type<const float*>(), "controls"),
    ctx.gcc.new_param(ctx.type<float*>(), "abus"),
  };
  auto process = ctx.gcc.new_function(GCC_JIT_FUNCTION_EXPORTED,
    t_void, ctx.graph_symbol("process"), process_args, 0
  );
  {
    auto entry = process.new_block("entry");
    auto p_state = process.get_param(0);
    for (size_t i = 0; i < nodes.size(); i++) {
      nodes[i]->emit_proc(process, entry, make_node_state(p_state, i));
    }
    entry.end_with_return();
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
        end - begin
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
