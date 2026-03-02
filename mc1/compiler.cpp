#include "compiler.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
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
template<> inline constexpr enum gcc_jit_types jit_type_v<float>  = GCC_JIT_TYPE_FLOAT;
template<> inline constexpr enum gcc_jit_types jit_type_v<size_t> = GCC_JIT_TYPE_SIZE_T;

struct Context
{
  gccjit::context gcc;
  unsigned int sample_rate;
  size_t block_size;
  DAG const& graph;

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

  gccjit::function sinf;

  std::unordered_map<std::string, gccjit::function> kernelCache;
  std::unordered_map<std::string, std::unique_ptr<StateTypeBase>> stateCache;

  Context(unsigned int sample_rate, size_t block_size, DAG const& graph)
  : gcc{gccjit::context::acquire()}
  , sample_rate{sample_rate}, block_size{block_size}, graph{graph}
  , sinf{[&]{
      auto params = std::vector{ gcc.new_param(type<float>(), "x") };
      return gcc.new_function(GCC_JIT_FUNCTION_IMPORTED,
        type<float>(), "sinf", params, 0
      );
    }()}
  , kernelCache{}, stateCache{}
  {}

  ~Context() { gcc.release(); }

  Context(Context const&) = delete;
  Context& operator=(Context const&) = delete;

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

  gccjit::rvalue new_float(float value)
  {
    return gcc.new_rvalue(type<float>(), static_cast<double>(value));
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
  std::optional<gccjit::lvalue> state;

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

  virtual void emit_init(gccjit::block) {};
  virtual void emit_proc(gccjit::function, gccjit::block) = 0;
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
    gccjit::function f, gccjit::block b, std::variant<gccjit::function, gccjit::rvalue> v
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
        call_args.reserve(args.size() + (rate() == 'a') + (state ? 1 : 0));

        if (state) call_args.push_back(state->get_address());
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

  void emit_proc(gccjit::function, gccjit::block) override
  {
    assert(args.size() == 1);
    rvalue = ctx.new_float(ctx.graph.constants[args.front()]);
  }
};

struct Control final : NonGraphArgs
{
  using NonGraphArgs::NonGraphArgs;

  void emit_proc(gccjit::function f, gccjit::block) override
  {
    assert(args.size() == 1);
    rvalue = f.get_param(0)[args.front()];
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

  void emit_proc(gccjit::function f, gccjit::block) override
  {
    assert(sig == "ba");
    assert(args.size() == 1);

    auto abus = f.get_param(1);
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

  void emit_proc(gccjit::function f, gccjit::block b) override
  {
    assert(sig == "baa");
    assert(args.size() == 2);
    assert(kernel);

    auto abus = f.get_param(1);

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

  void emit_proc(gccjit::function f, gccjit::block b) override
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

      auto c_bs_f = ctx.gcc.new_rvalue(ctx.type<float>(), double(ctx.block_size));

      auto tau = ctx.gcc.new_rvalue(ctx.type<float>(), 2.0 * std::numbers::pi_v<double>);
      auto tau_over_sr = ctx.gcc.new_rvalue(
        ctx.type<float>(), (2.0 * std::numbers::pi_v<double>) / double(ctx.sample_rate)
      );

      auto after_loop = ctx.loop(k, entry, [&](gccjit::block body, gccjit::block cont, gccjit::lvalue lv_i) {
        gccjit::rvalue freq  = (args[0]->rate() == 'a') ? p_freq[lv_i] : p_freq;
        gccjit::rvalue phofs = (args[1]->rate() == 'a') ? p_phase[lv_i] : p_phase;

        auto outv = ctx.sinf(lv_phase + phofs);
        body.add_assignment(p_out[lv_i], outv);

        // phase += tau*freq/sample_rate
        auto delta = freq * tau_over_sr;
        body.add_assignment(lv_phase, lv_phase + delta);

        // branchless wrap: phase -= tau * (phase >= tau)
        auto t_int = ctx.gcc.get_type(GCC_JIT_TYPE_INT);
        auto wrapped = ctx.gcc.new_cast(ctx.gcc.new_comparison(GCC_JIT_COMPARISON_GE, lv_phase, tau), t_int);
        auto wrapped_f = ctx.gcc.new_cast(wrapped, ctx.type<float>());
        body.add_assignment(lv_phase, lv_phase - (tau * wrapped_f));
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

  void emit_init(gccjit::block b) override
  {
    state = ctx.gcc.new_global(GCC_JIT_GLOBAL_INTERNAL, ST.st, std::format("s{}", vertex_index));

    b.add_assignment(state->access_field(ST.phase), ctx.gcc.zero(ctx.type<float>()));
  }

  void emit_proc(gccjit::function f, gccjit::block b) override
  {
    assert(rate() == 'a');
    assert(kernel);
    rvalue = new_proc_local(f, b, *kernel);
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

  void emit_proc(gccjit::function f, gccjit::block b) override
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

  std::vector<gccjit::param> init_args{};
  auto init = ctx.gcc.new_function(GCC_JIT_FUNCTION_EXPORTED,
    t_void, "init", init_args, 0
  );
  {
    auto entry = init.new_block("entry");
    for (auto &op: nodes) op->emit_init(entry);
    entry.end_with_return();
  }

  // process(const float *controls, float *abus)
  auto process_args = std::vector{
    ctx.gcc.new_param(ctx.type<const float*>(), "controls"),
    ctx.gcc.new_param(ctx.type<float*>(), "abus"),
  };
  auto process = ctx.gcc.new_function(GCC_JIT_FUNCTION_EXPORTED,
    t_void, "process", process_args, 0
  );
  {
    auto entry = process.new_block("entry");
    for (auto &op: nodes) op->emit_proc(process, entry);
    entry.end_with_return();
  }

  return Result{ctx.gcc.compile()};
}

}
