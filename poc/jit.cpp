#include <cassert>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

#include <libgccjit++.h>

struct CacheKey {
  std::string name;
  std::string sig;
  bool operator==(CacheKey const&) const = default;
};

struct CacheKeyHash {
  size_t operator()(CacheKey const& k) const noexcept
  {
    return std::hash<std::string>{}(k.name) ^ (std::hash<std::string>{}(k.sig) << 1);
  }
};

template<class T>
using Cache = std::unordered_map<CacheKey, T, CacheKeyHash>;

static Cache<gccjit::function> kernelCache;
static Cache<gccjit::struct_> stateCache;

struct Vertex
{
  std::string name;
  char rate;
  std::vector<size_t> args;
};

struct Graph
{
  std::vector<float> constants;
  std::vector<Vertex> vertices;
};

struct Compiler
{
  gccjit::context &gcc;
  unsigned int sample_rate;
  size_t block_size;
  Graph const& graph;
};

class Opcode
{
protected:
  Compiler const& c;
  std::string name, sig;
  size_t vertex_index;
  gccjit::rvalue rvalue;

  gccjit::rvalue new_float(float value) const
  {
    auto type = c.gcc.get_type(GCC_JIT_TYPE_FLOAT);
    return c.gcc.new_rvalue(type, static_cast<double>(value));
  }

  void loop(gccjit::function fn, gccjit::block entry, auto &&body_fn) const
  {
    auto t_size_t = c.gcc.get_type(GCC_JIT_TYPE_SIZE_T);

    auto c_0_size  = c.gcc.zero(t_size_t);
    auto c_BS_size = c.gcc.new_rvalue(t_size_t, long(c.block_size));

    auto cond  = fn.new_block("cond");
    auto body  = fn.new_block("body");
    auto inc   = fn.new_block("inc");
    auto done  = fn.new_block("done");

    auto lv_i = fn.new_local(t_size_t, "i");
    entry.add_assignment(lv_i, c_0_size);
    entry.end_with_jump(cond);

    auto cnd = c.gcc.new_comparison(GCC_JIT_COMPARISON_LT, lv_i, c_BS_size);
    cond.end_with_conditional(cnd, body, done);

    body_fn(body, lv_i);
    body.end_with_jump(inc);

    auto i_next = c.gcc.new_binary_op(GCC_JIT_BINARY_OP_PLUS, t_size_t, lv_i, c.gcc.one(t_size_t));
    inc.add_assignment(lv_i, i_next);
    inc.end_with_jump(cond);

    done.end_with_return();
  }

public:
  Opcode(
    Compiler const& c,
    std::string name, std::string sig, size_t vertex_index
  )
  : c{c}, name{std::move(name)}, sig{std::move(sig)}, vertex_index{vertex_index}
  , rvalue{}
  {}

  virtual ~Opcode() = default;

  char rate() const { return sig.back(); }

  gccjit::rvalue get_rvalue() const
  {
    assert(rvalue.get_inner_rvalue() != nullptr);
    return rvalue;
  }

  virtual void emit_init(gccjit::block) {};
  virtual void emit_proc(gccjit::function, gccjit::block) = 0;
};

static gccjit::rvalue get_address_of_first_array_element(gccjit::lvalue lvalue)
{
  auto gcc = lvalue.get_context();
  auto t_size_t = gcc.get_type(GCC_JIT_TYPE_SIZE_T);
  return gcc.new_array_access(lvalue, gcc.zero(t_size_t)).get_address();
}

class NonGraphArgs : public Opcode
{
protected:
  std::vector<size_t> args;

public:
  NonGraphArgs(
    Compiler const& c,
    std::string name, std::string sig, size_t vertex_index,
    std::vector<size_t> args
  )
  : Opcode(c, std::move(name), std::move(sig), vertex_index)
  , args{std::move(args)}
  {}
};

template<class T> concept SpecialIndices = std::is_base_of_v<NonGraphArgs, T>;

class GraphArgs : public Opcode
{
protected:
  std::vector<Opcode*> args;

public:
  GraphArgs(
    Compiler const& c,
    std::string name, std::string sig, size_t vertex_index,
    std::vector<Opcode*> args
  )
  : Opcode(c, std::move(name), std::move(sig), vertex_index)
  , args{std::move(args)}
  {}
};

class Const final : public NonGraphArgs
{
public:
  using NonGraphArgs::NonGraphArgs;

  void emit_proc(gccjit::function, gccjit::block) override
  {
    assert(args.size() == 1);
    rvalue = new_float(c.graph.constants[args.front()]);
  }
};

class Control final : public NonGraphArgs
{
public:
  using NonGraphArgs::NonGraphArgs;

  void emit_proc(gccjit::function f, gccjit::block) override
  {
    assert(args.size() == 1);
    auto controls = f.get_param(0);
    auto index = c.gcc.new_rvalue(c.gcc.get_type(GCC_JIT_TYPE_SIZE_T), long(args.front()));
    rvalue = c.gcc.new_array_access(controls, index);
  }
};

class In final : public GraphArgs
{
public:
  In(
    Compiler const& c,
    std::string name, std::string sig, size_t vertex_index,
    std::vector<Opcode*> args
  )
  : GraphArgs(c, std::move(name), std::move(sig), vertex_index, std::move(args))
  {}

  void emit_proc(gccjit::function f, gccjit::block) override
  {
    assert(sig == "ba");
    assert(args.size() == 1);

    auto t_size_t  = c.gcc.get_type(GCC_JIT_TYPE_SIZE_T);
    auto t_float   = c.gcc.get_type(GCC_JIT_TYPE_FLOAT);

    auto abus = f.get_param(1);

    // roundf(arg0) -> size_t index
    auto idx_f = args[0]->get_rvalue();

    auto roundf_args = std::vector{ c.gcc.new_param(t_float, "x") };
    auto fn_roundf = c.gcc.new_function(GCC_JIT_FUNCTION_IMPORTED,
      t_float, "roundf", roundf_args, 0
    );

    auto idx_rf = c.gcc.new_call(fn_roundf, { idx_f });
    auto idx = c.gcc.new_cast(idx_rf, t_size_t);

    auto c_bs = c.gcc.new_rvalue(t_size_t, long(c.block_size));
    auto off = c.gcc.new_binary_op(GCC_JIT_BINARY_OP_MULT, t_size_t, idx, c_bs);

    rvalue = c.gcc.new_array_access(abus, off).get_address();
  }
};

class Out final : public GraphArgs
{
  gccjit::function kernel;

  gccjit::function make_kernel() const
  {
    // Only support: Out baa
    if (sig != "baa") return {};

    auto t_void   = c.gcc.get_type(GCC_JIT_TYPE_VOID);
    auto t_float  = c.gcc.get_type(GCC_JIT_TYPE_FLOAT);

    auto t_float_ptr = t_float.get_pointer();
    auto t_const_float_ptr = t_float.get_const().get_pointer();

    auto fn_name = std::format("{}_{}", name, sig);

    // void Out_baa(float *dst, const float *src)
    auto p_dst = c.gcc.new_param(t_float_ptr, "dst");
    auto p_src = c.gcc.new_param(t_const_float_ptr, "src");
    auto params = std::vector{p_dst, p_src};
    auto kernel = c.gcc.new_function(GCC_JIT_FUNCTION_INTERNAL,
      t_void, fn_name, params, 0
    );
    {
      auto entry = kernel.new_block("entry");
      loop(kernel, entry, [&](gccjit::block body, gccjit::lvalue lv_i) {
        auto dst_i = c.gcc.new_array_access(p_dst, lv_i);
        auto src_i = c.gcc.new_array_access(p_src, lv_i);
        body.add_assignment(dst_i, src_i);
      });
    }

    return kernel;
  }

  gccjit::function get_or_make_kernel() const
  {
    CacheKey key{name, sig};
    if (auto it = kernelCache.find(key); it != kernelCache.end())
      return it->second;

    auto created = make_kernel();
    kernelCache.emplace(std::move(key), created);
    return created;
  }

public:
  Out(
    Compiler const& c,
    std::string name, std::string sig, size_t vertex_index,
    std::vector<Opcode*> args
  )
  : GraphArgs(c, std::move(name), std::move(sig), vertex_index, std::move(args))
  , kernel{get_or_make_kernel()}
  {}

  void emit_proc(gccjit::function f, gccjit::block b) override
  {
    assert(sig == "baa");
    assert(args.size() == 2);

    auto t_size_t  = c.gcc.get_type(GCC_JIT_TYPE_SIZE_T);
    auto t_float   = c.gcc.get_type(GCC_JIT_TYPE_FLOAT);

    auto abus = f.get_param(1);

    // roundf(arg0) -> size_t index
    auto idx_f = args[0]->get_rvalue();

    auto roundf_args = std::vector{ c.gcc.new_param(t_float, "x") };
    auto fn_roundf = c.gcc.new_function(GCC_JIT_FUNCTION_IMPORTED,
      t_float, "roundf", roundf_args, 0
    );

    auto idx_rf = c.gcc.new_call(fn_roundf, { idx_f });
    auto idx = c.gcc.new_cast(idx_rf, t_size_t);

    auto c_bs = c.gcc.new_rvalue(t_size_t, long(c.block_size));
    auto off = c.gcc.new_binary_op(GCC_JIT_BINARY_OP_MULT, t_size_t, idx, c_bs);
    auto dst = c.gcc.new_array_access(abus, off).get_address();

    auto src = args[1]->get_rvalue();
    auto call_args = std::vector{ dst, src };
    b.add_eval(c.gcc.new_call(kernel, call_args));

    rvalue = src;
  }
};

class BinOp final : public GraphArgs
{
  enum gcc_jit_binary_op op_kind;
  gccjit::function kernel;

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

  gccjit::function make_kernel() const
  {
    if (sig == "bbb") return {};

    auto t_void   = c.gcc.get_type(GCC_JIT_TYPE_VOID);
    auto t_float  = c.gcc.get_type(GCC_JIT_TYPE_FLOAT);

    auto t_float_ptr = t_float.get_pointer();
    auto t_const_float_ptr = t_float.get_const().get_pointer();

    auto fn_name = std::format("{}_{}", name, sig);

    // void f(float *r, <a>, <b>) where <a>/<b> are float or float* depending
    gccjit::param p_r = c.gcc.new_param(t_float_ptr, "r");
    gccjit::type t_a = (sig[0] == 'a') ? t_const_float_ptr : t_float;
    gccjit::type t_b = (sig[1] == 'a') ? t_const_float_ptr : t_float;
    gccjit::param p_a = c.gcc.new_param(t_a, "a");
    gccjit::param p_b = c.gcc.new_param(t_b, "b");

    std::vector<gccjit::param> params{p_r, p_a, p_b};
    auto kernel = c.gcc.new_function(GCC_JIT_FUNCTION_INTERNAL, t_void, fn_name, params, 0);

    auto entry = kernel.new_block("entry");
    loop(kernel, entry, [&](gccjit::block body, gccjit::lvalue lv_i) {
      auto lv_r_i = c.gcc.new_array_access(p_r, lv_i);

      gccjit::rvalue ra =
        (sig[0] == 'a')
          ? c.gcc.new_array_access(p_a, lv_i)
          : p_a;

      gccjit::rvalue rb =
        (sig[1] == 'a')
          ? c.gcc.new_array_access(p_b, lv_i)
          : p_b;

      auto expr = c.gcc.new_binary_op(op_kind, t_float, ra, rb);
      body.add_assignment(lv_r_i, expr);
    });

    return kernel;
  }

  gccjit::function get_or_make_kernel() const
  {
    CacheKey key{name, sig};
    if (auto it = kernelCache.find(key); it != kernelCache.end())
      return it->second;

    auto created = make_kernel();
    kernelCache.emplace(std::move(key), created);
    return created;
  }

public:
  BinOp(
    Compiler const& c,
    std::string name, std::string sig, size_t vertex_index,
    std::vector<Opcode*> args
  )
  : GraphArgs(c, std::move(name), std::move(sig), vertex_index, std::move(args))
  , op_kind{kind_from_name(this->name)}
  , kernel{get_or_make_kernel()}
  {}

  void emit_proc(gccjit::function f, gccjit::block b) override
  {
    assert(args.size() == 2);
    assert(sig.size() == 3);
    assert(is_sig_supported(sig));

    auto t_float = c.gcc.get_type(GCC_JIT_TYPE_FLOAT);

    if (sig == "bbb") {
      assert(rate() == 'b');
      auto lv_tmp = f.new_local(t_float, std::format("e{}", vertex_index));
      auto expr = c.gcc.new_binary_op(op_kind,
        t_float, args[0]->get_rvalue(), args[1]->get_rvalue()
      );
      b.add_assignment(lv_tmp, expr);
      rvalue = lv_tmp;
      return;
    }

    assert(rate() == 'a');

    auto t_float_array_BS = c.gcc.new_array_type(t_float, int(c.block_size));
    auto lv_buf = f.new_local(t_float_array_BS, std::format("e{}", vertex_index).c_str());

    std::vector<gccjit::rvalue> call_args{
      get_address_of_first_array_element(lv_buf),
      args[0]->get_rvalue(),
      args[1]->get_rvalue()
    };
    b.add_eval(c.gcc.new_call(kernel, call_args));
    rvalue = get_address_of_first_array_element(lv_buf);
  }
};

class Registry
{
  using nongraph_args_t = std::vector<size_t>;
  using graph_args_t = std::vector<Opcode*>;
  using make_nongraph_fn = std::unique_ptr<Opcode>(Compiler const&,std::string,std::string,size_t,nongraph_args_t);
  using make_graph_fn = std::unique_ptr<Opcode>(Compiler const&,std::string,std::string,size_t,graph_args_t);

  std::unordered_map<std::string, std::variant<make_nongraph_fn*, make_graph_fn*>> maker;

  template<class T> static std::unique_ptr<Opcode> make_nongraph(
    Compiler const& c,
    std::string name, std::string sig, size_t vertex_index,
    nongraph_args_t args
  )
  { return std::make_unique<T>(c, name, sig, vertex_index, std::move(args)); }

  template<class T> static std::unique_ptr<Opcode> make_graph(
    Compiler const& c,
    std::string name, std::string sig, size_t vertex_index,
    graph_args_t args
  )
  { return std::make_unique<T>(c, name, sig, vertex_index, std::move(args)); }

public:
  Registry()
  {
    kernelCache.clear();
    stateCache.clear();

    emplace<Const>("Const");
    emplace<Control>("Control");
    emplace<In>("In");
    emplace<Out>("Out");

    emplace<BinOp>("Mul");
    emplace<BinOp>("Add");
    emplace<BinOp>("Sub");
    emplace<BinOp>("Div");
  }

  bool is_nongraph_args(std::string const& name) const
  {
    auto it = maker.find(name);
    if (it == maker.end())
      throw std::runtime_error(std::format("Opcode '{}' not found", name));
    return std::holds_alternative<make_nongraph_fn*>(it->second);
  }

  std::string compute_signature(const Graph &g, size_t i) const
  {
    auto const &v = g.vertices.at(i);

    size_t n_in = is_nongraph_args(v.name) ? 0 : v.args.size();

    std::string sig;
    sig.reserve(n_in + 1);

    for (size_t k = 0; k < n_in; k++) {
      size_t arg_i = v.args.at(k);
      sig.push_back(g.vertices.at(arg_i).rate);
    }

    sig.push_back(v.rate);
    return sig;
  }

  std::unique_ptr<Opcode> create(
    Compiler const& c,
    std::string name, std::string sig, size_t vertex_index,
    nongraph_args_t args
  )
  {
    auto iter = maker.find(name);
    if (iter == maker.end()) throw std::runtime_error("Not found");

    auto *fn = std::get<make_nongraph_fn *>(iter->second);
    return fn(c, std::move(name), std::move(sig), vertex_index, std::move(args));
  }

  std::unique_ptr<Opcode> create(
    Compiler const& c,
    std::string name, std::string sig, size_t vertex_index,
    graph_args_t args
  )
  {
    auto iter = maker.find(name);
    if (iter == maker.end()) throw std::runtime_error("Not found");

    auto *fn = std::get<make_graph_fn *>(iter->second);
    return fn(c, std::move(name), std::move(sig), vertex_index, std::move(args));
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

gcc_jit_result *compile(const Graph &g, unsigned int sample_rate, size_t block_size)
{
  auto gcc = gccjit::context::acquire();
  gcc.set_bool_option(GCC_JIT_BOOL_OPTION_DUMP_INITIAL_GIMPLE, true);
  Compiler compiler{gcc, sample_rate, block_size, g};
  Registry opcodes;
  std::vector<std::unique_ptr<Opcode>> ops;
  ops.reserve(g.vertices.size());
  for (size_t i = 0; i < g.vertices.size(); i++) {
    auto &vertex = g.vertices[i];
    auto sig = opcodes.compute_signature(g, i);
    if (opcodes.is_nongraph_args(vertex.name)) {
      ops.push_back(opcodes.create(compiler, vertex.name, sig, i, vertex.args));
      continue;
    }
    std::vector<Opcode*> args;
    for (auto arg: vertex.args) {
      assert(arg < i);
      args.push_back(ops[arg].get());
    }
    ops.push_back(opcodes.create(compiler, vertex.name, sig, i, std::move(args)));
  }

  std::vector<gccjit::param> init_args{};
  auto init = gcc.new_function(GCC_JIT_FUNCTION_EXPORTED,
    gcc.get_type(GCC_JIT_TYPE_VOID), "init", init_args, 0
  );
  {
    auto entry = init.new_block("entry");
    for (auto &op: ops) op->emit_init(entry);
    entry.end_with_return();
  }

  auto t_void  = gcc.get_type(GCC_JIT_TYPE_VOID);
  auto t_float = gcc.get_type(GCC_JIT_TYPE_FLOAT);

  auto t_const_float_ptr = t_float.get_const().get_pointer();
  auto t_float_ptr = t_float.get_pointer();

  // process(const float *controls, float *abus)
  auto process_args = std::vector{
    gcc.new_param(t_const_float_ptr, "controls"),
    gcc.new_param(t_float_ptr, "abus"),
  };

  auto process = gcc.new_function(GCC_JIT_FUNCTION_EXPORTED,
    t_void, "process", process_args, 0
  );
  {
    auto entry = process.new_block("entry");
    for (auto &op: ops) op->emit_proc(process, entry);
    entry.end_with_return();
  }

  gcc_jit_result *result = gcc.compile();
  gcc.release();
  return result;
}

int main()
{
  Graph g{
    .constants = {0.0f, 0.2f},
    .vertices = {
      {"Control", 'b', {0}},
      {"Const",   'b', {0}},
      {"SinOsc",  'a', {0, 1}},
      {"Const",   'b', {1}},
      {"Mul",     'a', {2, 3}},
    }
  };

  constexpr size_t BS_V = 128;

  gcc_jit_result *r = compile(g, 44100u, BS_V);
  (void)r;
}
