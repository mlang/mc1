#include <cassert>
#include <memory>
#include <print>
#include <regex>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <variant>

#include <libgccjit++.h>

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

class Opcode
{
protected:
  mutable gccjit::context gcc;
  std::string name, sig;
  size_t vertex_index;
  gccjit::rvalue rvalue;

  gccjit::rvalue new_float(float value) const
  {
    auto type = gcc.get_type(GCC_JIT_TYPE_FLOAT);
    return gcc.new_rvalue(type, static_cast<double>(value));
  }

public:
  Opcode(
    gccjit::context gcc,
    std::string name,
    std::string sig,
    size_t vertex_index
  )
  : gcc{gcc},
    name{std::move(name)},
    sig{std::move(sig)},
    vertex_index{vertex_index},
    rvalue{}
  {}

  virtual ~Opcode() = default;

  char rate() const { return sig.back(); }

  gccjit::rvalue get_rvalue() const
  {
    assert(rvalue.get_inner_rvalue() != nullptr);
    return rvalue;
  }

  virtual void emit_proc(const Graph &) = 0;
};

class NonGraphArgs : public Opcode
{
protected:
  std::vector<size_t> args;

public:
  NonGraphArgs(
    gccjit::context gcc,
    std::string name,
    std::string sig,
    size_t vertex_index,
    std::vector<size_t> args
  )
  : Opcode(gcc, std::move(name), std::move(sig), vertex_index),
    args{std::move(args)}
  {}
};

template<class T>
concept SpecialIndices = std::is_base_of_v<NonGraphArgs, T>;

class GraphArgs : public Opcode
{
protected:
  std::vector<Opcode*> args;

public:
  GraphArgs(
    gccjit::context gcc,
    std::string name,
    std::string sig,
    size_t vertex_index,
    std::vector<Opcode*> args
  )
  : Opcode(gcc, std::move(name), std::move(sig), vertex_index),
    args{std::move(args)}
  {}
};

class Const : public NonGraphArgs
{
public:
  using NonGraphArgs::NonGraphArgs;

  void emit_proc(const Graph &g) override
  {
    assert(args.size() == 1);
    rvalue = new_float(g.constants[args.front()]);
  }
};

class Registry
{
  using nongraph_args_t = std::vector<size_t>;
  using graph_args_t = std::vector<Opcode*>;
  using make_nongraph_fn = std::unique_ptr<Opcode>(gccjit::context,std::string,std::string,size_t,nongraph_args_t);
  using make_graph_fn = std::unique_ptr<Opcode>(gccjit::context,std::string,std::string,size_t,graph_args_t);

  struct SpecialEntry {
    std::regex sig_re;
    make_nongraph_fn *make_fn;
  };

  struct Entry {
    std::regex sig_re;
    make_graph_fn *make_fn;
  };

  using SpecialEntries = std::vector<SpecialEntry>;
  using Entries = std::vector<Entry>;

  std::unordered_map<std::string, std::variant<SpecialEntries, Entries>> maker;

  template<class T>
  static std::unique_ptr<Opcode> make_nongraph(
    gccjit::context gcc, std::string name, std::string sig, size_t vertex_index, nongraph_args_t args
  )
  { return std::make_unique<T>(gcc, name, sig, vertex_index, std::move(args)); }

  template<class T>
  static std::unique_ptr<Opcode> make_graph(
    gccjit::context gcc, std::string name, std::string sig, size_t vertex_index, graph_args_t args
  )
  { return std::make_unique<T>(gcc, name, sig, vertex_index, std::move(args)); }

  SpecialEntry const* find_special_entry(std::string const& name, std::string const& sig) const
  {
    auto it = maker.find(name);
    if (it == maker.end()) return nullptr;
    auto const* entries = std::get_if<SpecialEntries>(&it->second);
    if (!entries) return nullptr;

    for (auto const& e : *entries) {
      if (std::regex_match(sig, e.sig_re))
        return &e;
    }

    return nullptr;
  }

  Entry const* find_entry(std::string const& name, std::string const& sig) const
  {
    auto it = maker.find(name);
    if (it == maker.end()) return nullptr;
    auto const* entries = std::get_if<Entries>(&it->second);
    if (!entries) return nullptr;

    for (auto const& e : *entries) {
      if (std::regex_match(sig, e.sig_re))
        return &e;
    }

    return nullptr;
  }

public:
  bool is_nongraph_args(std::string const& name) const
  {
    auto it = maker.find(name);
    if (it == maker.end()) throw std::runtime_error("Not found");
    return std::holds_alternative<SpecialEntries>(it->second);
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
    gccjit::context gcc,
    std::string name,
    std::string sig,
    size_t vertex_index,
    nongraph_args_t args
  )
  {
    auto const* e = find_special_entry(name, sig);
    if (!e) throw std::runtime_error("Not found");

    return e->make_fn(gcc, std::move(name), std::move(sig), vertex_index, std::move(args));
  }

  std::unique_ptr<Opcode> create(
    gccjit::context gcc,
    std::string name,
    std::string sig,
    size_t vertex_index,
    graph_args_t args
  )
  {
    auto const* e = find_entry(name, sig);
    if (!e) throw std::runtime_error("Not found");

    return e->make_fn(gcc, std::move(name), std::move(sig), vertex_index, std::move(args));
  }

  template<class T>
  requires SpecialIndices<T>
  void emplace(std::string name, std::string sig_regex)
  {
    auto it = maker.find(name);
    if (it == maker.end()) {
      maker.emplace(
        std::move(name),
        SpecialEntries{ SpecialEntry{std::regex(std::move(sig_regex)), &make_nongraph<T>} }
      );
      return;
    }

    auto &v = it->second;
    if (auto *entries = std::get_if<SpecialEntries>(&v)) {
      entries->push_back(SpecialEntry{std::regex(std::move(sig_regex)), &make_nongraph<T>});
      return;
    }

    throw std::runtime_error("Registry name already used for graph-args opcode");
  }

  template<class T>
  requires (!SpecialIndices<T>)
  void emplace(std::string name, std::string sig_regex)
  {
    auto it = maker.find(name);
    if (it == maker.end()) {
      maker.emplace(
        std::move(name),
        Entries{ Entry{std::regex(std::move(sig_regex)), &make_graph<T>} }
      );
      return;
    }

    auto &v = it->second;
    if (auto *entries = std::get_if<Entries>(&v)) {
      entries->push_back(Entry{std::regex(std::move(sig_regex)), &make_graph<T>});
      return;
    }

    throw std::runtime_error("Registry name already used for special-indices opcode");
  }
};

int main()
{
  Registry opcodes;
  opcodes.emplace<Const>("Const", "b");
  assert(opcodes.is_nongraph_args("Const"));
}
