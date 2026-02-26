#include <cassert>
#include <memory>
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

  struct Entry {
    std::regex sig_re;
    std::variant<make_nongraph_fn*, make_graph_fn*> make_fn;
  };

  std::unordered_map<std::string, std::vector<Entry>> maker;

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

  Entry const* find_entry(std::string const& name, std::string const& sig) const
  {
    auto it = maker.find(name);
    if (it == maker.end()) return nullptr;

    for (auto const& e : it->second) {
      if (std::regex_match(sig, e.sig_re))
        return &e;
    }

    return nullptr;
  }

public:
  bool is_nongraph_args(std::string const& name, std::string const& sig) const
  {
    auto const* e = find_entry(name, sig);
    if (!e) throw std::runtime_error("Not found");
    return std::get_if<make_nongraph_fn*>(&e->make_fn) != nullptr;
  }

  std::unique_ptr<Opcode> create(
    gccjit::context gcc,
    std::string name,
    std::string sig,
    size_t vertex_index,
    nongraph_args_t args
  )
  {
    auto const* e = find_entry(name, sig);
    if (!e) throw std::runtime_error("Not found");

    auto *fn = std::get<make_nongraph_fn*>(e->make_fn);
    return fn(gcc, std::move(name), std::move(sig), vertex_index, std::move(args));
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

    auto *fn = std::get<make_graph_fn*>(e->make_fn);
    return fn(gcc, std::move(name), std::move(sig), vertex_index, std::move(args));
  }

  template<class T>
  requires SpecialIndices<T>
  void emplace(std::string name, std::string sig_regex)
  {
    maker[std::move(name)].push_back(
      Entry{std::regex(std::move(sig_regex)), &make_nongraph<T>}
    );
  }

  template<class T>
  requires (!SpecialIndices<T>)
  void emplace(std::string name, std::string sig_regex)
  {
    maker[std::move(name)].push_back(
      Entry{std::regex(std::move(sig_regex)), &make_graph<T>}
    );
  }
};

int main()
{
  Registry opcodes;
  opcodes.emplace<Const>("Const", "b");
}
