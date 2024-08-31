#include <pipewire/pipewire.h>
#include <libgccjit++.h>

#include <iostream>
#include <utility>
#include <boost/asio.hpp>

using boost::asio::ip::udp;
using boost::asio::awaitable;
using boost::asio::buffer;
using boost::asio::co_spawn;
using boost::asio::detached;
using boost::asio::use_awaitable;
using boost::asio::posix::stream_descriptor::wait_read;
using boost::asio::posix::stream_descriptor;
namespace this_coro = boost::asio::this_coro;



#include "bytes.hpp"
using mlang::get_pstring;
using mlang::get_value;
using mlang::get_values;
#include "math.hpp"
using mlang::tau;

namespace {

struct dag {
  std::vector<float> constants;
  std::vector<float> controls;
  struct op {
    std::string name;
    char rate;
    std::vector<unsigned short> args;

    static std::optional<op> parse(std::span<const std::byte> &bytes)
    {
      if (auto name = get_pstring(bytes)) {
        if (auto rate = get_value<char>(bytes)) {
          if (auto nargs = get_value<unsigned short>(bytes)) {
            if (auto args = get_values<unsigned short>(bytes, nargs.value())) {
              return op{
                std::move(name.value()),
                std::move(rate.value()),
                std::move(args.value())
              };
            }
          }
        }
      }
      return std::nullopt;
    }
  };
  std::vector<op> ops;

  static std::optional<dag> parse(std::span<const std::byte> bytes)
  {
    if (auto n = get_value<unsigned short>(bytes)) {
      if (auto consts = get_values<float>(bytes, n.value())) {
        if (auto nctrlvals = get_value<unsigned short>(bytes)) {
          if (auto ctrlvals = get_values<float>(bytes, nctrlvals.value())) {
            if (auto nops = get_value<unsigned short>(bytes)) {
              std::vector<op> ops;
              ops.reserve(nops.value());
              for (int i = 0; i != nops.value(); i++) {
                auto op = op::parse(bytes);
                if (!op) return std::nullopt;
                ops.emplace_back(std::move(op.value()));
              }

              if (bytes.empty()) {
                return dag{
                  std::move(consts.value()),
                  std::move(ctrlvals.value()),
                  std::move(ops)
                };
              }
            }
	  }
        }
      }
    }
    return std::nullopt;
  }
};

// Overload for << operator to print the dag structure
std::ostream& operator<<(std::ostream& os, const dag& d) {
  os << "Constants: ";
  for (const auto& constant : d.constants) {
    os << constant << " ";
  }
  os << "\nControls: ";
  for (const auto& control : d.controls) {
    os << control << " ";
  }
  os << "\nOperations:\n";
  for (const auto& operation : d.ops) {
    os << "  Name: " << operation.name << "\n";
    os << "  Rate: " << operation.rate << "\n";
    os << "  Args: ";
    for (const auto& arg : operation.args) {
      os << arg << " ";
    }
    os << "\n";
  }
  return os;
}

namespace pipewire {

using main_loop_ptr = std::unique_ptr<pw_main_loop, void (*)(pw_main_loop*)>;
using stream_ptr = std::unique_ptr<pw_stream, void (*)(pw_stream*)>;

main_loop_ptr make_main_loop(const spa_dict *props = nullptr)
{ return {pw_main_loop_new(props), pw_main_loop_destroy}; }

pw_loop *get_loop(main_loop_ptr const &main_loop)
{ return pw_main_loop_get_loop(main_loop.get()); }

stream_ptr make_stream(main_loop_ptr const &main_loop,
  const char *name, pw_properties *props, const pw_stream_events *events,
  void *data
) {
  return {
    pw_stream_new_simple(get_loop(main_loop), name, props, events, data),
    pw_stream_destroy
  };
}

}

class engine {
  pipewire::main_loop_ptr main_loop;

public:
  engine() : main_loop{pipewire::make_main_loop()} {}

  awaitable<void> pipewire() {
    auto loop = pipewire::get_loop(main_loop);
    stream_descriptor fd(co_await this_coro::executor, pw_loop_get_fd(loop));

    try {
      for (;;) {
        co_await fd.async_wait(wait_read, use_awaitable);
        pw_loop_iterate(loop, -1);
      }
    } catch (std::exception& e) {
      std::cerr << e.what() << std::endl;
    }
  }

  awaitable<void> udp_server(udp::socket socket)
  {
    std::byte data[1024];
    try {
      for (;;) {
        udp::endpoint sender;
        size_t n = co_await socket.async_receive_from(buffer(data), sender, use_awaitable);
        packet_received(std::span(&data[0], n));
      }
    } catch (std::exception& e) {
      std::cerr << e.what() << std::endl;
    }
  }

  void packet_received(std::span<const std::byte> view) {
    if (auto i = get_value<unsigned short>(view)) {
      switch (*i) {
      case 0: std::cout << "quit" << std::endl; break;
      case 1:
        if (auto dag = dag::parse(view)) {
          std::cout << dag.value();
        }
        break;
      default:
        std::cout << *i << std::endl;
      }
    }
  }
};

}

template <typename F>
gccjit::lvalue make_table(gccjit::context gcc, std::string name, size_t n, F f)
{
  auto fp_type = gcc.get_type(GCC_JIT_TYPE_DOUBLE);
  auto array_type = gcc.new_array_type(fp_type.get_const(), n + 1);
  auto lvalue = gcc.new_global(GCC_JIT_GLOBAL_INTERNAL, array_type, "tab_" + name);
  std::vector<gccjit::rvalue> init;
  for (size_t i = 0; i < n; i++) {
    init.push_back(gcc.new_rvalue(fp_type, static_cast<double>(f(i, n))));
  }
  init.push_back(init.front());
  lvalue.set_initializer_rvalue(gcc.new_array_ctor(array_type, init));

  return lvalue;  
}

gccjit::function lookup(gccjit::context gcc, gccjit::rvalue ptr) {
  auto fp_type = gcc.get_type(GCC_JIT_TYPE_DOUBLE);
  auto index_type = gcc.get_type(GCC_JIT_TYPE_SIZE_T);
  auto param = std::vector<gccjit::param> {
    gcc.new_param(fp_type, "index")
  };
  auto func = gcc.new_function(GCC_JIT_FUNCTION_EXPORTED, fp_type, "ix_sin", param, 0);
  auto i = func.new_local(index_type, "i");
  auto a = func.new_local(fp_type, "a");
  auto b = func.new_local(fp_type, "b");
  auto block = func.new_block("entry");
  block.add_assignment(i, gcc.new_cast(func.get_param(0), i.get_type())); 
  block.add_assignment(a, gcc.new_array_access(ptr, i));
  block.add_assignment(b, gcc.new_array_access(ptr, i + index_type.one()));
  block.end_with_return(
    gcc.get_builtin_function("__builtin_fma")(
      func.get_param(0) - gcc.new_cast(i, fp_type), b - a, a
    )
  );
  return func;
}

int main(int argc, char *argv[]) {
  pw_init(&argc, &argv);
  auto gcc = gccjit::context::acquire();
  gcc.set_int_option(GCC_JIT_INT_OPTION_OPTIMIZATION_LEVEL, 3);
  auto t = make_table(gcc, "sin", 1024, [](size_t i, size_t n) {
    return std::sin(tau<double> * i / n);
  });
  auto ix_sin = lookup(gcc, t);
  gcc.dump_to_file("xxx.gimple", false);
  gcc.compile_to_file(GCC_JIT_OUTPUT_KIND_ASSEMBLER, "xxx.s");
  auto result = gcc.compile();

  gcc.release();


  boost::asio::io_context io;
  engine world;

  udp::socket socket{io, udp::endpoint(udp::v4(), std::atoi(argv[1]))};
  co_spawn(io, world.udp_server(std::move(socket)), detached);
  co_spawn(io, world.pipewire(), detached);

  io.run();

  pw_deinit();

  return 0;
}
