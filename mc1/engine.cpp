#include <iostream>
#include <ranges>
#include <utility>

#include <boost/asio.hpp>

#include "mlang/bytes.hpp"
#include "compiler.hpp"
#include "dag.hpp"
#include "mlang/gccjit.hpp"
#include "mlang/math.hpp"

using boost::asio::ip::udp;
using boost::asio::awaitable;
using boost::asio::buffer;
using boost::asio::co_spawn;
using boost::asio::detached;
using boost::asio::use_awaitable;

using mlang::numbers::tau;
using std::views::transform;
using mlang::views::sampled_interval;

namespace mc1 {

class engine final
{
  gccjit::context gcc;
  boost::asio::thread_pool compiler;

public:
  engine()
  : gcc{gccjit::context::acquire()}
  , compiler{1}
  {
    gcc.set_int_option(GCC_JIT_INT_OPTION_OPTIMIZATION_LEVEL, 3);
    gcc.set_bool_option(GCC_JIT_BOOL_OPTION_DUMP_INITIAL_GIMPLE, true);
    gcc.set_bool_option(GCC_JIT_BOOL_OPTION_DUMP_SUMMARY, true);
    make_tabled_function(gcc, "fast_sin", tau, 256, std::sin);
  }

  ~engine() { gcc.release(); }

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

  void packet_received(std::span<const std::byte> bytes)
  {
    if (auto i = mlang::get_value<unsigned short>(bytes)) {
      switch (*i) {
      case 0: std::cout << "quit" << std::endl; break;
      case 1:
        if (auto dag = DAG::parse(bytes)) {
          if (bytes.empty()) {
            post(compiler, [dag = std::move(dag.value()), this]()
            {
              compile_synth(std::move(dag));
            });
          }
        }
        break;
      default:
        std::cout << *i << std::endl;
      }
    }
  }

  void compile_synth(DAG dag)
  {
    std::cout << dag;
    compile(dag, 44100, 32);
  }
};

}


int main(int argc, char *argv[])
{
  boost::asio::io_context io;

  {
    mc1::engine world;

    {
      udp::socket socket{io, udp::endpoint(udp::v4(), std::atoi(argv[1]))};
      co_spawn(io, world.udp_server(std::move(socket)), detached);
    }

    io.run();
  }

  std::cout << "ended" << std::endl;

  return EXIT_SUCCESS;
}
