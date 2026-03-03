#include <iostream>
#include <ranges>
#include <utility>

#include <boost/asio.hpp>

#include "mlang/bytes.hpp"
#include "compiler.hpp"
#include "dag.hpp"

using boost::asio::ip::udp;
using boost::asio::awaitable;
using boost::asio::buffer;
using boost::asio::co_spawn;
using boost::asio::detached;
using boost::asio::use_awaitable;

namespace mc1 {

class engine final
{
  boost::asio::thread_pool compiler;
  bool running = true;

public:
  engine() : compiler{1} {}

  awaitable<void> udp_server(udp::socket socket)
  {
    std::byte data[1024];
    try {
      while (running) {
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
      case 0: running = false; break;
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

  return EXIT_SUCCESS;
}
