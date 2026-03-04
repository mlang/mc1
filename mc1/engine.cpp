#include <chrono>
#include <iostream>
#include <print>
#include <ranges>
#include <utility>
#include <vector>

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
    constexpr unsigned int SR = 44100;
    constexpr size_t BS = 32;

    auto r = compile(dag, SR, BS);
    auto s = r[dag.name];
    s.init();

    std::vector<float> controls = dag.controls;

    // Two-channel audiobus: channel-major layout [ch0 block][ch1 block]
    std::vector<float> abus(2 * BS, 0.0f);

    for (int iter = 0; iter < 10; ++iter) {
      s.process(controls.data(), abus.data());

      std::println("process call {}", iter);
      std::println("i\tch0\tch1");
      for (size_t i = 0; i < BS; ++i) {
        float ch0 = abus[i];
        float ch1 = abus[BS + i];
        std::println("{}\t{}\t{}", i, ch0, ch1);
      }
      std::println("");
    }

    const size_t nblocks = SR / BS;

    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < nblocks; ++i) {
      s.process(controls.data(), abus.data());
    }
    auto t1 = std::chrono::steady_clock::now();

    std::chrono::duration<double> elapsed = t1 - t0;
    double seconds = elapsed.count();
    double ratio = 1.0 / seconds;

    std::println("perf: {} blocks in {} s (x{})", nblocks, seconds, ratio);
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
