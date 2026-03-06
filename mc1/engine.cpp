#include <array>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>
#include <print>
#include <ranges>
#include <utility>
#include <vector>

#include <boost/asio.hpp>

#include "mlang/bytes.hpp"
#include "audio.hpp"
#include "compiler.hpp"
#include "dag.hpp"
#include "osc.hpp"

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
  std::unique_ptr<audio_device> audio;
  bool running = true;

  static void send_message(udp::socket &socket, udp::endpoint sender, std::vector<std::byte> bytes)
  {
    boost::asio::post(socket.get_executor(),
      [&socket, sender = std::move(sender), bytes = std::move(bytes)]() {
      boost::system::error_code ec;
      socket.send_to(buffer(bytes), sender, 0, ec);
      if (ec) std::cerr << "send_to failed: " << ec.message() << std::endl;
    });
  }

public:
  engine() : compiler{1} {}

  awaitable<void> udp_server(udp::socket socket)
  {
    std::byte data[65536];
    try {
      while (running) {
        udp::endpoint sender;
        size_t n = co_await socket.async_receive_from(buffer(data), sender, use_awaitable);
        packet_received(socket, sender, std::span(&data[0], n));
      }
    } catch (std::exception& e) {
      std::cerr << e.what() << std::endl;
    }
  }

  void packet_received(udp::socket &socket, udp::endpoint sender, std::span<const std::byte> bytes)
  {
    auto packet = osc::decode_packet(bytes);
    if (!packet) return;

    auto msg = std::get_if<osc::message>(&packet.value());
    if (!msg) return;

    if (msg->address == "/mc1/quit") {
      running = false;
      return;
    }

    if (msg->address == "/mc1/compile") {
      if (msg->args.size() != 1) return;
      auto blob = std::get_if<osc::blob>(&msg->args[0]);
      if (!blob) return;

      auto dag_bytes = std::span<const std::byte>(blob->data);
      if (auto dag = DAG::parse(dag_bytes)) {
        if (dag_bytes.empty()) {
          post(compiler, [dag = std::move(dag.value()), this]()
          {
            compile_synth(std::move(dag));
          });
        }
      }
      return;
    }

    if (msg->address == "/mc1/sync") {
      // Barrier against the compile queue: reply only after all prior jobs ran.
      post(compiler, [&socket, sender = std::move(sender)]()
      {
        std::vector<osc::argument> args;
        send_message(socket, sender, osc::encode_message("/mc1/done", args));
      });
    }
  }

  void compile_synth(DAG dag)
  {
    std::cout << dag;
    constexpr unsigned int SR = 44100;
    constexpr size_t BS = 32;

    auto r = compile(dag, SR, BS);
    auto s = r[dag.name];

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
