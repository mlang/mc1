#include <iostream>
#include <ranges>
#include <utility>

#include <boost/asio.hpp>

#include "mlang/bytes.hpp"
#include "dag.hpp"
#include "mlang/gccjit.hpp"
#include "mlang/math.hpp"
#include "mlang/pipewire.hpp"

using boost::asio::ip::udp;
using boost::asio::awaitable;
using boost::asio::buffer;
using boost::asio::co_spawn;
using boost::asio::detached;
using boost::asio::use_awaitable;

using mlang::tau;
using std::views::transform;
using mlang::views::sampled_interval;


namespace {

class osc final {
  double phase;
public:
  template<size_t Size>
  void process(std::span<float, Size> buffer, spa_io_position &position)
  {
    const double diff = tau<decltype(phase)> * 440 / position.clock.rate.denom;
    for (auto &sample: buffer) {
      sample = std::sin(phase) * 0.2;
      phase += diff;
      while (phase >= tau<decltype(phase)>) phase -= tau<decltype(phase)>;
    }
  }
};

class engine final : mlang::pipewire::make_filter_events<engine> {
  mlang::pipewire::main_loop_ptr main_loop;
  mlang::pipewire::filter_ptr filter;
  osc *out;

  static pw_properties *filter_props() {
    return pw_properties_new(
      PW_KEY_MEDIA_TYPE, "Audio",
      PW_KEY_MEDIA_CATEGORY, "Source",
      PW_KEY_MEDIA_ROLE, "DSP",
      PW_KEY_MEDIA_CLASS, "Stream/Output/Audio",
      PW_KEY_NODE_AUTOCONNECT, "true",
      nullptr
    );
  }
  static pw_properties *port_props() {
    return pw_properties_new(
      PW_KEY_FORMAT_DSP, "32 bit float mono audio",
      PW_KEY_PORT_NAME, "output",
      nullptr
    );
  }

  friend class mlang::pipewire::make_filter_events<engine>;
  void process(spa_io_position &position)
  {
    std::cout << position.clock.rate.num << '/' << position.clock.rate.denom << std::endl;
    mlang::pipewire::process_port(out, position);
  }

public:
  engine()
  : main_loop{mlang::pipewire::make_main_loop()}
  , filter{mlang::pipewire::make_filter(main_loop, "dsp", filter_props(), &filter_events, this)}
  , out{mlang::pipewire::add_port<osc>(filter, PW_DIRECTION_OUTPUT, PW_FILTER_PORT_FLAG_MAP_BUFFERS, filter_props())}
  { new(out)osc{}; }

  std::error_code connect()
  { return mlang::pipewire::connect(filter, PW_FILTER_FLAG_RT_PROCESS); }

  awaitable<void> pipewire() { co_await mlang::pipewire::run(main_loop); }

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
    if (auto i = mlang::get_value<unsigned short>(view)) {
      switch (*i) {
      case 0: std::cout << "quit" << std::endl; break;
      case 1:
        if (auto dag = MiniCollider::dag::parse(view)) {
          if (view.empty()) std::cout << dag.value();
        }
        break;
      default:
        std::cout << *i << std::endl;
      }
    }
  }
};

}


int main(int argc, char *argv[])
{
  pw_init(&argc, &argv);
  auto gcc = gccjit::context::acquire();
  gcc.set_int_option(GCC_JIT_INT_OPTION_OPTIMIZATION_LEVEL, 3);
  gcc.set_bool_option(GCC_JIT_BOOL_OPTION_DUMP_INITIAL_GIMPLE, true);
  gcc.set_bool_option(GCC_JIT_BOOL_OPTION_DUMP_SUMMARY, true);
  make_tabled_function(gcc, "fast_sin", tau<double>, 256, std::sin);
  gcc.dump_to_file(".fast_sin.gimple", false);
  {
    auto result = gccjit::compile_shared(gcc);
    auto sin = gccjit::get_code<double(double)>(result, "fast_sin");
    const auto diff = std::ranges::max
    ( sampled_interval(tau<double>, 48000)
    | transform([sin=*sin](double x) { return std::abs(std::sin(x) - sin(x)); })
    );
    std::cout << "diff = " << diff << std::endl;
  }

  gcc.release();


  boost::asio::io_context io;
  engine world;

  if (auto errc = world.connect()) {
    std::cerr << errc.message() << std::endl;
    return EXIT_FAILURE;
  }

  {
    udp::socket socket{io, udp::endpoint(udp::v4(), std::atoi(argv[1]))};
    co_spawn(io, world.udp_server(std::move(socket)), detached);
  }
  co_spawn(io, world.pipewire(), detached);

  io.run();

  pw_deinit();

  return EXIT_SUCCESS;
}
