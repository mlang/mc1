#include <iostream>
#include <utility>

#include <boost/asio.hpp>

#include "bytes.hpp"
#include "dag.hpp"
#include "gccjit.hpp"
#include "math.hpp"
#include "pipewire.hpp"

using boost::asio::ip::udp;
using boost::asio::awaitable;
using boost::asio::buffer;
using boost::asio::co_spawn;
using boost::asio::detached;
using boost::asio::use_awaitable;

using mlang::tau;

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
      if (phase >= tau<decltype(phase)>) phase -= tau<decltype(phase)>;
    }
  }
};

class engine final : pipewire::make_filter_events<engine> {
  pipewire::main_loop_ptr main_loop;
  pipewire::filter_ptr filter;
  osc *out;

  static pw_properties *filter_props() {
    return pw_properties_new(
      PW_KEY_MEDIA_TYPE, "Audio",
      PW_KEY_MEDIA_CATEGORY, "Source",
      PW_KEY_MEDIA_ROLE, "DSP",
      PW_KEY_MEDIA_CLASS, "Stream/Output/Audio",
//      PW_KEY_NODE_AUTOCONNECT, "true",
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

  friend class pipewire::make_filter_events<engine>;
  void process(spa_io_position &position)
  {
    std::cout << position.clock.rate.num << '/' << position.clock.rate.denom << std::endl;
    pipewire::process_port(out, position);
  }

public:
  engine()
  : main_loop{pipewire::make_main_loop()}
  , filter{pipewire::make_filter(main_loop, "dsp", filter_props(), &filter_events, this)}
  , out{pipewire::add_port<osc>(filter, PW_DIRECTION_OUTPUT, PW_FILTER_PORT_FLAG_MAP_BUFFERS, filter_props())}
  { new(out)osc{}; }

  std::error_code connect()
  { return pipewire::connect(filter, PW_FILTER_FLAG_RT_PROCESS); }

  awaitable<void> pipewire() { co_await pipewire::run(main_loop); }

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

template<typename T>
gccjit::function
tabled_function
(gccjit::context gcc, std::string name, T period, size_t n, T(*f)(T))
{
  auto fp_type = gccjit::get_type<T>(gcc);
  auto array_type = gcc.new_array_type(fp_type.get_const(), n + 1);
  auto table = gcc.new_global(GCC_JIT_GLOBAL_INTERNAL, array_type, "tab_" + name);
  std::vector<gccjit::rvalue> init;
  init.reserve(n + 1);
  for (size_t i = 0; i < n; i++)
    init.push_back(gcc.new_rvalue(fp_type, f(period * i / n)));
  init.push_back(init.front());
  table.set_initializer_rvalue(gcc.new_array_ctor(array_type, init));

  auto param = std::vector{gcc.new_param(fp_type, "x")};
  auto func = gcc.new_function(GCC_JIT_FUNCTION_EXPORTED,
    fp_type, name, param, 0
  );
  auto index_type = gcc.get_type(GCC_JIT_TYPE_SIZE_T);
  auto i = func.new_local(index_type, "i");
  auto a = func.new_local(fp_type, "a");
  auto b = func.new_local(fp_type, "b");
  auto block = func.new_block("entry");
  block.add_assignment(func.get_param(0),
    func.get_param(0) * gcc.new_rvalue(fp_type, static_cast<T>(n) / period)
  );
  block.add_assignment(i, gcc.new_cast(func.get_param(0), i.get_type())); 
  block.add_assignment(a, gcc.new_array_access(table, i));
  block.add_assignment(b, gcc.new_array_access(table, i + index_type.one()));
  block.end_with_return(
    gcc.get_builtin_function("__builtin_fma")(
      func.get_param(0) - gcc.new_cast(i, fp_type), b - a, a
    )
  );

  return func;
}

int main(int argc, char *argv[])
{
  pw_init(&argc, &argv);
  auto gcc = gccjit::context::acquire();
  gcc.set_int_option(GCC_JIT_INT_OPTION_OPTIMIZATION_LEVEL, 3);
  gcc.set_bool_option(GCC_JIT_BOOL_OPTION_DUMP_INITIAL_GIMPLE, true);
  gcc.set_bool_option(GCC_JIT_BOOL_OPTION_DUMP_SUMMARY, true);
  tabled_function(gcc, "ix_sin", tau<double>, 1024, std::sin);
  {
    auto sin = gccjit::get_code<double(double)>(gccjit::compile(gcc), "ix_sin");
    std::cout << (*sin)(0) << ' ' << (*sin)(M_PI) << std::endl;
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
