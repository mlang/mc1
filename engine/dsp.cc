#include <iostream>
#include <utility>

#include <libgccjit++.h>
#include <boost/asio.hpp>

#include "bytes.hpp"
#include "math.hpp"
#include "pipewire.hpp"

using boost::asio::ip::udp;
using boost::asio::awaitable;
using boost::asio::buffer;
using boost::asio::co_spawn;
using boost::asio::detached;
using boost::asio::use_awaitable;

using mlang::get_pstring;
using mlang::get_value;
using mlang::get_values;
using mlang::tau;

namespace {

struct dag final {
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
std::ostream& operator<<(std::ostream& os, const dag& d)
{
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

class osc {
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

  return 0;
}
