#pragma once

#include <iostream>
#include <memory>
#include <utility>
#include <boost/asio/awaitable.hpp>
#include <boost/asio/posix/stream_descriptor.hpp>
#include <boost/asio/this_coro.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <pipewire/pipewire.h>

namespace pipewire {

using main_loop_ptr = std::unique_ptr<pw_main_loop, void (*)(pw_main_loop*)>;
using filter_ptr = std::unique_ptr<pw_filter, void (*)(pw_filter*)>;

main_loop_ptr make_main_loop(const spa_dict *props = nullptr)
{ return {pw_main_loop_new(props), pw_main_loop_destroy}; }

pw_loop *get_loop(main_loop_ptr const &main_loop)
{ return pw_main_loop_get_loop(main_loop.get()); }

filter_ptr make_filter(main_loop_ptr const &main_loop,
  const char *name, pw_properties *props, const pw_filter_events *events,
  void *data
) {
  return {
    pw_filter_new_simple(get_loop(main_loop), name, props, events, data),
    pw_filter_destroy
  };
}

template<typename T>
T *add_port(filter_ptr const &filter,
  enum pw_direction direction, enum pw_filter_port_flags flags,
  pw_properties *props, std::vector<const spa_pod *> params = {}
) {
  return static_cast<T *>(
    pw_filter_add_port(filter.get(),
      direction, flags, sizeof(T), props, params.data(), params.size()
    )
  );
}

boost::asio::awaitable<void> run(main_loop_ptr const &main_loop) {
  auto loop = get_loop(main_loop);
  boost::asio::posix::stream_descriptor fd(
    co_await boost::asio::this_coro::executor, pw_loop_get_fd(loop)
  );

  try {
    pw_loop_enter(loop);
    while (true) {
      co_await fd.async_wait(boost::asio::posix::stream_descriptor::wait_read, boost::asio::use_awaitable);
      pw_loop_iterate(loop, -1);
    }
    pw_loop_leave(loop);
  } catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
  }
}

namespace process_impl {

template<typename T, size_t I>
void call(T &obj, std::span<float> span, spa_io_position &position)
{ obj.template process<I>(std::span<float, I>(span), position); }

}

template<size_t... Is, typename T>
void process(std::index_sequence<Is...> seq, T &obj, std::span<float> span, spa_io_position &position)
{
  static constexpr void(*proc[])(T &, std::span<float>, spa_io_position &) = {
    &process_impl::call<T, Is>...
  };
  return proc[span.size()](obj, span, position);
}

template<typename T>
void process_port(T *port, spa_io_position &position)
{
  process(
    std::index_sequence<8192>{},
    *port,
    std::span(
      static_cast<float *>(
        pw_filter_get_dsp_buffer(port, position.clock.duration)
      ),
      position.clock.duration
    ),
    position
  );
}

}
