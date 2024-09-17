#pragma once

#include <iostream>
#include <memory>
#include <system_error>
#include <utility>
#include <boost/asio/awaitable.hpp>
#include <boost/asio/posix/stream_descriptor.hpp>
#include <boost/asio/this_coro.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <pipewire/pipewire.h>

namespace mlang::pipewire {

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
  auto ptr = pw_filter_new_simple(get_loop(main_loop), name, props, events, data);
  if (!ptr)
    throw std::system_error(errno, std::system_category(), "pw_filter_new_simple");
  return { ptr, pw_filter_destroy };
}

template<typename T>
using port_ptr = std::unique_ptr<T, void(*)(T*)>;

template<typename T>
port_ptr<T> make_port
( filter_ptr const &filter
, enum pw_direction direction, enum pw_filter_port_flags flags
, pw_properties *props, std::vector<const spa_pod *> params = {}
)
{
  auto const placement = static_cast<T *>(pw_filter_add_port(filter.get(),
    direction, flags, sizeof(T), props, params.data(), params.size()
  ));
  if (!placement)
    throw std::system_error(errno, std::system_category(), "pw_filter_add_port");
  new(placement)T{};
  return { placement,
    [](T* port)
    {
      port->~T();
      if (const int result = pw_filter_remove_port(port))
        throw std::system_error(-result, std::generic_category(), "pw_filter_remove_port");
    }
  };
}

std::error_code connect(filter_ptr const& filter,
  enum pw_filter_flags flags, std::vector<const spa_pod *> params = {}
) {
  const int result = pw_filter_connect(filter.get(),
    flags, params.data(), params.size()
  );
  if (result < 0) return { -result, std::generic_category() };
  return {};
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

template<size_t... Is, typename T, typename U, typename... Args>
void
process(std::index_sequence<Is...>,
  T &obj, std::span<U> span, Args&&... args)
{
  static constexpr void(* const proc[])(T &, std::span<U>, Args&&...) = {
    [](T &obj, std::span<U> span, Args&&... args) {
      obj.process(std::span<U, Is>(span), std::forward<Args>(args)...);
    }...
  };
  assert(std::size(proc) > span.size());
  proc[span.size()](obj, span, std::forward<Args>(args)...);
}

template<typename T>
void process_port(port_ptr<T> const &port, spa_io_position &position)
{
  auto const sample_count = position.clock.duration;
  auto const buffer = static_cast<float *>(
    pw_filter_get_dsp_buffer(port.get(), sample_count)
  );
  assert(buffer);
  auto const span = std::span(buffer, sample_count);
  process(std::make_index_sequence<1024>{}, *port, span, position);
}

template<typename Derived>
class make_filter_events
{
  static void do_process(void *data, spa_io_position *position)
  { static_cast<Derived *>(data)->process(*position); }

protected:
  static constexpr pw_filter_events filter_events = {
    PW_VERSION_FILTER_EVENTS,
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    make_filter_events::do_process,
    nullptr, nullptr
  };
};

}
