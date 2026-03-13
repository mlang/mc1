#pragma once

#include <boost/lockfree/spsc_queue.hpp>

#include <cstddef>

namespace mc1 {

template<typename T, size_t Capacity>
class fixed_spsc_queue final
{
public:
  static constexpr size_t capacity = Capacity;

  bool try_push(const T& value) noexcept
  {
    return queue_.push(value);
  }

  bool try_pop(T& value) noexcept
  {
    return queue_.pop(value);
  }

private:
  boost::lockfree::spsc_queue<T, boost::lockfree::capacity<Capacity>> queue_{};
};

} // namespace mc1
