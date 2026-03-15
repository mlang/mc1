#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <new>
#include <span>
#include <utility>

namespace mc1 {

inline constexpr size_t audio_buffer_alignment = 64;

inline bool is_aligned(const void* ptr, size_t alignment) noexcept
{
  auto address = reinterpret_cast<std::uintptr_t>(ptr);
  return address % alignment == 0;
}

class aligned_float_buffer final
{
public:
  aligned_float_buffer() = default;

  explicit aligned_float_buffer(size_t size)
  : size_{size}
  , data_{size_ == 0
      ? nullptr
      : static_cast<float*>(::operator new[](
            sizeof(float) * size_,
            std::align_val_t{audio_buffer_alignment}))}
  {
    assert(data_ == nullptr || is_aligned(data_, audio_buffer_alignment));
  }

  ~aligned_float_buffer()
  {
    reset();
  }

  aligned_float_buffer(const aligned_float_buffer&) = delete;
  aligned_float_buffer& operator=(const aligned_float_buffer&) = delete;

  aligned_float_buffer(aligned_float_buffer&& other) noexcept
  : size_{std::exchange(other.size_, 0)}
  , data_{std::exchange(other.data_, nullptr)}
  {}

  aligned_float_buffer& operator=(aligned_float_buffer&& other) noexcept
  {
    if (this == &other) return *this;

    reset();
    size_ = std::exchange(other.size_, 0);
    data_ = std::exchange(other.data_, nullptr);
    return *this;
  }

  float* data() noexcept { return data_; }
  const float* data() const noexcept { return data_; }

  size_t size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }

  std::span<float> span() noexcept { return {data_, size_}; }
  std::span<const float> span() const noexcept { return {data_, size_}; }

  float* begin() noexcept { return data_; }
  float* end() noexcept { return data_ + size_; }
  const float* begin() const noexcept { return data_; }
  const float* end() const noexcept { return data_ + size_; }

  float& operator[](size_t index) noexcept
  {
    assert(index < size_);
    return data_[index];
  }

  const float& operator[](size_t index) const noexcept
  {
    assert(index < size_);
    return data_[index];
  }

  void fill(float value) noexcept
  {
    std::fill(begin(), end(), value);
  }

private:
  size_t size_{};
  float* data_{};

  void reset() noexcept
  {
    if (data_ == nullptr) return;

    ::operator delete[](data_, std::align_val_t{audio_buffer_alignment});
    data_ = nullptr;
    size_ = 0;
  }
};

} // namespace mc1
