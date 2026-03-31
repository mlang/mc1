// Unicode braille waterfall

#include <fftw3.h>
#include <sndfile.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <numbers>
#include <string>
#include <thread>
#include <vector>

namespace {

float hann(int n, int N)
{
  constexpr auto tau = 2.0f * std::numbers::pi_v<float>;
  return 0.5f - 0.5f * std::cos(tau * n / (N - 1));
}

int quant5(float x01)
{
  x01 = std::clamp(x01, 0.0f, 1.0f);
  int q = int(std::floor(x01 * 5.0f)); // 0..5
  return std::min(q, 4);               // 0..4
}

float amplitude_dbfs_from_fft_bin(
  fftwf_complex value, int bin, int bins, float window_sum
)
{
  float mag = std::hypot(value[0], value[1]);
  if (bin != 0 && bin != bins - 1) mag *= 2.0f;
  float amplitude = mag / window_sum;
  return 20.0f * std::log10(std::max(amplitude, std::numeric_limits<float>::epsilon()));
}

float dbfs_to_unit(float dbfs, float min_dbfs, float max_dbfs)
{
  if (max_dbfs <= min_dbfs) return 0.0f;
  return (dbfs - min_dbfs) / (max_dbfs - min_dbfs);
}

std::string glyph2x4(int L, int R)
{
  // Use Unicode Braille patterns U+2800..U+28FF.
  auto set_if = [](int &mask, bool on, int dot) {
    if (on) mask |= 1 << (dot - 1); // dot 1 -> bit0, dot 8 -> bit7
  };

  int mask = 0;

  // Left column
  set_if(mask, L >= 1, 1);
  set_if(mask, L >= 2, 2);
  set_if(mask, L >= 3, 3);
  set_if(mask, L == 4, 7);

  // Right column
  set_if(mask, R >= 1, 4);
  set_if(mask, R >= 2, 5);
  set_if(mask, R >= 3, 6);
  set_if(mask, R == 4, 8);

  auto cp = static_cast<char32_t>(0x2800 + mask);

  std::string s;
  if (cp <= 0x7F) {
    s.push_back(static_cast<char>(cp));
  } else if (cp <= 0x7FF) {
    s.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp <= 0xFFFF) {
    s.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
    s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    s.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
    s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
  return s;
}

std::string logfreq(
  int width,
  int N, int bins, int samplerate,
  const std::vector<float>& dbfs,
  float min_dbfs,
  float max_dbfs,
  float f_min = 20.0f,           // lowest displayed frequency (Hz)
  float f_max = -1.0f            // highest displayed frequency (Hz); <=0 means Nyquist
)
{
  auto clampi = [](int v, int lo, int hi) { return std::max(lo, std::min(v, hi)); };

  const float Fs = float(samplerate);
  const float nyq = 0.5f * Fs;
  if (f_max <= 0.0f || f_max > nyq) f_max = nyq;

  // Avoid log(0) and invalid ranges
  f_min = std::max(1.0f, std::min(f_min, f_max * 0.999f));

  // Convert frequency (Hz) to nearest FFT bin index.
  auto hz_to_bin = [&](float f_hz) -> int {
    float k = f_hz * float(N) / Fs;          // bin index (float)
    int ki = int(std::lround(k));
    return clampi(ki, 0, bins - 1);
  };

  // Log-frequency edges for this character column: [f0, f2], split at f1 for L/R.
  // We use edges (not centers) so the whole [f_min,f_max] is covered.
  auto edge_hz = [&](int edgeIndex /*0..2*width*/) -> float {
    float t = float(edgeIndex) / float(2 * width);            // 0..1
    float r = std::log(f_max / f_min);
    return f_min * std::exp(r * t);
  };

  // Reduce a bin range [a,b] (inclusive) into one level.
  // We combine as: max over bins of dBFS, then normalize to the fixed display range.
  // (Max works well for thin tonal lines; average works better for noise.)
  auto level_for_range = [&](int a, int b) -> int
  {
    a = clampi(a, 0, bins - 1);
    b = clampi(b, 0, bins - 1);
    if (b < a) std::swap(a, b);

    float best_dbfs = std::numeric_limits<float>::lowest();
    for (int k = a; k <= b; ++k) {
      best_dbfs = std::max(best_dbfs, dbfs[k]);
    }

    return quant5(dbfs_to_unit(best_dbfs, min_dbfs, max_dbfs));
  };

  std::string result;
  for (size_t x = 0; x < width; x++) {
    float f0 = edge_hz(2 * x + 0);
    float f1 = edge_hz(2 * x + 1);
    float f2 = edge_hz(2 * x + 2);

    int k0 = hz_to_bin(f0);
    int k1 = hz_to_bin(f1);
    int k2 = hz_to_bin(f2);

    // Ensure non-decreasing bin edges.
    if (k1 < k0) k1 = k0;
    if (k2 < k1) k2 = k1;

    result += glyph2x4(level_for_range(k0, k1), level_for_range(k1, k2));
  }

  return result;
}

} // namespace

int main(int argc, char* argv[])
{
  std::string path = (argc > 1) ? argv[1] : "";
  unsigned int fps = 30;
  unsigned int width = 79;
  float display_min_dbfs = -60.0f;
  float display_max_dbfs = 0.0f;

  SF_INFO info{};
  SNDFILE* sf = sf_open(path.c_str(), SFM_READ, &info);
  const size_t hop = info.samplerate / fps;
  const size_t N = std::bit_ceil(std::bit_ceil(hop + 1) + 1);
  const size_t bins = N / 2 + 1;

  std::vector<float> interleaved(N * info.channels);
  if (sf_readf_float(sf, interleaved.data(), N) < N) {
    sf_close(sf);
    return EXIT_SUCCESS;
  }

  std::vector<float> in(N);
  std::vector<fftwf_complex> out(bins);
  auto plan = fftwf_plan_dft_r2c_1d(N, in.data(), out.data(), FFTW_ESTIMATE);

  // Precompute Hann window
  std::vector<float> w(N);
  for (int i = 0; i < N; ++i) w[i] = hann(i, N);
  float window_sum = 0.0f;
  for (float wi: w) window_sum += wi;

  std::vector<float> dbfs(bins);

  using clock = std::chrono::steady_clock;
  auto time = clock::now();
  const auto hop_duration = clock::duration(std::chrono::seconds(hop)) / info.samplerate;
  const int overlap = N - hop;
  do {
    for (sf_count_t i = 0; i < N; ++i) {
      float s = 0.0f;
      for (int c = 0; c < info.channels; ++c) s += interleaved[i * info.channels + c];
      s /= info.channels;
      in[i] = s * w[i];
    }

    fftwf_execute(plan);

    for (int k = 0; k < bins; ++k) {
      dbfs[k] = amplitude_dbfs_from_fft_bin(out[k], k, bins, window_sum);
    }

    std::cout << logfreq(width,
      N, bins, info.samplerate, dbfs, display_min_dbfs, display_max_dbfs, 20.0f, -1.0f
    );
    std::cout.flush();

    time += hop_duration;
    std::this_thread::sleep_for(time - clock::now());
    std::cout << '\n';

    std::memmove(
      interleaved.data(),
      interleaved.data() + hop * info.channels,
      size_t(overlap) * info.channels * sizeof(float)
    );
  } while (sf_readf_float(sf, interleaved.data() + overlap * info.channels, hop) == hop);

  sf_close(sf);
  fftwf_destroy_plan(plan);
  return EXIT_SUCCESS;
}
