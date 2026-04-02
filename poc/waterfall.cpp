// Unicode braille waterfall

#include <fftw3.h>
#include <sndfile.hh>
#include <jack/jack.h>
#include <boost/lockfree/spsc_queue.hpp>


#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <numbers>
#include <print>
#include <ranges>
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
    int k0 = hz_to_bin(edge_hz(2 * x + 0));
    int k1 = hz_to_bin(edge_hz(2 * x + 1));
    int k2 = hz_to_bin(edge_hz(2 * x + 2));

    // Ensure non-decreasing bin edges.
    if (k1 < k0) k1 = k0;
    if (k2 < k1) k2 = k1;

    result += glyph2x4(level_for_range(k0, k1), level_for_range(k1, k2));
  }

  return result;
}

class FFT {
  std::vector<float> in;
  std::vector<fftwf_complex> out;
  fftwf_plan plan;

public:
  FFT(size_t n)
  : in(n), out(n / 2 + 1)
  , plan{fftwf_plan_dft_r2c_1d(n, in.data(), out.data(), FFTW_ESTIMATE)}
  {}
  FFT(FFT const &) = delete;
  FFT &operator=(FFT const &) = delete;
  ~FFT() { fftwf_destroy_plan(plan); }

  std::span<float> input() { return {in}; }
  void execute() { fftwf_execute(plan); }
  std::span<const fftwf_complex> output() const { return {out}; }
};

inline auto mono(size_t channels)
{
  constexpr auto mean = [](auto frame)
  {
    using V = std::ranges::range_value_t<decltype(frame)>;
    return std::ranges::fold_left(frame, V{}, std::plus<>{}) / frame.size();
  };
  return std::views::chunk(channels) | std::views::transform(mean);
}

class AudioSource
{
public:
  virtual ~AudioSource() = default;
  virtual size_t channels() const = 0;
  virtual unsigned int samplerate() const = 0;
  virtual size_t readf(float *interleaved, size_t frames) = 0;
};

class SoundFile : public AudioSource
{
  SndfileHandle sf;

public:
  SoundFile(std::string path)
  : sf(path)
  { if (sf.error()) throw std::runtime_error(sf.strError()); }

  size_t channels() const override { return sf.channels(); }
  unsigned int samplerate() const override { return sf.samplerate(); }
  size_t readf(float *interleaved, size_t frames) override
  { return sf.readf(interleaved, frames); }
};

class JACK : public AudioSource
{
  jack_client_t* client{};
  std::vector<jack_port_t*> inports;   // non-RT
  std::vector<float> tmp;              // non-RT, reused in RT
  unsigned int fs{};

  boost::lockfree::spsc_queue<float, boost::lockfree::capacity<1 << 18>> q;

  static int process_cb(jack_nframes_t nframes, void* arg)
  {
    auto& self = *static_cast<JACK*>(arg);
    const size_t C = self.inports.size();
    const size_t n = size_t(nframes) * C;
    assert(n == self.tmp.size());

    for (auto [ch, port]: self.inports | std::views::enumerate) {
      auto* in = static_cast<const float*>(jack_port_get_buffer(port, nframes));
      for (jack_nframes_t i = 0; i < nframes; ++i)
        self.tmp[size_t(i) * C + ch] = in[i];
    }

    self.q.push(self.tmp.begin(), self.tmp.end());
    return 0;
  }

public:
  explicit JACK(size_t inputs = 1)
  : inports(inputs)
  {
    client = jack_client_open("braille-waterfall", JackNoStartServer, nullptr);
    fs = jack_get_sample_rate(client);

    for (size_t ch = 0; ch < inputs; ++ch) {
      auto name = "in_" + std::to_string(ch + 1);
      inports[ch] = jack_port_register(client, name.c_str(),
                                       JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput, 0);
    }

    tmp.resize(size_t(jack_get_buffer_size(client)) * inputs);

    jack_set_process_callback(client, &JACK::process_cb, this);
    jack_activate(client);
  }

  ~JACK() override
  {
    jack_deactivate(client);
    jack_client_close(client);
  }

  size_t channels() const override { return inports.size(); }
  unsigned int samplerate() const override { return fs; }

  size_t readf(float* interleaved, size_t frames) override
  {
    const size_t want = frames * channels();
    while (q.read_available() < want)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    q.pop(interleaved, want);
    return frames;
  }
};

} // namespace

int main(int argc, char* argv[])
{
  std::string path = (argc > 1) ? argv[1] : "";
  unsigned int fps = 30;
  unsigned int width = 79;
  float display_min_dbfs = -60.0f;
  float display_max_dbfs = 0.0f;

  std::unique_ptr<AudioSource> src_ptr;
  src_ptr = std::make_unique<SoundFile>(path);
  auto &src = *src_ptr;
  const size_t hop = src.samplerate() / fps;
  const size_t N = std::bit_ceil(std::bit_ceil(hop + 1) + 1);
  const size_t bins = N / 2 + 1;

  std::vector<float> interleaved(N * src.channels());
  if (src.readf(interleaved.data(), N) < N) {
    return EXIT_SUCCESS;
  }

  FFT fft(N);

  // Precompute Hann window
  std::vector<float> w(N);
  for (int i = 0; i < N; ++i) w[i] = hann(i, N);
  float window_sum = 0.0f;
  for (float wi: w) window_sum += wi;

  std::vector<float> dbfs(fft.output().size());

  using clock = std::chrono::steady_clock;
  auto time = clock::now();
  const auto hop_duration = clock::duration(std::chrono::seconds(hop)) / src.samplerate();
  const int overlap = N - hop;
  do {
    std::ranges::copy(
      std::views::zip_transform(std::multiplies<>{},
        interleaved | mono(src.channels()), w
      ),
      fft.input().begin()
    );

    fft.execute();

    for (auto [k, value]: fft.output() | std::views::enumerate) {
      float mag = std::hypot(value[0], value[1]);
      if (k != 0 && k != fft.output().size() - 1) mag *= 2.0f;
      const float amplitude = mag / window_sum;
      dbfs[k] = 20.0f * std::log10(std::max(amplitude, std::numeric_limits<float>::epsilon()));
    }

    std::cout << logfreq(width,
      N, bins, src.samplerate(), dbfs, display_min_dbfs, display_max_dbfs, 20.0f, -1.0f
    );
    std::cout.flush();

    time += hop_duration;
    std::this_thread::sleep_for(time - clock::now());
    std::cout << '\n';

    std::memmove(
      interleaved.data(),
      interleaved.data() + hop * src.channels(),
      size_t(overlap) * src.channels() * sizeof(float)
    );
  } while (src.readf(interleaved.data() + overlap * src.channels(), hop) == hop);

  return EXIT_SUCCESS;
}
