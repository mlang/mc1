// Unicode braille waterfall

#include <boost/lockfree/spsc_queue.hpp>
#include <fftw3.h>
#include <jack/jack.h>
#include <sndfile.hh>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <charconv>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numbers>
#include <print>
#include <ranges>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

using brl_t = uint8_t;

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

brl_t glyph2x4(int L, int R)
{
  auto set_if = [](brl_t &mask, bool on, int dot) {
    if (on) mask |= 1 << (dot - 1); // dot 1 -> bit0, dot 8 -> bit7
  };

  brl_t mask = 0;

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

  return mask;
}

brl_t mirror_brl(brl_t cell)
{
  return static_cast<brl_t>(
    ((cell & 0x07u) << 3) |
    ((cell & 0x38u) >> 3) |
    ((cell & 0x40u) << 1) |
    ((cell & 0x80u) >> 1)
  );
}

std::string to_utf8(std::span<brl_t> cells)
{
  std::string s;
  s.reserve(cells.size() * 3);

  for (brl_t cell: cells) {
    auto cp = static_cast<char32_t>(0x2800u + cell);
    s.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
    s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }

  return s;
}

std::vector<brl_t> logfreq(
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

  std::vector<brl_t> result;
  result.reserve(std::max(width, 0));

  for (int x = 0; x < width; ++x) {
    int k0 = hz_to_bin(edge_hz(2 * x + 0));
    int k1 = hz_to_bin(edge_hz(2 * x + 1));
    int k2 = hz_to_bin(edge_hz(2 * x + 2));

    // Ensure non-decreasing bin edges.
    if (k1 < k0) k1 = k0;
    if (k2 < k1) k2 = k1;

    result.push_back(glyph2x4(level_for_range(k0, k1), level_for_range(k1, k2)));
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

  std::span<float> input() { return in; }
  void execute() { fftwf_execute(plan); }
  std::span<const fftwf_complex> output() const { return out; }
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

class SoundFile final : public AudioSource
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

class JACK final : public AudioSource
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
        self.tmp[C * i + ch] = in[i];
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

    tmp.resize(inputs * jack_get_buffer_size(client));

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
    const size_t want_samples = frames * channels();

    while (q.read_available() < want_samples) {
      const size_t avail_frames = q.read_available() / channels();
      const auto missing_frames = frames - avail_frames;

      using clock = std::chrono::steady_clock;
      auto dt = clock::duration(std::chrono::seconds(missing_frames)) / samplerate();

      dt -= std::chrono::microseconds(200);
      if (dt > std::chrono::microseconds(0)) std::this_thread::sleep_for(dt);
      else std::this_thread::yield();
    }

    q.pop(interleaved, want_samples);
    return frames;
  }
};

int waterfall(
  AudioSource &src,
  unsigned int fps = 30,
  unsigned int width = 79,
  float display_min_dbfs = -60.0f, float display_max_dbfs = 0.0f,
  float min_freq = 20.0f, float max_freq = -1.0f,
  bool stereo = false
)
{
  const size_t hop = src.samplerate() / fps;
  const size_t N = std::bit_ceil(std::bit_ceil(hop + 1) + 1);
  const size_t bins = N / 2 + 1;

  std::vector<float> interleaved(N * src.channels());
  if (src.readf(interleaved.data(), N) < N) {
    return EXIT_SUCCESS;
  }

  FFT fft(N);
  std::vector<float> w(N);
  for (int i = 0; i < N; ++i) w[i] = hann(i, N);
  float window_sum = 0.0f;
  for (float wi: w) window_sum += wi;

  std::vector<float> dbfs(fft.output().size());
  std::vector<float> dbfs_r;
  if (stereo) dbfs_r.resize(fft.output().size());

  using clock = std::chrono::steady_clock;
  auto time = clock::now();
  const auto hop_duration = clock::duration(std::chrono::seconds(hop)) / src.samplerate();
  const int overlap = N - hop;

  auto compute_dbfs = [&](auto fill_input, std::vector<float>& out_dbfs) {
    fill_input();
    fft.execute();

    for (auto [k, value]: fft.output() | std::views::enumerate) {
      float mag = std::hypot(value[0], value[1]);
      if (k != 0 && k != fft.output().size() - 1) mag *= 2.0f;
      const float amplitude = mag / window_sum;
      out_dbfs[k] = 20.0f * std::log10(std::max(amplitude, std::numeric_limits<float>::epsilon()));
    }
  };

  do {
    std::cout << '\n';

    std::vector<brl_t> glyphs;
    if (!stereo) {
      compute_dbfs([&] {
        std::ranges::copy(
          std::views::zip_transform(std::multiplies<>{},
            interleaved | mono(src.channels()), w
          ),
          fft.input().begin()
        );
      }, dbfs);

      glyphs = logfreq(width,
        N, bins, src.samplerate(), dbfs, display_min_dbfs, display_max_dbfs, min_freq, max_freq
      );
    } else {
      compute_dbfs([&] {
        for (size_t i = 0; i < N; ++i) fft.input()[i] = interleaved[2 * i] * w[i];
      }, dbfs);
      compute_dbfs([&] {
        for (size_t i = 0; i < N; ++i) fft.input()[i] = interleaved[2 * i + 1] * w[i];
      }, dbfs_r);

      const auto half_width = static_cast<int>(width / 2);
      auto left = logfreq(half_width,
        N, bins, src.samplerate(), dbfs, display_min_dbfs, display_max_dbfs, min_freq, max_freq
      );
      auto right = logfreq(half_width,
        N, bins, src.samplerate(), dbfs_r, display_min_dbfs, display_max_dbfs, min_freq, max_freq
      );

      std::ranges::reverse(left);
      std::ranges::transform(left, left.begin(), mirror_brl);

      glyphs.reserve(left.size() + right.size() + (width % 2));
      glyphs.insert(glyphs.end(), left.begin(), left.end());
      if (width % 2) glyphs.push_back(brl_t{0});
      glyphs.insert(glyphs.end(), right.begin(), right.end());
    }

    std::cout << to_utf8(glyphs);
    std::cout.flush();

    time += hop_duration;
    std::this_thread::sleep_for(time - clock::now());

    std::memmove(
      interleaved.data(),
      interleaved.data() + hop * src.channels(),
      size_t(overlap) * src.channels() * sizeof(float)
    );
  } while (src.readf(interleaved.data() + overlap * src.channels(), hop) == hop);

  return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char* argv[])
{
  auto usage = [&]() -> int {
    std::print(
      "usage:\n"
      "  {} [options] (--jack | <filename>)\n\n"
      "options:\n"
      "  --fps N                 Frames per second (default: 30)\n"
      "  --min-freq HZ           Min displayed frequency (default: 20)\n"
      "  --max-freq HZ           Max displayed frequency (default: nyquist)\n"
      "  --min-dbfs DB           Min displayed level in dBFS (default: -60)\n"
      "  --max-dbfs DB           Max displayed level in dBFS (default: 0)\n"
      "  --width N               Output width in braille glyphs (default: 79)\n"
      "  --jack                  Use JACK input instead of a file\n"
      "  --stereo                Mirror 2 channels around the center axis\n"
      "  --help                  Show this help\n",
      argv[0]
    );
    return EXIT_FAILURE;
  };

  auto die = [](std::string_view msg) -> int {
    std::println("error: {}", msg);
    return EXIT_FAILURE;
  };

  auto require_value = [&](int& i, std::string_view opt) -> std::string_view {
    if (i + 1 >= argc) throw opt;
    return argv[++i];
  };

  auto parse_u32 = [&](std::string_view s, std::string_view opt) -> unsigned {
    unsigned v{};
    auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc{} || p != s.data() + s.size()) {
      std::println("error: invalid value for {}: '{}'", opt, s);
      throw opt;
    }
    return v;
  };

  auto parse_f32 = [&](std::string_view s, std::string_view opt) -> float {
    float v{};
    auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc{} || p != s.data() + s.size()) {
      std::println("error: invalid value for {}: '{}'", opt, s);
      throw opt;
    }
    return v;
  };

  // Defaults (match your existing call)
  unsigned fps = 30;
  unsigned width = 79;
  float min_freq = 20.0f;
  float max_freq = -1.0f; // <=0 => Nyquist
  float min_dbfs = -60.0f;
  float max_dbfs = 0.0f;

  bool use_jack = false;
  bool stereo = false;
  std::string filename;

  try {
    for (int i = 1; i < argc; ++i) {
      std::string_view a = argv[i];

      if (a == "--help") {
        return usage();
      } else if (a == "--fps") {
        fps = parse_u32(require_value(i, "--fps"), "--fps");
      } else if (a == "--width") {
        width = parse_u32(require_value(i, "--width"), "--width");
      } else if (a == "--min-freq") {
        min_freq = parse_f32(require_value(i, "--min-freq"), "--min-freq");
      } else if (a == "--max-freq") {
        max_freq = parse_f32(require_value(i, "--max-freq"), "--max-freq");
      } else if (a == "--min-dbfs") {
        min_dbfs = parse_f32(require_value(i, "--min-dbfs"), "--min-dbfs");
      } else if (a == "--max-dbfs") {
        max_dbfs = parse_f32(require_value(i, "--max-dbfs"), "--max-dbfs");
      } else if (a == "--jack") {
        use_jack = true;
      } else if (a == "--stereo") {
        stereo = true;
      } else if (!a.empty() && a.front() == '-') {
        std::println("error: unknown option: {}", a);
        return usage();
      } else {
        if (!filename.empty()) {
          std::println("error: multiple filenames given: '{}' and '{}'", filename, a);
          return usage();
        }
        filename = std::string(a);
      }
    }
  } catch (std::string_view opt) {
    std::println("error: missing value for {}", opt);
    return usage();
  }

  if (use_jack && !filename.empty())
    return die("choose either --jack or <filename>, not both");
  if (!use_jack && filename.empty())
    return die("missing input source: specify --jack or <filename>");

  if (fps == 0)   return die("--fps must be > 0");
  if (width == 0) return die("--width must be > 0");

  std::unique_ptr<AudioSource> src;
  try {
    if (use_jack) src = std::make_unique<JACK>(stereo ? 2 : 1);
    else          src = std::make_unique<SoundFile>(filename);
  } catch (const std::exception& e) {
    std::println("error: failed to open input: {}", e.what());
    return EXIT_FAILURE;
  }

  if (stereo && src->channels() != 2)
    return die("--stereo requires exactly 2 channels");

  return waterfall(*src, fps, width, min_dbfs, max_dbfs, min_freq, max_freq, stereo);
}
