// audio -> STFT -> 2-bin-per-char Unicode braille waterfall
#include <print>
// Requires: libsndfile, fftw3f
// Build (example):
//   g++ -std=c++23 -O2 waterfall.cpp -lsndfile -lfftw3f -lm -o waterfall
// Run:
//   ./waterfall input.wav

#ifndef WATERFALL_UNIT_TEST
#include <fftw3.h>
#include <sndfile.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <numbers>
#include <string>
#include <thread>
#include <vector>

namespace {

float hann(int n, int N) {
    constexpr auto tau = 2.0f * std::numbers::pi_v<float>;
    return 0.5f - 0.5f * std::cos(tau * n / (N - 1));
}

int quant5(float x01) { // x in [0,1]
    x01 = std::clamp(x01, 0.0f, 1.0f);
    int q = int(std::floor(x01 * 5.0f)); // 0..5
    return std::min(q, 4);               // 0..4
}

float amplitude_dbfs_from_fft_bin(
    float re,
    float im,
    int bin,
    int bins,
    float window_sum,
    float eps = 1e-12f
) {
    float mag = std::hypot(re, im);
    if (bin != 0 && bin != bins - 1) mag *= 2.0f;
    float amplitude = mag / std::max(window_sum, eps);
    return 20.0f * std::log10(std::max(amplitude, eps));
}

float dbfs_to_unit(float dbfs, float min_dbfs, float max_dbfs) {
    if (max_dbfs <= min_dbfs) return 0.0f;
    return (dbfs - min_dbfs) / (max_dbfs - min_dbfs);
}

int quantized_level_from_dbfs(float dbfs, float min_dbfs, float max_dbfs) {
    return quant5(dbfs_to_unit(dbfs, min_dbfs, max_dbfs));
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
    auto level_for_range = [&](int a, int b) -> int {
        a = clampi(a, 0, bins - 1);
        b = clampi(b, 0, bins - 1);
        if (b < a) std::swap(a, b);

        float best_dbfs = std::numeric_limits<float>::lowest();
        for (int k = a; k <= b; ++k) {
            best_dbfs = std::max(best_dbfs, dbfs[k]);
        }

        return quantized_level_from_dbfs(best_dbfs, min_dbfs, max_dbfs);
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

#ifndef WATERFALL_UNIT_TEST
int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "";

    SF_INFO info{};
    SNDFILE* sf = sf_open(path.c_str(), SFM_READ, &info);

    const int N = 4096;          // FFT size
    const int hop = N / 4;       // hop size
    const int bins = N / 2 + 1;  // r2c bins

    std::vector<float> interleaved(info.frames * info.channels);
    sf_readf_float(sf, interleaved.data(), info.frames);
    sf_close(sf);

    std::vector<float> in(N);
    std::vector<fftwf_complex> out(bins);
    auto plan = fftwf_plan_dft_r2c_1d(N, in.data(), out.data(), FFTW_MEASURE);

    // Precompute Hann window
    std::vector<float> w(N);
    for (int i = 0; i < N; ++i) w[i] = hann(i, N);
    float window_sum = 0.0f;
    for (float wi : w) window_sum += wi;

    // Spectral amplitude in dBFS with a fixed display range.
    std::vector<float> dbfs(bins);
    constexpr float eps = 1e-12f;
    constexpr float display_min_dbfs = -60.0f;
    constexpr float display_max_dbfs = 0.0f;

    using clock = std::chrono::steady_clock;
    auto time = clock::now();
    const auto hop_duration = clock::duration(std::chrono::seconds(hop)) / info.samplerate;
    for (sf_count_t pos = 0; pos + N <= info.frames; pos += hop) {
        for (sf_count_t i = 0; i < N; ++i) {
            float s = 0.0f;
            for (int c = 0; c < info.channels; ++c) s += interleaved[(pos + i) * info.channels + c];
            s /= info.channels;
            in[i] = s * w[i];
        }

        fftwf_execute(plan);

        for (int k = 0; k < bins; ++k) {
            float re = out[k][0], im = out[k][1];
            dbfs[k] = amplitude_dbfs_from_fft_bin(re, im, k, bins, window_sum, eps);
        }

        std::cout << logfreq(79,
            N, bins, info.samplerate, dbfs, display_min_dbfs, display_max_dbfs, 20.0f, -1.0f
        );
        std::cout.flush();

        time += hop_duration;
        std::this_thread::sleep_for(time - clock::now());
        std::cout << '\n';
    }

    fftwf_destroy_plan(plan);
    return EXIT_SUCCESS;
}
#endif
