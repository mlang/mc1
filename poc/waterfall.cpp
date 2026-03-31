// C++23 PoC: audio -> STFT -> 2-bin-per-char Unicode waterfall using 4 heights.
// Requires: libsndfile, fftw3f
// Build (example):
//   g++ -std=c++23 -O2 waterfall.cpp -lsndfile -lfftw3f -lm -o waterfall
// Run:
//   ./waterfall input.wav


#include <sndfile.h>
#include <fftw3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <numbers>
#include <string>
#include <thread>
#include <vector>

static float hann(int n, int N) {
    return 0.5f - 0.5f * std::cos(2.0f * std::numbers::pi_v<float> * n / (N - 1));
}

static int quant4(float x01) { // x in [0,1]
    x01 = std::clamp(x01, 0.0f, 1.0f);
    int q = int(std::floor(x01 * 4.0f)); // 0..4
    return std::min(q, 3);               // 0..3
}

static std::string glyph2x4(int L, int R) {
    L = std::clamp(L, 0, 4);
    R = std::clamp(R, 0, 4);

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

static std::string glyph_logfreq_char(
    int width,
    int N, int bins, int samplerate,
    const std::vector<float>& db,
    const std::vector<float>& floor_db,
    float f_min = 20.0f,           // lowest displayed frequency (Hz)
    float f_max = -1.0f,           // highest displayed frequency (Hz); <=0 means Nyquist
    float range_db = 40.0f         // dB above floor that maps to "full"
) {
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
    // We combine as: max over bins of (db - floor_db), then normalize to [0,1].
    // (Max works well for thin tonal lines; average works better for noise.)
    auto level_for_range = [&](int a, int b) -> int {
	a = clampi(a, 0, bins - 1);
	b = clampi(b, 0, bins - 1);
	if (b < a) std::swap(a, b);

	float best_rel = -1e9f;
	for (int k = a; k <= b; ++k) {
	    best_rel = std::max(best_rel, db[k] - floor_db[k]);
	}
	return quant4(std::clamp(best_rel / range_db, 0.0f, 1.0f));
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

int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "";

    SF_INFO info{};
    SNDFILE* sf = sf_open(path.c_str(), SFM_READ, &info);

    const int N = 4096;          // FFT size
    const int hop = N / 4;         // hop size
    const int bins = N / 2 + 1;  // r2c bins
    const int chars = bins / 2;  // 2 bins per character

    std::vector<float> interleaved(info.frames * info.channels);
    sf_readf_float(sf, interleaved.data(), info.frames);
    sf_close(sf);

    // Mixdown to mono (simple average)
    std::vector<float> mono(info.frames);
    for (sf_count_t i = 0; i < info.frames; ++i) {
        float s = 0.0f;
        for (int c = 0; c < info.channels; ++c) s += interleaved[i * info.channels + c];
        mono[i] = s / std::max(1, info.channels);
    }

    std::vector<float> in(N);
    std::vector<fftwf_complex> out(bins);
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(N, in.data(), out.data(), FFTW_MEASURE);

    // Precompute Hann window
    std::vector<float> w(N);
    for (int i = 0; i < N; ++i) w[i] = hann(i, N);

    // A simple moving "noise floor" reference using an exponential smoother on log-power.
    std::vector<float> floor_db(bins, -80.0f);
    std::vector<float> db(bins);
    constexpr float alpha = 0.01f; // smoothing for floor
    constexpr float eps = 1e-12f;

    using clock = std::chrono::steady_clock;
    auto time = clock::now();
    auto hop_duration = clock::duration(std::chrono::seconds(hop)) / 44100;
    for (sf_count_t pos = 0; pos + N <= info.frames; pos += hop) {
        for (int i = 0; i < N; ++i) in[i] = mono[pos + i] * w[i];
        fftwf_execute(plan);

        for (int k = 0; k < bins; ++k) {
            float re = out[k][0], im = out[k][1];
            float p = re * re + im * im;
            float d = 10.0f * std::log10(p + eps);
            db[k] = d;
            floor_db[k] = (1.0f - alpha) * floor_db[k] + alpha * d; // track typical level
        }

        // Render one row: normalize relative to floor, compress to [0,1], quantize 4 levels.
        // You can tweak these to taste.
        float range_db = 40.0f; // how far above floor becomes "full"

        std::cout << glyph_logfreq_char(79,
            N, bins, info.samplerate, db, floor_db, 20.0f, -1.0f, range_db
        );
        std::cout.flush();

        time += hop_duration;
        std::this_thread::sleep_for(time - clock::now());
        std::cout << '\n';
    }

    fftwf_destroy_plan(plan);
    return EXIT_SUCCESS;
}
