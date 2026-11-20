#include "engine/community_models/kale_tts/source_filter.h"

#include "engine/framework/audio/fft.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace engine::models::kale_tts {
namespace {

constexpr double kPi = 3.14159265358979323846;
using Complex = std::complex<float>;

// [bins x points]: linear interpolation from points spaced on log(1 + f / 700) to the linear STFT bins
std::vector<float> warp_matrix(int64_t bins, int64_t points, int64_t sample_rate) {
    std::vector<float> out(static_cast<size_t>(bins * points), 0.0F);
    const float top = std::log1p(static_cast<float>(sample_rate) / 2.0F / 700.0F);
    for (int64_t b = 0; b < bins; ++b) {
        const float f = static_cast<float>(sample_rate) / 2.0F * static_cast<float>(b) / static_cast<float>(bins - 1);
        const float pos = std::log1p(f / 700.0F) / top * static_cast<float>(points - 1);
        const auto i = std::clamp<int64_t>(static_cast<int64_t>(std::floor(pos)), 0, points - 2);
        const float w = pos - static_cast<float>(i);
        out[static_cast<size_t>(b * points + i)] = 1.0F - w;
        out[static_cast<size_t>(b * points + i + 1)] += w;
    }
    return out;
}

std::vector<float> hann(int64_t n) {
    std::vector<float> w(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        w[static_cast<size_t>(i)] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(i) / static_cast<double>(n)));
    }
    return w;
}

// torch.stft(x, n_fft, hop, hann, center, reflect): [frames x bins]
std::vector<Complex> stft(const std::vector<float> & x, int64_t n_fft, int64_t hop) {
    const auto samples = static_cast<int64_t>(x.size());
    const int64_t pad = n_fft / 2;
    const int64_t frames = samples / hop + 1;
    const int64_t bins = n_fft / 2 + 1;
    const auto window = hann(n_fft);
    std::vector<float> framed(static_cast<size_t>(frames * n_fft));
    for (int64_t f = 0; f < frames; ++f) {
        for (int64_t i = 0; i < n_fft; ++i) {
            int64_t at = f * hop + i - pad;
            if (at < 0) at = -at;
            if (at >= samples) at = 2 * (samples - 1) - at;
            framed[static_cast<size_t>(f * n_fft + i)] = x[static_cast<size_t>(std::clamp<int64_t>(at, 0, samples - 1))] * window[static_cast<size_t>(i)];
        }
    }
    std::vector<Complex> out(static_cast<size_t>(frames * bins));
    const auto plan = audio::get_real_fft_plan(static_cast<size_t>(n_fft));
    plan->forward({static_cast<size_t>(frames), static_cast<size_t>(n_fft)},
                  {static_cast<std::ptrdiff_t>(n_fft * sizeof(float)), static_cast<std::ptrdiff_t>(sizeof(float))},
                  {static_cast<std::ptrdiff_t>(bins * sizeof(Complex)), static_cast<std::ptrdiff_t>(sizeof(Complex))}, 1, framed.data(), out.data());
    return out;
}

// torch.istft with the same settings, output trimmed to `samples`; adds into `y`
void istft_add(const std::vector<Complex> & spec, int64_t n_fft, int64_t hop, int64_t samples, std::vector<float> & y) {
    const int64_t bins = n_fft / 2 + 1;
    const auto frames = static_cast<int64_t>(spec.size()) / bins;
    const auto window = hann(n_fft);
    std::vector<float> framed(static_cast<size_t>(frames * n_fft));
    const auto plan = audio::get_real_fft_plan(static_cast<size_t>(n_fft));
    plan->inverse({static_cast<size_t>(frames), static_cast<size_t>(n_fft)},
                  {static_cast<std::ptrdiff_t>(bins * sizeof(Complex)), static_cast<std::ptrdiff_t>(sizeof(Complex))},
                  {static_cast<std::ptrdiff_t>(n_fft * sizeof(float)), static_cast<std::ptrdiff_t>(sizeof(float))}, 1, spec.data(), framed.data(),
                  1.0F / static_cast<float>(n_fft));
    const int64_t total = (frames - 1) * hop + n_fft;
    std::vector<float> sum(static_cast<size_t>(total), 0.0F);
    std::vector<float> envelope(static_cast<size_t>(total), 0.0F);
    for (int64_t f = 0; f < frames; ++f) {
        for (int64_t i = 0; i < n_fft; ++i) {
            const float w = window[static_cast<size_t>(i)];
            sum[static_cast<size_t>(f * hop + i)] += framed[static_cast<size_t>(f * n_fft + i)] * w;
            envelope[static_cast<size_t>(f * hop + i)] += w * w;
        }
    }
    const int64_t pad = n_fft / 2;
    for (int64_t t = 0; t < samples; ++t) {
        y[static_cast<size_t>(t)] += sum[static_cast<size_t>(t + pad)] / std::max(envelope[static_cast<size_t>(t + pad)], 1e-8F);
    }
}

}  // namespace

std::vector<float> source_filter(const SourceFilterConfig & c, const std::vector<float> & head, const std::vector<float> & burst,
                                 const std::vector<float> & f0, const std::vector<float> & noise) {
    const auto n = static_cast<int64_t>(f0.size());
    const int64_t up = c.f0_hop / c.hop;
    const int64_t T = n * up;
    const int64_t S = n * c.f0_hop;
    const int64_t bins = c.n_fft / 2 + 1;
    const int64_t channels = 4 + 2 * c.mod;
    const int64_t row = channels * c.points + 1;
    if (static_cast<int64_t>(head.size()) != n * up * row || static_cast<int64_t>(burst.size()) != n * 4 * c.burst_points ||
        static_cast<int64_t>(noise.size()) < S) {
        throw std::runtime_error("kale-tts source filter got inputs of the wrong size");
    }

    // the sine source: running phase per pitch frame (float32, wrapped), interpolated between frame centres
    std::vector<float> c0(static_cast<size_t>(n));
    std::vector<float> d(static_cast<size_t>(n), 0.0F);
    float running = 0.0F;
    for (int64_t i = 0; i < n; ++i) {
        const float r = f0[static_cast<size_t>(i)] * static_cast<float>(c.f0_hop) / static_cast<float>(c.sample_rate);
        running += r;
        c0[static_cast<size_t>(i)] = running - std::floor(running);
        if (i > 0) d[static_cast<size_t>(i - 1)] = r;
    }
    std::vector<float> e(static_cast<size_t>(S)), cycles(static_cast<size_t>(S)), voiced(static_cast<size_t>(S));
    for (int64_t t = 0; t < S; ++t) {
        const float p = (static_cast<float>(t) + 0.5F) / static_cast<float>(c.f0_hop) - 0.5F;
        const auto i = std::clamp<int64_t>(static_cast<int64_t>(std::floor(p)), 0, n - 1);
        const float frac = std::clamp(p - static_cast<float>(i), 0.0F, 1.0F);
        const float cyc = c0[static_cast<size_t>(i)] + frac * d[static_cast<size_t>(i)];
        const float hz = f0[static_cast<size_t>(t / c.f0_hop)];
        cycles[static_cast<size_t>(t)] = cyc;
        voiced[static_cast<size_t>(t)] = hz > 10.0F ? 1.0F : 0.0F;
        if (hz <= 10.0F) {
            e[static_cast<size_t>(t)] = 0.0F;
            continue;
        }
        // harmonics k = 1..K below 11 kHz, summed in closed form: sin(K x / 2) sin((K + 1) x / 2) / sin(x / 2)
        int64_t K = 0;
        while (K < c.harmonics && hz * static_cast<float>(K + 1) < 11000.0F) ++K;
        const double x = 2.0 * kPi * static_cast<double>(cyc);
        const double s = std::sin(x / 2.0);
        double sum = 0.0;
        if (std::fabs(s) > 1e-6) {
            sum = std::sin(static_cast<double>(K) * x / 2.0) * std::sin(static_cast<double>(K + 1) * x / 2.0) / s;
        } else {
            for (int64_t k = 1; k <= K; ++k) sum += std::sin(static_cast<double>(k) * x);
        }
        e[static_cast<size_t>(t)] = static_cast<float>(0.1 * sum);
    }

    // the network's filter per STFT frame (T + 1 frames: the centred STFT has one more)
    const auto M = warp_matrix(bins, c.points, c.sample_rate);
    auto o = [&](int64_t t) { return &head[static_cast<size_t>(std::min(t, T - 1) * row)]; };
    std::vector<Complex> E = stft(e, c.n_fft, c.hop);
    std::vector<float> plain(noise.begin(), noise.begin() + S);
    std::vector<Complex> Nz = stft(plain, c.n_fft, c.hop);
    const int64_t frames = T + 1;
    std::vector<float> g(static_cast<size_t>(channels * bins * frames));   // [channel][frame][bin]
    for (int64_t t = 0; t < frames; ++t) {
        const float * values = o(t);
        for (int64_t ch = 0; ch < channels; ++ch) {
            const float * v = values + ch * c.points;
            float * out = &g[static_cast<size_t>((ch * frames + t) * bins)];
            for (int64_t b = 0; b < bins; ++b) {
                const float * m = &M[static_cast<size_t>(b * c.points)];
                float acc = 0.0F;
                for (int64_t p = 0; p < c.points; ++p) acc += v[p] * m[p];
                out[b] = acc;
            }
        }
    }
    auto G = [&](int64_t ch, int64_t t, int64_t b) { return g[static_cast<size_t>((ch * frames + t) * bins + b)]; };
    int64_t j = 0;
    std::vector<float> pulsing(static_cast<size_t>(S));
    for (int64_t m = 1; m <= c.mod; ++m) {
        for (int k = 0; k < 2; ++k, ++j) {
            for (int64_t t = 0; t < S; ++t) {
                const double angle = 2.0 * kPi * static_cast<double>(m) * static_cast<double>(cycles[static_cast<size_t>(t)]);
                pulsing[static_cast<size_t>(t)] = plain[static_cast<size_t>(t)] * voiced[static_cast<size_t>(t)] *
                                                  static_cast<float>(k == 0 ? std::cos(angle) : std::sin(angle));
            }
            const auto P = stft(pulsing, c.n_fft, c.hop);
            for (int64_t t = 0; t < frames; ++t) {
                for (int64_t b = 0; b < bins; ++b) {
                    Nz[static_cast<size_t>(t * bins + b)] += G(4 + j, t, b) * P[static_cast<size_t>(t * bins + b)];
                }
            }
        }
    }
    std::vector<Complex> Y(static_cast<size_t>(frames * bins));
    for (int64_t t = 0; t < frames; ++t) {
        const float hz = std::max(f0[static_cast<size_t>(std::min(t / up, n - 1))], 60.0F);
        const float shift = 0.5F * std::tanh(o(t)[row - 1]);
        for (int64_t b = 0; b < bins; ++b) {
            const float gh = std::exp(std::min(G(0, t, b), 6.0F));
            const float gn = std::exp(std::min(G(3, t, b), 6.0F));
            const float ur0 = G(1, t, b);
            const float ui0 = G(2, t, b);
            const float um = std::max(std::sqrt(ur0 * ur0 + ui0 * ui0), 1e-4F);
            const float freq = static_cast<float>(c.sample_rate) / 2.0F * static_cast<float>(b) / static_cast<float>(bins - 1);
            const float angle = 2.0F * static_cast<float>(kPi) * (freq / hz) * shift;
            const Complex H = gh * Complex(ur0 / um, ui0 / um) * Complex(std::cos(angle), std::sin(angle));
            const auto at = static_cast<size_t>(t * bins + b);
            Y[at] = E[at] * H + Nz[at] * gn;
        }
    }
    std::vector<float> y(static_cast<size_t>(S), 0.0F);
    istft_add(Y, c.n_fft, c.hop, S, y);

    // the short-window noise burst path (n_fft 150, hop 75: 4 frames per trunk frame)
    const int64_t burst_bins = 150 / 2 + 1;
    const auto Mb = warp_matrix(burst_bins, c.burst_points, c.sample_rate);
    std::vector<float> reversed(plain.rbegin(), plain.rend());
    auto Nb = stft(reversed, 150, 75);
    const int64_t burst_frames = 4 * n + 1;
    for (int64_t q = 0; q < burst_frames; ++q) {
        const float * v = &burst[static_cast<size_t>(std::min(q, 4 * n - 1) * c.burst_points)];
        for (int64_t b = 0; b < burst_bins; ++b) {
            float acc = 0.0F;
            for (int64_t p = 0; p < c.burst_points; ++p) acc += v[p] * Mb[static_cast<size_t>(b * c.burst_points + p)];
            Nb[static_cast<size_t>(q * burst_bins + b)] *= std::exp(std::min(acc, 6.0F));
        }
    }
    istft_add(Nb, 150, 75, S, y);
    return y;
}

}  // namespace engine::models::kale_tts
