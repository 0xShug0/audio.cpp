// SanoTtsNeonDecoder (sanotts.cpu_decoder=neon) against a plain double-precision reference of
// the same decoder, on small random weights. No model files needed. On builds without AArch64
// NEON it only checks that the decoder reports itself unavailable and refuses to construct.

#include "neon_decoder.h"

#include "test_assert.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

using engine::models::sanotts::SanoTtsConfig;
using engine::models::sanotts::SanoTtsNeonDecoder;
using Mat = std::vector<double>;   // channel-major (C, T)

SanoTtsConfig small_config() {
    SanoTtsConfig c;
    c.acoustic_hidden = 8;
    c.acoustic_depth = 2;
    c.acoustic_kernel = 5;
    c.mels = 6;
    c.dim = 12;
    c.pw_hidden = 20;
    c.blocks = 2;
    c.noise_channels = 4;
    c.dw_kernel = 7;
    c.embed_kernel = 7;
    c.n_fft = 30;   // head output n_fft + 2 = 32
    return c;
}

std::map<std::string, std::vector<float>> random_weights(const SanoTtsConfig & c, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-0.5F, 0.5F);
    std::map<std::string, std::vector<float>> w;
    auto add = [&](const std::string & name, size_t n, float scale = 1.0F, float offset = 0.0F) {
        auto & v = w[name];
        v.resize(n);
        for (auto & x : v) {
            x = offset + scale * u(rng);
        }
    };
    const size_t H = c.acoustic_hidden, M = c.mels, D = c.dim, P = c.pw_hidden, N = c.noise_channels;
    add("acoustic.frame_input_proj.weight", H * (H + 3));
    add("acoustic.frame_input_proj.bias", H);
    for (int64_t i = 0; i < c.acoustic_depth; ++i) {
        const std::string q = "acoustic.frame_blocks." + std::to_string(i);
        add(q + ".net.0.weight", H * H * c.acoustic_kernel, 0.4F);
        add(q + ".net.0.bias", H);
        add(q + ".net.2.weight", H * H * c.acoustic_kernel, 0.4F);
        add(q + ".net.2.bias", H);
        add(q + ".scale", 1, 0.2F, 0.5F);
    }
    add("acoustic.output.weight", M * H);
    add("acoustic.output.bias", M);
    add("decoder.embed.weight", D * M * c.embed_kernel, 0.4F);
    add("decoder.embed.bias", D);
    add("decoder.noise_adapter.weight", D * N * c.embed_kernel, 0.4F);
    add("decoder.noise_adapter.bias", D);
    add("decoder.norm.weight", D, 0.2F, 1.0F);
    add("decoder.norm.bias", D);
    for (int64_t i = 0; i < c.blocks; ++i) {
        const std::string q = "decoder.blocks." + std::to_string(i);
        add(q + ".dwconv.weight", D * c.dw_kernel);
        add(q + ".dwconv.bias", D);
        add(q + ".norm.weight", D, 0.2F, 1.0F);
        add(q + ".norm.bias", D);
        add(q + ".pwconv1.weight", P * D);
        add(q + ".pwconv1.bias", P);
        add(q + ".pwconv2.weight", D * P);
        add(q + ".pwconv2.bias", D);
        add(q + ".gamma", D, 0.2F, 0.3F);
    }
    add("decoder.final_norm.weight", D, 0.2F, 1.0F);
    add("decoder.final_norm.bias", D);
    add("decoder.head.weight", static_cast<size_t>(c.n_fft + 2) * D);
    add("decoder.head.bias", static_cast<size_t>(c.n_fft + 2));
    return w;
}

// y (O, T) = conv1d(x (C, T), w (O, C, K)) + b, zero padding K / 2, as the GGML graph's conv1d
Mat conv1d(const Mat & x, int C, int T, const std::vector<float> & w, const std::vector<float> & b, int O, int K) {
    Mat y(static_cast<size_t>(O) * T);
    const int pad = K / 2;
    for (int o = 0; o < O; ++o) {
        for (int t = 0; t < T; ++t) {
            double s = b[o];
            for (int ci = 0; ci < C; ++ci) {
                for (int k = 0; k < K; ++k) {
                    const int ti = t + k - pad;
                    if (ti >= 0 && ti < T) {
                        s += static_cast<double>(w[(static_cast<size_t>(o) * C + ci) * K + k]) * x[ci * T + ti];
                    }
                }
            }
            y[static_cast<size_t>(o) * T + t] = s;
        }
    }
    return y;
}

// LayerNorm over the C channels of each frame, eps 1e-6 (nn.LayerNorm(eps=1e-6), as runtime.cpp)
void layer_norm(Mat & x, int C, int T, const std::vector<float> & g, const std::vector<float> & b) {
    for (int t = 0; t < T; ++t) {
        double mean = 0.0, var = 0.0;
        for (int c = 0; c < C; ++c) {
            mean += x[c * T + t];
        }
        mean /= C;
        for (int c = 0; c < C; ++c) {
            var += (x[c * T + t] - mean) * (x[c * T + t] - mean);
        }
        const double inv = 1.0 / std::sqrt(var / C + 1.0e-6);
        for (int c = 0; c < C; ++c) {
            x[c * T + t] = (x[c * T + t] - mean) * inv * g[c] + b[c];
        }
    }
}

// The decoder as build_decoder_graph() defines it, unfused, in double precision.
// Returns frames x (n_fft + 2) rows.
std::vector<double> reference(const SanoTtsConfig & c, std::map<std::string, std::vector<float>> & w,
                              const std::vector<float> & context, const std::vector<float> & feats,
                              const std::vector<float> & noise, int T) {
    const int H = static_cast<int>(c.acoustic_hidden), M = static_cast<int>(c.mels);
    const int D = static_cast<int>(c.dim), P = static_cast<int>(c.pw_hidden);
    const int N = static_cast<int>(c.noise_channels), OUT = static_cast<int>(c.n_fft + 2);
    Mat in(static_cast<size_t>(H + 3) * T);
    for (size_t i = 0; i < context.size(); ++i) in[i] = context[i];
    for (size_t i = 0; i < feats.size(); ++i) in[context.size() + i] = feats[i];
    Mat f = conv1d(in, H + 3, T, w["acoustic.frame_input_proj.weight"], w["acoustic.frame_input_proj.bias"], H, 1);
    for (int64_t i = 0; i < c.acoustic_depth; ++i) {
        const std::string q = "acoustic.frame_blocks." + std::to_string(i);
        const int K = static_cast<int>(c.acoustic_kernel);
        Mat h = conv1d(f, H, T, w[q + ".net.0.weight"], w[q + ".net.0.bias"], H, K);
        for (auto & v : h) v = v / (1.0 + std::exp(-v));   // SiLU
        h = conv1d(h, H, T, w[q + ".net.2.weight"], w[q + ".net.2.bias"], H, K);
        const double s = w[q + ".scale"][0];
        for (size_t k = 0; k < f.size(); ++k) f[k] += s * h[k];
    }
    const Mat mel = conv1d(f, H, T, w["acoustic.output.weight"], w["acoustic.output.bias"], M, 1);
    const int EK = static_cast<int>(c.embed_kernel);
    Mat x = conv1d(mel, M, T, w["decoder.embed.weight"], w["decoder.embed.bias"], D, EK);
    Mat nz(noise.begin(), noise.end());
    const Mat na = conv1d(nz, N, T, w["decoder.noise_adapter.weight"], w["decoder.noise_adapter.bias"], D, EK);
    for (size_t k = 0; k < x.size(); ++k) x[k] += na[k];
    layer_norm(x, D, T, w["decoder.norm.weight"], w["decoder.norm.bias"]);
    const int DK = static_cast<int>(c.dw_kernel);
    for (int64_t i = 0; i < c.blocks; ++i) {
        const std::string q = "decoder.blocks." + std::to_string(i);
        Mat d(static_cast<size_t>(D) * T);
        const auto & dw = w[q + ".dwconv.weight"];
        const auto & db = w[q + ".dwconv.bias"];
        for (int ch = 0; ch < D; ++ch) {
            for (int t = 0; t < T; ++t) {
                double s = db[ch];
                for (int k = 0; k < DK; ++k) {
                    const int ti = t + k - DK / 2;
                    if (ti >= 0 && ti < T) s += static_cast<double>(dw[static_cast<size_t>(ch) * DK + k]) * x[ch * T + ti];
                }
                d[static_cast<size_t>(ch) * T + t] = s;
            }
        }
        layer_norm(d, D, T, w[q + ".norm.weight"], w[q + ".norm.bias"]);
        Mat h = conv1d(d, D, T, w[q + ".pwconv1.weight"], w[q + ".pwconv1.bias"], P, 1);
        for (auto & v : h) v = 0.5 * v * (1.0 + std::erf(v / std::sqrt(2.0)));   // exact GELU
        const Mat y = conv1d(h, P, T, w[q + ".pwconv2.weight"], w[q + ".pwconv2.bias"], D, 1);
        const auto & gamma = w[q + ".gamma"];
        for (int ch = 0; ch < D; ++ch) {
            for (int t = 0; t < T; ++t) x[ch * T + t] += gamma[ch] * y[ch * T + t];
        }
    }
    layer_norm(x, D, T, w["decoder.final_norm.weight"], w["decoder.final_norm.bias"]);
    const Mat head = conv1d(x, D, T, w["decoder.head.weight"], w["decoder.head.bias"], OUT, 1);
    std::vector<double> rows(static_cast<size_t>(T) * OUT);
    for (int t = 0; t < T; ++t) {
        for (int o = 0; o < OUT; ++o) rows[static_cast<size_t>(t) * OUT + o] = head[static_cast<size_t>(o) * T + t];
    }
    return rows;
}

std::vector<float> random_vec(size_t n, std::mt19937 & rng) {
    std::normal_distribution<float> g(0.0F, 1.0F);
    std::vector<float> v(n);
    for (auto & x : v) x = g(rng);
    return v;
}

}  // namespace

int main() try {
    const SanoTtsConfig cfg = small_config();
    auto weights = random_weights(cfg, 20261010);
    const auto reader = [&weights](const std::string & name) {
        const auto found = weights.find(name);
        engine::test::require(found != weights.end(), "test has no tensor " + name);
        return found->second;
    };

    if (!SanoTtsNeonDecoder::available()) {
        bool threw = false;
        try {
            SanoTtsNeonDecoder decoder(cfg, reader, 1);
        } catch (const std::exception &) {
            threw = true;
        }
        engine::test::require(threw, "NEON decoder must refuse to construct without AArch64 NEON");
        std::cout << "sanotts_neon_decoder_test: NEON not available in this build; checked that it refuses\n";
        return 0;
    }

    SanoTtsNeonDecoder decoder(cfg, reader, 2);
    const int H = static_cast<int>(cfg.acoustic_hidden), N = static_cast<int>(cfg.noise_channels);
    const int OUT = static_cast<int>(cfg.n_fft + 2);
    std::mt19937 rng(7);

    // 1. Matches the reference for frame counts around the 16-frame block size, incl. 1 frame.
    for (const int T : {1, 2, 15, 16, 17, 63, 100}) {
        const auto context = random_vec(static_cast<size_t>(H) * T, rng);
        const auto feats = random_vec(static_cast<size_t>(3) * T, rng);
        const auto noise = random_vec(static_cast<size_t>(N) * T, rng);
        const auto got = decoder.decode(context, feats, noise, T, 3);
        const auto want = reference(cfg, weights, context, feats, noise, T);
        engine::test::require_eq(got.size(), want.size(), "output size for T=" + std::to_string(T));
        double max_ref = 0.0, max_diff = 0.0;
        for (size_t i = 0; i < got.size(); ++i) {
            max_ref = std::max(max_ref, std::abs(want[i]));
            max_diff = std::max(max_diff, std::abs(static_cast<double>(got[i]) - want[i]));
        }
        engine::test::require(max_diff <= 1.0e-4 * std::max(1.0, max_ref),
                              "T=" + std::to_string(T) + ": max difference " + std::to_string(max_diff) +
                                  " from the reference (max |ref| " + std::to_string(max_ref) + ")");
    }

    // 2. Same input -> byte-identical output for any thread count, and across buffer reuse
    //    (a long call, a short one, then the long one again on the same instance).
    const int T = 100;
    const auto context = random_vec(static_cast<size_t>(H) * T, rng);
    const auto feats = random_vec(static_cast<size_t>(3) * T, rng);
    const auto noise = random_vec(static_cast<size_t>(N) * T, rng);
    const auto first = decoder.decode(context, feats, noise, T, 1);
    for (const int threads : {2, 3, 4, 7}) {
        engine::test::require(decoder.decode(context, feats, noise, T, threads) == first,
                              "output changed with " + std::to_string(threads) + " threads");
    }
    const auto short_ctx = random_vec(static_cast<size_t>(H) * 17, rng);
    const auto short_feats = random_vec(static_cast<size_t>(3) * 17, rng);
    const auto short_noise = random_vec(static_cast<size_t>(N) * 17, rng);
    (void)decoder.decode(short_ctx, short_feats, short_noise, 17, 3);
    engine::test::require(decoder.decode(context, feats, noise, T, 3) == first,
                          "output changed after reusing the instance for a shorter input");
    engine::test::require_eq(static_cast<int>(first.size()), T * OUT, "row layout frames x (n_fft + 2)");

    // 3. Wrong input sizes are rejected.
    bool threw = false;
    try {
        (void)decoder.decode(context, feats, noise, T + 1, 3);
    } catch (const std::exception &) {
        threw = true;
    }
    engine::test::require(threw, "decode must reject inputs whose size does not match frames");

    std::cout << "sanotts_neon_decoder_test: ok\n";
    return 0;
} catch (const std::exception & e) {
    std::cerr << "sanotts_neon_decoder_test failed: " << e.what() << "\n";
    return 1;
}
