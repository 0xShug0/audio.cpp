// SanoTtsNeonDecoder against a plain double-precision reference of the same
// decoder and front-end stages, plus fused decode_audio() against decode() +
// SanoTtsNeonIstft + sequential DC blocker, on small random weights. No model
// files needed. On builds without AArch64 NEON it only checks that the decoder
// reports itself unavailable and refuses to construct.

#include "engine/community_models/sanotts/neon_decoder.h"
#include "engine/community_models/sanotts/neon_istft.h"

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
using engine::models::sanotts::SanoTtsNeonIstft;
using Mat = std::vector<double>;  // channel-major (C, T)

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr float kDcBlockPole = 0.9973F;

SanoTtsConfig small_config() {
    SanoTtsConfig c;
    c.vocab_size = 19;
    c.duration_hidden = 10;
    c.duration_depth = 2;
    c.duration_kernel = 5;
    c.acoustic_hidden = 8;
    c.acoustic_token_depth = 2;
    c.acoustic_depth = 2;
    c.acoustic_kernel = 5;
    c.mels = 6;
    c.dim = 12;
    c.pw_hidden = 20;
    c.blocks = 2;
    c.noise_channels = 4;
    c.dw_kernel = 7;
    c.embed_kernel = 7;
    c.n_fft = 30;  // head output n_fft + 2 = 32
    c.hop_length = 8;
    return c;
}

SanoTtsConfig audio_config() {
    SanoTtsConfig c = small_config();
    c.n_fft = 1024;
    c.hop_length = 256;
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
    const size_t V = c.vocab_size, DH = c.duration_hidden;
    const size_t H = c.acoustic_hidden, M = c.mels, D = c.dim, P = c.pw_hidden, N = c.noise_channels;

    // Front-end: duration
    add("duration.embedding.weight", V * DH);
    add("duration.input_proj.weight", DH * (DH + 3));
    add("duration.input_proj.bias", DH);
    for (int64_t i = 0; i < c.duration_depth; ++i) {
        const std::string q = "duration.blocks." + std::to_string(i);
        add(q + ".net.0.weight", DH * DH * c.duration_kernel, 0.4F);
        add(q + ".net.0.bias", DH);
        add(q + ".net.2.weight", DH * DH * c.duration_kernel, 0.4F);
        add(q + ".net.2.bias", DH);
        add(q + ".scale", 1, 0.2F, 0.5F);
    }
    add("duration.output.weight", DH);
    add("duration.output.bias", 1);

    // Front-end: acoustic token stage
    add("acoustic.embedding.weight", V * H);
    add("acoustic.token_input_proj.weight", H * (H + 2));
    add("acoustic.token_input_proj.bias", H);
    for (int64_t i = 0; i < c.acoustic_token_depth; ++i) {
        const std::string q = "acoustic.token_blocks." + std::to_string(i);
        add(q + ".net.0.weight", H * H * c.acoustic_kernel, 0.4F);
        add(q + ".net.0.bias", H);
        add(q + ".net.2.weight", H * H * c.acoustic_kernel, 0.4F);
        add(q + ".net.2.bias", H);
        add(q + ".scale", 1, 0.2F, 0.5F);
    }

    // Frame stage + ConvNeXt decoder
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
    add("decoder.head.weight", static_cast<size_t>(c.n_fft + 2) * D, 0.3F);
    add("decoder.head.bias", static_cast<size_t>(c.n_fft + 2), 0.3F);
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

void layer_norm(Mat & x, int C, int T, const std::vector<float> & g, const std::vector<float> & b) {
    for (int t = 0; t < T; ++t) {
        double mean = 0.0, var = 0.0;
        for (int c = 0; c < C; ++c) mean += x[c * T + t];
        mean /= C;
        for (int c = 0; c < C; ++c) var += (x[c * T + t] - mean) * (x[c * T + t] - mean);
        const double inv = 1.0 / std::sqrt(var / C + 1.0e-6);
        for (int c = 0; c < C; ++c) x[c * T + t] = (x[c * T + t] - mean) * inv * g[c] + b[c];
    }
}

std::vector<double> front_reference(
    std::map<std::string, std::vector<float>> & w,
    const std::string & emb_name, const std::string & proj_name, const std::string & blk_prefix,
    const char * out_name, int H, int feat_rows, int depth, int K,
    const std::vector<int32_t> & tokens, const std::vector<float> & feats) {
    const int N = static_cast<int>(tokens.size());
    const auto & emb = w[emb_name];
    Mat in(static_cast<size_t>(H + feat_rows) * N);
    for (int t = 0; t < N; ++t) {
        for (int c = 0; c < H; ++c) in[static_cast<size_t>(c) * N + t] = emb[static_cast<size_t>(tokens[t]) * H + c];
        for (int c = 0; c < feat_rows; ++c) in[static_cast<size_t>(H + c) * N + t] = feats[static_cast<size_t>(c) * N + t];
    }
    Mat f = conv1d(in, H + feat_rows, N, w[proj_name + ".weight"], w[proj_name + ".bias"], H, 1);
    for (int i = 0; i < depth; ++i) {
        const std::string q = blk_prefix + std::to_string(i);
        Mat h = conv1d(f, H, N, w[q + ".net.0.weight"], w[q + ".net.0.bias"], H, K);
        for (auto & v : h) v = v / (1.0 + std::exp(-v));
        h = conv1d(h, H, N, w[q + ".net.2.weight"], w[q + ".net.2.bias"], H, K);
        const double s = w[q + ".scale"][0];
        for (size_t k = 0; k < f.size(); ++k) f[k] += s * h[k];
    }
    if (out_name != nullptr) {
        return conv1d(f, H, N, w[std::string(out_name) + ".weight"], w[std::string(out_name) + ".bias"], 1, 1);
    }
    return f;
}

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
        for (auto & v : h) v = v / (1.0 + std::exp(-v));
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
        for (auto & v : h) v = 0.5 * v * (1.0 + std::erf(v / std::sqrt(2.0)));
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

std::vector<int32_t> random_tokens(size_t n, int vocab, std::mt19937 & rng) {
    std::uniform_int_distribution<int32_t> d(0, vocab - 1);
    std::vector<int32_t> v(n);
    for (auto & x : v) x = d(rng);
    return v;
}

std::vector<float> periodic_hann_window(int64_t n_fft) {
    std::vector<float> w(static_cast<size_t>(n_fft));
    for (int64_t i = 0; i < n_fft; ++i) {
        w[static_cast<size_t>(i)] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(i) / static_cast<double>(n_fft)));
    }
    return w;
}

void dc_block_ref(std::vector<float> & samples) {
    float x1 = 0.0F, y1 = 0.0F;
    for (float & s : samples) {
        const float x = s;
        const float y = x - x1 + kDcBlockPole * y1;
        x1 = x;
        y1 = y;
        s = y;
    }
}

void check_close(const std::vector<float> & got, const std::vector<double> & want, double rel_tol, const std::string & label) {
    engine::test::require_eq(got.size(), want.size(), "output size for " + label);
    double max_ref = 0.0, max_diff = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        max_ref = std::max(max_ref, std::abs(want[i]));
        max_diff = std::max(max_diff, std::abs(static_cast<double>(got[i]) - want[i]));
    }
    engine::test::require(max_diff <= rel_tol * std::max(1.0, max_ref),
                          label + ": max difference " + std::to_string(max_diff) +
                              " from reference (max |ref| " + std::to_string(max_ref) + ")");
}

void check_close_f(const std::vector<float> & got, const std::vector<float> & want, double rel_tol, const std::string & label) {
    engine::test::require_eq(got.size(), want.size(), "output size for " + label);
    double max_ref = 0.0, max_diff = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        max_ref = std::max(max_ref, static_cast<double>(std::fabs(want[i])));
        max_diff = std::max(max_diff, static_cast<double>(std::fabs(got[i] - want[i])));
    }
    engine::test::require(max_diff <= rel_tol * std::max(1.0, max_ref),
                          label + ": max difference " + std::to_string(max_diff) +
                              " from reference (max |ref| " + std::to_string(max_ref) + ")");
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
#ifdef ENGINE_HAVE_NEON
            SanoTtsNeonDecoder decoder(cfg, engine::models::sanotts::load_sanotts_neon_weights(cfg, reader, 1));
#else
            SanoTtsNeonDecoder decoder(cfg, reader, 1);
#endif
        } catch (const std::exception &) {
            threw = true;
        }
        engine::test::require(threw, "NEON decoder must refuse to construct without AArch64 NEON");
        std::cout << "sanotts_neon_decoder_test: NEON not available in this build; checked that it refuses\n";
        return 0;
    }

#ifdef ENGINE_HAVE_NEON
    const auto bound = engine::models::sanotts::load_sanotts_neon_weights(cfg, reader, 2);
    SanoTtsNeonDecoder decoder(cfg, bound);
#else
    SanoTtsNeonDecoder decoder(cfg, reader, 2);
#endif
    const int H = static_cast<int>(cfg.acoustic_hidden), N = static_cast<int>(cfg.noise_channels);
    const int OUT = static_cast<int>(cfg.n_fft + 2);
    std::mt19937 rng(7);

    // 1. decode() matches the reference across block boundaries, incl. 1 frame.
    for (const int T : {1, 2, 11, 12, 13, 15, 16, 17, 63, 100}) {
        const auto context = random_vec(static_cast<size_t>(H) * T, rng);
        const auto feats = random_vec(static_cast<size_t>(3) * T, rng);
        const auto noise = random_vec(static_cast<size_t>(N) * T, rng);
        const auto got = decoder.decode(context, feats, noise, T, 3);
        const auto want = reference(cfg, weights, context, feats, noise, T);
        check_close(got, want, 1.0e-4, "decode T=" + std::to_string(T));
    }

    // 2. decode() is byte-identical for any thread count and across buffer reuse.
    const int T = 100;
    const auto context = random_vec(static_cast<size_t>(H) * T, rng);
    const auto feats = random_vec(static_cast<size_t>(3) * T, rng);
    const auto noise = random_vec(static_cast<size_t>(N) * T, rng);
    const auto first = decoder.decode(context, feats, noise, T, 1);
    for (const int threads : {2, 3, 4, 7}) {
        engine::test::require(decoder.decode(context, feats, noise, T, threads) == first,
                              "decode output changed with " + std::to_string(threads) + " threads");
    }
    const auto short_ctx = random_vec(static_cast<size_t>(H) * 17, rng);
    const auto short_feats = random_vec(static_cast<size_t>(3) * 17, rng);
    const auto short_noise = random_vec(static_cast<size_t>(N) * 17, rng);
    (void)decoder.decode(short_ctx, short_feats, short_noise, 17, 3);
    engine::test::require(decoder.decode(context, feats, noise, T, 3) == first,
                          "decode output changed after reusing the instance for a shorter input");
    engine::test::require_eq(static_cast<int>(first.size()), T * OUT, "row layout frames x (n_fft + 2)");

    // 3. Wrong input sizes are rejected.
    bool threw = false;
    try {
        (void)decoder.decode(context, feats, noise, T + 1, 3);
    } catch (const std::exception &) {
        threw = true;
    }
    engine::test::require(threw, "decode must reject inputs whose size does not match frames");

#ifndef ENGINE_HAVE_NEON
    // 4. NEON front-end (predict_duration + predict_acoustic): reference, thread invariance, reuse, errors.
    engine::test::require(!decoder.has_frontend(), "front-end should start uninitialized");
    engine::test::require(!decoder.can_decode_audio(), "n_fft=30 must not claim can_decode_audio");
    decoder.init_frontend(cfg, reader);
    engine::test::require(decoder.has_frontend(), "front-end should report ready after init_frontend");

    for (const int tok_n : {1, 2, 11, 12, 13, 47, 48, 49, 100}) {
        const auto toks = random_tokens(static_cast<size_t>(tok_n), static_cast<int>(cfg.vocab_size), rng);
        const auto dfeats = random_vec(static_cast<size_t>(3) * tok_n, rng);
        const auto afeats = random_vec(static_cast<size_t>(2) * tok_n, rng);

        const auto dur_want = front_reference(
            weights, "duration.embedding.weight", "duration.input_proj", "duration.blocks.",
            "duration.output", static_cast<int>(cfg.duration_hidden), 3,
            static_cast<int>(cfg.duration_depth), static_cast<int>(cfg.duration_kernel), toks, dfeats);
        const auto dur_1t = decoder.predict_duration(toks, dfeats, 1);
        check_close(dur_1t, dur_want, 1.0e-4, "predict_duration N=" + std::to_string(tok_n));
        for (const int threads : {2, 4, 7}) {
            engine::test::require(decoder.predict_duration(toks, dfeats, threads) == dur_1t,
                                  "predict_duration output changed with " + std::to_string(threads) +
                                      " threads at N=" + std::to_string(tok_n));
        }

        const auto aco_want = front_reference(
            weights, "acoustic.embedding.weight", "acoustic.token_input_proj", "acoustic.token_blocks.",
            nullptr, static_cast<int>(cfg.acoustic_hidden), 2,
            static_cast<int>(cfg.acoustic_token_depth), static_cast<int>(cfg.acoustic_kernel), toks, afeats);
        const auto aco_1t = decoder.predict_acoustic(toks, afeats, 1);
        check_close(aco_1t, aco_want, 1.0e-4, "predict_acoustic N=" + std::to_string(tok_n));
        for (const int threads : {2, 4, 7}) {
            engine::test::require(decoder.predict_acoustic(toks, afeats, threads) == aco_1t,
                                  "predict_acoustic output changed with " + std::to_string(threads) +
                                      " threads at N=" + std::to_string(tok_n));
        }
    }

    // Front-end buffer reuse (long -> short -> long)
    {
        const auto t100 = random_tokens(100, static_cast<int>(cfg.vocab_size), rng);
        const auto d100 = random_vec(300, rng);
        const auto a100 = random_vec(200, rng);
        const auto t13 = random_tokens(13, static_cast<int>(cfg.vocab_size), rng);
        const auto d13 = random_vec(39, rng);
        const auto a13 = random_vec(26, rng);
        const auto dur_a = decoder.predict_duration(t100, d100, 4);
        const auto aco_a = decoder.predict_acoustic(t100, a100, 4);
        (void)decoder.predict_duration(t13, d13, 4);
        (void)decoder.predict_acoustic(t13, a13, 4);
        engine::test::require(decoder.predict_duration(t100, d100, 4) == dur_a, "predict_duration changed after reuse");
        engine::test::require(decoder.predict_acoustic(t100, a100, 4) == aco_a, "predict_acoustic changed after reuse");
    }

    // Front-end invalid inputs rejected
    threw = false;
    try { (void)decoder.predict_duration({}, {}, 1); } catch (const std::exception &) { threw = true; }
    engine::test::require(threw, "predict_duration must reject empty tokens");
    threw = false;
    try { (void)decoder.predict_duration({0, static_cast<int32_t>(cfg.vocab_size)}, random_vec(6, rng), 1); } catch (const std::exception &) { threw = true; }
    engine::test::require(threw, "predict_duration must reject out-of-range token id");
    threw = false;
    try { (void)decoder.predict_acoustic({0, 1}, random_vec(3, rng), 1); } catch (const std::exception &) { threw = true; }
    engine::test::require(threw, "predict_acoustic must reject mismatched feature size");

    // 5. Fused decode_audio() (n_fft=1024, hop=256): reference, thread invariance (1/2/4/7), reuse, errors.
    {
        const SanoTtsConfig acfg = audio_config();
        auto aweights = random_weights(acfg, 20261126);
        const auto areader = [&aweights](const std::string & name) {
            const auto found = aweights.find(name);
            engine::test::require(found != aweights.end(), "audio test has no tensor " + name);
            return found->second;
        };
        SanoTtsNeonDecoder adec(acfg, areader, 2);
        engine::test::require(adec.can_decode_audio(), "n_fft=1024/hop=256 must support decode_audio");
        SanoTtsNeonIstft istft;
        const auto window = periodic_hann_window(acfg.n_fft);

        for (const int AT : {2, 3, 4, 11, 12, 13, 25, 64}) {
            const auto actx = random_vec(static_cast<size_t>(acfg.acoustic_hidden) * AT, rng);
            const auto aft = random_vec(static_cast<size_t>(3) * AT, rng);
            const auto anz = random_vec(static_cast<size_t>(acfg.noise_channels) * AT, rng);

            // Reference: unfused decode() -> SanoTtsNeonIstft::compute() -> sequential DC blocker.
            const auto spec = adec.decode(actx, aft, anz, AT, 1);
            auto want_audio = istft.compute(spec, AT, window, 1);
            dc_block_ref(want_audio);

            const auto got_1t = adec.decode_audio(actx, aft, anz, AT, 1);
            engine::test::require_eq(got_1t.size(), static_cast<size_t>(AT - 1) * 256,
                                     "decode_audio sample count for T=" + std::to_string(AT));
            check_close_f(got_1t, want_audio, 1.0e-4, "decode_audio T=" + std::to_string(AT));

            // Byte-identical across 1, 2, 4, 7 threads!
            for (const int threads : {2, 4, 7}) {
                engine::test::require(adec.decode_audio(actx, aft, anz, AT, threads) == got_1t,
                                      "decode_audio output changed with " + std::to_string(threads) +
                                          " threads at T=" + std::to_string(AT));
            }
        }

        // Buffer reuse (T=64 -> T=13 -> T=64)
        const auto c64 = random_vec(static_cast<size_t>(acfg.acoustic_hidden) * 64, rng);
        const auto f64 = random_vec(3 * 64, rng);
        const auto n64 = random_vec(static_cast<size_t>(acfg.noise_channels) * 64, rng);
        const auto c13 = random_vec(static_cast<size_t>(acfg.acoustic_hidden) * 13, rng);
        const auto f13 = random_vec(3 * 13, rng);
        const auto n13 = random_vec(static_cast<size_t>(acfg.noise_channels) * 13, rng);
        const auto a64 = adec.decode_audio(c64, f64, n64, 64, 4);
        (void)adec.decode_audio(c13, f13, n13, 13, 4);
        engine::test::require(adec.decode_audio(c64, f64, n64, 64, 4) == a64,
                              "decode_audio changed after buffer reuse");

        // Invalid inputs rejected
        threw = false;
        try { (void)adec.decode_audio(random_vec(acfg.acoustic_hidden, rng), random_vec(3, rng), random_vec(acfg.noise_channels, rng), 1, 1); }
        catch (const std::exception &) { threw = true; }
        engine::test::require(threw, "decode_audio must reject frames < 2");
        threw = false;
        try { (void)adec.decode_audio(c13, f13, n13, 14, 1); }
        catch (const std::exception &) { threw = true; }
        engine::test::require(threw, "decode_audio must reject mismatched input sizes");
    }
#endif

    std::cout << "sanotts_neon_decoder_test: ok\n";
    return 0;
} catch (const std::exception & e) {
    std::cerr << "sanotts_neon_decoder_test failed: " << e.what() << "\n";
    return 1;
}
