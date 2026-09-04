#include "engine/community_models/sanotts/runtime.h"

#include "graph_common.h"

#include "engine/framework/audio/istft_graph.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::sanotts {
namespace {

namespace core = engine::core;
namespace modules = engine::modules;
using namespace engine::models::sanotts::graph;   // shared graph helpers

constexpr size_t kIoArenaBytes = 8ULL * 1024ULL * 1024ULL;
constexpr size_t kGraphArenaBytes = 128ULL * 1024ULL * 1024ULL;
constexpr size_t kWeightArenaBytes = 32ULL * 1024ULL * 1024ULL;

// The decoder's norms are nn.LayerNorm(eps=1e-6), NOT torch's 1e-5 default.
// The difference compounds through the ConvNeXt blocks and is then amplified
// by the exp() in the magnitude head; the reference implementations document
// losing 0.06 of correlation and a third of the output amplitude to exactly
// this constant.
constexpr float kLayerNormEps = 1.0e-6F;
constexpr float kDcBlockPole = 0.9973F;
constexpr double kPi = 3.14159265358979323846;

/** Tensor count implied by the config -- 103 for heart-nano, 117 for heart.
 *  Kept in lockstep with the inventory validate_nano_tensors() builds. */
size_t expected_tensor_count(const SanoTtsConfig & config) {
    const auto duration = 5 + 5 * config.duration_depth;
    const auto acoustic =
        7 + 5 * (config.acoustic_token_depth + config.acoustic_depth);
    const auto decoder = 10 + 9 * config.blocks;
    return static_cast<size_t>(duration + acoustic + decoder);
}

core::TensorValue channel_last_layer_norm(
    core::ModuleBuildContext & ctx,
    const SanoTtsBackendWeights & weights,
    const core::TensorValue & input,
    const std::string & prefix,
    int64_t channels) {
    return modules::LayerNormModule({channels, kLayerNormEps, true, true}).build(
        ctx,
        input,
        {
            weight(weights, prefix + ".weight"),
            weight(weights, prefix + ".bias"),
        });
}

// ---- ATen-compatible noise ----------------------------------------------
//
// The decoder is noise-fed, so a rendering is only reproducible if the noise
// stream is. This is PyTorch's CPU path exactly: MT19937 seeded from the low
// 32 bits, a 24-bit float uniform, and Box-Muller in blocks of 16
// (aten/src/ATen/native/DistributionTemplates.h). The integer half is
// bit-exact against torch.rand; the Gaussian half goes through logf/cosf/
// sinf, which differ by a few ulp across libm builds -- the reference
// implementations document the same bound (at most ~2e-6 per draw).

constexpr int kMtN = 624;
constexpr int kMtM = 397;

struct AtenMt19937 {
    uint32_t state[kMtN];
    int left = 1;
    int next = 0;

    explicit AtenMt19937(uint64_t seed) {
        // ATen mt19937_engine::init_with_uint32 -- only the low 32 bits used.
        state[0] = static_cast<uint32_t>(seed & 0xFFFFFFFFULL);
        for (int i = 1; i < kMtN; ++i) {
            state[i] = 1812433253U * (state[i - 1] ^ (state[i - 1] >> 30)) +
                       static_cast<uint32_t>(i);
        }
    }

    static uint32_t twist(uint32_t u, uint32_t v) {
        const uint32_t mixed = (u & 0x80000000U) | (v & 0x7FFFFFFFU);
        return (mixed >> 1) ^ ((v & 1U) != 0U ? 0x9908B0DFU : 0U);
    }

    void next_state() {
        left = kMtN;
        next = 0;
        int i = 0;
        for (; i < kMtN - kMtM; ++i) {
            state[i] = state[i + kMtM] ^ twist(state[i], state[i + 1]);
        }
        for (; i < kMtN - 1; ++i) {
            state[i] = state[i + kMtM - kMtN] ^ twist(state[i], state[i + 1]);
        }
        state[kMtN - 1] = state[kMtM - 1] ^ twist(state[kMtN - 1], state[0]);
    }

    uint32_t random() {
        if (--left <= 0) {
            next_state();
        }
        uint32_t y = state[next++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9D2C5680U;
        y ^= (y << 15) & 0xEFC60000U;
        y ^= (y >> 18);
        return y;
    }

    /** at::uniform_real_distribution<float>: (raw & (2^24 - 1)) * 2^-24. */
    float uniform() {
        return static_cast<float>(random() & 0xFFFFFFU) * (1.0F / 16777216.0F);
    }
};

/** ATen normal_fill_16, mean 0 std 1, float32 throughout. In place. */
void normal_fill_16(float * d) {
    for (int j = 0; j < 8; ++j) {
        const float u1 = 1.0F - d[j];
        const float u2 = d[j + 8];
        const float radius = std::sqrt(-2.0F * std::log(u1));
        const float theta = static_cast<float>(2.0 * kPi) * u2;
        d[j] = radius * std::cos(theta);
        d[j + 8] = radius * std::sin(theta);
    }
}

std::vector<float> seeded_noise(uint64_t seed, int64_t channels, int64_t frames) {
    const int64_t size = channels * frames;
    if (size < 16) {
        // torch dispatches sizes under 16 to a scalar path this does not
        // implement; the decoder always asks for channels*frames above that.
        throw std::runtime_error("sanoTTS seeded noise needs at least 16 values");
    }
    AtenMt19937 gen(seed);
    std::vector<float> out(static_cast<size_t>(size));
    for (auto & value : out) {
        value = gen.uniform();
    }
    int64_t i = 0;
    for (; i + 16 <= size; i += 16) {
        normal_fill_16(out.data() + i);
    }
    if (size % 16 != 0) {
        // Torch draws a FRESH block of 16 (continuing the same stream) and
        // overwrites the last 16 values with it; the loop's remainder is
        // discarded, so the tail is not simply its leftover.
        float tail[16];
        for (float & value : tail) {
            value = gen.uniform();
        }
        normal_fill_16(tail);
        std::memcpy(out.data() + size - 16, tail, sizeof(tail));
    }
    return out;
}

// ---- decoder graph -------------------------------------------------------

struct DecoderGraph : GraphResources {
    int64_t frames = 0;
    ggml_tensor * context = nullptr;
    ggml_tensor * feats = nullptr;
    ggml_tensor * noise = nullptr;
    ggml_tensor * spectrum = nullptr;
};

/** Frame-stage acoustic blocks -> mel-100 -> ConvNeXt decoder -> the
 *  [log-magnitude | phase] spectrum rows the host iSTFT consumes. */
std::unique_ptr<DecoderGraph> build_decoder_graph(
    const SanoTtsBackendWeights & weights,
    const SanoTtsConfig & config,
    ggml_backend_t backend,
    core::BackendType backend_type,
    int64_t frames) {
    auto out = std::make_unique<DecoderGraph>();
    out->backend = backend;
    out->frames = frames;
    out->io_context.reset(ggml_init({kIoArenaBytes, nullptr, true}));
    out->graph_context.reset(ggml_init({kGraphArenaBytes, nullptr, true}));
    if (out->io_context == nullptr || out->graph_context == nullptr) {
        throw std::runtime_error("sanoTTS failed to create decoder graph contexts");
    }
    core::ModuleBuildContext io_ctx{
        out->io_context.get(),
        "sanotts.decoder.io",
        backend_type,
    };
    core::ModuleBuildContext ctx{
        out->graph_context.get(),
        "sanotts.decoder",
        backend_type,
    };
    auto context = core::make_tensor(
        io_ctx,
        GGML_TYPE_F32,
        core::TensorShape::from_dims({1, config.acoustic_hidden, frames}));
    auto feats = core::make_tensor(
        io_ctx,
        GGML_TYPE_F32,
        core::TensorShape::from_dims({1, 3, frames}));
    auto noise = core::make_tensor(
        io_ctx,
        GGML_TYPE_F32,
        core::TensorShape::from_dims({1, config.noise_channels, frames}));
    ggml_set_input(context.tensor);
    ggml_set_input(feats.tensor);
    ggml_set_input(noise.tensor);

    auto mel = acoustic_frame_stage(
        ctx,
        weights,
        context,
        feats,
        config.acoustic_hidden,
        config.acoustic_depth,
        config.acoustic_kernel,
        config.mels);

    // ConvNeXt decoder. Noise-fed: the noise adapter's output is added to the
    // mel embedding before the first norm.
    const int embed_padding = static_cast<int>(config.embed_kernel / 2);
    auto value = conv1d(
        ctx,
        weights,
        mel,
        "decoder.embed",
        config.dim,
        config.embed_kernel,
        embed_padding);
    value = add(
        ctx,
        value,
        conv1d(
            ctx,
            weights,
            noise,
            "decoder.noise_adapter",
            config.dim,
            config.embed_kernel,
            embed_padding));

    // Channel-last from here: LayerNorm and the pointwise projections act on
    // the channel axis, exactly as the PyTorch modules do.
    auto value_cl = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, value);
    value_cl = channel_last_layer_norm(ctx, weights, value_cl, "decoder.norm", config.dim);
    for (int64_t block = 0; block < config.blocks; ++block) {
        const std::string prefix = "decoder.blocks." + std::to_string(block);
        const auto residual = value_cl;
        auto branch_cm = contiguous(
            ctx,
            modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, value_cl));
        auto branch = core::wrap_tensor(
            ggml_conv_1d_dw(
                ctx.ggml,
                contiguous(ctx, weight(weights, prefix + ".dwconv.weight")).tensor,
                branch_cm.tensor,
                1,
                static_cast<int>(config.dw_kernel / 2),
                1),
            core::TensorShape::from_dims({1, config.dim, frames}),
            GGML_TYPE_F32);
        auto * dw_bias = ggml_reshape_3d(
            ctx.ggml,
            weight(weights, prefix + ".dwconv.bias").tensor,
            1,
            config.dim,
            1);
        branch = core::wrap_tensor(
            ggml_add(ctx.ggml, branch.tensor, dw_bias),
            branch.shape,
            GGML_TYPE_F32);
        auto branch_cl = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, branch);
        branch_cl = channel_last_layer_norm(ctx, weights, branch_cl, prefix + ".norm", config.dim);
        branch_cl = modules::LinearModule({config.dim, config.pw_hidden, true}).build(
            ctx,
            branch_cl,
            {
                weight(weights, prefix + ".pwconv1.weight"),
                weight(weights, prefix + ".pwconv1.bias"),
            });
        // Exact erf GELU -- what nn.GELU() computes by default, NOT the tanh
        // approximation.
        branch_cl = modules::GeluModule({modules::GeluApproximation::ExactErf})
                        .build(ctx, branch_cl);
        branch_cl = modules::LinearModule({config.pw_hidden, config.dim, true}).build(
            ctx,
            branch_cl,
            {
                weight(weights, prefix + ".pwconv2.weight"),
                weight(weights, prefix + ".pwconv2.bias"),
            });
        const auto branch_source = contiguous(ctx, branch_cl);
        branch_cl = core::wrap_tensor(
            ggml_mul(
                ctx.ggml,
                branch_source.tensor,
                weight(weights, prefix + ".gamma").tensor),
            branch_cl.shape,
            GGML_TYPE_F32);
        value_cl = add(ctx, residual, branch_cl);
    }
    value_cl = channel_last_layer_norm(ctx, weights, value_cl, "decoder.final_norm", config.dim);
    auto spectrum = modules::LinearModule({config.dim, config.n_fft + 2, true}).build(
        ctx,
        value_cl,
        {
            weight(weights, "decoder.head.weight"),
            weight(weights, "decoder.head.bias"),
        });
    spectrum = contiguous(ctx, spectrum);
    out->context = context.tensor;
    out->feats = feats.tensor;
    out->noise = noise.tensor;
    out->spectrum = spectrum.tensor;
    ggml_set_output(out->spectrum);
    out->graph = ggml_new_graph_custom(ctx.ggml, 16384, false);
    ggml_build_forward_expand(out->graph, out->spectrum);
    allocate_graph(*out);
    return out;
}

// ---- SHA-256 for the derive-from-text seed -------------------------------

constexpr uint32_t kShaK[64] = {
    0x428A2F98U, 0x71374491U, 0xB5C0FBCFU, 0xE9B5DBA5U, 0x3956C25BU, 0x59F111F1U,
    0x923F82A4U, 0xAB1C5ED5U, 0xD807AA98U, 0x12835B01U, 0x243185BEU, 0x550C7DC3U,
    0x72BE5D74U, 0x80DEB1FEU, 0x9BDC06A7U, 0xC19BF174U, 0xE49B69C1U, 0xEFBE4786U,
    0x0FC19DC6U, 0x240CA1CCU, 0x2DE92C6FU, 0x4A7484AAU, 0x5CB0A9DCU, 0x76F988DAU,
    0x983E5152U, 0xA831C66DU, 0xB00327C8U, 0xBF597FC7U, 0xC6E00BF3U, 0xD5A79147U,
    0x06CA6351U, 0x14292967U, 0x27B70A85U, 0x2E1B2138U, 0x4D2C6DFCU, 0x53380D13U,
    0x650A7354U, 0x766A0ABBU, 0x81C2C92EU, 0x92722C85U, 0xA2BFE8A1U, 0xA81A664BU,
    0xC24B8B70U, 0xC76C51A3U, 0xD192E819U, 0xD6990624U, 0xF40E3585U, 0x106AA070U,
    0x19A4C116U, 0x1E376C08U, 0x2748774CU, 0x34B0BCB5U, 0x391C0CB3U, 0x4ED8AA4AU,
    0x5B9CCA4FU, 0x682E6FF3U, 0x748F82EEU, 0x78A5636FU, 0x84C87814U, 0x8CC70208U,
    0x90BEFFFAU, 0xA4506CEBU, 0xBEF9A3F7U, 0xC67178F2U,
};

uint32_t rotr32(uint32_t v, int n) {
    return (v >> n) | (v << (32 - n));
}

void sha256_block(uint32_t * h, const unsigned char * block) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (static_cast<uint32_t>(block[4 * i]) << 24) |
               (static_cast<uint32_t>(block[4 * i + 1]) << 16) |
               (static_cast<uint32_t>(block[4 * i + 2]) << 8) |
               static_cast<uint32_t>(block[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0];
    uint32_t b = h[1];
    uint32_t c = h[2];
    uint32_t d = h[3];
    uint32_t e = h[4];
    uint32_t f = h[5];
    uint32_t g = h[6];
    uint32_t hh = h[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t s1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        const uint32_t ch = (e & f) ^ ((~e) & g);
        const uint32_t s0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t1 = hh + s1 + ch + kShaK[i] + w[i];
        const uint32_t t2 = s0 + maj;
        hh = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
}

// ---- host post-processing ------------------------------------------------

std::vector<float> periodic_hann_window(int64_t n_fft) {
    std::vector<float> window(static_cast<size_t>(n_fft));
    for (int64_t i = 0; i < n_fft; ++i) {
        window[static_cast<size_t>(i)] = static_cast<float>(
            0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(i) /
                                 static_cast<double>(n_fft)));
    }
    return window;
}

/** H(z) = (1 - z^-1) / (1 - R z^-1), 2 MAC/sample, zero initial state --
 *  the same DC blocker the reference runtimes apply after the iSTFT. */
void dc_block_in_place(std::vector<float> & samples) {
    float x1 = 0.0F;
    float y1 = 0.0F;
    for (float & sample : samples) {
        const float x = sample;
        const float y = x - x1 + kDcBlockPole * y1;
        x1 = x;
        y1 = y;
        sample = y;
    }
}

}  // namespace

uint64_t sanotts_text_seed(const std::string & text) {
    uint32_t h[8] = {0x6A09E667U, 0xBB67AE85U, 0x3C6EF372U, 0xA54FF53AU,
                     0x510E527FU, 0x9B05688CU, 0x1F83D9ABU, 0x5BE0CD19U};
    const auto * data = reinterpret_cast<const unsigned char *>(text.data());
    const size_t len = text.size();
    const size_t full = len / 64;
    const size_t rem = len % 64;
    for (size_t i = 0; i < full; ++i) {
        sha256_block(h, data + i * 64);
    }
    unsigned char tail[128] = {0};
    std::memcpy(tail, data + full * 64, rem);
    tail[rem] = 0x80;
    const size_t tail_len = rem < 56 ? 64 : 128;
    const uint64_t bits = static_cast<uint64_t>(len) * 8U;
    for (int i = 0; i < 8; ++i) {
        tail[tail_len - 1 - static_cast<size_t>(i)] =
            static_cast<unsigned char>((bits >> (8 * i)) & 0xFFU);
    }
    sha256_block(h, tail);
    if (tail_len == 128) {
        sha256_block(h, tail + 64);
    }
    // == int.from_bytes(sha256(text).digest()[:8], "big"); ATen seeding then
    // keeps only the low 32 bits, exactly as the reference runtimes do.
    return (static_cast<uint64_t>(h[0]) << 32) | static_cast<uint64_t>(h[1]);
}

struct SanoTtsNativeRuntime::State : BackendState {
    State(const std::shared_ptr<const SanoTtsAssets> & assets_in,
          core::BackendConfig backend_config)
        : BackendState(
              assets_in,
              backend_config,
              assets_in == nullptr ? 0 : expected_tensor_count(assets_in->config),
              kWeightArenaBytes) {}

    runtime::CacheSlots<int64_t, std::unique_ptr<FrontGraph>> duration_graphs{4};
    runtime::CacheSlots<int64_t, std::unique_ptr<FrontGraph>> token_graphs{4};
    runtime::CacheSlots<int64_t, std::unique_ptr<DecoderGraph>> decoder_graphs{2};
};

SanoTtsNativeRuntime::SanoTtsNativeRuntime(
    std::shared_ptr<const SanoTtsAssets> assets,
    core::BackendConfig backend_config)
    : state_(std::make_unique<State>(std::move(assets), backend_config)) {}

SanoTtsNativeRuntime::~SanoTtsNativeRuntime() = default;

runtime::AudioBuffer SanoTtsNativeRuntime::synthesize(
    const std::vector<int32_t> & token_ids,
    const SanoTtsGenerationOptions & options) {
    const auto & config = state_->assets->config;
    const auto token_count = static_cast<int64_t>(token_ids.size());
    if (token_count <= 0) {
        throw std::runtime_error("sanoTTS requires at least one phoneme token");
    }
    const auto total_start = std::chrono::steady_clock::now();

    // -- durations ---------------------------------------------------------
    const auto duration_start = std::chrono::steady_clock::now();
    auto & duration = cached_graph(
        state_->duration_graphs,
        token_count,
        "sanotts.duration_graph.cache_hit",
        [&] {
            return build_front_graph(
                state_->duration_weights_ref(),
                duration_stage_spec(
                    config.vocab_size,
                    config.duration_hidden,
                    config.duration_depth,
                    config.duration_kernel),
                state_->duration_backend_value(),
                state_->duration_backend_type(),
                token_count,
                kIoArenaBytes,
                kGraphArenaBytes);
        });
    write_i32_input(
        duration.tokens, core::TensorShape::from_dims({1, token_count}), token_ids);
    write_f32_input(
        duration.feats,
        core::TensorShape::from_dims({1, 3, token_count}),
        duration_features(token_count, config.duration_max_tokens));
    compute_graph(duration, "sanoTTS duration");
    const auto log_duration = core::read_tensor_f32(duration.output);
    if (static_cast<int64_t>(log_duration.size()) != token_count) {
        throw std::runtime_error("sanoTTS duration graph returned invalid output");
    }
    int64_t frames = 0;
    const auto durations = round_durations(
        log_duration,
        static_cast<double>(options.speaking_rate),
        config.duration_max_frames,
        config.duration_max_tokens * config.duration_max_frames,
        frames);
    if (frames < 2) {
        throw std::runtime_error("sanoTTS duration predictor produced too few frames");
    }
    engine::debug::timing_log_scalar(
        "sanotts.duration_ms",
        engine::debug::elapsed_ms(duration_start));

    // -- token-stage acoustic context --------------------------------------
    const auto acoustic_start = std::chrono::steady_clock::now();
    auto & token_graph = cached_graph(
        state_->token_graphs,
        token_count,
        "sanotts.token_graph.cache_hit",
        [&] {
            return build_front_graph(
                *state_->weights,
                token_stage_spec(
                    config.vocab_size,
                    config.acoustic_hidden,
                    config.acoustic_token_depth,
                    config.acoustic_kernel),
                state_->backend.value,
                state_->backend_type,
                token_count,
                kIoArenaBytes,
                kGraphArenaBytes);
        });
    write_i32_input(
        token_graph.tokens, core::TensorShape::from_dims({1, token_count}), token_ids);
    write_f32_input(
        token_graph.feats,
        core::TensorShape::from_dims({1, 2, token_count}),
        token_features(token_count, durations));
    compute_graph(token_graph, "sanoTTS token context");
    const auto token_context = core::read_tensor_f32(token_graph.output);
    if (static_cast<int64_t>(token_context.size()) !=
        config.acoustic_hidden * token_count) {
        throw std::runtime_error("sanoTTS token graph returned invalid output");
    }
    const auto noise = seeded_noise(options.seed, config.noise_channels, frames);
    engine::debug::timing_log_scalar(
        "sanotts.acoustic_ms",
        engine::debug::elapsed_ms(acoustic_start));

    // -- frame stage + decoder ---------------------------------------------
    const auto decoder_start = std::chrono::steady_clock::now();
    auto & decoder = cached_graph(
        state_->decoder_graphs,
        frames,
        "sanotts.decoder_graph.cache_hit",
        [&] {
            return build_decoder_graph(
                *state_->weights,
                config,
                state_->backend.value,
                state_->backend_type,
                frames);
        });
    write_f32_input(
        decoder.context,
        core::TensorShape::from_dims({1, config.acoustic_hidden, frames}),
        expand_context(token_context, durations, config.acoustic_hidden, token_count, frames));
    write_f32_input(
        decoder.feats,
        core::TensorShape::from_dims({1, 3, frames}),
        frame_features(token_count, durations, frames));
    write_f32_input(
        decoder.noise,
        core::TensorShape::from_dims({1, config.noise_channels, frames}),
        noise);
    compute_graph(decoder, "sanoTTS decoder");
    auto spectrum = core::read_tensor_f32(decoder.spectrum);
    const int64_t out_dim = config.n_fft + 2;
    if (static_cast<int64_t>(spectrum.size()) != frames * out_dim) {
        throw std::runtime_error("sanoTTS decoder graph returned invalid output");
    }
    engine::debug::timing_log_scalar(
        "sanotts.decoder_ms",
        engine::debug::elapsed_ms(decoder_start));

    // -- iSTFT + DC block ---------------------------------------------------
    const auto istft_start = std::chrono::steady_clock::now();
    // Bin 0 and Nyquist stay zeroed: the mag*(cos,sin) parametrisation
    // phase-collapses at bin 0, which is where the frame-DC artefact came
    // from. -inf log-magnitude exps to exactly 0.
    const int64_t bins = config.n_fft / 2 + 1;
    for (int64_t frame = 0; frame < frames; ++frame) {
        float * row = spectrum.data() + frame * out_dim;
        row[0] = -std::numeric_limits<float>::infinity();
        row[bins - 1] = -std::numeric_limits<float>::infinity();
    }
    engine::audio::HostLogMagnitudePhaseISTFTConfig istft_config;
    istft_config.frames = frames;
    istft_config.n_fft = config.n_fft;
    istft_config.hop_length = config.hop_length;
    istft_config.out_dim = out_dim;
    istft_config.threads = static_cast<size_t>(state_->threads);
    engine::audio::HostLogMagnitudePhaseISTFT istft(istft_config);
    auto istft_result = istft.compute(spectrum, periodic_hann_window(config.n_fft));
    // The framework trims (n_fft - hop)/2 per side; torch.istft(center=True)
    // trims n_fft/2. Drop the extra hop/2 per side so lengths and content
    // match the reference exactly: (frames - 1) * hop samples.
    const auto edge = static_cast<size_t>(config.hop_length / 2);
    const size_t expected = static_cast<size_t>(frames) * static_cast<size_t>(config.hop_length);
    if (istft_result.audio.size() != expected || istft_result.audio.size() < 2 * edge) {
        throw std::runtime_error("sanoTTS iSTFT returned an unexpected sample count");
    }
    std::vector<float> samples(
        istft_result.audio.begin() + static_cast<std::ptrdiff_t>(edge),
        istft_result.audio.end() - static_cast<std::ptrdiff_t>(edge));
    dc_block_in_place(samples);
    engine::debug::timing_log_scalar(
        "sanotts.istft_ms",
        engine::debug::elapsed_ms(istft_start));

    runtime::AudioBuffer out;
    out.sample_rate = static_cast<int>(config.sample_rate);
    out.channels = 1;
    out.samples = std::move(samples);
    engine::debug::trace_log_scalar("sanotts.token_count", token_count);
    engine::debug::trace_log_scalar("sanotts.frames", frames);
    engine::debug::trace_log_scalar(
        "sanotts.output_samples",
        static_cast<int64_t>(out.samples.size()));
    engine::debug::timing_log_scalar(
        "session.wall_ms",
        engine::debug::elapsed_ms(total_start));
    return out;
}

}  // namespace engine::models::sanotts
