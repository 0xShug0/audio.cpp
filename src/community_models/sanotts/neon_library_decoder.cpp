#include "engine/community_models/sanotts/neon_decoder.h"
#include "engine/framework/neon/blocks.h"

#include <stdexcept>

#if defined(__aarch64__) && defined(__ARM_NEON)

#include <algorithm>
#include <optional>

namespace engine::models::sanotts {
namespace cpu = engine::neon;

struct SanoTtsNeonDecoder::Impl {
    int hidden = 0;
    int noise = 0;
    cpu::ExecutionPlan plan;
};

bool SanoTtsNeonDecoder::available() {
    return true;
}

SanoTtsNeonDecoder::SanoTtsNeonDecoder(const SanoTtsConfig &config, const SanoTtsNeonWeights &weights)
    : impl_(std::make_unique<Impl>()) {
    auto &p = *impl_;
    p.hidden = static_cast<int>(config.acoustic_hidden);
    p.noise = static_cast<int>(config.noise_channels);
    const int channels = static_cast<int>(config.dim);
    const int projection_channels = p.hidden + 3;
    const int embedding_channels = p.hidden + 1 + p.noise;
    const int acoustic_kernel = static_cast<int>(config.acoustic_kernel);
    const int embedding_kernel = static_cast<int>(config.embed_kernel);
    if (weights.acoustic_blocks.size() != static_cast<size_t>(config.acoustic_depth) ||
        weights.decoder_blocks.size() != static_cast<size_t>(config.blocks)) {
        throw std::invalid_argument("sanoTTS NEON decoder block inventory mismatch");
    }

    const auto context = p.plan.input();
    const auto features = p.plan.input();
    std::optional<cpu::ExecutionPlan::Value> noise;
    if (p.noise > 0) {
        noise = p.plan.input();
    }
    auto value = p.plan.add(cpu::ConcatChannels(), {context, features});
    value = p.plan.add(
        cpu::Conv1d({projection_channels, p.hidden, 1}, weights.projection.weight, weights.projection.bias), {value});
    for (const auto &block : weights.acoustic_blocks) {
        value = cpu::build_residual_conv(p.plan, value, p.hidden, p.hidden, acoustic_kernel, cpu::ActivationKind::SiLU,
                                         block);
    }

    const auto inside = p.plan.add(cpu::ConstantChannels(1, 1.0F), {value});
    value = noise ? p.plan.add(cpu::ConcatChannels(), {value, inside, *noise})
                  : p.plan.add(cpu::ConcatChannels(), {value, inside});
    value = p.plan.add(
        cpu::Conv1d({embedding_channels, channels, embedding_kernel, 1, 1, embedding_kernel / 2, embedding_kernel / 2},
                    weights.embedding.weight, weights.embedding.bias),
        {value});
    value = p.plan.add(cpu::LayerNorm(weights.input_norm.weight, weights.input_norm.bias, 1e-6F), {value});
    for (const auto &block : weights.decoder_blocks) {
        value = cpu::build_convnext(p.plan, value, channels, static_cast<int>(config.pw_hidden),
                                    static_cast<int>(config.dw_kernel), 1e-6F, block);
    }
    value = p.plan.add(cpu::LayerNorm(weights.final_norm.weight, weights.final_norm.bias, 1e-6F), {value});
    value = p.plan.add(
        cpu::Conv1d({channels, static_cast<int>(config.n_fft + 2), 1}, weights.head.weight, weights.head.bias),
        {value});
    p.plan.set_output(value, cpu::Layout::FrameMajor);
}

SanoTtsNeonDecoder::~SanoTtsNeonDecoder() = default;

std::vector<float> SanoTtsNeonDecoder::decode(const std::vector<float> &context, const std::vector<float> &feats,
                                              const std::vector<float> &noise, int64_t frames, int threads) {
    auto &p = *impl_;
    const auto T = static_cast<size_t>(frames);
    if (frames < 1 || frames > (1 << 24) || context.size() != p.hidden * T || feats.size() != 3 * T ||
        noise.size() != p.noise * T) {
        throw std::runtime_error("sanoTTS NEON decoder: invalid input sizes");
    }
    const int frame_count = static_cast<int>(frames);
    const cpu::TensorView context_view(context, {p.hidden, frame_count});
    const cpu::TensorView feature_view(feats, {3, frame_count});
    if (p.noise == 0) {
        return p.plan.run({context_view, feature_view}, std::max(threads, 1)).values;
    }
    return p.plan
        .run({context_view, feature_view, cpu::TensorView(noise, {p.noise, frame_count})}, std::max(threads, 1))
        .values;
}

} // namespace engine::models::sanotts

#else // no AArch64 NEON

namespace engine::models::sanotts {

struct SanoTtsNeonDecoder::Impl {};

bool SanoTtsNeonDecoder::available() {
    return false;
}

SanoTtsNeonDecoder::SanoTtsNeonDecoder(const SanoTtsConfig &, const SanoTtsNeonWeights &) {
    throw std::runtime_error(
        "sanotts.exec_plan.decoder: impl neon needs an AArch64 build with NEON; select ggml instead");
}

SanoTtsNeonDecoder::~SanoTtsNeonDecoder() = default;

std::vector<float> SanoTtsNeonDecoder::decode(const std::vector<float> &, const std::vector<float> &,
                                              const std::vector<float> &, int64_t, int) {
    throw std::runtime_error("sanoTTS NEON decoder is not available in this build");
}

} // namespace engine::models::sanotts

#endif
