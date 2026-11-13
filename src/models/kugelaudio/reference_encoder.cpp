#include "engine/models/kugelaudio/reference_encoder.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/runtime/errors.h"

#include <ggml-alloc.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace engine::models::kugelaudio {
namespace {
constexpr int kSampleRate = 24000;
constexpr int kMaxReferenceSeconds = 30;

struct BlockWeights {
    modules::NormWeights norm, ffn_norm;
    modules::DepthwiseConv1dWeights mixer;
    modules::LinearWeights expansion, projection;
    core::TensorValue gamma, ffn_gamma;
};

void validate_config(const ReferenceEncoderConfig & config) {
    if (config.depths.empty() || config.depths.size() != config.ratios.size() + 1 ||
        config.depths.size() > 10 || config.filters <= 0 || config.latent_size <= 0 ||
        !std::isfinite(config.norm_eps) || config.norm_eps <= 0 ||
        !std::isfinite(config.fixed_std) || config.fixed_std < 0)
        throw std::runtime_error("invalid KugelAudio reference encoder dimensions or normalization");
    // Bound graph metadata and prevent channel/kernel arithmetic overflow before
    // loading weights. The supported checkpoint has 26 blocks across 7 stages.
    int total_depth = 0;
    for (int depth : config.depths) {
        if (depth < 0 || depth > 64 || (total_depth += depth) > 64)
            throw std::runtime_error("unsupported KugelAudio reference encoder block count");
    }
    for (int ratio : config.ratios) {
        if (ratio <= 0 || ratio > std::numeric_limits<int>::max() / 2)
            throw std::runtime_error("invalid KugelAudio reference encoder stride");
    }
    const int64_t channel_multiplier = int64_t{4} << (config.depths.size() - 1);
    if (config.filters > std::numeric_limits<int64_t>::max() / channel_multiplier)
        throw std::runtime_error("KugelAudio reference encoder channel count overflows");
}

core::TensorValue causal_pad(core::ModuleBuildContext & ctx, const core::TensorValue & x,
                           int64_t kernel, int64_t stride) {
    const int64_t left = kernel - stride;
    const int64_t frames = x.shape.dims[2];
    const int64_t right = (stride - frames % stride) % stride;
    auto shape = x.shape;
    shape.dims[2] += left + right;
    return core::wrap_tensor(ggml_pad_ext(ctx.ggml, x.tensor, int(left), int(right), 0, 0, 0, 0, 0, 0),
                             shape, GGML_TYPE_F32);
}

core::TensorValue rms(core::ModuleBuildContext & ctx, const core::TensorValue & x,
                      const modules::NormWeights & weights, int64_t channels, float eps) {
    auto transposed = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    transposed = modules::RMSNormModule({channels, eps, true, false}).build(ctx, transposed, weights);
    return modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, transposed);
}
} // namespace

ReferenceEncoderConfig parse_reference_encoder_config(const io::json::Value & tokenizer) {
    namespace json = io::json;
    auto is_string = [&](const char * key, const char * expected) {
        const auto * value = tokenizer.find(key);
        return value && value->is_string() && value->as_string() == expected;
    };
    if (!json::optional_bool(tokenizer, "causal", false) || json::optional_i64(tokenizer, "channels", 1) != 1 ||
        !json::optional_bool(tokenizer, "conv_bias", true) || !json::optional_bool(tokenizer, "disable_last_norm", false) ||
        !json::optional_bool(tokenizer, "layernorm_elementwise_affine", true) ||
        !is_string("layernorm", "RMSNorm") || !is_string("mixer_layer", "depthwise_conv") ||
        !is_string("conv_norm", "none") || !is_string("pad_mode", "constant"))
        throw std::runtime_error("KugelAudio reference encoder configuration is not supported");
    ReferenceEncoderConfig config;
    config.latent_size = json::require_i64(tokenizer, "vae_dim");
    config.filters = json::optional_i64(tokenizer, "encoder_n_filters", 32);
    config.norm_eps = json::require_f32(tokenizer, "layernorm_eps");
    if (tokenizer.find("fix_std")) config.fixed_std = json::require_f32(tokenizer, "fix_std");
    if (tokenizer.find("encoder_ratios")) config.ratios = json::number_array_as<int>(tokenizer.require("encoder_ratios"));
    const auto & depths = tokenizer.require("encoder_depths");
    if (depths.is_string()) {
        config.depths.clear();
        std::istringstream stream(depths.as_string());
        std::string depth;
        while (std::getline(stream, depth, '-')) {
            size_t consumed = 0;
            const int value = std::stoi(depth, &consumed);
            if (consumed != depth.size()) throw std::runtime_error("invalid KugelAudio reference encoder depth");
            config.depths.push_back(value);
        }
    } else config.depths = json::number_array_as<int>(depths);
    validate_config(config);
    return config;
}

std::vector<float> prepare_reference_audio(const runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty() ||
        audio.samples.size() % static_cast<size_t>(audio.channels) != 0)
        throw runtime::InvalidRequestError("KugelAudio reference audio must contain complete, non-empty audio frames");
    const auto frames = audio.samples.size() / static_cast<size_t>(audio.channels);
    if (frames > uint64_t(audio.sample_rate) * kMaxReferenceSeconds)
        throw runtime::InvalidRequestError("KugelAudio reference audio must be at most 30 seconds; select a shorter recording");
    for (float value : audio.samples)
        if (!std::isfinite(value)) throw runtime::InvalidRequestError("KugelAudio reference audio contains non-finite samples");
    auto samples = audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio.samples, audio.sample_rate, audio.channels, kSampleRate);
    if (samples.empty() || samples.size() > size_t(kSampleRate) * kMaxReferenceSeconds)
        throw runtime::InvalidRequestError("KugelAudio reference audio has an invalid resampled length");
    double square_sum = 0;
    for (float value : samples) square_sum += double(value) * value;
    const double level = std::sqrt(square_sum / samples.size());
    if (!std::isfinite(level) || level <= 1e-8)
        throw runtime::InvalidRequestError("KugelAudio reference audio is silent; provide a speech recording");
    const double gain = std::pow(10.0, -25.0 / 20.0) / (level + 1e-6);
    float peak = 0;
    for (float & value : samples) {
        value = static_cast<float>(value * gain);
        peak = std::max(peak, std::abs(value));
    }
    if (peak > 1) for (float & value : samples) value /= peak + 1e-6F;
    return samples;
}

struct ReferenceEncoder::Impl {
    core::ExecutionContext & execution;
    ReferenceEncoderConfig config;
    core::BackendWeightStore weights;
    std::vector<modules::Conv1dWeights> downsample;
    std::vector<std::vector<BlockWeights>> blocks;
    modules::Conv1dWeights head;

    Impl(const assets::TensorSource & source, core::ExecutionContext & execution_, ReferenceEncoderConfig config_)
        : execution(execution_), config(std::move(config_)),
          weights(execution.backend(), execution.backend_type(), "kugelaudio.reference_encoder.weights", 4 * 1024 * 1024) {
        validate_config(config);
        namespace binding = modules::binding;
        const std::string prefix = "model.acoustic_tokenizer.encoder.";
        for (size_t stage = 0; stage < config.depths.size(); ++stage) {
            const int64_t channels = config.filters * (int64_t{1} << stage);
            const int stride = stage ? config.ratios[config.ratios.size() - stage] : 1;
            // Preserve source BF16 values exactly in F32. Encoder preparation
            // is brief; the session releases this store before speech generation.
            constexpr auto storage = assets::TensorStorageType::F32;
            downsample.push_back(binding::conv1d_from_source(weights, source,
                prefix + "downsample_layers." + std::to_string(stage) + ".0.conv.conv", storage,
                channels, stage ? channels / 2 : 1, stage ? 2 * stride : 7, true));
            blocks.emplace_back();
            for (int layer = 0; layer < config.depths[stage]; ++layer) {
                const auto name = prefix + "stages." + std::to_string(stage) + "." + std::to_string(layer) + ".";
                BlockWeights block;
                block.norm = binding::norm_weight_from_source(weights, source, name + "norm", channels);
                block.mixer = binding::depthwise_conv1d_from_source(weights, source, name + "mixer.conv.conv.conv",
                    storage, channels, 7, true);
                block.gamma = weights.load_f32_tensor(source, name + "gamma", {channels});
                block.ffn_norm = binding::norm_weight_from_source(weights, source, name + "ffn_norm", channels);
                block.expansion = binding::linear_from_source(weights, source, name + "ffn.linear1", storage,
                    4 * channels, channels, true);
                block.projection = binding::linear_from_source(weights, source, name + "ffn.linear2", storage,
                    channels, 4 * channels, true);
                block.ffn_gamma = weights.load_f32_tensor(source, name + "ffn_gamma", {channels});
                blocks.back().push_back(std::move(block));
            }
        }
        const int64_t last_channels = config.filters * (int64_t{1} << (config.depths.size() - 1));
        head = binding::conv1d_from_source(weights, source, prefix + "head.conv.conv", assets::TensorStorageType::F32,
            config.latent_size, last_channels, 7, true);
        weights.upload();
    }

    std::vector<float> encode(const std::vector<float> & waveform) {
        if (waveform.empty() || waveform.size() > size_t(kSampleRate) * kMaxReferenceSeconds)
            throw std::runtime_error("invalid KugelAudio reference encoder sample count");
        for (float value : waveform) if (!std::isfinite(value))
            throw std::runtime_error("non-finite KugelAudio reference encoder input");
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({16 * 1024 * 1024, nullptr, true}), ggml_free);
        if (!context) throw std::bad_alloc();
        core::ModuleBuildContext ctx{context.get(), "kugelaudio.reference_encoder", execution.backend_type()};
        auto input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, int64_t(waveform.size())}));
        ggml_set_input(input.tensor);
        auto x = input;
        for (size_t stage = 0; stage < blocks.size(); ++stage) {
            const int64_t channels = config.filters * (int64_t{1} << stage);
            const int stride = stage ? config.ratios[config.ratios.size() - stage] : 1;
            const int kernel = stage ? 2 * stride : 7;
            x = causal_pad(ctx, x, kernel, stride);
            x = modules::Conv1dModule({stage ? channels / 2 : 1, channels, kernel, stride, 0, 1, true})
                .build(ctx, x, downsample[stage]);
            for (const auto & block : blocks[stage]) {
                auto residual = x;
                x = rms(ctx, x, block.norm, channels, config.norm_eps);
                x = causal_pad(ctx, x, 7, 1);
                x = modules::DepthwiseConv1dModule({channels, 7, 1, 0, 1, true}).build(ctx, x, block.mixer);
                auto gamma = core::reshape_tensor(ctx, block.gamma, core::TensorShape::from_dims({1, channels, 1}));
                x = modules::MulModule{}.build(ctx, x, modules::RepeatModule({x.shape}).build(ctx, gamma));
                x = modules::AddModule{}.build(ctx, residual, x);
                residual = x;
                x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
                x = modules::RMSNormModule({channels, config.norm_eps, true, false}).build(ctx, x, block.ffn_norm);
                x = modules::LinearModule({channels, 4 * channels, true}).build(ctx, x, block.expansion);
                x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
                x = modules::LinearModule({4 * channels, channels, true}).build(ctx, x, block.projection);
                auto ffn_gamma = core::reshape_tensor(ctx, block.ffn_gamma, core::TensorShape::from_dims({1, 1, channels}));
                x = modules::MulModule{}.build(ctx, x, modules::RepeatModule({x.shape}).build(ctx, ffn_gamma));
                x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
                x = modules::AddModule{}.build(ctx, residual, x);
            }
        }
        const int64_t last_channels = config.filters * (int64_t{1} << (config.depths.size() - 1));
        x = causal_pad(ctx, x, 7, 1);
        x = modules::Conv1dModule({last_channels, config.latent_size, 7, 1, 0, 1, true}).build(ctx, x, head);
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        auto output = core::ensure_backend_addressable_layout(ctx, x);
        ggml_set_output(output.tensor);
        auto * graph = ggml_new_graph_custom(context.get(), 8192, false);
        ggml_build_forward_expand(graph, output.tensor);
        // The trained voice features are sensitive to accumulated rounding
        // through 26 residual blocks. Request F32 accumulation and explicit F32
        // inputs where supported. CUDA retains its backend TF32 operand policy;
        // global defaults and the existing speech generator are unaffected.
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            if (node->op != GGML_OP_MUL_MAT) continue;
            ggml_mul_mat_set_prec(node, GGML_PREC_F32);
            if (execution.backend_type() == core::BackendType::Vulkan)
                ggml_mul_mat_set_lowering(node, GGML_MUL_MAT_LOWERING_VULKAN_F32_INPUTS);
            if (execution.backend_type() == core::BackendType::Metal)
                ggml_mul_mat_set_lowering(node, GGML_MUL_MAT_LOWERING_METAL_F32_INPUTS);
        }
        core::validate_backend_graph_supported(execution.backend(), graph, "KugelAudio reference encoder");
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())), ggml_gallocr_free);
        core::HostGraphPlan plan;
        // Release any backend graph captures while their tensors/buffer live,
        // including allocation/compute/readback exceptions.
        struct GraphRelease {
            core::ExecutionContext & execution;
            ggml_cgraph * graph;
            ~GraphRelease() { core::release_backend_graph_resources(execution.backend(), graph, true); }
        } release{execution, graph};
        if (!ggml_gallocr_alloc_graph(allocator.get(), graph))
            throw std::runtime_error("KugelAudio reference encoder allocation failed");
        core::prepare_host_graph_plan(execution, graph, plan);
        core::write_tensor_f32(input, waveform);
        if (core::compute_graph(execution, graph, plan, "KugelAudio reference encoder") != GGML_STATUS_SUCCESS)
            throw std::runtime_error("KugelAudio reference encoder compute failed");
        auto values = core::read_tensor_f32(output.tensor);
        for (float value : values) if (!std::isfinite(value))
            throw std::runtime_error("KugelAudio reference encoder produced non-finite features");
        return values;
    }
};

ReferenceEncoder::ReferenceEncoder(const assets::TensorSource & source, core::ExecutionContext & execution,
                                 ReferenceEncoderConfig config)
    : impl_(std::make_unique<Impl>(source, execution, std::move(config))) {}
ReferenceEncoder::~ReferenceEncoder() = default;
std::vector<float> ReferenceEncoder::encode(const std::vector<float> & waveform) { return impl_->encode(waveform); }

} // namespace engine::models::kugelaudio
