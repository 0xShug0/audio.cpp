#include "engine/community_models/latentsync/whisper.h"

#include "engine/framework/audio/mel_spectrogram_frontend.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ggml-alloc.h>
#include <stdexcept>

namespace engine::models::latentsync {

modules::WhisperEmbeddingWeights load_whisper_encoder_weights(
    core::BackendWeightStore & store, const assets::TensorSource & source,
    const modules::WhisperEmbeddingConfig & config, assets::TensorStorageType type) {
    namespace binding = modules::binding;
    modules::WhisperEmbeddingWeights weights;
    const int64_t channels = config.n_audio_state;
    weights.conv1 = binding::conv1d_from_source(
        store, source, "whisper.encoder.conv1", type, channels, config.n_mels, 3, true);
    weights.conv2 = binding::conv1d_from_source(
        store, source, "whisper.encoder.conv2", type, channels, channels, 3, true);
    weights.positional_embedding = store.load_f32_tensor(source,
        "whisper.encoder.positional_embedding", {config.n_audio_ctx, channels});
    for (int64_t index = 0; index < config.n_audio_layer; ++index) {
        const auto prefix = "whisper.encoder.blocks." + std::to_string(index);
        modules::WhisperEncoderLayerWeights layer;
        layer.attention_norm = binding::norm_from_source(store, source, prefix + ".attn_ln", channels);
        layer.mlp_norm = binding::norm_from_source(store, source, prefix + ".mlp_ln", channels);
        layer.attention.query = binding::linear_from_source(store, source, prefix + ".attn.query", type, channels, channels, true);
        layer.attention.key = binding::linear_from_source(store, source, prefix + ".attn.key", type, channels, channels, false);
        layer.attention.value = binding::linear_from_source(store, source, prefix + ".attn.value", type, channels, channels, true);
        layer.attention.out = binding::linear_from_source(store, source, prefix + ".attn.out", type, channels, channels, true);
        const auto fc1 = binding::linear_from_source(store, source, prefix + ".mlp.0", type, channels * 4, channels, true);
        const auto fc2 = binding::linear_from_source(store, source, prefix + ".mlp.2", type, channels, channels * 4, true);
        layer.mlp = {fc1.weight, fc1.bias, fc2.weight, fc2.bias};
        weights.layers.push_back(std::move(layer));
    }
    return weights;
}

LatentSyncWhisperEncoder::LatentSyncWhisperEncoder(modules::WhisperEmbeddingConfig config)
    : config_(config) {
    if (config_.n_audio_state <= 0 || config_.n_audio_head <= 0 ||
        config_.n_audio_state % config_.n_audio_head != 0 || config_.n_audio_ctx <= 0 ||
        config_.n_mels <= 0 || config_.n_audio_layer <= 0) {
        throw std::runtime_error("Invalid LatentSync Whisper encoder dimensions");
    }
}

core::TensorValue LatentSyncWhisperEncoder::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & mel,
    const modules::WhisperEmbeddingWeights & weights) const {
    using namespace modules;
    core::validate_shape(mel,
        core::TensorShape::from_dims({1, config_.n_mels, 2 * config_.n_audio_ctx}), "mel");
    if (weights.layers.size() != static_cast<size_t>(config_.n_audio_layer)) {
        throw std::runtime_error("LatentSync Whisper layer count does not match weights");
    }
    const int64_t channels = config_.n_audio_state;
    const int64_t frames = config_.n_audio_ctx;
    const int64_t heads = config_.n_audio_head;
    const int64_t head_dim = channels / heads;
    const GeluModule gelu({GeluApproximation::ExactErf});
    auto x = Conv1dModule({config_.n_mels, channels, 3, 1, 1, 1, true})
                 .build(ctx, mel, weights.conv1);
    x = gelu.build(ctx, x);
    x = Conv1dModule({channels, channels, 3, 2, 1, 1, true}).build(ctx, x, weights.conv2);
    x = gelu.build(ctx, x);
    x = TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
    x = core::ensure_backend_addressable_layout(ctx, x);
    const auto positions = core::reshape_tensor(ctx, weights.positional_embedding,
        core::TensorShape::from_dims({1, frames, channels}));
    x = AddModule().build(ctx, x, positions);

    auto embeddings = core::reshape_tensor(ctx, x,
        core::TensorShape::from_dims({1, frames, 1, channels}));
    const LayerNormModule norm({channels, config_.layer_norm_eps, true, true});
    const FeedForwardModule mlp({channels, channels * 4, true, GeluApproximation::ExactErf});
    const ScaledDotProductAttentionModule attention({
        head_dim, ScaledDotProductAttentionLowering::Flash, GGML_PREC_F32});
    for (const auto & layer : weights.layers) {
        const auto normalized = norm.build(ctx, x, layer.attention_norm);
        const auto project_heads = [&](const LinearWeights & projection, bool bias) {
            auto projected = LinearModule({channels, channels, bias}).build(ctx, normalized, projection);
            projected = core::reshape_tensor(ctx, projected,
                core::TensorShape::from_dims({1, frames, heads, head_dim}));
            return TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, projected);
        };
        const auto q = project_heads(layer.attention.query, true);
        const auto k = project_heads(layer.attention.key, false);
        const auto v = project_heads(layer.attention.value, true);
        auto attended = attention.build(ctx, q, k, v);
        attended = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, attended),
            core::TensorShape::from_dims({1, frames, channels}));
        attended = LinearModule({channels, channels, true}).build(ctx, attended, layer.attention.out);
        x = AddModule().build(ctx, x, attended);
        x = AddModule().build(ctx, x, mlp.build(ctx, norm.build(ctx, x, layer.mlp_norm), layer.mlp));
        const auto layer_embedding = core::reshape_tensor(ctx, x,
            core::TensorShape::from_dims({1, frames, 1, channels}));
        embeddings = ConcatModule({2}).build(ctx, embeddings, layer_embedding);
    }
    return embeddings;
}

struct LatentSyncWhisperRuntime::Impl {
    core::ExecutionContext & execution;
    core::BackendWeightStore store;
    modules::WhisperEmbeddingWeights weights;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator{nullptr, ggml_gallocr_free};
    ggml_cgraph * graph = nullptr;
    core::TensorValue input, output;
    std::shared_ptr<const audio::MelSpectrogramFrontend> frontend;

    Impl(core::ExecutionContext & execution_in, const assets::TensorSource & source,
        modules::WhisperEmbeddingConfig config, assets::TensorStorageType type)
        : execution(execution_in),
          store(execution.backend(), execution.backend_type(), "latentsync.whisper.weights", 4 * 1024 * 1024),
          weights(load_whisper_encoder_weights(store, source, config, type)) {
        store.upload();
        constexpr size_t nodes = 4096;
        context.reset(ggml_init({nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true}));
        if (!context) throw std::runtime_error("LatentSync Whisper graph context allocation failed");
        core::ModuleBuildContext ctx{context.get(), "latentsync.whisper", execution.backend_type()};
        input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config.n_mels, config.n_audio_ctx * 2}));
        ggml_set_input(input.tensor);
        output = LatentSyncWhisperEncoder(config).build(ctx, input, weights);
        ggml_set_output(output.tensor);
        graph = ggml_new_graph_custom(context.get(), nodes, false);
        ggml_build_forward_expand(graph, output.tensor);
        allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) throw std::runtime_error("LatentSync Whisper graph allocation failed");
        debug::timing_log_scalar("latentsync.whisper.graph_buffer_mb", ggml_gallocr_get_buffer_size(allocator.get(), 0) / 1048576.0);
        audio::MelSpectrogramFrontendConfig mel;
        mel.sample_rate = 16000;
        mel.n_fft = mel.win_length = 400;
        mel.hop_length = 160;
        mel.n_mels = 80;
        mel.stft_center = true;
        mel.waveform_padding = audio::MelWaveformPadding::None;
        mel.spectrum_mode = audio::MelSpectrumMode::PowerDuringProjection;
        mel.value_transform = audio::MelValueTransform::Log10;
        mel.log_floor = 1e-10;
        mel.drop_last_frames = 1;
        mel.log_dynamic_range = 8;
        mel.log_shift = 4;
        mel.log_divisor = 4;
        frontend = audio::get_cached_mel_spectrogram_frontend(mel);
    }
    ~Impl() { core::release_backend_graph_resources(execution.backend(), graph, true); }
};

LatentSyncWhisperRuntime::LatentSyncWhisperRuntime(core::ExecutionContext & execution,
    const assets::TensorSource & source, modules::WhisperEmbeddingConfig config, assets::TensorStorageType weight_type)
    : impl_(std::make_unique<Impl>(execution, source, config, weight_type)) {}
LatentSyncWhisperRuntime::~LatentSyncWhisperRuntime() = default;

std::vector<float> LatentSyncWhisperRuntime::encode_mel(const std::vector<float> & mel) {
    const auto start = std::chrono::steady_clock::now();
    core::write_tensor_f32(impl_->input, mel);
    if (core::compute_backend_graph(impl_->execution.backend(), impl_->graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("LatentSync Whisper execution failed");
    }
    auto output = core::read_tensor_f32(impl_->output.tensor);
    debug::timing_log_scalar("latentsync.whisper.wall_ms",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    return output;
}

std::vector<float> LatentSyncWhisperRuntime::encode_audio(const std::vector<float> & mono_16khz, size_t threads) {
    const auto start = std::chrono::steady_clock::now();
    const auto mel = impl_->frontend->extract_mono(mono_16khz, threads);
    if (mel.frames < 2) throw std::runtime_error("LatentSync audio is too short for Whisper conditioning");
    constexpr int64_t window = 3000, row = 5 * 384;
    std::vector<float> output;
    output.reserve(static_cast<size_t>(mel.frames / 2) * row);
    std::vector<float> segment(80 * window);
    for (int64_t offset = 0; offset < mel.frames; offset += window) {
        std::fill(segment.begin(), segment.end(), 0.0F);
        const int64_t count = std::min(window, mel.frames - offset);
        for (int64_t bin = 0; bin < 80; ++bin) {
            std::copy_n(mel.values.begin() + bin * mel.frames + offset, count, segment.begin() + bin * window);
        }
        const auto embeddings = encode_mel(segment);
        output.insert(output.end(), embeddings.begin(), embeddings.begin() + count / 2 * row);
    }
    debug::timing_log_scalar("latentsync.audio_conditioning.wall_ms",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    return output;
}

std::vector<float> latentsync_audio_windows(const std::vector<float> & features,
    int64_t first_frame, int64_t frames, double fps) {
    constexpr int64_t row = 5 * 384;
    if (features.empty() || features.size() % row || first_frame < 0 || frames <= 0 || !std::isfinite(fps) || fps <= 0) {
        throw std::runtime_error("Invalid LatentSync audio window dimensions");
    }
    const int64_t length = features.size() / row;
    std::vector<float> output(static_cast<size_t>(frames) * 10 * row);
    for (int64_t frame = 0; frame < frames; ++frame) {
        const int64_t center = static_cast<int64_t>((first_frame + frame) * 50 / fps);
        for (int64_t i = 0; i < 10; ++i) {
            const int64_t index = std::clamp<int64_t>(center - 4 + i, 0, length - 1);
            std::copy_n(features.begin() + index * row, row, output.begin() + (frame * 10 + i) * row);
        }
    }
    return output;
}

}  // namespace engine::models::latentsync
