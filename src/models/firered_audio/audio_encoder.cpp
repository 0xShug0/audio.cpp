#include "engine/models/firered_audio/audio_encoder.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/feed_forward.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace engine::models::firered_audio {
namespace {

namespace assets = engine::assets;
namespace binding = engine::modules::binding;
namespace core = engine::core;
namespace modules = engine::modules;

using Clock = std::chrono::steady_clock;

constexpr size_t kAudioEncoderWeightContextBytes = 256ull * 1024ull * 1024ull;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct GgmlGallocrDeleter {
    void operator()(ggml_gallocr_t alloc) const noexcept {
        if (alloc != nullptr) {
            ggml_gallocr_free(alloc);
        }
    }
};

struct GraphMemory {
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr;
    ggml_backend_buffer_t input_buffer = nullptr;
    ggml_cgraph * graph = nullptr;

    ~GraphMemory() {
        reset(nullptr);
    }

    void reset(ggml_backend_t backend) {
        if (graph != nullptr && backend != nullptr) {
            core::release_backend_graph_resources(backend, graph);
        }
        graph = nullptr;
        gallocr.reset();
        if (input_buffer != nullptr) {
            ggml_backend_buffer_free(input_buffer);
            input_buffer = nullptr;
        }
        input_ctx.reset();
        ctx.reset();
    }
};

struct AudioAttentionWeights {
    modules::LinearWeights q_proj;
    modules::LinearWeights k_proj;
    modules::LinearWeights v_proj;
    modules::LinearWeights out_proj;
};

struct AudioEncoderLayerWeights {
    modules::NormWeights self_attn_norm;
    AudioAttentionWeights attention;
    modules::NormWeights final_norm;
    modules::FeedForwardWeights feed_forward;
};

struct AudioEncoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::Conv1dWeights conv1;
    modules::Conv1dWeights conv2;
    std::vector<AudioEncoderLayerWeights> layers;
    modules::Conv1dWeights adapter_conv3;
    modules::Conv1dWeights adapter_conv4;
    modules::NormWeights adapter_norm;
    modules::LinearWeights adapter_linear1;
    modules::LinearWeights adapter_linear2;
    core::TensorValue positions;
};

int64_t conv_stride2_length(int64_t input) {
    return (input - 1) / 2 + 1;
}

std::vector<float> sinusoidal_positions(int64_t length, int64_t channels) {
    if (channels % 2 != 0) {
        throw std::runtime_error("FireRedAudio audio encoder positional embedding requires even channel count");
    }
    std::vector<float> table(static_cast<size_t>(length * channels), 0.0F);
    const double increment = std::log(10000.0) / static_cast<double>(channels / 2 - 1);
    for (int64_t pos = 0; pos < length; ++pos) {
        for (int64_t dim = 0; dim < channels / 2; ++dim) {
            const double scaled = static_cast<double>(pos) * std::exp(-increment * static_cast<double>(dim));
            table[static_cast<size_t>(pos * channels + dim)] = static_cast<float>(std::sin(scaled));
            table[static_cast<size_t>(pos * channels + channels / 2 + dim)] = static_cast<float>(std::cos(scaled));
        }
    }
    return table;
}

modules::NormWeights layer_norm_from_source(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t hidden_size) {
    return {
        store.load_f32_tensor(source, prefix + ".weight", {hidden_size}),
        store.load_f32_tensor(source, prefix + ".bias", {hidden_size}),
    };
}

std::shared_ptr<AudioEncoderWeights> load_audio_encoder_weights(
    const FireRedAudioAssets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    const auto & c = assets.audio_encoder;
    const auto & source = *assets.model_weights;
    auto out = std::make_shared<AudioEncoderWeights>();
    out->store = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        "firered_audio.audio_encoder.weights",
        std::max(weight_context_bytes, kAudioEncoderWeightContextBytes));
    out->conv1 = binding::conv1d_from_source(
        *out->store,
        source,
        "audio_encoder.conv1",
        storage_type,
        c.d_model,
        c.num_mel_bins,
        3,
        true);
    out->conv2 = binding::conv1d_from_source(
        *out->store,
        source,
        "audio_encoder.conv2",
        storage_type,
        c.d_model,
        c.d_model,
        3,
        true);
    out->layers.reserve(static_cast<size_t>(c.encoder_layers));
    for (int64_t layer = 0; layer < c.encoder_layers; ++layer) {
        const std::string prefix = "audio_encoder.layers." + std::to_string(layer);
        AudioEncoderLayerWeights item;
        item.self_attn_norm = layer_norm_from_source(*out->store, source, prefix + ".self_attn_layer_norm", c.d_model);
        item.attention.q_proj = binding::linear_from_source(*out->store, source, prefix + ".self_attn.q_proj", storage_type, c.d_model, c.d_model, true);
        item.attention.k_proj = binding::linear_from_source(*out->store, source, prefix + ".self_attn.k_proj", storage_type, c.d_model, c.d_model, false);
        item.attention.v_proj = binding::linear_from_source(*out->store, source, prefix + ".self_attn.v_proj", storage_type, c.d_model, c.d_model, true);
        item.attention.out_proj = binding::linear_from_source(*out->store, source, prefix + ".self_attn.out_proj", storage_type, c.d_model, c.d_model, true);
        item.final_norm = layer_norm_from_source(*out->store, source, prefix + ".final_layer_norm", c.d_model);
        const auto fc1 = binding::linear_from_source(*out->store, source, prefix + ".fc1", storage_type, c.encoder_ffn_dim, c.d_model, true);
        const auto fc2 = binding::linear_from_source(*out->store, source, prefix + ".fc2", storage_type, c.d_model, c.encoder_ffn_dim, true);
        item.feed_forward = {fc1.weight, fc1.bias, fc2.weight, fc2.bias};
        out->layers.push_back(std::move(item));
    }
    out->adapter_conv3 = binding::conv1d_from_source(
        *out->store,
        source,
        "audio_encoder.adapter.conv3",
        storage_type,
        c.d_model,
        c.d_model,
        3,
        true);
    out->adapter_conv4 = binding::conv1d_from_source(
        *out->store,
        source,
        "audio_encoder.adapter.conv4",
        storage_type,
        c.d_model,
        c.d_model,
        3,
        true);
    out->adapter_norm = layer_norm_from_source(*out->store, source, "audio_encoder.adapter.layer_norm", c.d_model);
    out->adapter_linear1 = binding::linear_from_source(*out->store, source, "audio_encoder.adapter.linear1", storage_type, c.output_dim, c.d_model, true);
    out->adapter_linear2 = binding::linear_from_source(*out->store, source, "audio_encoder.adapter.linear2", storage_type, c.output_dim, c.output_dim, true);
    out->positions = out->store->make_f32(
        core::TensorShape::from_dims({c.max_source_positions, c.d_model}),
        sinusoidal_positions(c.max_source_positions, c.d_model));
    out->store->upload();
    return out;
}

core::TensorValue reshape_audio_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t dim) {
    return core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, input),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

core::TensorValue audio_self_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const AudioAttentionWeights & weights,
    const FireRedAudioAudioEncoderConfig & config) {
    const int64_t dim = config.d_model / config.encoder_attention_heads;
    auto q = modules::LinearModule({config.d_model, config.d_model, true}).build(ctx, input, weights.q_proj);
    auto k = modules::LinearModule({config.d_model, config.d_model, false}).build(ctx, input, weights.k_proj);
    auto v = modules::LinearModule({config.d_model, config.d_model, true}).build(ctx, input, weights.v_proj);
    q = reshape_audio_heads(ctx, q, config.encoder_attention_heads, dim);
    k = reshape_audio_heads(ctx, k, config.encoder_attention_heads, dim);
    v = reshape_audio_heads(ctx, v, config.encoder_attention_heads, dim);
    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    auto context = modules::ScaledDotProductAttentionModule({
        dim,
        modules::ScaledDotProductAttentionLowering::Explicit,
        GGML_PREC_F32,
    }).build(ctx, q_heads, k_heads, v_heads, std::nullopt);
    context = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, context),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.d_model}));
    return modules::LinearModule({config.d_model, config.d_model, true}).build(ctx, context, weights.out_proj);
}

core::TensorValue encoder_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const AudioEncoderLayerWeights & weights,
    const FireRedAudioAudioEncoderConfig & config) {
    auto h = modules::LayerNormModule({config.d_model, config.layer_norm_eps, true, true}).build(ctx, input, weights.self_attn_norm);
    h = audio_self_attention(ctx, h, weights.attention, config);
    auto x = modules::AddModule().build(ctx, input, h);
    h = modules::LayerNormModule({config.d_model, config.layer_norm_eps, true, true}).build(ctx, x, weights.final_norm);
    h = modules::FeedForwardModule({
        config.d_model,
        config.encoder_ffn_dim,
        true,
        modules::GeluApproximation::ExactErf,
    }).build(ctx, h, weights.feed_forward);
    return modules::AddModule().build(ctx, x, h);
}

std::vector<float> prepare_understanding_audio(
    const runtime::AudioBuffer & audio,
    const FireRedAudioAudioEncoderConfig & config) {
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty()) {
        throw std::runtime_error("FireRedAudio understanding audio input is invalid");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("FireRedAudio understanding audio sample count is not divisible by channels");
    }
    return engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio.samples,
        audio.sample_rate,
        audio.channels,
        config.sample_rate);
}

struct FrontendOutput {
    std::vector<float> values;
    int64_t frames = 0;
};

FrontendOutput extract_log_mel(
    const runtime::AudioBuffer & audio,
    const FireRedAudioAudioEncoderConfig & config,
    engine::audio::WhisperLogMelExtractor & extractor) {
    auto samples = prepare_understanding_audio(audio, config);
    const int64_t real_frames = static_cast<int64_t>(samples.size()) / config.hop_length;
    samples.resize(samples.size() + static_cast<size_t>(config.sample_rate), 0.0F);
    auto features = extractor.compute(samples);
    if (features.mel_bins != config.num_mel_bins || features.frames < real_frames) {
        throw std::runtime_error("FireRedAudio understanding frontend produced unexpected shape");
    }
    FrontendOutput out;
    out.frames = real_frames;
    out.values.assign(static_cast<size_t>(config.num_mel_bins * real_frames), 0.0F);
    for (int64_t mel = 0; mel < config.num_mel_bins; ++mel) {
        const auto * src = features.values.data() + static_cast<size_t>(mel * features.frames);
        auto * dst = out.values.data() + static_cast<size_t>(mel * real_frames);
        std::copy(src, src + real_frames, dst);
    }
    return out;
}

class ChunkEncoderGraph {
public:
    ChunkEncoderGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const AudioEncoderWeights> weights,
        FireRedAudioAudioEncoderConfig config,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          config_(config),
          graph_arena_bytes_(graph_arena_bytes) {}

    ~ChunkEncoderGraph() {
        mem_.reset(execution_.backend());
    }

    std::vector<float> run(const std::vector<float> & features, int64_t frames) {
        if (frames <= 0 || static_cast<int64_t>(features.size()) != config_.num_mel_bins * frames) {
            throw std::runtime_error("FireRedAudio audio encoder chunk input size mismatch");
        }
        ensure(frames);
        core::write_tensor_f32(core::wrap_tensor(input_, core::TensorShape::from_dims({1, config_.num_mel_bins, frames})), features);
        if (core::compute_backend_graph(execution_.backend(), mem_.graph, nullptr, "firered_audio.audio_encoder.chunk") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("FireRedAudio audio encoder chunk graph compute failed");
        }
        return core::read_tensor_f32(output_);
    }

    void release_graph() {
        mem_.reset(execution_.backend());
        frames_ = 0;
        input_ = nullptr;
        output_ = nullptr;
    }

private:
    void ensure(int64_t frames) {
        if (mem_.graph != nullptr && frames_ == frames) {
            return;
        }
        mem_.reset(execution_.backend());
        ggml_init_params params{graph_arena_bytes_, nullptr, true};
        mem_.ctx.reset(ggml_init(params));
        ggml_init_params input_params{16ull * 1024ull * 1024ull, nullptr, true};
        mem_.input_ctx.reset(ggml_init(input_params));
        core::ModuleBuildContext ctx{mem_.ctx.get(), "firered_audio.audio_encoder.chunk", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{mem_.input_ctx.get(), "firered_audio.audio_encoder.chunk.inputs", execution_.backend_type()};
        auto x = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config_.num_mel_bins, frames}));
        input_ = x.tensor;
        ggml_set_input(input_);
        x = modules::Conv1dModule({config_.num_mel_bins, config_.d_model, 3, 1, 1, 1, true}).build(ctx, x, weights_->conv1);
        x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
        x = modules::Conv1dModule({config_.d_model, config_.d_model, 3, 2, 1, 1, true}).build(ctx, x, weights_->conv2);
        x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
        x = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
        const int64_t tokens = conv_stride2_length(frames);
        if (tokens > config_.max_source_positions) {
            throw std::runtime_error("FireRedAudio audio encoder chunk exceeds positional embedding length");
        }
        auto pos = modules::SliceModule({0, 0, tokens}).build(ctx, weights_->positions);
        pos = core::reshape_tensor(ctx, pos, core::TensorShape::from_dims({1, tokens, config_.d_model}));
        x = modules::AddModule().build(ctx, x, pos);
        for (const auto & layer : weights_->layers) {
            x = encoder_layer(ctx, x, layer, config_);
        }
        output_ = core::ensure_backend_addressable_layout(ctx, x).tensor;
        ggml_set_output(output_);
        mem_.graph = ggml_new_graph_custom(mem_.ctx.get(), 800000, false);
        ggml_build_forward_expand(mem_.graph, output_);
        mem_.input_buffer = ggml_backend_alloc_ctx_tensors(mem_.input_ctx.get(), execution_.backend());
        mem_.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend())));
        if (mem_.input_buffer == nullptr || mem_.gallocr == nullptr ||
            !ggml_gallocr_reserve(mem_.gallocr.get(), mem_.graph) ||
            !ggml_gallocr_alloc_graph(mem_.gallocr.get(), mem_.graph)) {
            mem_.reset(execution_.backend());
            throw std::runtime_error("failed to allocate FireRedAudio audio encoder chunk graph");
        }
        frames_ = frames;
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const AudioEncoderWeights> weights_;
    FireRedAudioAudioEncoderConfig config_;
    size_t graph_arena_bytes_;
    GraphMemory mem_;
    int64_t frames_ = 0;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
};

class AdapterGraph {
public:
    AdapterGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const AudioEncoderWeights> weights,
        FireRedAudioAudioEncoderConfig config,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          config_(config),
          graph_arena_bytes_(graph_arena_bytes) {}

    ~AdapterGraph() {
        mem_.reset(execution_.backend());
    }

    std::vector<float> run(const std::vector<float> & hidden, int64_t tokens) {
        if (tokens <= 0 || static_cast<int64_t>(hidden.size()) != tokens * config_.d_model) {
            throw std::runtime_error("FireRedAudio audio encoder adapter input size mismatch");
        }
        ensure(tokens);
        core::write_tensor_f32(core::wrap_tensor(input_, core::TensorShape::from_dims({1, tokens, config_.d_model})), hidden);
        if (core::compute_backend_graph(execution_.backend(), mem_.graph, nullptr, "firered_audio.audio_encoder.adapter") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("FireRedAudio audio encoder adapter graph compute failed");
        }
        return core::read_tensor_f32(output_);
    }

    void release_graph() {
        mem_.reset(execution_.backend());
        tokens_ = 0;
        input_ = nullptr;
        output_ = nullptr;
    }

private:
    void ensure(int64_t tokens) {
        if (mem_.graph != nullptr && tokens_ == tokens) {
            return;
        }
        mem_.reset(execution_.backend());
        ggml_init_params params{graph_arena_bytes_, nullptr, true};
        mem_.ctx.reset(ggml_init(params));
        ggml_init_params input_params{16ull * 1024ull * 1024ull, nullptr, true};
        mem_.input_ctx.reset(ggml_init(input_params));
        core::ModuleBuildContext ctx{mem_.ctx.get(), "firered_audio.audio_encoder.adapter", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{mem_.input_ctx.get(), "firered_audio.audio_encoder.adapter.inputs", execution_.backend_type()};
        auto x = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, tokens, config_.d_model}));
        input_ = x.tensor;
        ggml_set_input(input_);
        x = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
        x = modules::Conv1dModule({config_.d_model, config_.d_model, 3, 2, 1, 1, true}).build(ctx, x, weights_->adapter_conv3);
        x = modules::Conv1dModule({config_.d_model, config_.d_model, 3, 2, 1, 1, true}).build(ctx, x, weights_->adapter_conv4);
        x = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
        x = modules::LayerNormModule({config_.d_model, config_.layer_norm_eps, true, true}).build(ctx, x, weights_->adapter_norm);
        x = modules::LinearModule({config_.d_model, config_.output_dim, true}).build(ctx, x, weights_->adapter_linear1);
        x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
        x = modules::LinearModule({config_.output_dim, config_.output_dim, true}).build(ctx, x, weights_->adapter_linear2);
        output_ = core::ensure_backend_addressable_layout(ctx, x).tensor;
        ggml_set_output(output_);
        mem_.graph = ggml_new_graph_custom(mem_.ctx.get(), 4096, false);
        ggml_build_forward_expand(mem_.graph, output_);
        mem_.input_buffer = ggml_backend_alloc_ctx_tensors(mem_.input_ctx.get(), execution_.backend());
        mem_.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend())));
        if (mem_.input_buffer == nullptr || mem_.gallocr == nullptr ||
            !ggml_gallocr_reserve(mem_.gallocr.get(), mem_.graph) ||
            !ggml_gallocr_alloc_graph(mem_.gallocr.get(), mem_.graph)) {
            mem_.reset(execution_.backend());
            throw std::runtime_error("failed to allocate FireRedAudio audio encoder adapter graph");
        }
        tokens_ = tokens;
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const AudioEncoderWeights> weights_;
    FireRedAudioAudioEncoderConfig config_;
    size_t graph_arena_bytes_;
    GraphMemory mem_;
    int64_t tokens_ = 0;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
};

}  // namespace

class FireRedAudioAudioEncoderRuntime::Impl {
public:
    Impl(
        std::shared_ptr<const FireRedAudioAssets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : assets_(std::move(assets)),
          extractor_({
              assets_->audio_encoder.sample_rate,
              assets_->audio_encoder.n_fft,
              assets_->audio_encoder.hop_length,
              assets_->audio_encoder.num_mel_bins,
              engine::audio::STFTFamily::Default,
          }),
          weights_(load_audio_encoder_weights(*assets_, execution, weight_context_bytes, storage_type)),
          chunk_(execution, weights_, assets_->audio_encoder, graph_arena_bytes),
          adapter_(execution, weights_, assets_->audio_encoder, graph_arena_bytes) {
        if (assets_ == nullptr) {
            throw std::runtime_error("FireRedAudio audio encoder runtime requires assets");
        }
    }

    FireRedAudioUnderstandFeatures encode(const runtime::AudioBuffer & audio) {
        const auto frontend_start = Clock::now();
        const auto frontend = extract_log_mel(audio, assets_->audio_encoder, extractor_);
        const auto frontend_end = Clock::now();
        const auto & config = assets_->audio_encoder;
        std::vector<float> encoded;
        int64_t encoder_tokens = 0;
        for (int64_t offset = 0; offset < frontend.frames; offset += config.n_window * 2) {
            const int64_t chunk_frames = std::min(config.n_window * 2, frontend.frames - offset);
            std::vector<float> chunk_values(static_cast<size_t>(config.num_mel_bins * chunk_frames));
            for (int64_t mel = 0; mel < config.num_mel_bins; ++mel) {
                const auto * src = frontend.values.data() + static_cast<size_t>(mel * frontend.frames + offset);
                auto * dst = chunk_values.data() + static_cast<size_t>(mel * chunk_frames);
                std::copy(src, src + chunk_frames, dst);
            }
            auto chunk_hidden = chunk_.run(chunk_values, chunk_frames);
            const int64_t chunk_tokens = conv_stride2_length(chunk_frames);
            encoded.insert(encoded.end(), chunk_hidden.begin(), chunk_hidden.end());
            encoder_tokens += chunk_tokens;
        }
        if (static_cast<int64_t>(encoded.size()) != encoder_tokens * config.d_model) {
            throw std::runtime_error("FireRedAudio audio encoder produced an invalid hidden size");
        }
        const auto adapter_start = Clock::now();
        auto adapted = adapter_.run(encoded, encoder_tokens);
        const auto adapter_end = Clock::now();
        FireRedAudioUnderstandFeatures out;
        out.tokens = conv_stride2_length(conv_stride2_length(conv_stride2_length(frontend.frames)));
        if (static_cast<int64_t>(adapted.size()) != out.tokens * config.output_dim) {
            throw std::runtime_error("FireRedAudio audio encoder adapter produced an invalid output size");
        }
        out.embeddings = std::move(adapted);
        engine::debug::timing_log_scalar("firered_audio.audio_encoder.frontend_ms", engine::debug::elapsed_ms(frontend_start, frontend_end));
        engine::debug::timing_log_scalar("firered_audio.audio_encoder.adapter_ms", engine::debug::elapsed_ms(adapter_start, adapter_end));
        return out;
    }

    void release_graphs() {
        chunk_.release_graph();
        adapter_.release_graph();
    }

private:
    std::shared_ptr<const FireRedAudioAssets> assets_;
    engine::audio::WhisperLogMelExtractor extractor_;
    std::shared_ptr<AudioEncoderWeights> weights_;
    ChunkEncoderGraph chunk_;
    AdapterGraph adapter_;
};

FireRedAudioAudioEncoderRuntime::FireRedAudioAudioEncoderRuntime(
    std::shared_ptr<const FireRedAudioAssets> assets,
    engine::core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, graph_arena_bytes, weight_context_bytes, storage_type)) {}

FireRedAudioAudioEncoderRuntime::~FireRedAudioAudioEncoderRuntime() = default;

FireRedAudioUnderstandFeatures FireRedAudioAudioEncoderRuntime::encode(const engine::runtime::AudioBuffer & audio) {
    return impl_->encode(audio);
}

void FireRedAudioAudioEncoderRuntime::release_graphs() {
    impl_->release_graphs();
}

}  // namespace engine::models::firered_audio
