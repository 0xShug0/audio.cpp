#include "engine/community_models/glm_tts/speech_tokenizer.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/feed_forward.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/speech_encoders/whisper_embedding.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::glm_tts {
namespace {

namespace binding = modules::binding;
struct SpeechTokenizerWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::WhisperEmbeddingConfig whisper_config;
    modules::WhisperEmbeddingWeights whisper;
    modules::LinearWeights codebook_scores;
};

modules::LinearWeights load_linear(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t out_features,
    int64_t in_features,
    bool use_bias) {
    return binding::linear_from_source(
        store,
        source,
        prefix,
        storage_type,
        out_features,
        in_features,
        use_bias);
}

SpeechTokenizerWeights load_weights(
    const GlmTTSAssets & model_assets,
    core::ExecutionContext & execution,
    assets::TensorStorageType storage_type,
    size_t weight_context_bytes) {
    const auto & config = model_assets.config.speech_tokenizer;
    const auto & source = *model_assets.speech_tokenizer_weights;
    SpeechTokenizerWeights out;
    out.whisper_config.n_mels = config.feature_size;
    out.whisper_config.n_audio_ctx = config.max_source_positions;
    out.whisper_config.n_audio_state = config.d_model;
    out.whisper_config.n_audio_head = config.encoder_attention_heads;
    out.whisper_config.n_audio_layer = config.pooling_position;
    out.whisper_config.layer_norm_eps = 1.0e-5F;
    out.store = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        "glm_tts.speech_tokenizer.weights",
        weight_context_bytes);

    auto & whisper = out.whisper;
    whisper.conv1 = {
        out.store->load_tensor(
            source,
            "model.encoder.conv1.weight",
            storage_type,
            {config.d_model, config.feature_size, 3}),
        out.store->load_f32_tensor(
            source, "model.encoder.conv1.bias", {config.d_model}),
    };
    whisper.conv2 = {
        out.store->load_tensor(
            source,
            "model.encoder.conv2.weight",
            storage_type,
            {config.d_model, config.d_model, 3}),
        out.store->load_f32_tensor(
            source, "model.encoder.conv2.bias", {config.d_model}),
    };
    whisper.positional_embedding = out.store->load_f32_tensor(
        source,
        "model.encoder.embed_positions.weight",
        {config.max_source_positions, config.d_model});
    whisper.layers.reserve(static_cast<size_t>(config.pooling_position));
    for (int64_t layer = 0; layer < config.pooling_position; ++layer) {
        const std::string prefix =
            "model.encoder.layers." + std::to_string(layer);
        modules::WhisperEncoderLayerWeights layer_weights;
        layer_weights.attention_norm = binding::norm_from_source(
            *out.store,
            source,
            prefix + ".self_attn_layer_norm",
            config.d_model);
        layer_weights.attention.query = load_linear(
            *out.store,
            source,
            prefix + ".self_attn.q_proj",
            storage_type,
            config.d_model,
            config.d_model,
            true);
        layer_weights.attention.key = load_linear(
            *out.store,
            source,
            prefix + ".self_attn.k_proj",
            storage_type,
            config.d_model,
            config.d_model,
            false);
        layer_weights.attention.value = load_linear(
            *out.store,
            source,
            prefix + ".self_attn.v_proj",
            storage_type,
            config.d_model,
            config.d_model,
            true);
        layer_weights.attention.out = load_linear(
            *out.store,
            source,
            prefix + ".self_attn.out_proj",
            storage_type,
            config.d_model,
            config.d_model,
            true);
        layer_weights.mlp_norm = binding::norm_from_source(
            *out.store,
            source,
            prefix + ".final_layer_norm",
            config.d_model);
        layer_weights.mlp.fc1_weight = out.store->load_tensor(
            source,
            prefix + ".fc1.weight",
            storage_type,
            {config.encoder_ffn_dim, config.d_model});
        layer_weights.mlp.fc1_bias = out.store->load_f32_tensor(
            source,
            prefix + ".fc1.bias",
            {config.encoder_ffn_dim});
        layer_weights.mlp.fc2_weight = out.store->load_tensor(
            source,
            prefix + ".fc2.weight",
            storage_type,
            {config.d_model, config.encoder_ffn_dim});
        layer_weights.mlp.fc2_bias = out.store->load_f32_tensor(
            source,
            prefix + ".fc2.bias",
            {config.d_model});
        whisper.layers.push_back(std::move(layer_weights));
    }
    const auto codebook = source.require_f32(
        "model.encoder.codebook.weight",
        {config.quantize_vocab_size, config.d_model});
    std::vector<float> score_bias(
        static_cast<size_t>(config.quantize_vocab_size), 0.0F);
    for (int64_t token = 0; token < config.quantize_vocab_size; ++token) {
        double norm = 0.0;
        const size_t offset =
            static_cast<size_t>(token * config.d_model);
        for (int64_t channel = 0; channel < config.d_model; ++channel) {
            const double value =
                codebook[offset + static_cast<size_t>(channel)];
            norm += value * value;
        }
        score_bias[static_cast<size_t>(token)] =
            static_cast<float>(-0.5 * norm);
    }
    out.codebook_scores.weight = out.store->load_tensor(
        source,
        "model.encoder.codebook.weight",
        storage_type,
        {config.quantize_vocab_size, config.d_model});
    out.codebook_scores.bias = out.store->make_f32(
        core::TensorShape::from_dims({config.quantize_vocab_size}),
        std::move(score_bias));
    out.store->upload();
    return out;
}

core::TensorValue average_pool_time(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t kernel) {
    if (input.shape.rank != 3 || kernel <= 0) {
        throw std::runtime_error(
            "GLM-TTS speech tokenizer pooling expects [batch,time,hidden]");
    }
    const int64_t batch = input.shape.dims[0];
    const int64_t time = input.shape.dims[1];
    const int64_t hidden = input.shape.dims[2];
    if (time % kernel != 0) {
        throw std::runtime_error(
            "GLM-TTS speech tokenizer input must be aligned to the pooling "
            "kernel");
    }
    const int64_t padded_time = time;
    auto x = core::ensure_backend_addressable_layout(ctx, input);
    x = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, x),
        core::TensorShape::from_dims(
            {batch, padded_time / kernel, kernel, hidden}));
    auto pooled = modules::ReduceMeanModule({2}).build(ctx, x);
    return core::reshape_tensor(
        ctx,
        pooled,
        core::TensorShape::from_dims(
            {batch, padded_time / kernel, hidden}));
}

core::TensorValue argmax_last_dim(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & logits) {
    const int64_t rows =
        logits.shape.num_elements() / logits.shape.last_dim();
    auto flat = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, logits),
        core::TensorShape::from_dims({rows, logits.shape.last_dim()}));
    auto ids = core::wrap_tensor(
        ggml_argmax(ctx.ggml, flat.tensor),
        core::TensorShape::from_dims({rows}),
        GGML_TYPE_I32);
    return core::reshape_tensor(
        ctx,
        ids,
        core::TensorShape::from_dims(
            {logits.shape.dims[0], logits.shape.dims[1]}));
}

std::vector<float> normalize_reference_audio(
    const runtime::AudioBuffer & audio,
    int target_sample_rate) {
    if (audio.sample_rate <= 0 || audio.channels <= 0 ||
        audio.samples.empty() ||
        audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("GLM-TTS reference audio is invalid");
    }
    auto mono = engine::audio::extract_interleaved_channel(
        audio.samples, audio.channels, 0);
    if (audio.sample_rate != target_sample_rate) {
        engine::audio::TorchaudioSincHannResampleOptions options;
        options.kernel_mode =
            engine::audio::TorchaudioSincHannKernelMode::
                Float64ComputationStoredAsFloat32;
        options.accumulation =
            engine::audio::TorchaudioSincHannAccumulation::Float32;
        mono = engine::audio::resample_mono_torchaudio_sinc_hann(
            mono, audio.sample_rate, target_sample_rate, options);
    }
    return mono;
}

}  // namespace

struct GlmTTSSpeechTokenizer::Impl {
    class Graph {
    public:
        Graph(
            const GlmTTSSpeechTokenizerConfig & config,
            const SpeechTokenizerWeights & weights,
            const core::ExecutionContext & execution,
            int64_t mel_frames,
            size_t graph_context_bytes)
            : execution_(&execution),
              mel_frames_(mel_frames) {
            ggml_init_params params{graph_context_bytes, nullptr, true};
            ctx_ = ggml_init(params);
            if (ctx_ == nullptr) {
                throw std::runtime_error(
                    "failed to create GLM-TTS speech tokenizer graph");
            }
            core::ModuleBuildContext build{
                ctx_,
                "glm_tts.speech_tokenizer",
                execution.backend_type()};
            input_ = core::make_tensor(
                build,
                GGML_TYPE_F32,
                core::TensorShape::from_dims(
                    {1, config.feature_size, mel_frames_}));
            auto hidden =
                modules::WhisperEmbeddingModule(weights.whisper_config)
                    .build_pre_norm(build, input_, weights.whisper);
            hidden = average_pool_time(
                build, hidden, config.pooling_kernel_size);
            auto scores = modules::LinearModule(
                              {config.d_model,
                               config.quantize_vocab_size,
                               true})
                              .build(
                                  build,
                                  hidden,
                                  weights.codebook_scores);
            output_ = argmax_last_dim(build, scores);
            ggml_set_output(output_.tensor);
            graph_ = ggml_new_graph_custom(ctx_, 65536, false);
            ggml_build_forward_expand(graph_, output_.tensor);
            allocator_ = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(execution.backend()));
            if (allocator_ == nullptr ||
                !ggml_gallocr_reserve(allocator_, graph_) ||
                !ggml_gallocr_alloc_graph(allocator_, graph_)) {
                throw std::runtime_error(
                    "failed to allocate GLM-TTS speech tokenizer graph");
            }
        }

        ~Graph() {
            if (execution_ != nullptr && graph_ != nullptr) {
                core::release_backend_graph_resources(
                    execution_->backend(), graph_);
            }
            if (allocator_ != nullptr) {
                ggml_gallocr_free(allocator_);
            }
            if (ctx_ != nullptr) {
                ggml_free(ctx_);
            }
        }

        bool matches(int64_t mel_frames) const noexcept {
            return mel_frames_ == mel_frames;
        }

        std::vector<int32_t> run(const std::vector<float> & log_mel) {
            core::write_tensor_f32(input_, log_mel);
            core::set_backend_threads(
                execution_->backend(),
                std::max(1, execution_->config().threads));
            if (core::compute_backend_graph(
                    execution_->backend(), graph_) !=
                GGML_STATUS_SUCCESS) {
                throw std::runtime_error(
                    "GLM-TTS speech tokenizer graph compute failed");
            }
            return core::read_tensor_i32(output_.tensor);
        }

    private:
        const core::ExecutionContext * execution_ = nullptr;
        int64_t mel_frames_ = 0;
        ggml_context * ctx_ = nullptr;
        ggml_gallocr_t allocator_ = nullptr;
        ggml_cgraph * graph_ = nullptr;
        core::TensorValue input_;
        core::TensorValue output_;
    };

    Impl(
        std::shared_ptr<const GlmTTSAssets> assets_in,
        core::BackendConfig backend,
        assets::TensorStorageType storage_type,
        size_t weight_context_bytes,
        size_t graph_context_bytes_in)
        : assets(std::move(assets_in)),
          execution(std::move(backend)),
          graph_context_bytes(graph_context_bytes_in),
          extractor({
              16000,
              400,
              160,
              128,
              engine::audio::STFTFamily::Kokoro,
          }) {
        if (assets == nullptr) {
            throw std::runtime_error(
                "GLM-TTS speech tokenizer requires assets");
        }
        weights = load_weights(
            *assets,
            execution,
            storage_type,
            weight_context_bytes);
    }

    std::vector<int32_t> encode(
        const runtime::AudioBuffer & audio) const {
        const auto & config = assets->config.speech_tokenizer;
        auto samples =
            normalize_reference_audio(audio, config.sampling_rate);
        const size_t max_chunk_samples =
            static_cast<size_t>(config.sampling_rate * 30);
        std::vector<int32_t> tokens;
        for (size_t offset = 0; offset < samples.size();
             offset += max_chunk_samples) {
            const size_t count =
                std::min(max_chunk_samples, samples.size() - offset);
            const size_t stride_samples = 640;
            const size_t padded_count =
                ((count + stride_samples - 1) / stride_samples) *
                stride_samples;
            std::vector<float> chunk(padded_count, 0.0F);
            std::copy_n(
                samples.data() + static_cast<std::ptrdiff_t>(offset),
                count,
                chunk.data());
            const auto features = extractor.compute(
                chunk,
                static_cast<size_t>(
                    std::max(1, execution.config().threads)));
            if (features.frames <= 0 ||
                features.frames >
                    config.max_source_positions * 2) {
                throw std::runtime_error(
                    "GLM-TTS speech tokenizer feature length is invalid");
            }
            std::lock_guard<std::mutex> lock(mutex);
            if (graph == nullptr ||
                !graph->matches(features.frames)) {
                graph = std::make_unique<Graph>(
                    config,
                    weights,
                    execution,
                    features.frames,
                    graph_context_bytes);
            }
            auto chunk_tokens = graph->run(features.values);
            tokens.insert(
                tokens.end(),
                chunk_tokens.begin(),
                chunk_tokens.end());
        }
        if (tokens.empty()) {
            throw std::runtime_error(
                "GLM-TTS speech tokenizer produced no tokens");
        }
        debug::trace_log_scalar(
            "glm_tts.speech_tokenizer.tokens",
            static_cast<int64_t>(tokens.size()));
        return tokens;
    }

    void release_graph() const {
        std::lock_guard<std::mutex> lock(mutex);
        graph.reset();
    }

    std::shared_ptr<const GlmTTSAssets> assets;
    core::ExecutionContext execution;
    size_t graph_context_bytes = 0;
    engine::audio::WhisperLogMelExtractor extractor;
    SpeechTokenizerWeights weights;
    mutable std::mutex mutex;
    mutable std::unique_ptr<Graph> graph;
};

GlmTTSSpeechTokenizer::GlmTTSSpeechTokenizer(
    std::shared_ptr<const GlmTTSAssets> assets,
    core::BackendConfig backend,
    assets::TensorStorageType weight_storage_type,
    size_t weight_context_bytes,
    size_t graph_context_bytes)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          std::move(backend),
          weight_storage_type,
          weight_context_bytes,
          graph_context_bytes)) {}

GlmTTSSpeechTokenizer::~GlmTTSSpeechTokenizer() = default;

std::vector<int32_t> GlmTTSSpeechTokenizer::encode(
    const runtime::AudioBuffer & audio) const {
    return impl_->encode(audio);
}

void GlmTTSSpeechTokenizer::release_runtime_graph() const {
    impl_->release_graph();
}

}  // namespace engine::models::glm_tts
