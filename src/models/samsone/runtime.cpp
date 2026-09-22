#include "engine/models/samsone/runtime.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/mel_spectrogram_frontend.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/audio/waveform_ops.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/framework/io/text.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/speech_encoders/whisper_frontend.h"
#include "engine/framework/modules/transformers/qwen_causal_decode_runtime.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::samsone {
namespace {

using Storage = assets::TensorStorageType;
namespace binding = modules::binding;

constexpr int64_t kSampleRate = 16000;
constexpr int64_t kMaxSamples = 480000;
constexpr int64_t kWhisperFrames = 1500;
constexpr int64_t kWhisperChannels = 384;
constexpr int64_t kAudioTokens = 50;
constexpr int64_t kPoolSize = kWhisperFrames / kAudioTokens;
constexpr int64_t kLookupCapacity = 512;

struct ContextDeleter {
    void operator()(ggml_context * value) const { ggml_free(value); }
};

struct AllocatorDeleter {
    void operator()(ggml_gallocr_t value) const { ggml_gallocr_free(value); }
};

std::string normalize_prompt(std::string prompt) {
    std::transform(prompt.begin(), prompt.end(), prompt.begin(), [](unsigned char value) {
        return value < 128 ? static_cast<char>(std::tolower(value)) : '\0';
    });
    prompt.erase(std::remove(prompt.begin(), prompt.end(), '\0'), prompt.end());
    prompt = std::regex_replace(prompt, std::regex("\\s{4,}"), " ");
    prompt = std::regex_replace(prompt, std::regex("-{3,}"), "-");
    return std::regex_replace(prompt, std::regex("#{3,}"), "#");
}

audio::MelSpectrogramFrontendConfig whisper_mel_config() {
    audio::MelSpectrogramFrontendConfig config;
    config.sample_rate = kSampleRate;
    config.n_fft = 400;
    config.hop_length = 160;
    config.win_length = 400;
    config.n_mels = 80;
    config.stft_center = true;
    config.waveform_padding = audio::MelWaveformPadding::None;
    config.spectrum_mode = audio::MelSpectrumMode::PowerDuringProjection;
    config.value_transform = audio::MelValueTransform::Log10;
    config.log_floor = 1.0e-10;
    config.max_frames = 3000;
    config.log_dynamic_range = 8.0F;
    config.log_shift = 4.0F;
    config.log_divisor = 4.0F;
    return config;
}

std::vector<float> pool_whisper(const std::vector<float> & encoded) {
    if (encoded.size() != static_cast<size_t>(kWhisperFrames * kWhisperChannels)) {
        throw std::runtime_error("SAMSONE Whisper encoder output shape mismatch");
    }
    std::vector<float> pooled(static_cast<size_t>(kAudioTokens * kWhisperChannels), 0.0F);
    for (int64_t token = 0; token < kAudioTokens; ++token) {
        for (int64_t frame = 0; frame < kPoolSize; ++frame) {
            const float * source = encoded.data() + (token * kPoolSize + frame) * kWhisperChannels;
            float * target = pooled.data() + token * kWhisperChannels;
            for (int64_t channel = 0; channel < kWhisperChannels; ++channel) {
                target[channel] += source[channel] / static_cast<float>(kPoolSize);
            }
        }
    }
    return pooled;
}

}  // namespace

class SamsoneRuntime::Impl {
public:
    Impl(std::shared_ptr<const SamsoneAssets> assets, core::ExecutionContext & execution)
        : assets_(std::move(assets)), execution_(execution),
          store_(execution.backend(), execution.backend_type(), "samsone.weights", 8 * 1024 * 1024),
          mel_(audio::get_cached_mel_spectrogram_frontend(whisper_mel_config())) {
        if (assets_ == nullptr) {
            throw std::runtime_error("SAMSONE runtime requires assets");
        }
        tokenizers::LlamaBpeTokenizerSpec tokenizer_config;
        tokenizer_config.tokenizer_config_path = assets_->resources.require_file("tokenizer_config");
        tokenizer_config.tokenizer_json_path = assets_->resources.require_file("tokenizer_json");
        tokenizer_config.pre_type = tokenizers::LlamaBpePreTokenizer::Smollm;
        tokenizer_ = tokenizers::load_llama_bpe_tokenizer(tokenizer_config);
        load_token_map();

        modules::WhisperFrontendComponentConfig whisper_config;
        whisper_config.name = "samsone.whisper";
        whisper_config.weight_context_bytes = 4 * 1024 * 1024;
        whisper_config.graph_context_bytes = 16 * 1024 * 1024;
        whisper_ = modules::WhisperFrontendComponent::load_openai_layout(
            assets_->weights,
            execution.config(),
            {80, kWhisperFrames, kWhisperChannels, 6, 4, 1.0e-5F},
            whisper_config);
        if (whisper_.channels() != kWhisperChannels || whisper_.config().n_audio_ctx != kWhisperFrames) {
            throw std::runtime_error("SAMSONE requires the Whisper-tiny encoder");
        }

        const auto & config = assets_->config;
        token_embedding_ = store_.load_tensor(
            *assets_->weights, "model.embed_tokens.weight", Storage::Native,
            {config.vocab_size, config.hidden_size});
        modules::QwenCausalDecodeRuntimeWeights decoder_weights;
        decoder_weights.token_embedding = token_embedding_;
        decoder_weights.stack.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
        const int64_t head_dim = config.hidden_size / config.num_attention_heads;
        for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
            const std::string prefix = "model.layers." + std::to_string(layer);
            modules::QwenDecoderLayerWeights weights;
            weights.input_norm = binding::norm_weight_from_source(store_, *assets_->weights, prefix + ".input_layernorm", config.hidden_size);
            weights.post_norm = binding::norm_weight_from_source(store_, *assets_->weights, prefix + ".post_attention_layernorm", config.hidden_size);
            weights.self_attention.q_weight = store_.load_tensor(*assets_->weights, prefix + ".self_attn.q_proj.weight", Storage::Native, {config.hidden_size, config.hidden_size});
            weights.self_attention.k_weight = store_.load_tensor(*assets_->weights, prefix + ".self_attn.k_proj.weight", Storage::Native, {config.num_key_value_heads * head_dim, config.hidden_size});
            weights.self_attention.v_weight = store_.load_tensor(*assets_->weights, prefix + ".self_attn.v_proj.weight", Storage::Native, {config.num_key_value_heads * head_dim, config.hidden_size});
            weights.self_attention.out_weight = store_.load_tensor(*assets_->weights, prefix + ".self_attn.o_proj.weight", Storage::Native, {config.hidden_size, config.hidden_size});
            weights.mlp.gate_proj = binding::linear_from_source(store_, *assets_->weights, prefix + ".mlp.gate_proj", Storage::Native, config.intermediate_size, config.hidden_size, false);
            weights.mlp.up_proj = binding::linear_from_source(store_, *assets_->weights, prefix + ".mlp.up_proj", Storage::Native, config.intermediate_size, config.hidden_size, false);
            weights.mlp.down_proj = binding::linear_from_source(store_, *assets_->weights, prefix + ".mlp.down_proj", Storage::Native, config.hidden_size, config.intermediate_size, false);
            decoder_weights.stack.layers.push_back(std::move(weights));
        }
        decoder_weights.final_norm = binding::norm_weight_from_source(store_, *assets_->weights, "model.norm", config.hidden_size);
        decoder_weights.lm_head = modules::LinearWeights{token_embedding_, std::nullopt};

        modules::QwenCausalDecodeRuntimeConfig decoder_config;
        decoder_config.trace_name = "samsone.decoder";
        decoder_config.prefill_graph_arena_bytes = 64 * 1024 * 1024;
        decoder_config.decode_graph_arena_bytes = 32 * 1024 * 1024;
        decoder_config.decoder.logits_size = config.vocab_size;
        decoder_config.decoder.static_cache_type = GGML_TYPE_F16;
        auto & stack = decoder_config.decoder.stack;
        stack.hidden_size = config.hidden_size;
        stack.num_attention_heads = config.num_attention_heads;
        stack.num_key_value_heads = config.num_key_value_heads;
        stack.head_dim = head_dim;
        stack.intermediate_size = config.intermediate_size;
        stack.layers = config.num_hidden_layers;
        stack.rms_norm_eps = config.rms_norm_eps;
        stack.rope_theta = config.rope_theta;
        stack.use_qk_norm = false;
        stack.runtime.attention.prefill_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
        stack.runtime.attention.static_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
        stack.runtime.static_cache.update_mode = modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
        stack.runtime.static_cache.set_rows_mode = modules::QwenDecoderStaticCacheSetRowsMode::BackendViewOptimized;

        projector_linear1_ = binding::linear_from_source(store_, *assets_->weights, "projector.linear1", Storage::Native, config.hidden_size, kWhisperChannels, false);
        projector_linear2_ = binding::linear_from_source(store_, *assets_->weights, "projector.linear2", Storage::Native, config.hidden_size, config.hidden_size, false);
        projector_norm_ = binding::norm_from_source(store_, *assets_->weights, "projector.layer_norm", config.hidden_size);
        separator_ = store_.load_f32_tensor(*assets_->weights, "sep_token", {config.hidden_size});
        store_.upload();
        decoder_ = std::make_unique<modules::QwenCausalDecodeRuntime>(execution_, decoder_config, std::move(decoder_weights));
        build_auxiliary_graphs();
        assets_->weights->release_storage();
    }

    ~Impl() {
        release_auxiliary_graphs();
    }

    std::string generate(
        const runtime::AudioBuffer & audio,
        const std::string & prompt,
        int64_t max_tokens,
        size_t threads) {
        if (audio.samples.empty() || max_tokens <= 0) {
            throw std::runtime_error("SAMSONE requires non-empty audio and positive max_tokens");
        }
        const auto frontend_start = std::chrono::steady_clock::now();
        auto mono = audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
        if (audio.sample_rate != kSampleRate) {
            audio::SoxrResampleOptions options;
            options.output_length_policy = audio::SoxrOutputLengthPolicy::ExactExpected;
            options.require_full_input = true;
            const auto converted = audio::try_resample_mono_soxr(mono, audio.sample_rate, kSampleRate, options);
            if (!converted.has_value()) {
                throw std::runtime_error("SAMSONE requires SOXR for non-16-kHz audio");
            }
            mono = *converted;
        }
        mono.resize(std::min<size_t>(mono.size(), kMaxSamples));
        mono.resize(kMaxSamples, 0.0F);
        auto mel = mel_->extract_mono(mono, threads);
        for (float & value : mel.values) {
            value = ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
        }
        const auto encoded = whisper_.encode_log_mel(mel.values);
        const auto pooled = pool_whisper(encoded);
        auto projected = project(pooled);
        debug::timing_log_scalar("samsone.frontend_ms", debug::elapsed_ms(frontend_start));

        const auto prompt_start = std::chrono::steady_clock::now();
        auto original_ids = tokenizer_->encode(normalize_prompt(prompt) + " answer: ");
        std::vector<int32_t> prompt_ids;
        prompt_ids.reserve(original_ids.size());
        for (int32_t id : original_ids) {
            if (id < 0 || static_cast<size_t>(id) >= original_to_pruned_.size() || original_to_pruned_[id] < 0) {
                throw std::runtime_error("SAMSONE prompt contains a token unavailable in the pruned vocabulary");
            }
            prompt_ids.push_back(original_to_pruned_[id]);
        }
        auto text_embeddings = embed(prompt_ids);
        std::vector<float> embeddings;
        embeddings.reserve(static_cast<size_t>((kAudioTokens + 2 + prompt_ids.size()) * assets_->config.hidden_size));
        embeddings.insert(embeddings.end(), separator_values_.begin(), separator_values_.end());
        embeddings.insert(embeddings.end(), projected.begin(), projected.end());
        embeddings.insert(embeddings.end(), separator_values_.begin(), separator_values_.end());
        embeddings.insert(embeddings.end(), text_embeddings.begin(), text_embeddings.end());
        const int64_t steps = kAudioTokens + 2 + static_cast<int64_t>(prompt_ids.size());
        if (steps + max_tokens > assets_->config.max_position_embeddings) {
            throw std::runtime_error("SAMSONE prompt and output exceed the model context");
        }
        debug::timing_log_scalar("samsone.prompt_ms", debug::elapsed_ms(prompt_start));

        const auto decode_start = std::chrono::steady_clock::now();
        auto logits = decoder_->prefill_embeddings_into_cache(embeddings, steps, steps + max_tokens, 128).logits;
        std::vector<int32_t> generated;
        generated.reserve(static_cast<size_t>(max_tokens));
        for (int64_t index = 0; index < max_tokens; ++index) {
            const auto best = std::max_element(logits.begin(), logits.end());
            const int32_t token = static_cast<int32_t>(std::distance(logits.begin(), best));
            if (token == assets_->config.eos_token_id) {
                break;
            }
            generated.push_back(token);
            logits = decoder_->decode_token(token).logits;
        }
        std::vector<int32_t> decoded;
        decoded.reserve(generated.size());
        for (int32_t token : generated) {
            decoded.push_back(pruned_to_original_.at(static_cast<size_t>(token)));
        }
        debug::timing_log_scalar("samsone.decode_ms", debug::elapsed_ms(decode_start));
        debug::trace_log_scalar("samsone.prompt_tokens", steps);
        debug::trace_log_scalar("samsone.generated_tokens", static_cast<int64_t>(generated.size()));
        return io::trim_ascii_whitespace(tokenizer_->decode(decoded, true));
    }

private:
    void load_token_map() {
        const auto map = assets_->resources.parse_json("token_map");
        pruned_to_original_.assign(static_cast<size_t>(assets_->config.vocab_size), -1);
        int32_t max_original = 0;
        for (const auto & [key, value] : map.as_object()) {
            const int32_t pruned = std::stoi(key);
            const int32_t original = static_cast<int32_t>(value.as_i64());
            if (pruned < 0 || static_cast<size_t>(pruned) >= pruned_to_original_.size() || original < 0) {
                throw std::runtime_error("SAMSONE token map contains an invalid ID");
            }
            pruned_to_original_[static_cast<size_t>(pruned)] = original;
            max_original = std::max(max_original, original);
        }
        original_to_pruned_.assign(static_cast<size_t>(max_original + 1), -1);
        for (size_t pruned = 0; pruned < pruned_to_original_.size(); ++pruned) {
            const int32_t original = pruned_to_original_[pruned];
            if (original < 0) {
                throw std::runtime_error("SAMSONE token map is incomplete");
            }
            original_to_pruned_[static_cast<size_t>(original)] = static_cast<int32_t>(pruned);
        }
    }

    void build_auxiliary_graphs() {
        ggml_.reset(ggml_init({8 * 1024 * 1024, nullptr, true}));
        if (!ggml_) {
            throw std::runtime_error("SAMSONE auxiliary graph context allocation failed");
        }
        core::ModuleBuildContext ctx{ggml_.get(), "samsone.auxiliary", execution_.backend_type()};
        auto projector_input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({kAudioTokens, kWhisperChannels}));
        projector_input_ = projector_input;
        auto residual = modules::LinearModule({kWhisperChannels, assets_->config.hidden_size, false}).build(ctx, projector_input, projector_linear1_);
        auto hidden = modules::GeluModule{}.build(ctx, residual);
        hidden = modules::LinearModule({assets_->config.hidden_size, assets_->config.hidden_size, false}).build(ctx, hidden, projector_linear2_);
        hidden = modules::AddModule{}.build(ctx, residual, hidden);
        projector_output_ = modules::LayerNormModule({assets_->config.hidden_size, 1.0e-5F, true, true}).build(ctx, hidden, projector_norm_).tensor;
        ggml_set_input(projector_input_.tensor);
        ggml_set_output(projector_output_);

        auto ids = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({kLookupCapacity}));
        lookup_input_ = ids;
        lookup_output_ = modules::EmbeddingModule({assets_->config.vocab_size, assets_->config.hidden_size}).build(ctx, ids, token_embedding_).tensor;
        ggml_set_input(lookup_input_.tensor);
        ggml_set_output(lookup_output_);

        separator_values_ = core::read_tensor_f32(separator_.tensor);
        graph_ = ggml_new_graph_custom(ggml_.get(), 512, false);
        ggml_build_forward_expand(graph_, projector_output_);
        ggml_build_forward_expand(graph_, lookup_output_);
        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend())));
        if (!allocator_ || !ggml_gallocr_alloc_graph(allocator_.get(), graph_)) {
            throw std::runtime_error("SAMSONE auxiliary graph allocation failed");
        }
        core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    std::vector<float> project(const std::vector<float> & pooled) {
        core::write_tensor_f32(projector_input_, pooled);
        if (core::compute_graph(execution_, graph_, plan_, "samsone.projector") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("SAMSONE projector graph compute failed");
        }
        return core::read_tensor_f32(projector_output_);
    }

    std::vector<float> embed(const std::vector<int32_t> & ids) {
        if (ids.size() > kLookupCapacity) {
            throw std::runtime_error("SAMSONE prompt is too long");
        }
        std::vector<int32_t> padded(kLookupCapacity, 0);
        std::copy(ids.begin(), ids.end(), padded.begin());
        core::write_tensor_i32(lookup_input_, padded);
        if (core::compute_graph(execution_, graph_, plan_, "samsone.embed") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("SAMSONE token embedding graph compute failed");
        }
        auto values = core::read_tensor_f32(lookup_output_);
        values.resize(ids.size() * static_cast<size_t>(assets_->config.hidden_size));
        return values;
    }

    void release_auxiliary_graphs() {
        if (graph_) {
            core::release_backend_graph_resources(execution_.backend(), graph_, true);
        }
        allocator_.reset();
        ggml_.reset();
        plan_.reset();
        graph_ = nullptr;
    }

    std::shared_ptr<const SamsoneAssets> assets_;
    core::ExecutionContext & execution_;
    core::BackendWeightStore store_;
    std::shared_ptr<const audio::MelSpectrogramFrontend> mel_;
    modules::WhisperFrontendComponent whisper_;
    std::shared_ptr<tokenizers::LlamaBpeTokenizer> tokenizer_;
    std::unique_ptr<modules::QwenCausalDecodeRuntime> decoder_;
    std::vector<int32_t> pruned_to_original_;
    std::vector<int32_t> original_to_pruned_;
    std::vector<float> separator_values_;
    core::TensorValue token_embedding_;
    core::TensorValue separator_;
    modules::LinearWeights projector_linear1_;
    modules::LinearWeights projector_linear2_;
    modules::NormWeights projector_norm_;
    std::unique_ptr<ggml_context, ContextDeleter> ggml_;
    std::unique_ptr<ggml_gallocr, AllocatorDeleter> allocator_;
    core::HostGraphPlan plan_;
    ggml_cgraph * graph_ = nullptr;
    core::TensorValue projector_input_;
    ggml_tensor * projector_output_ = nullptr;
    core::TensorValue lookup_input_;
    ggml_tensor * lookup_output_ = nullptr;
};

SamsoneRuntime::SamsoneRuntime(
    std::shared_ptr<const SamsoneAssets> assets,
    core::ExecutionContext & execution)
    : impl_(std::make_unique<Impl>(std::move(assets), execution)) {}

SamsoneRuntime::~SamsoneRuntime() = default;

std::string SamsoneRuntime::generate(
    const runtime::AudioBuffer & audio,
    const std::string & prompt,
    int64_t max_tokens,
    size_t threads) {
    return impl_->generate(audio, prompt, max_tokens, threads);
}

}  // namespace engine::models::samsone
