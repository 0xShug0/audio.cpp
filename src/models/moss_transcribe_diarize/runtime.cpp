#include "engine/models/moss_transcribe_diarize/runtime.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/json.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/speech_encoders/whisper_frontend.h"
#include "engine/framework/modules/transformers/qwen_causal_decode_runtime.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/runtime/partial_text.h"
#include "engine/framework/sampling/hf_sampler.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace engine::models::moss_transcribe_diarize {
namespace {

using Storage = assets::TensorStorageType;
namespace binding = modules::binding;
constexpr int64_t kHidden = 1024;
constexpr int64_t kAudioFrames = 375;
constexpr int64_t kLookupSteps = 512;
constexpr int64_t kChunkSamples = 480000;
constexpr int32_t kAudioToken = 151671;
constexpr int32_t kEosToken = 151645;

const char * const kDefaultPrompt =
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。";

struct ContextDeleter {
    void operator()(ggml_context * value) const { ggml_free(value); }
};
struct AllocatorDeleter {
    void operator()(ggml_gallocr_t value) const { ggml_gallocr_free(value); }
};

}  // namespace

class TranscribeRuntime::Impl {
public:
    Impl(const assets::ResourceBundle & resources, core::ExecutionContext & execution)
        : execution_(execution), source_(resources.open_tensor_source("weights")),
          store_(execution.backend(), execution.backend_type(), "moss_transcribe_diarize.weights", 2 * 1024 * 1024),
          mel_({16000, 400, 160, 80, audio::STFTFamily::Default}) {
        const auto config = resources.parse_json("config");
        if (io::json::require_string(config, "model_type") != "moss_transcribe_diarize") {
            throw std::runtime_error("MOSS-Transcribe-Diarize model_type mismatch");
        }
        const auto & text = config.require("text_config");
        const auto & encoder = config.require("audio_config");
        if (io::json::require_i64(text, "hidden_size") != 1024 ||
            io::json::require_i64(text, "num_hidden_layers") != 28 ||
            io::json::require_i64(encoder, "encoder_layers") != 24 ||
            io::json::require_i64(encoder, "encoder_attention_heads") != 16 ||
            io::json::require_i64(config, "audio_merge_size") != 4 ||
            io::json::require_i64(config, "audio_token_id") != kAudioToken) {
            throw std::runtime_error("Unsupported MOSS-Transcribe-Diarize architecture");
        }
        vocab_ = io::json::require_i64(text, "vocab_size");
        max_context_ = io::json::require_i64(text, "max_position_embeddings");
        const auto processor = resources.parse_json("processor_config");
        audio_rate_ = io::json::require_f32(processor, "audio_tokens_per_second");
        marker_seconds_ = io::json::require_i64(processor, "time_marker_every_seconds");
        time_markers_ = io::json::require_bool(processor, "enable_time_marker");
        tokenizers::LlamaBpeTokenizerSpec tokenizer_config;
        tokenizer_config.vocab_path = resources.require_file("vocab");
        tokenizer_config.merges_path = resources.require_file("merges");
        tokenizer_config.tokenizer_config_path = resources.require_file("tokenizer_config");
        tokenizer_config.tokenizer_json_path = resources.require_file("tokenizer_json");
        tokenizer_config.pre_type = tokenizers::LlamaBpePreTokenizer::Qwen2;
        tokenizer_ = tokenizers::load_llama_bpe_tokenizer(tokenizer_config);

        modules::WhisperFrontendComponentConfig whisper_config;
        whisper_config.name = "moss_transcribe_diarize.encoder";
        whisper_config.weight_context_bytes = 2 * 1024 * 1024;
        whisper_config.graph_context_bytes = 16 * 1024 * 1024;
        whisper_config.conv_weight_storage_type = Storage::F32;
        whisper_ = modules::WhisperFrontendComponent::load_openai_layout(
            source_, execution.config(), {80, 1500, 1024, 16, 24, 1e-5f}, whisper_config);

        modules::QwenCausalDecodeRuntimeConfig ar_config;
        ar_config.trace_name = "moss_transcribe_diarize.decoder";
        ar_config.prefill_graph_arena_bytes = 32 * 1024 * 1024;
        ar_config.decode_graph_arena_bytes = 32 * 1024 * 1024;
        ar_config.evict_cuda_graph_cache_on_release = true;
        auto & stack = ar_config.decoder.stack;
        stack.hidden_size = kHidden;
        stack.num_attention_heads = io::json::require_i64(text, "num_attention_heads");
        stack.num_key_value_heads = io::json::require_i64(text, "num_key_value_heads");
        stack.head_dim = io::json::require_i64(text, "head_dim");
        stack.intermediate_size = io::json::require_i64(text, "intermediate_size");
        stack.layers = 28;
        stack.rms_norm_eps = 1e-6f;
        stack.rope_theta = 1000000.f;
        stack.runtime.attention.prefill_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
        stack.runtime.attention.static_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
        stack.runtime.static_cache.update_mode = modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
        stack.runtime.static_cache.set_rows_mode = modules::QwenDecoderStaticCacheSetRowsMode::BackendViewOptimized;
        ar_config.decoder.static_cache_type = GGML_TYPE_F16;
        ar_config.decoder.logits_size = vocab_;

        modules::QwenCausalDecodeRuntimeWeights ar_weights;
        const std::string prefix = "model.language_model";
        ar_weights.token_embedding = store_.load_tensor(*source_, prefix + ".embed_tokens.weight", Storage::Native, {vocab_, kHidden});
        for (int64_t i = 0; i < stack.layers; ++i) {
            const std::string p = prefix + ".layers." + std::to_string(i);
            modules::QwenDecoderLayerWeights layer;
            layer.input_norm = binding::norm_weight_from_source(store_, *source_, p + ".input_layernorm", kHidden);
            layer.post_norm = binding::norm_weight_from_source(store_, *source_, p + ".post_attention_layernorm", kHidden);
            layer.q_norm = binding::norm_weight_from_source(store_, *source_, p + ".self_attn.q_norm", stack.head_dim);
            layer.k_norm = binding::norm_weight_from_source(store_, *source_, p + ".self_attn.k_norm", stack.head_dim);
            layer.self_attention.q_weight = store_.load_tensor(*source_, p + ".self_attn.q_proj.weight", Storage::Native,
                {stack.num_attention_heads * stack.head_dim, kHidden});
            layer.self_attention.k_weight = store_.load_tensor(*source_, p + ".self_attn.k_proj.weight", Storage::Native,
                {stack.num_key_value_heads * stack.head_dim, kHidden});
            layer.self_attention.v_weight = store_.load_tensor(*source_, p + ".self_attn.v_proj.weight", Storage::Native,
                {stack.num_key_value_heads * stack.head_dim, kHidden});
            layer.self_attention.out_weight = store_.load_tensor(*source_, p + ".self_attn.o_proj.weight", Storage::Native,
                {kHidden, stack.num_attention_heads * stack.head_dim});
            layer.mlp.gate_proj = binding::linear_from_source(store_, *source_, p + ".mlp.gate_proj", Storage::Native,
                stack.intermediate_size, kHidden, false);
            layer.mlp.up_proj = binding::linear_from_source(store_, *source_, p + ".mlp.up_proj", Storage::Native,
                stack.intermediate_size, kHidden, false);
            layer.mlp.down_proj = binding::linear_from_source(store_, *source_, p + ".mlp.down_proj", Storage::Native,
                kHidden, stack.intermediate_size, false);
            ar_weights.stack.layers.push_back(std::move(layer));
        }
        ar_weights.final_norm = binding::norm_weight_from_source(store_, *source_, prefix + ".norm", kHidden);
        ar_weights.lm_head = modules::LinearWeights{ar_weights.token_embedding, std::nullopt};
        const auto first = binding::linear_from_source(store_, *source_, "model.vq_adaptor.layers.0", Storage::Native, kHidden, 4096, true);
        const auto second = binding::linear_from_source(store_, *source_, "model.vq_adaptor.layers.2", Storage::Native, kHidden, kHidden, true);
        const auto norm = binding::norm_from_source(store_, *source_, "model.vq_adaptor.layers.3", kHidden);
        store_.upload();
        source_->release_storage();
        ar_ = std::make_unique<modules::QwenCausalDecodeRuntime>(execution_, ar_config, ar_weights);

        ctx_.reset(ggml_init({2 * 1024 * 1024, nullptr, true}));
        if (!ctx_) {
            throw std::runtime_error("MOSS-Transcribe-Diarize adaptor context allocation failed");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "moss_transcribe_diarize", execution.backend_type()};
        adaptor_input_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, kAudioFrames, 4096}));
        ggml_set_input(adaptor_input_.tensor);
        auto projected = modules::LinearModule({4096, kHidden, true}).build(ctx, adaptor_input_, first);
        projected = modules::SiluModule{}.build(ctx, projected);
        projected = modules::LinearModule({kHidden, kHidden, true}).build(ctx, projected, second);
        adaptor_output_ = modules::LayerNormModule({kHidden, 1e-6f, true, true}).build(ctx, projected, norm);
        ggml_set_output(adaptor_output_.tensor);
        adaptor_graph_ = ggml_new_graph_custom(ctx_.get(), 256, false);
        ggml_build_forward_expand(adaptor_graph_, adaptor_output_.tensor);
        adaptor_allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!ggml_gallocr_alloc_graph(adaptor_allocator_.get(), adaptor_graph_)) {
            throw std::runtime_error("MOSS-Transcribe-Diarize adaptor allocation failed");
        }
        core::prepare_host_graph_plan(execution_, adaptor_graph_, adaptor_plan_);
        lookup_input_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({kLookupSteps}));
        ggml_set_input(lookup_input_.tensor);
        lookup_output_ = modules::EmbeddingModule({vocab_, kHidden}).build(ctx, lookup_input_, ar_weights.token_embedding);
        ggml_set_output(lookup_output_.tensor);
        lookup_graph_ = ggml_new_graph_custom(ctx_.get(), 32, false);
        ggml_build_forward_expand(lookup_graph_, lookup_output_.tensor);
        lookup_allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!ggml_gallocr_alloc_graph(lookup_allocator_.get(), lookup_graph_)) {
            throw std::runtime_error("MOSS-Transcribe-Diarize embedding allocation failed");
        }
        core::prepare_host_graph_plan(execution_, lookup_graph_, lookup_plan_);
    }

    ~Impl() {
        core::release_backend_graph_resources(execution_.backend(), adaptor_graph_, true);
        core::release_backend_graph_resources(execution_.backend(), lookup_graph_, true);
    }

    void start(const runtime::AudioBuffer & audio, const std::string & instruction, int64_t max_tokens) {
        reset();
        const auto started = std::chrono::steady_clock::now();
        auto samples = audio::convert_interleaved_audio_to_mono_linear_resampled(
            audio.samples, audio.sample_rate, audio.channels, audio.sample_rate);
        if (audio.sample_rate != 16000) {
            audio::SoxrResampleOptions options;
            options.output_length_policy = audio::SoxrOutputLengthPolicy::ExactExpected;
            options.require_full_input = true;
            auto resampled = audio::try_resample_mono_soxr(samples, audio.sample_rate, 16000, options);
            if (!resampled) {
                throw std::runtime_error("MOSS-Transcribe-Diarize requires SOXR for non-16-kHz audio");
            }
            samples = std::move(*resampled);
        }
        if (samples.empty() || max_tokens <= 0) {
            throw std::runtime_error("MOSS-Transcribe-Diarize requires nonempty audio and positive max_tokens");
        }
        const int64_t audio_tokens = (static_cast<int64_t>(samples.size()) + 1279) / 1280;
        auto ids = tokenizer_->encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|audio_start|>");
        std::vector<size_t> audio_positions;
        audio_positions.reserve(audio_tokens);
        const int64_t marker_stride = static_cast<int64_t>(audio_rate_ * marker_seconds_);
        const int64_t markers = time_markers_ && marker_seconds_ > 0 && marker_stride > 0
            ? static_cast<int64_t>(audio_tokens / audio_rate_) / marker_seconds_ : 0;
        for (int64_t token = 0; token < audio_tokens; ++token) {
            audio_positions.push_back(ids.size());
            ids.push_back(kAudioToken);
            if (markers > 0 && (token + 1) % marker_stride == 0 && (token + 1) / marker_stride <= markers) {
                const auto marker = tokenizer_->encode(std::to_string((token + 1) / marker_stride * marker_seconds_));
                ids.insert(ids.end(), marker.begin(), marker.end());
            }
        }
        const auto suffix = tokenizer_->encode(std::string("<|audio_end|>\n") +
            (instruction.empty() ? kDefaultPrompt : instruction) + "<|im_end|>\n<|im_start|>assistant\n");
        ids.insert(ids.end(), suffix.begin(), suffix.end());
        if (static_cast<int64_t>(ids.size()) + max_tokens > max_context_) {
            throw std::runtime_error("MOSS-Transcribe-Diarize audio/prompt plus max_tokens exceeds context");
        }
        std::vector<float> embeddings(ids.size() * kHidden);
        std::vector<int32_t> block(kLookupSteps, 0);
        for (size_t offset = 0; offset < ids.size(); offset += kLookupSteps) {
            const size_t count = std::min<size_t>(kLookupSteps, ids.size() - offset);
            std::copy_n(ids.begin() + offset, count, block.begin());
            core::write_tensor_i32(lookup_input_, block);
            if (core::compute_graph(execution_, lookup_graph_, lookup_plan_) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("MOSS-Transcribe-Diarize embedding compute failed");
            }
            const auto values = core::read_tensor_f32(lookup_output_.tensor);
            std::copy_n(values.begin(), count * kHidden, embeddings.begin() + offset * kHidden);
        }
        std::vector<float> chunk(kChunkSamples, 0.f);
        size_t audio_offset = 0;
        for (size_t offset = 0; offset < samples.size(); offset += kChunkSamples) {
            const size_t count = std::min<size_t>(kChunkSamples, samples.size() - offset);
            std::fill(chunk.begin(), chunk.end(), 0.f);
            std::copy_n(samples.begin() + offset, count, chunk.begin());
            auto mel = mel_.compute(chunk, execution_.config().threads);
            const auto encoded = whisper_.encode_log_mel(mel.values);
            core::write_tensor_f32(adaptor_input_, encoded);
            if (core::compute_graph(execution_, adaptor_graph_, adaptor_plan_) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("MOSS-Transcribe-Diarize adaptor compute failed");
            }
            const auto adapted = core::read_tensor_f32(adaptor_output_.tensor);
            const size_t tokens = (count + 1279) / 1280;
            for (size_t t = 0; t < tokens; ++t) {
                std::copy_n(adapted.begin() + t * kHidden, kHidden,
                    embeddings.begin() + audio_positions.at(audio_offset++) * kHidden);
            }
        }
        debug::timing_log_scalar("moss_transcribe_diarize.frontend_ms", debug::elapsed_ms(started));
        const auto prefill_started = std::chrono::steady_clock::now();
        logits_ = ar_->prefill_embeddings_into_cache(embeddings, static_cast<int64_t>(ids.size()),
            static_cast<int64_t>(ids.size()) + max_tokens, 128).logits;
        debug::timing_log_scalar("moss_transcribe_diarize.prefill_ms", debug::elapsed_ms(prefill_started));
        max_tokens_ = max_tokens;
        active_ = true;
        debug::trace_log_scalar("moss_transcribe_diarize.prompt_tokens", static_cast<int64_t>(ids.size()));
    }

    std::optional<std::string> next_text() {
        if (!active_) {
            return std::nullopt;
        }
        while (static_cast<int64_t>(generated_.size()) < max_tokens_) {
            if (!generated_.empty()) {
                logits_ = ar_->decode_token(generated_.back()).logits;
            }
            const int32_t token = sampling::HfLogitsProcessor::argmax(logits_.data(), logits_.size(), "MOSS-Transcribe-Diarize");
            if (token == kEosToken) {
                active_ = false;
                debug::trace_log_scalar("moss_transcribe_diarize.generated_tokens", static_cast<int64_t>(generated_.size()));
                if (partials_.published().size() != decoded_text_.size()) {
                    throw std::runtime_error("MOSS-Transcribe-Diarize ended with incomplete UTF-8 text");
                }
                return std::nullopt;
            }
            generated_.push_back(token);
            // BPE decoding is byte-concatenative; the publisher holds incomplete UTF-8 tails.
            decoded_text_ += tokenizer_->decode({token}, true);
            auto delta = partials_.publish(decoded_text_);
            if (!delta.empty()) {
                return delta;
            }
        }
        throw std::runtime_error("MOSS-Transcribe-Diarize reached max_tokens before EOS; increase max_tokens");
    }

    void reset() {
        active_ = false;
        generated_.clear();
        logits_.clear();
        decoded_text_.clear();
        partials_.reset();
        max_tokens_ = 0;
    }

private:
    core::ExecutionContext & execution_;
    std::shared_ptr<const assets::TensorSource> source_;
    core::BackendWeightStore store_;
    audio::WhisperLogMelExtractor mel_;
    modules::WhisperFrontendComponent whisper_;
    std::shared_ptr<tokenizers::LlamaBpeTokenizer> tokenizer_;
    std::unique_ptr<modules::QwenCausalDecodeRuntime> ar_;
    int64_t vocab_ = 0;
    bool active_ = false;
    int64_t max_tokens_ = 0;
    std::string decoded_text_;
    runtime::PartialTextPublisher partials_;
    std::vector<int32_t> generated_;
    std::vector<float> logits_;
    int64_t max_context_ = 0;
    float audio_rate_ = 12.5f;
    int64_t marker_seconds_ = 0;
    bool time_markers_ = false;
    std::unique_ptr<ggml_context, ContextDeleter> ctx_;
    std::unique_ptr<ggml_gallocr, AllocatorDeleter> adaptor_allocator_;
    std::unique_ptr<ggml_gallocr, AllocatorDeleter> lookup_allocator_;
    core::HostGraphPlan adaptor_plan_;
    core::HostGraphPlan lookup_plan_;
    core::TensorValue adaptor_input_, adaptor_output_, lookup_input_, lookup_output_;
    ggml_cgraph * adaptor_graph_ = nullptr;
    ggml_cgraph * lookup_graph_ = nullptr;
};

TranscribeRuntime::TranscribeRuntime(const assets::ResourceBundle & resources, core::ExecutionContext & execution)
    : impl_(std::make_unique<Impl>(resources, execution)) {}
TranscribeRuntime::~TranscribeRuntime() = default;
std::string TranscribeRuntime::transcribe(const runtime::AudioBuffer & audio, const std::string & instruction, int64_t max_tokens) {
    start(audio, instruction, max_tokens);
    std::string text;
    while (auto delta = next_text()) {
        text += *delta;
    }
    return text;
}
void TranscribeRuntime::start(const runtime::AudioBuffer & audio, const std::string & instruction, int64_t max_tokens) {
    impl_->start(audio, instruction, max_tokens);
}
std::optional<std::string> TranscribeRuntime::next_text() { return impl_->next_text(); }
void TranscribeRuntime::reset() { impl_->reset(); }

}  // namespace engine::models::moss_transcribe_diarize
