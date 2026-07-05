#include "engine/models/index_tts2/qwen_emotion.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/runtime/kv_cache.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::index_tts2 {
namespace {

namespace binding = engine::modules::binding;
namespace core = engine::core;
namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

constexpr int64_t kHidden = 1024;
constexpr int64_t kIntermediate = 3072;
constexpr int64_t kLayers = 28;
constexpr int64_t kAttentionHeads = 16;
constexpr int64_t kKvHeads = 8;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kVocab = 151936;
constexpr float kRmsEps = 1.0e-6F;
constexpr float kRopeTheta = 1000000.0F;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

modules::QwenDecoderStackConfig qwen_config() {
    return {
        kHidden,
        kAttentionHeads,
        kKvHeads,
        kHeadDim,
        kIntermediate,
        kLayers,
        kRmsEps,
        kRopeTheta,
        GGML_PREC_F32,
        GGML_PREC_DEFAULT,
    };
}

std::vector<float> causal_mask(int64_t steps) {
    std::vector<float> mask(static_cast<size_t>(steps * steps), 0.0F);
    for (int64_t q = 0; q < steps; ++q) {
        for (int64_t k = q + 1; k < steps; ++k) {
            mask[static_cast<size_t>(q * steps + k)] = -std::numeric_limits<float>::infinity();
        }
    }
    return mask;
}

int32_t argmax_index(const std::vector<float> & logits) {
    if (logits.empty()) {
        throw std::runtime_error("IndexTTS2 Qwen emotion argmax requires non-empty logits");
    }
    return static_cast<int32_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

std::string chat_prompt_text(const std::string & text) {
    return "System: 文本情感分类<|endoftext|>\n"
           "Human: " + text + "<|endoftext|>\n"
           "Assistant:";
}

float clamp_emotion(float value) {
    return std::clamp(value, 0.0F, 1.2F);
}

float parse_named_score(const std::string & content, const std::string & key) {
    const std::regex pattern("\"?" + key + "\"?\\s*:\\s*([0-9]+(?:\\.[0-9]+)?)");
    std::smatch match;
    if (!std::regex_search(content, match, pattern)) {
        return 0.0F;
    }
    return clamp_emotion(std::stof(match[1].str()));
}

IndexTTS2EmotionVector convert_emotion_json(const std::string & content, const std::string & source_text) {
    IndexTTS2EmotionVector out;
    out.values = {
        parse_named_score(content, "高兴"),
        parse_named_score(content, "愤怒"),
        parse_named_score(content, "悲伤"),
        parse_named_score(content, "恐惧"),
        parse_named_score(content, "反感"),
        parse_named_score(content, "低落"),
        parse_named_score(content, "惊讶"),
        parse_named_score(content, "自然"),
    };
    std::string lower = source_text;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (lower.find("低落") != std::string::npos ||
        lower.find("melancholy") != std::string::npos ||
        lower.find("melancholic") != std::string::npos ||
        lower.find("depression") != std::string::npos ||
        lower.find("depressed") != std::string::npos ||
        lower.find("gloomy") != std::string::npos) {
        std::swap(out.values[2], out.values[5]);
    }
    const bool all_zero = std::all_of(out.values.begin(), out.values.end(), [](float value) { return value <= 0.0F; });
    if (all_zero) {
        out.values[7] = 1.0F;
    }
    return out;
}

engine::modules::QwenDecoderLayerWeights load_qwen_layer(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    int64_t layer_index,
    engine::assets::TensorStorageType storage_type) {
    const std::string prefix = "model.layers." + std::to_string(layer_index);
    engine::modules::QwenDecoderLayerWeights layer;
    layer.input_norm = binding::norm_weight_from_source(store, source, prefix + ".input_layernorm", kHidden);
    layer.self_attention.q_weight = store.load_tensor(
        source,
        prefix + ".self_attn.q_proj.weight",
        storage_type,
        {kAttentionHeads * kHeadDim, kHidden});
    layer.self_attention.k_weight = store.load_tensor(
        source,
        prefix + ".self_attn.k_proj.weight",
        storage_type,
        {kKvHeads * kHeadDim, kHidden});
    layer.self_attention.v_weight = store.load_tensor(
        source,
        prefix + ".self_attn.v_proj.weight",
        storage_type,
        {kKvHeads * kHeadDim, kHidden});
    layer.self_attention.out_weight = store.load_tensor(
        source,
        prefix + ".self_attn.o_proj.weight",
        storage_type,
        {kHidden, kAttentionHeads * kHeadDim});
    layer.q_norm = binding::norm_weight_from_source(store, source, prefix + ".self_attn.q_norm", kHeadDim);
    layer.k_norm = binding::norm_weight_from_source(store, source, prefix + ".self_attn.k_norm", kHeadDim);
    layer.post_norm = binding::norm_weight_from_source(store, source, prefix + ".post_attention_layernorm", kHidden);
    layer.mlp.gate_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.gate_proj",
        storage_type,
        kIntermediate,
        kHidden,
        false);
    layer.mlp.up_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.up_proj",
        storage_type,
        kIntermediate,
        kHidden,
        false);
    layer.mlp.down_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.down_proj",
        storage_type,
        kHidden,
        kIntermediate,
        false);
    return layer;
}

}  // namespace

std::shared_ptr<const IndexTTS2QwenEmotionWeights> load_index_tts2_qwen_emotion_weights(
    const IndexTTS2Assets & assets,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    engine::assets::TensorStorageType storage_type,
    size_t weight_context_bytes) {
    if (assets.qwen_emotion_weights == nullptr) {
        throw std::runtime_error("IndexTTS2 Qwen emotion requires tensor source");
    }
    auto weights = std::make_shared<IndexTTS2QwenEmotionWeights>();
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        backend,
        backend_type,
        "index_tts2.qwen_emotion.weights",
        weight_context_bytes);

    const auto & source = *assets.qwen_emotion_weights;
    weights->token_embedding = weights->store->load_tensor(
        source,
        "model.embed_tokens.weight",
        storage_type,
        {kVocab, kHidden});
    weights->decoder.layers.reserve(static_cast<size_t>(kLayers));
    for (int64_t layer = 0; layer < kLayers; ++layer) {
        weights->decoder.layers.push_back(load_qwen_layer(*weights->store, source, layer, storage_type));
    }
    weights->final_norm = binding::norm_weight_from_source(*weights->store, source, "model.norm", kHidden);
    weights->store->upload();
    assets.qwen_emotion_weights->release_storage();
    return weights;
}

IndexTTS2QwenEmotionTokenizer::IndexTTS2QwenEmotionTokenizer(std::shared_ptr<const IndexTTS2Assets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("IndexTTS2 Qwen emotion tokenizer requires assets");
    }
    engine::tokenizers::LlamaBpeTokenizerSpec spec;
    spec.vocab_path = assets->paths.qwen_emotion_vocab_path;
    spec.merges_path = assets->paths.qwen_emotion_merges_path;
    spec.tokenizer_config_path = assets->paths.qwen_emotion_tokenizer_config_path;
    spec.tokenizer_json_path = assets->paths.qwen_emotion_tokenizer_json_path;
    spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen2;
    tokenizer_ = engine::tokenizers::load_llama_bpe_tokenizer(spec);
    if (const auto id = tokenizer_->find_token_id("<|endoftext|>"); id.has_value()) {
        eos_token_id_ = *id;
    }
    if (const auto id = tokenizer_->find_token_id("</think>"); id.has_value()) {
        think_end_token_id_ = *id;
    }
}

std::vector<int32_t> IndexTTS2QwenEmotionTokenizer::encode_chat_prompt(const std::string & text) const {
    return tokenizer_->encode(chat_prompt_text(text), true);
}

std::string IndexTTS2QwenEmotionTokenizer::decode(const std::vector<int32_t> & token_ids, bool skip_special_tokens) const {
    return tokenizer_->decode(token_ids, skip_special_tokens);
}

int32_t IndexTTS2QwenEmotionTokenizer::eos_token_id() const noexcept {
    return eos_token_id_;
}

int32_t IndexTTS2QwenEmotionTokenizer::think_end_token_id() const noexcept {
    return think_end_token_id_;
}

class IndexTTS2QwenEmotionRuntime::PrefillGraph {
public:
    PrefillGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const IndexTTS2QwenEmotionWeights> weights,
        int64_t prompt_steps,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          prompt_steps_(prompt_steps) {
        if (prompt_steps_ <= 0) {
            throw std::runtime_error("IndexTTS2 Qwen emotion prefill graph requires prompt tokens");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 Qwen emotion prefill graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "index_tts2.qwen_emotion.prefill", execution_.backend_type()};
        token_ids_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1, prompt_steps_})).tensor;
        positions_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({prompt_steps_})).tensor;
        mask_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, prompt_steps_, prompt_steps_})).tensor;
        auto x = modules::EmbeddingModule({kVocab, kHidden}).build(
            ctx,
            core::wrap_tensor(token_ids_, core::TensorShape::from_dims({1, prompt_steps_}), GGML_TYPE_I32),
            weights_->token_embedding);
        auto outputs = modules::QwenDecoderStackModule(qwen_config()).build(
            ctx,
            x,
            core::wrap_tensor(positions_, core::TensorShape::from_dims({prompt_steps_}), GGML_TYPE_I32),
            weights_->decoder,
            std::nullopt,
            core::wrap_tensor(mask_, core::TensorShape::from_dims({1, 1, prompt_steps_, prompt_steps_}), GGML_TYPE_F32));
        for (const auto & layer : outputs.state.layers) {
            keys_.push_back(layer.key->tensor);
            values_.push_back(layer.value->tensor);
        }
        build_step_source_views();
        auto last = modules::SliceModule({1, prompt_steps_ - 1, 1}).build(ctx, outputs.output);
        last = modules::RMSNormModule({kHidden, kRmsEps, true, false}).build(ctx, last, weights_->final_norm);
        auto logits = modules::LinearModule({kHidden, kVocab, false}).build(ctx, last, {weights_->token_embedding, std::nullopt});
        logits_ = logits.tensor;
        ggml_set_output(logits_);
        graph_ = ggml_new_graph_custom(ctx_.get(), static_cast<size_t>(std::max<int64_t>(131072, prompt_steps_ * 4096)), false);
        ggml_build_forward_expand(graph_, logits_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), execution_.backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate IndexTTS2 Qwen emotion prefill graph");
        }
        std::vector<int32_t> positions(static_cast<size_t>(prompt_steps_));
        for (int64_t i = 0; i < prompt_steps_; ++i) {
            positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        core::write_tensor_i32(core::wrap_tensor(positions_, core::TensorShape::from_dims({prompt_steps_}), GGML_TYPE_I32), positions);
        const auto mask = causal_mask(prompt_steps_);
        core::write_tensor_f32(core::wrap_tensor(mask_, core::TensorShape::from_dims({1, 1, prompt_steps_, prompt_steps_}), GGML_TYPE_F32), mask);
        debug::timing_log_scalar("index_tts2.qwen_emotion.prefill.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("index_tts2.qwen_emotion.prefill.prompt_tokens", prompt_steps_);
    }

    ~PrefillGraph() {
        core::release_backend_graph_resources(execution_.backend(), graph_);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool matches(const IndexTTS2QwenEmotionWeights & weights, ggml_backend_t backend, int64_t prompt_steps) const noexcept {
        return weights_.get() == &weights && execution_.backend() == backend && prompt_steps_ == prompt_steps;
    }

    std::vector<float> run(const std::vector<int32_t> & prompt_ids) {
        if (static_cast<int64_t>(prompt_ids.size()) != prompt_steps_) {
            throw std::runtime_error("IndexTTS2 Qwen emotion prompt length mismatch");
        }
        auto timing_start = Clock::now();
        ggml_backend_tensor_set(token_ids_, prompt_ids.data(), 0, prompt_ids.size() * sizeof(int32_t));
        debug::timing_log_scalar("index_tts2.qwen_emotion.prefill.input_upload_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        timing_start = Clock::now();
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph_, nullptr, "IndexTTS2 Qwen emotion prefill");
        ggml_backend_synchronize(execution_.backend());
        debug::timing_log_scalar("index_tts2.qwen_emotion.prefill.graph.compute_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("IndexTTS2 Qwen emotion prefill graph compute failed");
        }
        timing_start = Clock::now();
        auto logits = core::read_tensor_f32(logits_);
        debug::timing_log_scalar("index_tts2.qwen_emotion.prefill.output_read_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        return logits;
    }

    int64_t prompt_steps() const noexcept {
        return prompt_steps_;
    }

    size_t layer_count() const noexcept {
        return keys_.size();
    }

    ggml_tensor * key_step_source(size_t step, size_t layer) const {
        return key_step_sources_.at(step).at(layer);
    }

    ggml_tensor * value_step_source(size_t step, size_t layer) const {
        return value_step_sources_.at(step).at(layer);
    }

private:
    void build_step_source_views() {
        const int64_t step_elems = kKvHeads * kHeadDim;
        key_step_sources_.assign(static_cast<size_t>(prompt_steps_), {});
        value_step_sources_.assign(static_cast<size_t>(prompt_steps_), {});
        for (int64_t step = 0; step < prompt_steps_; ++step) {
            const size_t byte_offset = static_cast<size_t>(step * step_elems) * sizeof(float);
            auto & key_slot = key_step_sources_[static_cast<size_t>(step)];
            auto & value_slot = value_step_sources_[static_cast<size_t>(step)];
            key_slot.reserve(keys_.size());
            value_slot.reserve(values_.size());
            for (size_t layer = 0; layer < keys_.size(); ++layer) {
                key_slot.push_back(ggml_view_1d(ctx_.get(), keys_[layer], step_elems, byte_offset));
                value_slot.push_back(ggml_view_1d(ctx_.get(), values_[layer], step_elems, byte_offset));
            }
        }
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const IndexTTS2QwenEmotionWeights> weights_;
    int64_t prompt_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_ids_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * mask_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    std::vector<ggml_tensor *> keys_;
    std::vector<ggml_tensor *> values_;
    std::vector<std::vector<ggml_tensor *>> key_step_sources_;
    std::vector<std::vector<ggml_tensor *>> value_step_sources_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

class IndexTTS2QwenEmotionRuntime::DecodeGraph {
public:
    DecodeGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const IndexTTS2QwenEmotionWeights> weights,
        int64_t cache_steps,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          cache_steps_(cache_steps) {
        if (cache_steps_ <= 0) {
            throw std::runtime_error("IndexTTS2 Qwen emotion decode graph requires cache capacity");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 Qwen emotion decode graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "index_tts2.qwen_emotion.decode", execution_.backend_type()};
        token_id_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1, 1})).tensor;
        position_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1})).tensor;
        mask_ = core::make_tensor(ctx, GGML_TYPE_F16, core::TensorShape::from_dims({1, 1, 1, cache_steps_ + 1})).tensor;
        graph_ = ggml_new_graph_custom(ctx_.get(), 131072, false);
        std::vector<core::TensorValue> cache_keys;
        std::vector<core::TensorValue> cache_values;
        auto x = modules::EmbeddingModule({kVocab, kHidden}).build(
            ctx,
            core::wrap_tensor(token_id_, core::TensorShape::from_dims({1, 1}), GGML_TYPE_I32),
            weights_->token_embedding);
        const auto cfg = qwen_config();
        const auto mask = core::wrap_tensor(mask_, core::TensorShape::from_dims({1, 1, 1, cache_steps_ + 1}), GGML_TYPE_F16);
        for (const auto & layer : weights_->decoder.layers) {
            cache_keys.push_back(core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, cache_steps_ + 1, kKvHeads, kHeadDim})));
            cache_values.push_back(core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, cache_steps_ + 1, kKvHeads, kHeadDim})));
            auto out = modules::QwenDecoderLayerModule({
                kHidden,
                kAttentionHeads,
                kKvHeads,
                kHeadDim,
                kIntermediate,
                kRmsEps,
                kRopeTheta,
                GGML_PREC_F32,
                GGML_PREC_DEFAULT,
            }).build_with_static_cache_tail(
                ctx,
                graph_,
                x,
                core::wrap_tensor(position_, core::TensorShape::from_dims({1}), GGML_TYPE_I32),
                layer,
                cache_keys.back(),
                cache_values.back(),
                mask);
            x = out.output;
            key_sources_.push_back(ggml_view_1d(ctx_.get(), out.key.tensor, kKvHeads * kHeadDim, 0));
            value_sources_.push_back(ggml_view_1d(ctx_.get(), out.value.tensor, kKvHeads * kHeadDim, 0));
        }
        (void)cfg;
        kv_cache_ = engine::runtime::TransformerKVCache(cache_steps_ + 1, kKvHeads * kHeadDim, std::move(cache_keys), std::move(cache_values));
        build_transfer_views();
        x = modules::RMSNormModule({kHidden, kRmsEps, true, false}).build(ctx, x, weights_->final_norm);
        auto logits = modules::LinearModule({kHidden, kVocab, false}).build(ctx, x, {weights_->token_embedding, std::nullopt});
        logits_ = logits.tensor;
        ggml_set_output(logits_);
        ggml_build_forward_expand(graph_, logits_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), execution_.backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate IndexTTS2 Qwen emotion decode graph");
        }
        mask_values_.assign(static_cast<size_t>(cache_steps_ + 1), ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        debug::timing_log_scalar("index_tts2.qwen_emotion.decode.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("index_tts2.qwen_emotion.decode.cache_steps", cache_steps_);
    }

    ~DecodeGraph() {
        core::release_backend_graph_resources(execution_.backend(), graph_);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool can_run(const IndexTTS2QwenEmotionWeights & weights, ggml_backend_t backend, int64_t required_steps) const noexcept {
        return weights_.get() == &weights && execution_.backend() == backend && cache_steps_ >= required_steps;
    }

    void import_state(const PrefillGraph & prefill) {
        if (prefill.layer_count() != key_sources_.size() || prefill.prompt_steps() > cache_steps_) {
            throw std::runtime_error("IndexTTS2 Qwen emotion decode prefill state shape mismatch");
        }
        kv_cache_.retain_prefix(0);
        for (int64_t step = 0; step < prefill.prompt_steps(); ++step) {
            const size_t slot = static_cast<size_t>(step);
            for (size_t layer = 0; layer < key_sources_.size(); ++layer) {
                ggml_backend_tensor_copy(prefill.key_step_source(slot, layer), key_destinations_[slot][layer]);
                ggml_backend_tensor_copy(prefill.value_step_source(slot, layer), value_destinations_[slot][layer]);
            }
        }
        kv_cache_.advance_after_direct_append(prefill.prompt_steps());
    }

    std::vector<float> run_step(int32_t token) {
        if (kv_cache_.valid_steps() >= cache_steps_) {
            throw std::runtime_error("IndexTTS2 Qwen emotion decode cache exhausted");
        }
        ggml_backend_tensor_set(token_id_, &token, 0, sizeof(int32_t));
        const int32_t position = static_cast<int32_t>(kv_cache_.current_end());
        ggml_backend_tensor_set(position_, &position, 0, sizeof(int32_t));
        std::fill(mask_values_.begin(), mask_values_.end(), ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        for (int64_t i = 0; i < kv_cache_.valid_steps(); ++i) {
            mask_values_[static_cast<size_t>(i)] = ggml_fp32_to_fp16(0.0F);
        }
        mask_values_[static_cast<size_t>(cache_steps_)] = ggml_fp32_to_fp16(0.0F);
        ggml_backend_tensor_set(mask_, mask_values_.data(), 0, mask_values_.size() * sizeof(ggml_fp16_t));
        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph_, nullptr, "IndexTTS2 Qwen emotion decode");
        ggml_backend_synchronize(execution_.backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("IndexTTS2 Qwen emotion decode graph compute failed");
        }
        auto logits = core::read_tensor_f32(logits_);
        const size_t dst_slot = static_cast<size_t>(kv_cache_.valid_steps());
        for (size_t layer = 0; layer < key_sources_.size(); ++layer) {
            ggml_backend_tensor_copy(key_sources_[layer], key_destinations_[dst_slot][layer]);
            ggml_backend_tensor_copy(value_sources_[layer], value_destinations_[dst_slot][layer]);
        }
        kv_cache_.advance_after_direct_append(1);
        return logits;
    }

private:
    void build_transfer_views() {
        const int64_t step_elems = kKvHeads * kHeadDim;
        key_destinations_.assign(static_cast<size_t>(cache_steps_), {});
        value_destinations_.assign(static_cast<size_t>(cache_steps_), {});
        for (int64_t slot = 0; slot < cache_steps_; ++slot) {
            const size_t byte_offset = static_cast<size_t>(slot * step_elems) * sizeof(float);
            auto & key_slot = key_destinations_[static_cast<size_t>(slot)];
            auto & value_slot = value_destinations_[static_cast<size_t>(slot)];
            key_slot.reserve(key_sources_.size());
            value_slot.reserve(value_sources_.size());
            for (size_t layer = 0; layer < key_sources_.size(); ++layer) {
                key_slot.push_back(ggml_view_1d(ctx_.get(), kv_cache_.key_tensor(layer).tensor, step_elems, byte_offset));
                value_slot.push_back(ggml_view_1d(ctx_.get(), kv_cache_.value_tensor(layer).tensor, step_elems, byte_offset));
            }
        }
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const IndexTTS2QwenEmotionWeights> weights_;
    int64_t cache_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_id_ = nullptr;
    ggml_tensor * position_ = nullptr;
    ggml_tensor * mask_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    std::vector<ggml_tensor *> key_sources_;
    std::vector<ggml_tensor *> value_sources_;
    std::vector<std::vector<ggml_tensor *>> key_destinations_;
    std::vector<std::vector<ggml_tensor *>> value_destinations_;
    std::vector<ggml_fp16_t> mask_values_;
    engine::runtime::TransformerKVCache kv_cache_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

IndexTTS2QwenEmotionRuntime::IndexTTS2QwenEmotionRuntime(
    std::shared_ptr<const IndexTTS2Assets> assets,
    core::ExecutionContext & execution,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type)
    : assets_(std::move(assets)),
      execution_(&execution),
      prefill_graph_arena_bytes_(prefill_graph_arena_bytes),
      decode_graph_arena_bytes_(decode_graph_arena_bytes),
      tokenizer_(assets_) {
    if (assets_ == nullptr) {
        throw std::runtime_error("IndexTTS2 Qwen emotion runtime requires assets");
    }
    if (prefill_graph_arena_bytes_ == 0 || decode_graph_arena_bytes_ == 0) {
        throw std::runtime_error("IndexTTS2 Qwen emotion graph arenas must be non-zero");
    }
    weights_ = load_index_tts2_qwen_emotion_weights(
        *assets_,
        execution.backend(),
        execution.backend_type(),
        storage_type,
        weight_context_bytes);
}

IndexTTS2QwenEmotionRuntime::~IndexTTS2QwenEmotionRuntime() = default;

IndexTTS2EmotionVector IndexTTS2QwenEmotionRuntime::infer(const std::string & text, int64_t max_new_tokens) {
    if (execution_ == nullptr) {
        throw std::runtime_error("IndexTTS2 Qwen emotion runtime execution context is missing");
    }
    if (max_new_tokens <= 0) {
        throw std::runtime_error("IndexTTS2 Qwen emotion max_new_tokens must be positive");
    }
    const auto prompt_ids = tokenizer_.encode_chat_prompt(text);
    const int64_t prompt_steps = static_cast<int64_t>(prompt_ids.size());
    if (prefill_graph_ == nullptr || !prefill_graph_->matches(*weights_, execution_->backend(), prompt_steps)) {
        prefill_graph_.reset();
        prefill_graph_ = std::make_unique<PrefillGraph>(*execution_, weights_, prompt_steps, prefill_graph_arena_bytes_);
    }
    const auto prefill_logits = prefill_graph_->run(prompt_ids);
    const int64_t required_cache_steps = prompt_steps + max_new_tokens;
    if (decode_graph_ == nullptr || !decode_graph_->can_run(*weights_, execution_->backend(), required_cache_steps)) {
        decode_graph_.reset();
        decode_graph_ = std::make_unique<DecodeGraph>(*execution_, weights_, required_cache_steps, decode_graph_arena_bytes_);
    }
    decode_graph_->import_state(*prefill_graph_);

    std::vector<int32_t> generated;
    generated.reserve(static_cast<size_t>(max_new_tokens));
    int32_t token = argmax_index(prefill_logits);
    double decode_run_ms = 0.0;
    for (int64_t step = 0; step < max_new_tokens; ++step) {
        if (token == tokenizer_.eos_token_id()) {
            break;
        }
        generated.push_back(token);
        if (step + 1 >= max_new_tokens) {
            break;
        }
        const auto decode_start = Clock::now();
        token = argmax_index(decode_graph_->run_step(token));
        decode_run_ms += engine::debug::elapsed_ms(decode_start, Clock::now());
    }

    size_t start = 0;
    for (size_t i = generated.size(); i > 0; --i) {
        if (generated[i - 1] == tokenizer_.think_end_token_id()) {
            start = i;
            break;
        }
    }
    const std::vector<int32_t> answer(generated.begin() + static_cast<std::ptrdiff_t>(start), generated.end());
    const std::string content = tokenizer_.decode(answer, true);
    debug::timing_log_scalar("index_tts2.qwen_emotion.decode.run_ms", decode_run_ms);
    debug::trace_log_scalar("index_tts2.qwen_emotion.generated_tokens", generated.size());
    return convert_emotion_json(content, text);
}

void IndexTTS2QwenEmotionRuntime::release_graphs() {
    prefill_graph_.reset();
    decode_graph_.reset();
}

}  // namespace engine::models::index_tts2
