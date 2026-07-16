#include "engine/models/index_tts2/qwen_emotion.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/runtime/kv_cache.h"

#include <ggml-alloc.h>

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
    modules::QwenDecoderStackConfig config;
    config.hidden_size = kHidden;
    config.num_attention_heads = kAttentionHeads;
    config.num_key_value_heads = kKvHeads;
    config.head_dim = kHeadDim;
    config.intermediate_size = kIntermediate;
    config.layers = kLayers;
    config.rms_norm_eps = kRmsEps;
    config.rope_theta = kRopeTheta;
    config.attention_precision = GGML_PREC_F32;
    config.projection_precision = GGML_PREC_DEFAULT;
    return config;
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
    spec.vocab_path = assets->resources.require_file("qwen_emotion_vocab");
    spec.merges_path = assets->resources.require_file("qwen_emotion_merges");
    spec.tokenizer_config_path = assets->resources.require_file("qwen_emotion_tokenizer_config");
    spec.tokenizer_json_path = assets->resources.require_file("qwen_emotion_tokenizer");
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
    std::vector<int32_t> ids = tokenizer_->encode("System: 文本情感分类", true);
    ids.push_back(eos_token_id_);
    std::string user_text = "\nHuman: " + text;
    size_t trailing_spaces = 0;
    while (trailing_spaces < user_text.size() && user_text[user_text.size() - trailing_spaces - 1] == ' ') {
        ++trailing_spaces;
    }
    if (trailing_spaces > 0) {
        user_text.resize(user_text.size() - trailing_spaces);
    }
    auto user = tokenizer_->encode(user_text, true);
    ids.insert(ids.end(), user.begin(), user.end());
    if (trailing_spaces > 0) {
        const auto space_token = tokenizer_->find_token_id("Ġ");
        if (!space_token.has_value()) {
            throw std::runtime_error("IndexTTS2 Qwen emotion tokenizer missing space token");
        }
        ids.insert(ids.end(), trailing_spaces, *space_token);
    }
    ids.push_back(eos_token_id_);
    auto assistant = tokenizer_->encode("\nAssistant:", true);
    ids.insert(ids.end(), assistant.begin(), assistant.end());
    return ids;
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
        ggml_init_params input_params{64ull * 1024ull * 1024ull, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 Qwen emotion prefill input context");
        }
        ggml_init_params output_params{16ull * 1024ull * 1024ull, nullptr, true};
        output_ctx_.reset(ggml_init(output_params));
        if (output_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 Qwen emotion prefill output context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "index_tts2.qwen_emotion.prefill", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{
            input_ctx_.get(),
            "index_tts2.qwen_emotion.prefill.inputs",
            execution_.backend_type()};
        core::ModuleBuildContext output_ctx{
            output_ctx_.get(),
            "index_tts2.qwen_emotion.prefill.outputs",
            execution_.backend_type()};
        token_ids_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1, prompt_steps_})).tensor;
        positions_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({prompt_steps_})).tensor;
        mask_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, prompt_steps_, prompt_steps_})).tensor;
        ggml_set_input(token_ids_);
        ggml_set_input(positions_);
        ggml_set_input(mask_);
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
        graph_ = ggml_new_graph_custom(ctx_.get(), static_cast<size_t>(std::max<int64_t>(131072, prompt_steps_ * 4096)), false);
        for (const auto & layer : outputs.state.layers) {
            auto key = core::ensure_backend_addressable_layout(ctx, *layer.key);
            auto value = core::ensure_backend_addressable_layout(ctx, *layer.value);
            auto * key_output = core::make_tensor(output_ctx, GGML_TYPE_F32, key.shape).tensor;
            auto * value_output = core::make_tensor(output_ctx, GGML_TYPE_F32, value.shape).tensor;
            keys_.push_back(key_output);
            values_.push_back(value_output);
            ggml_build_forward_expand(graph_, ggml_cpy(ctx_.get(), key.tensor, key_output));
            ggml_build_forward_expand(graph_, ggml_cpy(ctx_.get(), value.tensor, value_output));
        }
        auto last = modules::SliceModule({1, prompt_steps_ - 1, 1}).build(ctx, outputs.output);
        last = modules::RMSNormModule({kHidden, kRmsEps, true, false}).build(ctx, last, weights_->final_norm);
        auto logits = modules::LinearModule({kHidden, kVocab, false}).build(ctx, last, {weights_->token_embedding, std::nullopt});
        auto flat_logits = core::reshape_tensor(
            ctx,
            core::ensure_backend_addressable_layout(ctx, logits),
            core::TensorShape::from_dims({1, kVocab}));
        auto * next_token_source = ggml_argmax(ctx.ggml, flat_logits.tensor);
        next_token_ = core::make_tensor(output_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1})).tensor;
        ggml_set_output(next_token_);
        ggml_build_forward_expand(graph_, ggml_cpy(ctx_.get(), next_token_source, next_token_));
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate IndexTTS2 Qwen emotion prefill input buffer");
        }
        output_buffer_ = ggml_backend_alloc_ctx_tensors(output_ctx_.get(), execution_.backend());
        if (output_buffer_ == nullptr) {
            clear_graph();
            throw std::runtime_error("failed to allocate IndexTTS2 Qwen emotion prefill output buffer");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            clear_graph();
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
        clear_graph();
    }

    bool matches(const IndexTTS2QwenEmotionWeights & weights, ggml_backend_t backend, int64_t prompt_steps) const noexcept {
        return weights_.get() == &weights && execution_.backend() == backend && prompt_steps_ == prompt_steps;
    }

    int32_t run(const std::vector<int32_t> & prompt_ids) {
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
        const auto next_token = core::read_tensor_i32(next_token_);
        debug::timing_log_scalar("index_tts2.qwen_emotion.prefill.output_read_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (next_token.size() != 1) {
            throw std::runtime_error("IndexTTS2 Qwen emotion prefill argmax output shape mismatch");
        }
        return next_token.front();
    }

    int64_t prompt_steps() const noexcept {
        return prompt_steps_;
    }

    size_t layer_count() const noexcept {
        return keys_.size();
    }

    void copy_layer_state_to(size_t layer, ggml_tensor * key_destination, ggml_tensor * value_destination) const {
        const int64_t values = prompt_steps_ * kKvHeads * kHeadDim;
        std::vector<float> key(static_cast<size_t>(values));
        std::vector<float> value(static_cast<size_t>(values));
        ggml_backend_tensor_get(keys_.at(layer), key.data(), 0, key.size() * sizeof(float));
        ggml_backend_tensor_get(values_.at(layer), value.data(), 0, value.size() * sizeof(float));
        ggml_backend_tensor_set(key_destination, key.data(), 0, key.size() * sizeof(float));
        ggml_backend_tensor_set(value_destination, value.data(), 0, value.size() * sizeof(float));
    }

private:
    void clear_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_.backend(), graph_);
            graph_ = nullptr;
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
            input_buffer_ = nullptr;
        }
        if (output_buffer_ != nullptr) {
            ggml_backend_buffer_free(output_buffer_);
            output_buffer_ = nullptr;
        }
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const IndexTTS2QwenEmotionWeights> weights_;
    int64_t prompt_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> output_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_ids_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * mask_ = nullptr;
    ggml_tensor * next_token_ = nullptr;
    std::vector<ggml_tensor *> keys_;
    std::vector<ggml_tensor *> values_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    ggml_backend_buffer_t output_buffer_ = nullptr;
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
        ggml_init_params input_params{16ull * 1024ull * 1024ull, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 Qwen emotion decode input context");
        }
        ggml_init_params state_params{128ull * 1024ull * 1024ull, nullptr, true};
        state_ctx_.reset(ggml_init(state_params));
        if (state_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 Qwen emotion decode state context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "index_tts2.qwen_emotion.decode", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{
            input_ctx_.get(),
            "index_tts2.qwen_emotion.decode.inputs",
            execution_.backend_type()};
        core::ModuleBuildContext state_ctx{
            state_ctx_.get(),
            "index_tts2.qwen_emotion.decode.state",
            execution_.backend_type()};
        token_id_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1, 1})).tensor;
        position_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1})).tensor;
        mask_ = core::make_tensor(input_ctx, GGML_TYPE_F16, core::TensorShape::from_dims({1, 1, 1, cache_steps_ + 1})).tensor;
        ggml_set_input(token_id_);
        ggml_set_input(position_);
        ggml_set_input(mask_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 131072, false);
        std::vector<core::TensorValue> cache_keys;
        std::vector<core::TensorValue> cache_values;
        auto x = modules::EmbeddingModule({kVocab, kHidden}).build(
            ctx,
            core::wrap_tensor(token_id_, core::TensorShape::from_dims({1, 1}), GGML_TYPE_I32),
            weights_->token_embedding);
        const auto cfg = qwen_config();
        const modules::QwenDecoderLayerModule layer_module(modules::qwen_decoder_layer_config_from_stack(cfg));
        const auto mask = core::wrap_tensor(mask_, core::TensorShape::from_dims({1, 1, 1, cache_steps_ + 1}), GGML_TYPE_F16);
        for (const auto & layer : weights_->decoder.layers) {
            cache_keys.push_back(core::make_tensor(
                state_ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, cache_steps_ + 1, kKvHeads, kHeadDim})));
            cache_values.push_back(core::make_tensor(
                state_ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, cache_steps_ + 1, kKvHeads, kHeadDim})));
            auto out = layer_module.build_with_static_cache_tail(
                ctx,
                graph_,
                x,
                core::wrap_tensor(position_, core::TensorShape::from_dims({1}), GGML_TYPE_I32),
                layer,
                cache_keys.back(),
                cache_values.back(),
                std::nullopt,
                mask);
            x = out.output;
        }
        kv_cache_ = engine::runtime::TransformerKVCache(cache_steps_ + 1, kKvHeads * kHeadDim, std::move(cache_keys), std::move(cache_values));
        build_transfer_views();
        x = modules::RMSNormModule({kHidden, kRmsEps, true, false}).build(ctx, x, weights_->final_norm);
        auto logits = modules::LinearModule({kHidden, kVocab, false}).build(ctx, x, {weights_->token_embedding, std::nullopt});
        auto flat_logits = core::reshape_tensor(
            ctx,
            core::ensure_backend_addressable_layout(ctx, logits),
            core::TensorShape::from_dims({1, kVocab}));
        next_token_ = ggml_argmax(ctx.ggml, flat_logits.tensor);
        ggml_set_output(next_token_);
        ggml_build_forward_expand(graph_, next_token_);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate IndexTTS2 Qwen emotion decode input buffer");
        }
        state_buffer_ = ggml_backend_alloc_ctx_tensors(state_ctx_.get(), execution_.backend());
        if (state_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate IndexTTS2 Qwen emotion decode state buffer");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            clear_graph();
            throw std::runtime_error("failed to allocate IndexTTS2 Qwen emotion decode graph");
        }
        mask_values_.assign(static_cast<size_t>(cache_steps_ + 1), ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        debug::timing_log_scalar("index_tts2.qwen_emotion.decode.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("index_tts2.qwen_emotion.decode.cache_steps", cache_steps_);
    }

    ~DecodeGraph() {
        clear_graph();
    }

    bool can_run(const IndexTTS2QwenEmotionWeights & weights, ggml_backend_t backend, int64_t required_steps) const noexcept {
        return weights_.get() == &weights && execution_.backend() == backend && cache_steps_ >= required_steps;
    }

    void import_state(const PrefillGraph & prefill) {
        if (prefill.layer_count() != key_sources_.size() || prefill.prompt_steps() > cache_steps_) {
            throw std::runtime_error("IndexTTS2 Qwen emotion decode prefill state shape mismatch");
        }
        kv_cache_.retain_prefix(0);
        const size_t prefix = static_cast<size_t>(prefill.prompt_steps());
        for (size_t layer = 0; layer < key_sources_.size(); ++layer) {
            prefill.copy_layer_state_to(layer, key_prefix_destinations_[prefix][layer], value_prefix_destinations_[prefix][layer]);
        }
        kv_cache_.advance_after_direct_append(prefill.prompt_steps());
    }

    int32_t run_step(int32_t token) {
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
        const auto next_token = core::read_tensor_i32(next_token_);
        if (next_token.size() != 1) {
            throw std::runtime_error("IndexTTS2 Qwen emotion decode argmax output shape mismatch");
        }
        const size_t dst_slot = static_cast<size_t>(kv_cache_.valid_steps());
        for (size_t layer = 0; layer < key_sources_.size(); ++layer) {
            ggml_backend_tensor_copy(key_sources_[layer], key_destinations_[dst_slot][layer]);
            ggml_backend_tensor_copy(value_sources_[layer], value_destinations_[dst_slot][layer]);
        }
        kv_cache_.advance_after_direct_append(1);
        return next_token.front();
    }

private:
    void clear_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_.backend(), graph_);
            graph_ = nullptr;
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (state_buffer_ != nullptr) {
            ggml_backend_buffer_free(state_buffer_);
            state_buffer_ = nullptr;
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
            input_buffer_ = nullptr;
        }
    }

    void build_transfer_views() {
        const int64_t step_elems = kKvHeads * kHeadDim;
        const size_t scratch_offset = static_cast<size_t>(cache_steps_ * step_elems) * sizeof(float);
        key_sources_.clear();
        value_sources_.clear();
        key_sources_.reserve(weights_->decoder.layers.size());
        value_sources_.reserve(weights_->decoder.layers.size());
        for (size_t layer = 0; layer < weights_->decoder.layers.size(); ++layer) {
            key_sources_.push_back(ggml_view_1d(state_ctx_.get(), kv_cache_.key_tensor(layer).tensor, step_elems, scratch_offset));
            value_sources_.push_back(ggml_view_1d(state_ctx_.get(), kv_cache_.value_tensor(layer).tensor, step_elems, scratch_offset));
        }
        key_destinations_.assign(static_cast<size_t>(cache_steps_), {});
        value_destinations_.assign(static_cast<size_t>(cache_steps_), {});
        key_prefix_destinations_.assign(static_cast<size_t>(cache_steps_ + 1), {});
        value_prefix_destinations_.assign(static_cast<size_t>(cache_steps_ + 1), {});
        for (int64_t slot = 0; slot < cache_steps_; ++slot) {
            const size_t byte_offset = static_cast<size_t>(slot * step_elems) * sizeof(float);
            auto & key_slot = key_destinations_[static_cast<size_t>(slot)];
            auto & value_slot = value_destinations_[static_cast<size_t>(slot)];
            key_slot.reserve(key_sources_.size());
            value_slot.reserve(value_sources_.size());
            for (size_t layer = 0; layer < key_sources_.size(); ++layer) {
                key_slot.push_back(ggml_view_1d(state_ctx_.get(), kv_cache_.key_tensor(layer).tensor, step_elems, byte_offset));
                value_slot.push_back(ggml_view_1d(state_ctx_.get(), kv_cache_.value_tensor(layer).tensor, step_elems, byte_offset));
            }
        }
        for (int64_t prefix = 1; prefix <= cache_steps_; ++prefix) {
            auto & key_prefix = key_prefix_destinations_[static_cast<size_t>(prefix)];
            auto & value_prefix = value_prefix_destinations_[static_cast<size_t>(prefix)];
            key_prefix.reserve(key_sources_.size());
            value_prefix.reserve(value_sources_.size());
            for (size_t layer = 0; layer < key_sources_.size(); ++layer) {
                key_prefix.push_back(ggml_view_1d(
                    state_ctx_.get(),
                    kv_cache_.key_tensor(layer).tensor,
                    prefix * step_elems,
                    0));
                value_prefix.push_back(ggml_view_1d(
                    state_ctx_.get(),
                    kv_cache_.value_tensor(layer).tensor,
                    prefix * step_elems,
                    0));
            }
        }
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const IndexTTS2QwenEmotionWeights> weights_;
    int64_t cache_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> state_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_id_ = nullptr;
    ggml_tensor * position_ = nullptr;
    ggml_tensor * mask_ = nullptr;
    ggml_tensor * next_token_ = nullptr;
    std::vector<ggml_tensor *> key_sources_;
    std::vector<ggml_tensor *> value_sources_;
    std::vector<std::vector<ggml_tensor *>> key_destinations_;
    std::vector<std::vector<ggml_tensor *>> value_destinations_;
    std::vector<std::vector<ggml_tensor *>> key_prefix_destinations_;
    std::vector<std::vector<ggml_tensor *>> value_prefix_destinations_;
    std::vector<ggml_fp16_t> mask_values_;
    engine::runtime::TransformerKVCache kv_cache_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    ggml_backend_buffer_t state_buffer_ = nullptr;
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
    const int64_t required_cache_steps = prompt_steps + max_new_tokens;
    if (decode_graph_ == nullptr || !decode_graph_->can_run(*weights_, execution_->backend(), required_cache_steps)) {
        decode_graph_.reset();
        decode_graph_ = std::make_unique<DecodeGraph>(*execution_, weights_, required_cache_steps, decode_graph_arena_bytes_);
    }
    const int32_t prefill_token = prefill_graph_->run(prompt_ids);
    decode_graph_->import_state(*prefill_graph_);

    std::vector<int32_t> generated;
    generated.reserve(static_cast<size_t>(max_new_tokens));
    int32_t token = prefill_token;
    double decode_run_ms = 0.0;
    bool saw_eos = false;
    for (int64_t step = 0; step < max_new_tokens; ++step) {
        if (token == tokenizer_.eos_token_id()) {
            saw_eos = true;
            break;
        }
        generated.push_back(token);
        if (step + 1 >= max_new_tokens) {
            break;
        }
        const auto decode_start = Clock::now();
        token = decode_graph_->run_step(token);
        decode_run_ms += engine::debug::elapsed_ms(decode_start, Clock::now());
    }
    if (!saw_eos && static_cast<int64_t>(generated.size()) >= max_new_tokens) {
        throw std::runtime_error("IndexTTS2 Qwen emotion decode reached max_new_tokens before EOS");
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
