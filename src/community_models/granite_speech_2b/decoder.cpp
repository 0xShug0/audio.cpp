// granite_speech_2b text decoder (Task C2): Granite binding + generation.
//
// Math reference (pinned transformers 4.57.0, IBM granite-speech-5.0-2b):
// - GraniteModel.forward: inputs_embeds = embed_tokens(input_ids) * 12
//   (modeling_granite.py, "inputs_embeds = inputs_embeds *
//   self.embedding_multiplier"). The x12 applies to the merged sequence:
//   audio rows are substituted into inputs_embeds BEFORE the multiply (via
//   get_merged_audio_embeddings in modeling_granite_speech.py), and generated
//   tokens pass through the same embed-then-scale path per step.
// - GraniteDecoderLayer.forward: residual + 0.22 * attn_out and
//   residual + 0.22 * mlp_out ("main diff with Llama").
// - GraniteAttention: softmax(Q @ K^T * 0.0078125) with the ordinary causal
//   mask (self.scaling = config.attention_multiplier; eager path
//   "attn_weights = torch.matmul(...) * scaling"). No Q/K norm modules, no
//   attention or MLP biases (attention_bias=false, mlp_bias=false in the
//   pinned config).
// - GraniteRotaryEmbedding + _compute_yarn_parameters (modeling_rope_utils.py):
//   128 NeoX channels, theta 10000, YaRN factor 4, original context 4096,
//   beta_fast/beta_slow 32/1 with interpolation blend, magnitude factor
//   0.1*ln(4)+1 ~= 1.13863 applied to cos/sin. Verified bit-near against the
//   ggml_rope_ext parametric path (max abs diff 3e-8 vs the HF inv_freq
//   buffer), so the C++ side forwards these as ggml YaRN parameters instead
//   of a frequency-factor tensor (the checkpoint carries no rope tensor).
// - GraniteForCausalLM.forward: logits = lm_head(hidden) / 8
//   ("logits = logits / self.config.logits_scaling"). Untied head.
// - EOS/budget stop: generation_config eos_token_id 100257,
//   max_position_embeddings 16384.

#include "engine/community_models/granite_speech_2b/decoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/transformers/decoder.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::community_models::granite_speech_2b {
namespace {

namespace modules = engine::modules;

using Clock = std::chrono::steady_clock;

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

void require_prompt_shapes(
    const runtime::GreedyCausalDecoderRuntime::Prompt & prompt,
    int64_t hidden_size) {
    const int64_t steps = static_cast<int64_t>(prompt.input_ids.size());
    const auto & injection = prompt.injection;
    if (injection.tokens < 0 || injection.tokens > steps ||
        static_cast<int64_t>(injection.positions.size()) != injection.tokens ||
        static_cast<int64_t>(injection.values.size()) != injection.tokens * hidden_size) {
        throw std::runtime_error("granite_speech_2b decoder injection shape does not match the prompt");
    }
    for (const int32_t position : injection.positions) {
        if (position < 0 || position >= steps) {
            throw std::runtime_error("granite_speech_2b decoder injection position is out of range");
        }
    }
}

modules::DecoderStackConfig make_granite_stack_config() {
    modules::DecoderStackConfig stack;
    stack.hidden_size = 2048;
    stack.num_attention_heads = 16;
    stack.num_key_value_heads = 4;
    stack.head_dim = 128;
    stack.intermediate_size = 4096;
    stack.layers = kGraniteSpeech2BDecoderLayers;
    stack.rms_norm_eps = 1e-5f;
    stack.rope_theta = 10000.0f;
    stack.rope_type = GGML_ROPE_TYPE_NEOX;
    stack.use_qk_norm = false;
    stack.attention_scale = kGraniteSpeech2BAttentionScale;
    stack.residual_scale = kGraniteSpeech2BResidualScale;
    stack.rope_freq_scale = kGraniteSpeech2BYarnFreqScale;
    stack.rope_ext_factor = kGraniteSpeech2BYarnExtFactor;
    stack.rope_attn_factor = kGraniteSpeech2BYarnAttnFactor;
    stack.rope_beta_fast = kGraniteSpeech2BYarnBetaFast;
    stack.rope_beta_slow = kGraniteSpeech2BYarnBetaSlow;
    stack.rope_orig_context = kGraniteSpeech2BYarnOrigContext;
    // Non-flash CPU reference policy: manual repeat-KV + matmul/softmax on
    // both the direct and static-cache paths.
    stack.runtime.attention.prefill_mode = modules::DecoderAttentionMode::ManualRepeat;
    stack.runtime.attention.static_mode = modules::DecoderAttentionMode::ManualRepeat;
    stack.runtime.attention.allow_flash_attention = false;
    stack.runtime.static_cache.update_mode = modules::DecoderStaticCacheUpdateMode::DirectSetRows;
    return stack;
}

int32_t argmax_index(const std::vector<float> & values) {
    if (values.empty()) {
        throw std::runtime_error("granite_speech_2b decoder cannot select from empty logits");
    }
    size_t best = 0;
    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] > values[best]) {
            best = i;
        }
    }
    return static_cast<int32_t>(best);
}

// Device-side token embedding lookup in 64-step blocks (same pattern as the
// shared greedy runtime: a small fixed-width graph also handles non-F32
// embedding storage, padding the final block and discarding it).
class PromptEmbeddingGraph {
public:
    static constexpr int64_t kSteps = 64;
    PromptEmbeddingGraph(
        ggml_backend_t backend,
        core::BackendType backend_type,
        int threads,
        const core::TensorValue & token_embedding,
        int64_t hidden_size,
        int64_t vocab_size)
        : backend_(backend),
          threads_(threads),
          hidden_size_(hidden_size) {
        ctx_.reset(ggml_init({1024 * 1024, nullptr, true}));
        if (!ctx_) {
            throw std::runtime_error("granite_speech_2b decoder failed to initialize embedding graph");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "granite_speech_2b_decoder.embedding", backend_type};
        ids_ = ggml_new_tensor_1d(ctx.ggml, GGML_TYPE_I32, kSteps);
        auto ids = core::wrap_tensor(ids_, core::TensorShape::from_dims({kSteps}), GGML_TYPE_I32);
        output_ = modules::EmbeddingModule({vocab_size, hidden_size}).build(ctx, ids, token_embedding).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx.ggml, 64, false);
        ggml_build_forward_expand(graph_, output_);
        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        if (!allocator_ || !ggml_gallocr_alloc_graph(allocator_.get(), graph_)) {
            throw std::runtime_error("granite_speech_2b decoder failed to allocate embedding graph");
        }
    }
    ~PromptEmbeddingGraph() {
        core::release_backend_graph_resources(backend_, graph_, true);
    }
    std::vector<float> run(const std::vector<int32_t> & ids) {
        std::vector<float> result(ids.size() * static_cast<size_t>(hidden_size_));
        std::vector<int32_t> block(kSteps, 0);
        core::set_backend_threads(backend_, threads_);
        for (size_t offset = 0; offset < ids.size(); offset += kSteps) {
            const size_t count = std::min<size_t>(kSteps, ids.size() - offset);
            std::fill(block.begin(), block.end(), 0);
            std::copy_n(ids.begin() + offset, count, block.begin());
            ggml_backend_tensor_set(ids_, block.data(), 0, block.size() * sizeof(int32_t));
            if (core::compute_backend_graph(backend_, graph_) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("granite_speech_2b decoder embedding graph compute failed");
            }
            ggml_backend_synchronize(backend_);
            ggml_backend_tensor_get(
                output_,
                result.data() + offset * hidden_size_,
                0,
                count * hidden_size_ * sizeof(float));
        }
        return result;
    }

private:
    ggml_backend_t backend_ = nullptr;
    int threads_ = 1;
    int64_t hidden_size_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * ids_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> allocator_;
};

struct DecoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue token_embedding;
    std::vector<modules::DecoderLayerWeights> layers;
    modules::NormWeights final_norm;
    modules::LinearWeights lm_head;
};

DecoderWeights load_decoder_weights(
    const assets::TensorSource & source,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    DecoderWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend, backend_type, "granite_speech_2b.decoder.weights", weight_context_bytes);
    auto & store = *weights.store;
    weights.token_embedding =
        store.load_tensor(source, "language_model.model.embed_tokens.weight", storage_type, {100353, 2048});
    weights.layers.reserve(static_cast<size_t>(kGraniteSpeech2BDecoderLayers));
    for (int64_t layer = 0; layer < kGraniteSpeech2BDecoderLayers; ++layer) {
        const std::string prefix = "language_model.model.layers." + std::to_string(layer);
        modules::DecoderLayerWeights w;
        w.input_norm = {store.load_f32_tensor(source, prefix + ".input_layernorm.weight", {2048}),
                        std::nullopt};
        // Separate bias-free Q/K/V projections; no Q/K norms on Granite.
        w.self_attention.q_weight =
            store.load_tensor(source, prefix + ".self_attn.q_proj.weight", storage_type, {2048, 2048});
        w.self_attention.k_weight =
            store.load_tensor(source, prefix + ".self_attn.k_proj.weight", storage_type, {512, 2048});
        w.self_attention.v_weight =
            store.load_tensor(source, prefix + ".self_attn.v_proj.weight", storage_type, {512, 2048});
        w.self_attention.out_weight =
            store.load_tensor(source, prefix + ".self_attn.o_proj.weight", storage_type, {2048, 2048});
        w.post_norm = {store.load_f32_tensor(source, prefix + ".post_attention_layernorm.weight", {2048}),
                       std::nullopt};
        w.mlp.gate_proj = {store.load_tensor(
                                source, prefix + ".mlp.gate_proj.weight", storage_type, {4096, 2048}),
                            std::nullopt};
        w.mlp.up_proj = {store.load_tensor(
                              source, prefix + ".mlp.up_proj.weight", storage_type, {4096, 2048}),
                          std::nullopt};
        w.mlp.down_proj = {store.load_tensor(
                                source, prefix + ".mlp.down_proj.weight", storage_type, {2048, 4096}),
                            std::nullopt};
        weights.layers.push_back(std::move(w));
    }
    weights.final_norm = {store.load_f32_tensor(source, "language_model.model.norm.weight", {2048}),
                          std::nullopt};
    // Untied LM head (no bias in the pinned checkpoint).
    weights.lm_head = {store.load_tensor(
                            source, "language_model.lm_head.weight", storage_type, {100353, 2048}),
                        std::nullopt};
    store.upload();
    return weights;
}

modules::CausalDecoderRuntimeWeights bind_runtime_weights(const DecoderWeights & weights) {
    modules::CausalDecoderRuntimeWeights out;
    out.token_embedding = weights.token_embedding;
    out.stack.layers = weights.layers;
    out.final_norm = weights.final_norm;
    out.lm_head = weights.lm_head;
    return out;
}

}  // namespace

modules::CausalDecoderConfig make_granite_speech_2b_causal_decoder_config() {
    modules::CausalDecoderConfig config;
    config.stack = make_granite_stack_config();
    config.logits_size = 100353;
    config.logits_mode = modules::CausalDecoderLogitsMode::LastStep;
    config.use_lm_head_bias = false;
    config.embedding_scale = kGraniteSpeech2BEmbedScale;
    config.logit_divisor = kGraniteSpeech2BLogitDivisor;
    config.static_cache_type = GGML_TYPE_F32;
    return config;
}

runtime::GreedyCausalDecoderSpec make_granite_speech_2b_greedy_decoder_spec() {
    runtime::GreedyCausalDecoderSpec spec;
    spec.decoder = make_granite_speech_2b_causal_decoder_config();
    spec.vocab_size = 100353;
    spec.max_position_embeddings = kGraniteSpeech2BMaxPositionEmbeddings;
    spec.tie_word_embeddings = false;
    spec.attention_bias = false;
    spec.packed_qkv = false;
    spec.token_embedding_tensor = "language_model.model.embed_tokens.weight";
    spec.lm_head_tensor = "language_model.lm_head.weight";
    spec.final_norm_tensor = "language_model.model.norm.weight";
    spec.layer_prefix = "language_model.model.layers";
    spec.eos_token_ids = {kGraniteSpeech2BEosTokenId};
    return spec;
}

void granite_speech_2b_require_generation_budget(
    int64_t prompt_len,
    int64_t max_new_tokens,
    int64_t max_position_embeddings) {
    if (prompt_len <= 0) {
        throw std::runtime_error("granite_speech_2b decoder prompt is empty");
    }
    if (max_new_tokens <= 0) {
        throw std::runtime_error("granite_speech_2b decoder max_new_tokens must be positive");
    }
    if (prompt_len + max_new_tokens > max_position_embeddings) {
        throw std::runtime_error(
            "granite_speech_2b decoder request exceeds max_position_embeddings (" +
            std::to_string(prompt_len + max_new_tokens) + " > " + std::to_string(max_position_embeddings) + ")");
    }
}

struct GraniteSpeech2BDecoderRuntime::Impl {
    Impl(
        std::shared_ptr<const assets::TensorSource> weights_source,
        core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type)
        : runtime_(
              std::move(weights_source),
              make_granite_speech_2b_greedy_decoder_spec(),
              execution,
              prefill_graph_arena_bytes,
              decode_graph_arena_bytes,
              weight_context_bytes,
              weight_storage_type) {}

    runtime::GreedyCausalDecoderRuntime runtime_;
};

GraniteSpeech2BDecoderRuntime::GraniteSpeech2BDecoderRuntime(
    std::shared_ptr<const assets::TensorSource> weights_source,
    core::ExecutionContext & execution,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(
          std::move(weights_source),
          execution,
          prefill_graph_arena_bytes,
          decode_graph_arena_bytes,
          weight_context_bytes,
          weight_storage_type)) {}

GraniteSpeech2BDecoderRuntime::~GraniteSpeech2BDecoderRuntime() = default;

std::vector<int32_t> GraniteSpeech2BDecoderRuntime::generate(
    const runtime::GreedyCausalDecoderRuntime::Prompt & prompt,
    int64_t max_new_tokens) {
    granite_speech_2b_require_generation_budget(
        static_cast<int64_t>(prompt.input_ids.size()), max_new_tokens);
    // One complete prompt per call, zero retained prefix steps.
    return impl_->runtime_.generate(prompt, max_new_tokens, /*reuse_graphs=*/true);
}

struct GraniteSpeech2BDecoder::Impl {
    Impl(
        std::shared_ptr<const assets::TensorSource> weights_source,
        core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type)
        : execution_(&execution),
          weights_(load_decoder_weights(
              *weights_source,
              execution.backend(),
              execution.backend_type(),
              weight_context_bytes,
              weight_storage_type)) {
        modules::CausalDecoderRuntimeConfig config;
        config.trace_name = "granite_speech_2b.decoder";
        config.decoder = make_granite_speech_2b_causal_decoder_config();
        config.prefill_graph_arena_bytes = prefill_graph_arena_bytes;
        config.decode_graph_arena_bytes = decode_graph_arena_bytes;
        runtime_ =
            std::make_unique<modules::CausalDecoderRuntime>(execution, config, bind_runtime_weights(weights_));
        embedding_graph_ = std::make_unique<PromptEmbeddingGraph>(
            execution.backend(),
            execution.backend_type(),
            std::max(1, execution.config().threads),
            weights_.token_embedding,
            2048,
            100353);
    }

    static constexpr int64_t kHiddenSize = 2048;

    GraniteSpeech2BDecoder::PrefillResult prefill(
        const runtime::GreedyCausalDecoderRuntime::Prompt & prompt,
        int64_t max_new_tokens,
        int64_t chunk_steps) {
        const int64_t steps = static_cast<int64_t>(prompt.input_ids.size());
        granite_speech_2b_require_generation_budget(steps, max_new_tokens);
        require_prompt_shapes(prompt, kHiddenSize);
        if (chunk_steps <= 0) {
            throw std::runtime_error("granite_speech_2b decoder chunk_steps must be positive");
        }
        const auto start = Clock::now();
        // Device lookup, then host audio substitution at the injected rows.
        std::vector<float> merged = embedding_graph_->run(prompt.input_ids);
        for (size_t i = 0; i < prompt.injection.positions.size(); ++i) {
            std::copy_n(
                prompt.injection.values.data() + i * kHiddenSize,
                kHiddenSize,
                merged.data() + prompt.injection.positions[i] * kHiddenSize);
        }
        const int64_t required = steps + max_new_tokens;
        const int64_t capacity = std::min<int64_t>(
            kGraniteSpeech2BMaxPositionEmbeddings, (required + 127) / 128 * 128);
        auto step = runtime_->prefill_embeddings_into_cache(merged, steps, capacity, chunk_steps, 0);
        engine::debug::timing_log_scalar(
            "granite_speech_2b.decoder.prefill_ms", engine::debug::elapsed_ms(start, Clock::now()));
        GraniteSpeech2BDecoder::PrefillResult out;
        out.steps = steps;
        out.merged = merged;
        out.scaled = merged;
        for (float & value : out.scaled) {
            value *= kGraniteSpeech2BEmbedScale;
        }
        out.last_logits = std::move(step.logits);
        return out;
    }

    core::ExecutionContext * execution_ = nullptr;
    DecoderWeights weights_;
    std::unique_ptr<modules::CausalDecoderRuntime> runtime_;
    std::unique_ptr<PromptEmbeddingGraph> embedding_graph_;
};

GraniteSpeech2BDecoder::GraniteSpeech2BDecoder(
    std::shared_ptr<const assets::TensorSource> weights_source,
    core::ExecutionContext & execution,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(
          std::move(weights_source),
          execution,
          prefill_graph_arena_bytes,
          decode_graph_arena_bytes,
          weight_context_bytes,
          weight_storage_type)) {}

GraniteSpeech2BDecoder::~GraniteSpeech2BDecoder() = default;

GraniteSpeech2BDecoder::PrefillResult GraniteSpeech2BDecoder::prefill(
    const runtime::GreedyCausalDecoderRuntime::Prompt & prompt,
    int64_t max_new_tokens,
    int64_t chunk_steps) {
    return impl_->prefill(prompt, max_new_tokens, chunk_steps);
}

std::vector<float> GraniteSpeech2BDecoder::decode_token(int32_t token) {
    return impl_->runtime_->decode_token(token).logits;
}

std::vector<int32_t> GraniteSpeech2BDecoder::generate_greedy(
    const runtime::GreedyCausalDecoderRuntime::Prompt & prompt,
    int64_t max_new_tokens) {
    const auto prefill = impl_->prefill(prompt, max_new_tokens, 64);
    std::vector<int32_t> out;
    std::vector<float> logits = prefill.last_logits;
    for (int64_t step = 0; step < max_new_tokens; ++step) {
        const int32_t token = argmax_index(logits);
        if (token == kGraniteSpeech2BEosTokenId) {
            break;
        }
        out.push_back(token);
        if (step + 1 < max_new_tokens) {
            logits = impl_->runtime_->decode_token(token).logits;
        }
    }
    return out;
}

}  // namespace engine::community_models::granite_speech_2b
