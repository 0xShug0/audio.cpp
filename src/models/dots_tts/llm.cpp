#include "engine/models/dots_tts/llm.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/transformers/qwen_causal_decode_runtime.h"
#include "engine/framework/modules/transformers/qwen_decoder.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/framework/core/backend.h"

#include <ggml-alloc.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::dots_tts {
namespace {

namespace modules = engine::modules;

constexpr size_t kWeightContextBytes = 128ull * 1024ull * 1024ull;
constexpr size_t kSmallGraphContextBytes = 32ull * 1024ull * 1024ull;
constexpr size_t kLargeGraphContextBytes = 128ull * 1024ull * 1024ull;

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

struct LlmWeights {
    DotsLlmConfig config;
    std::shared_ptr<core::ExecutionContext> execution_context;
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue token_embedding;
    modules::QwenDecoderStackWeights stack;
    modules::NormWeights final_norm;
    modules::LinearWeights eos_in;
    modules::LinearWeights eos_out;
};

int64_t head_dim(const DotsLlmConfig & config) {
    if (config.num_attention_heads <= 0 || config.num_key_value_heads <= 0) {
        throw std::runtime_error("DotTTS LLM attention heads must be positive");
    }
    if (config.hidden_size % config.num_attention_heads != 0) {
        throw std::runtime_error("DotTTS LLM hidden_size must be divisible by attention heads");
    }
    return config.hidden_size / config.num_attention_heads;
}

modules::QwenDecoderStackConfig stack_config(const DotsLlmConfig & config) {
    modules::QwenDecoderStackConfig out;
    out.hidden_size = config.hidden_size;
    out.num_attention_heads = config.num_attention_heads;
    out.num_key_value_heads = config.num_key_value_heads;
    out.head_dim = head_dim(config);
    out.intermediate_size = config.intermediate_size;
    out.layers = config.num_hidden_layers;
    out.rms_norm_eps = config.rms_norm_eps;
    out.rope_theta = config.rope_theta;
    out.attention_precision = GGML_PREC_DEFAULT;
    out.projection_precision = GGML_PREC_DEFAULT;
    out.use_qk_norm = false;
    out.runtime.static_cache.update_mode = modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
    out.runtime.attention.prefill_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.runtime.attention.static_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    return out;
}

modules::QwenCausalDecodeRuntimeConfig make_qwen_decode_runtime_config(const DotsLlmConfig & config) {
    modules::QwenCausalDecodeRuntimeConfig out;
    out.trace_name = "dots_tts.llm";
    out.decoder.stack = stack_config(config);
    out.decoder.logits_mode = modules::QwenCausalDecoderLogitsMode::AllSteps;
    out.prefill_graph_arena_bytes = kLargeGraphContextBytes;
    out.decode_graph_arena_bytes = kLargeGraphContextBytes;
    out.output_mode = modules::QwenCausalDecodeOutputMode::Hidden;
    out.return_hidden = true;
    out.readback_round_type = GGML_TYPE_BF16;
    return out;
}

modules::QwenCausalDecodeRuntimeWeights make_qwen_decode_runtime_weights(const LlmWeights & weights) {
    modules::QwenCausalDecodeRuntimeWeights out;
    out.token_embedding = weights.token_embedding;
    out.stack = weights.stack;
    out.final_norm = weights.final_norm;
    return out;
}

modules::QwenDecoderLayerWeights load_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const DotsLlmConfig & config,
    int64_t layer,
    assets::TensorStorageType storage_type) {
    const int64_t dim = head_dim(config);
    const std::string prefix = "llm.model.layers." + std::to_string(layer);
    modules::QwenDecoderLayerWeights out;
    out.input_norm = {store.load_f32_tensor(source, prefix + ".input_layernorm.weight", {config.hidden_size}), std::nullopt};
    out.self_attention.q_weight = store.load_tensor(source, prefix + ".self_attn.q_proj.weight", storage_type, {config.num_attention_heads * dim, config.hidden_size});
    out.self_attention.q_bias = store.load_f32_tensor(source, prefix + ".self_attn.q_proj.bias", {config.num_attention_heads * dim});
    out.self_attention.k_weight = store.load_tensor(source, prefix + ".self_attn.k_proj.weight", storage_type, {config.num_key_value_heads * dim, config.hidden_size});
    out.self_attention.k_bias = store.load_f32_tensor(source, prefix + ".self_attn.k_proj.bias", {config.num_key_value_heads * dim});
    out.self_attention.v_weight = store.load_tensor(source, prefix + ".self_attn.v_proj.weight", storage_type, {config.num_key_value_heads * dim, config.hidden_size});
    out.self_attention.v_bias = store.load_f32_tensor(source, prefix + ".self_attn.v_proj.bias", {config.num_key_value_heads * dim});
    out.self_attention.out_weight = store.load_tensor(source, prefix + ".self_attn.o_proj.weight", storage_type, {config.hidden_size, config.hidden_size});
    out.post_norm = {store.load_f32_tensor(source, prefix + ".post_attention_layernorm.weight", {config.hidden_size}), std::nullopt};
    out.mlp.gate_proj = {store.load_tensor(source, prefix + ".mlp.gate_proj.weight", storage_type, {config.intermediate_size, config.hidden_size}), std::nullopt};
    out.mlp.up_proj = {store.load_tensor(source, prefix + ".mlp.up_proj.weight", storage_type, {config.intermediate_size, config.hidden_size}), std::nullopt};
    out.mlp.down_proj = {store.load_tensor(source, prefix + ".mlp.down_proj.weight", storage_type, {config.hidden_size, config.intermediate_size}), std::nullopt};
    return out;
}

std::shared_ptr<LlmWeights> load_weights(
    std::shared_ptr<const assets::TensorSource> source,
    core::BackendConfig backend,
    DotsLlmConfig config,
    assets::TensorStorageType storage_type) {
    if (source == nullptr) {
        throw std::runtime_error("DotTTS LLM requires tensor source");
    }
    if (source->has_tensor("llm.model.layers.0.self_attn.q_norm.weight") ||
        source->has_tensor("llm.model.layers.0.self_attn.k_norm.weight")) {
        throw std::runtime_error("DotTTS LLM q/k norm checkpoint is not implemented");
    }
    auto weights = std::make_shared<LlmWeights>();
    weights->config = config;
    weights->execution_context = std::make_shared<core::ExecutionContext>(backend);
    weights->store = std::make_shared<core::BackendWeightStore>(
        weights->execution_context->backend(),
        weights->execution_context->backend_type(),
        "dots_tts.llm.weights",
        kWeightContextBytes);
    weights->token_embedding = weights->store->load_tensor(
        *source,
        "llm.model.embed_tokens.weight",
        storage_type,
        {config.vocab_size, config.hidden_size});
    weights->stack.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
    for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        weights->stack.layers.push_back(load_layer(*weights->store, *source, config, layer, storage_type));
    }
    weights->final_norm = {weights->store->load_f32_tensor(*source, "llm.model.norm.weight", {config.hidden_size}), std::nullopt};
    weights->eos_in = {
        weights->store->load_tensor(*source, "eos_proj.0.weight", storage_type, {config.hidden_size, config.hidden_size}),
        weights->store->load_f32_tensor(*source, "eos_proj.0.bias", {config.hidden_size}),
    };
    weights->eos_out = {
        weights->store->load_tensor(*source, "eos_proj.2.weight", storage_type, {2, config.hidden_size}),
        weights->store->load_f32_tensor(*source, "eos_proj.2.bias", {2}),
    };
    weights->store->upload();
    return weights;
}

class EmbeddingRunner {
public:
    explicit EmbeddingRunner(std::shared_ptr<const LlmWeights> weights)
        : weights_(std::move(weights)) {}

    ~EmbeddingRunner() { release_graph(); }

    std::vector<float> run(const std::vector<int32_t> & token_ids) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (token_ids.empty()) {
            throw std::runtime_error("DotTTS LLM token embedding requires at least one token");
        }
        ensure_graph(static_cast<int64_t>(token_ids.size()));
        ggml_backend_tensor_set(token_ids_, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));
        if (core::compute_graph(*weights_->execution_context, graph_, plan_, "dots_tts.llm.embed_tokens") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DotTTS LLM token embedding graph compute failed");
        }
        return core::read_tensor_f32(output_);
    }

    void release_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(weights_->execution_context->backend(), graph_);
        }
        gallocr_.reset();
        ggml_.reset();
        plan_.reset();
        graph_ = nullptr;
        token_ids_ = nullptr;
        output_ = nullptr;
        steps_ = 0;
    }

private:
    void ensure_graph(int64_t steps) {
        if (ggml_ != nullptr && steps_ == steps) {
            return;
        }
        release_graph();
        ggml_init_params params{kSmallGraphContextBytes, nullptr, true};
        ggml_.reset(ggml_init(params));
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize DotTTS LLM embedding graph context");
        }
        core::ModuleBuildContext build_ctx{ggml_.get(), "dots_tts.llm.embed_tokens", weights_->execution_context->backend_type()};
        auto ids = core::make_tensor(build_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
        token_ids_ = ids.tensor;
        auto embeddings = modules::EmbeddingModule({weights_->config.vocab_size, weights_->config.hidden_size})
                              .build(build_ctx, ids, weights_->token_embedding);
        output_ = embeddings.tensor;
        graph_ = ggml_new_graph_custom(ggml_.get(), 65536, false);
        ggml_build_forward_expand(graph_, output_);
        core::validate_backend_graph_supported(weights_->execution_context->backend(), graph_, "dots_tts.llm.embed_tokens");
        gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights_->execution_context->backend())));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_.get(), graph_) || !ggml_gallocr_alloc_graph(gallocr_.get(), graph_)) {
            release_graph();
            throw std::runtime_error("failed to allocate DotTTS LLM embedding graph memory");
        }
        core::prepare_host_graph_plan(*weights_->execution_context, graph_, plan_);
        steps_ = steps;
    }

    std::shared_ptr<const LlmWeights> weights_;
    std::mutex mutex_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ggml_;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr_;
    ggml_cgraph * graph_ = nullptr;
    core::HostGraphPlan plan_;
    ggml_tensor * token_ids_ = nullptr;
    ggml_tensor * output_ = nullptr;
    int64_t steps_ = 0;
};

class EosRunner {
public:
    explicit EosRunner(std::shared_ptr<const LlmWeights> weights)
        : weights_(std::move(weights)) {}

    ~EosRunner() { release_graph(); }

    float run(const std::vector<float> & hidden) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto & config = weights_->config;
        if (static_cast<int64_t>(hidden.size()) != config.hidden_size) {
            throw std::runtime_error("DotTTS EOS hidden size mismatch");
        }
        ensure_graph();
        ggml_backend_tensor_set(hidden_, hidden.data(), 0, hidden.size() * sizeof(float));
        if (core::compute_graph(*weights_->execution_context, graph_, plan_, "dots_tts.llm.eos") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DotTTS EOS graph compute failed");
        }
        const auto logits = core::read_tensor_f32(logits_);
        const float max_logit = std::max(logits[0], logits[1]);
        const float p0 = std::exp(logits[0] - max_logit);
        const float p1 = std::exp(logits[1] - max_logit);
        return p1 / (p0 + p1);
    }

    void release_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(weights_->execution_context->backend(), graph_);
        }
        gallocr_.reset();
        ggml_.reset();
        plan_.reset();
        graph_ = nullptr;
        hidden_ = nullptr;
        logits_ = nullptr;
    }

private:
    void ensure_graph() {
        if (ggml_ != nullptr) {
            return;
        }
        ggml_init_params params{kSmallGraphContextBytes, nullptr, true};
        ggml_.reset(ggml_init(params));
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize DotTTS EOS graph context");
        }
        const auto & config = weights_->config;
        core::ModuleBuildContext build_ctx{ggml_.get(), "dots_tts.llm.eos", weights_->execution_context->backend_type()};
        auto input = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config.hidden_size}));
        hidden_ = input.tensor;
        auto x = modules::LinearModule({config.hidden_size, config.hidden_size, true}).build(build_ctx, input, weights_->eos_in);
        x = modules::SiluModule().build(build_ctx, x);
        auto logits = modules::LinearModule({config.hidden_size, 2, true}).build(build_ctx, x, weights_->eos_out);
        logits_ = logits.tensor;
        graph_ = ggml_new_graph_custom(ggml_.get(), 65536, false);
        ggml_build_forward_expand(graph_, logits_);
        core::validate_backend_graph_supported(weights_->execution_context->backend(), graph_, "dots_tts.llm.eos");
        gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights_->execution_context->backend())));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_.get(), graph_) || !ggml_gallocr_alloc_graph(gallocr_.get(), graph_)) {
            release_graph();
            throw std::runtime_error("failed to allocate DotTTS EOS graph memory");
        }
        core::prepare_host_graph_plan(*weights_->execution_context, graph_, plan_);
    }

    std::shared_ptr<const LlmWeights> weights_;
    std::mutex mutex_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ggml_;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr_;
    ggml_cgraph * graph_ = nullptr;
    core::HostGraphPlan plan_;
    ggml_tensor * hidden_ = nullptr;
    ggml_tensor * logits_ = nullptr;
};

}  // namespace

struct DotsLlmState::Impl {
    runtime::TransformerKVState kv;
    int64_t capacity = 0;
    bool decode_advanced = false;
};

DotsLlmState::DotsLlmState() : impl_(std::make_unique<Impl>()) {}
DotsLlmState::~DotsLlmState() = default;
DotsLlmState::DotsLlmState(DotsLlmState &&) noexcept = default;
DotsLlmState & DotsLlmState::operator=(DotsLlmState &&) noexcept = default;

int64_t DotsLlmState::seq_len() const noexcept {
    return impl_ == nullptr ? 0 : impl_->kv.current_end;
}

int64_t DotsLlmState::capacity() const noexcept {
    return impl_ == nullptr ? 0 : impl_->capacity;
}

struct DotsLlmComponent::Impl {
    explicit Impl(std::shared_ptr<const LlmWeights> weights)
        : weights(std::move(weights)),
          embedding_runner(std::make_unique<EmbeddingRunner>(this->weights)),
          qwen_runtime(std::make_unique<modules::QwenCausalDecodeRuntime>(
              *this->weights->execution_context,
              make_qwen_decode_runtime_config(this->weights->config),
              make_qwen_decode_runtime_weights(*this->weights))),
          eos_runner(std::make_unique<EosRunner>(this->weights)) {}

    std::shared_ptr<const LlmWeights> weights;
    std::unique_ptr<EmbeddingRunner> embedding_runner;
    std::unique_ptr<modules::QwenCausalDecodeRuntime> qwen_runtime;
    std::unique_ptr<EosRunner> eos_runner;
    const void * active_decode_state = nullptr;
};

DotsLlmComponent DotsLlmComponent::load_from_tensor_source(
    std::shared_ptr<const assets::TensorSource> source,
    core::BackendConfig backend,
    DotsLlmConfig config,
    assets::TensorStorageType weight_storage_type) {
    DotsLlmComponent component;
    component.impl_ = std::make_unique<Impl>(load_weights(
        std::move(source),
        backend,
        std::move(config),
        weight_storage_type));
    return component;
}

DotsLlmComponent::DotsLlmComponent() = default;
DotsLlmComponent::~DotsLlmComponent() = default;
DotsLlmComponent::DotsLlmComponent(DotsLlmComponent &&) noexcept = default;
DotsLlmComponent & DotsLlmComponent::operator=(DotsLlmComponent &&) noexcept = default;

bool DotsLlmComponent::is_loaded() const noexcept {
    return impl_ != nullptr && impl_->weights != nullptr;
}

DotsLlmState DotsLlmComponent::create_state(int64_t max_sequence_length) const {
    if (impl_ == nullptr || impl_->weights == nullptr) {
        throw std::runtime_error("DotTTS LLM is not initialized");
    }
    if (max_sequence_length <= 0) {
        throw std::runtime_error("DotTTS LLM state capacity must be positive");
    }
    DotsLlmState state;
    state.impl_->capacity = max_sequence_length;
    state.impl_->kv.current_end = 0;
    state.impl_->kv.layers.resize(static_cast<size_t>(impl_->weights->config.num_hidden_layers));
    state.impl_->decode_advanced = false;
    return state;
}

std::vector<float> DotsLlmComponent::embed_tokens(const std::vector<int32_t> & token_ids) const {
    if (impl_ == nullptr || impl_->embedding_runner == nullptr) {
        throw std::runtime_error("DotTTS LLM is not initialized");
    }
    return impl_->embedding_runner->run(token_ids);
}

DotsLlmHidden DotsLlmComponent::prefill_embeddings(
    const std::vector<float> & embeddings,
    int64_t steps,
    DotsLlmState & state) const {
    if (impl_ == nullptr || impl_->qwen_runtime == nullptr) {
        throw std::runtime_error("DotTTS LLM is not initialized");
    }
    if (state.impl_ == nullptr || steps > state.impl_->capacity) {
        throw std::runtime_error("DotTTS LLM prefill exceeds state capacity");
    }
    auto result = impl_->qwen_runtime->prefill_embeddings(embeddings, steps);
    state.impl_->kv = std::move(result.state);
    state.impl_->decode_advanced = false;
    impl_->qwen_runtime->start_decode_embeddings(state.impl_->kv, state.impl_->capacity);
    impl_->active_decode_state = state.impl_.get();
    return {
        std::move(result.hidden),
        steps,
        impl_->weights->config.hidden_size,
    };
}

DotsLlmHidden DotsLlmComponent::decode_embedding(
    const std::vector<float> & embedding,
    DotsLlmState & state) const {
    if (impl_ == nullptr || impl_->qwen_runtime == nullptr) {
        throw std::runtime_error("DotTTS LLM is not initialized");
    }
    if (state.impl_ == nullptr) {
        throw std::runtime_error("DotTTS LLM decode requires initialized state");
    }
    if (impl_->active_decode_state != state.impl_.get()) {
        if (state.impl_->decode_advanced) {
            throw std::runtime_error("DotTTS LLM decode state is not active");
        }
        impl_->qwen_runtime->start_decode_embeddings(state.impl_->kv, state.impl_->capacity);
        impl_->active_decode_state = state.impl_.get();
    }
    if (state.impl_->kv.current_end >= state.impl_->capacity) {
        throw std::runtime_error("DotTTS LLM decode exceeds state capacity");
    }
    auto result = impl_->qwen_runtime->decode_embedding(embedding);
    state.impl_->kv.current_end = impl_->qwen_runtime->decode_current_end();
    for (auto & layer : state.impl_->kv.layers) {
        layer.valid_steps = impl_->qwen_runtime->decode_valid_steps();
    }
    state.impl_->decode_advanced = true;
    return {
        std::move(result.hidden),
        1,
        impl_->weights->config.hidden_size,
    };
}

float DotsLlmComponent::eos_probability(const std::vector<float> & hidden) const {
    if (impl_ == nullptr || impl_->eos_runner == nullptr) {
        throw std::runtime_error("DotTTS LLM is not initialized");
    }
    return impl_->eos_runner->run(hidden);
}

void DotsLlmComponent::release_runtime_graphs() {
    if (impl_ != nullptr && impl_->embedding_runner != nullptr) {
        impl_->embedding_runner->release_graph();
    }
    if (impl_ != nullptr && impl_->qwen_runtime != nullptr) {
        impl_->qwen_runtime->release_runtime_graphs();
        impl_->active_decode_state = nullptr;
    }
    if (impl_ != nullptr && impl_->eos_runner != nullptr) {
        impl_->eos_runner->release_graph();
    }
}

}  // namespace engine::models::dots_tts
