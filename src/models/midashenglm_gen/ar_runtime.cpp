#include "engine/models/midashenglm_gen/ar_runtime.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::midashenglm_gen {
namespace {

namespace binding = engine::modules::binding;
namespace core = engine::core;
namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

int64_t count_qwen_layers(const engine::assets::TensorSource & source) {
    int64_t layers = 0;
    while (source.has_tensor(
        "model.mdl_model.decoder.model.layers." + std::to_string(layers) +
        ".input_layernorm.weight")) {
        ++layers;
    }
    if (layers <= 0) {
        throw std::runtime_error("MiDashengLM-Gen Qwen decoder layers not found");
    }
    return layers;
}

modules::QwenCausalDecoderConfig qwen_config(
    const MiDashengLmGenConfig & config,
    const engine::assets::TensorSource & source) {
    const auto q = source.require_metadata("model.mdl_model.decoder.model.layers.0.self_attn.q_proj.weight");
    const auto k = source.require_metadata("model.mdl_model.decoder.model.layers.0.self_attn.k_proj.weight");
    const auto q_norm = source.require_metadata("model.mdl_model.decoder.model.layers.0.self_attn.q_norm.weight");
    const auto mlp = source.require_metadata("model.mdl_model.decoder.model.layers.0.mlp.gate_proj.weight");
    modules::QwenCausalDecoderConfig out;
    out.stack.hidden_size = config.hidden_size;
    out.stack.num_attention_heads = q.shape.at(0) / q_norm.shape.at(0);
    out.stack.num_key_value_heads = k.shape.at(0) / q_norm.shape.at(0);
    out.stack.head_dim = q_norm.shape.at(0);
    out.stack.intermediate_size = mlp.shape.at(0);
    out.stack.layers = count_qwen_layers(source);
    out.stack.rms_norm_eps = 1.0e-6F;
    out.stack.rope_theta = 1000000.0F;
    out.stack.rope_type = GGML_ROPE_TYPE_NEOX;
    out.stack.attention_precision = GGML_PREC_F32;
    out.stack.projection_precision = GGML_PREC_DEFAULT;
    out.stack.use_qk_norm = true;
    out.stack.runtime.attention.prefill_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.stack.runtime.attention.static_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.stack.runtime.static_cache.update_mode = modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
    out.logits_mode = modules::QwenCausalDecoderLogitsMode::LastStep;
    return out;
}

modules::QwenDecoderLayerWeights load_qwen_layer(
    core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const modules::QwenCausalDecoderConfig & config,
    engine::assets::TensorStorageType storage_type,
    int64_t layer) {
    const std::string prefix = "model.mdl_model.decoder.model.layers." + std::to_string(layer);
    modules::QwenDecoderLayerWeights out;
    out.input_norm = binding::norm_weight_from_source(store, source, prefix + ".input_layernorm", config.stack.hidden_size);
    out.self_attention.q_weight = store.load_tensor(source, prefix + ".self_attn.q_proj.weight", storage_type, {config.stack.num_attention_heads * config.stack.head_dim, config.stack.hidden_size});
    out.self_attention.k_weight = store.load_tensor(source, prefix + ".self_attn.k_proj.weight", storage_type, {config.stack.num_key_value_heads * config.stack.head_dim, config.stack.hidden_size});
    out.self_attention.v_weight = store.load_tensor(source, prefix + ".self_attn.v_proj.weight", storage_type, {config.stack.num_key_value_heads * config.stack.head_dim, config.stack.hidden_size});
    out.self_attention.out_weight = store.load_tensor(source, prefix + ".self_attn.o_proj.weight", storage_type, {config.stack.hidden_size, config.stack.num_attention_heads * config.stack.head_dim});
    out.q_norm = binding::norm_weight_from_source(store, source, prefix + ".self_attn.q_norm", config.stack.head_dim);
    out.k_norm = binding::norm_weight_from_source(store, source, prefix + ".self_attn.k_norm", config.stack.head_dim);
    out.post_norm = binding::norm_weight_from_source(store, source, prefix + ".post_attention_layernorm", config.stack.hidden_size);
    out.mlp.gate_proj = binding::linear_from_source(store, source, prefix + ".mlp.gate_proj", storage_type, config.stack.intermediate_size, config.stack.hidden_size, false);
    out.mlp.up_proj = binding::linear_from_source(store, source, prefix + ".mlp.up_proj", storage_type, config.stack.intermediate_size, config.stack.hidden_size, false);
    out.mlp.down_proj = binding::linear_from_source(store, source, prefix + ".mlp.down_proj", storage_type, config.stack.hidden_size, config.stack.intermediate_size, false);
    return out;
}

std::shared_ptr<const MiDashengLmGenARWeights> load_weights(
    const MiDashengLmGenAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type,
    modules::QwenCausalDecoderConfig & qwen) {
    auto weights = std::make_shared<MiDashengLmGenARWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "midashenglm_gen.ar.weights",
        weight_context_bytes);
    const auto & source = *assets.weights;
    qwen = qwen_config(assets.config, source);
    weights->qwen.token_embedding = weights->store->load_tensor(
        source,
        "model.mdl_model.decoder.model.embed_tokens.weight",
        storage_type,
        {assets.config.vocab_size, assets.config.hidden_size});
    weights->qwen.stack.layers.reserve(static_cast<size_t>(qwen.stack.layers));
    for (int64_t layer = 0; layer < qwen.stack.layers; ++layer) {
        weights->qwen.stack.layers.push_back(load_qwen_layer(*weights->store, source, qwen, storage_type, layer));
    }
    weights->qwen.final_norm = binding::norm_weight_from_source(
        *weights->store,
        source,
        "model.mdl_model.decoder.model.norm",
        assets.config.hidden_size);
    weights->audio_projector_in = binding::linear_from_source(
        *weights->store,
        source,
        "model.mdl_model.audio_projector.net.0",
        storage_type,
        assets.config.hidden_size,
        assets.config.target_embedding_size * assets.config.patch_size,
        true);
    weights->audio_projector_out = binding::linear_from_source(
        *weights->store,
        source,
        "model.mdl_model.audio_projector.net.2",
        storage_type,
        assets.config.hidden_size,
        assets.config.hidden_size,
        true);
    weights->stop_head = binding::linear_from_source(
        *weights->store,
        source,
        "model.stop_head",
        storage_type,
        2,
        assets.config.hidden_size,
        true);
    weights->store->upload();
    return weights;
}

std::vector<float> last_hidden_row(
    const std::vector<float> & hidden,
    int64_t steps,
    int64_t hidden_size) {
    if (static_cast<int64_t>(hidden.size()) == hidden_size) {
        return hidden;
    }
    if (static_cast<int64_t>(hidden.size()) != steps * hidden_size) {
        throw std::runtime_error("MiDashengLM-Gen AR hidden size mismatch");
    }
    return std::vector<float>(
        hidden.begin() + static_cast<std::ptrdiff_t>((steps - 1) * hidden_size),
        hidden.end());
}

float stop_probability(const std::vector<float> & logits) {
    if (logits.size() != 2) {
        throw std::runtime_error("MiDashengLM-Gen stop head expects two logits");
    }
    const float max_logit = std::max(logits[0], logits[1]);
    const float a = std::exp(logits[0] - max_logit);
    const float b = std::exp(logits[1] - max_logit);
    return b / (a + b);
}

}  // namespace

class MiDashengLmGenARRuntime::ProjectorGraph {
public:
    ProjectorGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const MiDashengLmGenARWeights> weights,
        MiDashengLmGenConfig config,
        int64_t batch,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          config_(config),
          batch_(batch) {
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        ggml_init_params input_params{8ull * 1024ull * 1024ull, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (ctx_ == nullptr || input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MiDashengLM-Gen audio projector graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "midashenglm_gen.ar.audio_projector", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{input_ctx_.get(), "midashenglm_gen.ar.audio_projector.inputs", execution_.backend_type()};
        input_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_, config_.patch_size, config_.target_embedding_size})).tensor;
        ggml_set_input(input_);
        auto x = core::wrap_tensor(input_, core::TensorShape::from_dims({batch_, config_.patch_size, config_.target_embedding_size}));
        x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x), core::TensorShape::from_dims({batch_, config_.patch_size * config_.target_embedding_size}));
        x = modules::LinearModule({config_.patch_size * config_.target_embedding_size, config_.hidden_size, true}).build(ctx, x, weights_->audio_projector_in);
        x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
        x = modules::LinearModule({config_.hidden_size, config_.hidden_size, true}).build(ctx, x, weights_->audio_projector_out);
        output_ = core::ensure_backend_addressable_layout(ctx, core::reshape_tensor(ctx, x, core::TensorShape::from_dims({batch_, 1, config_.hidden_size}))).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 4096, false);
        ggml_build_forward_expand(graph_, output_);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (input_buffer_ == nullptr || gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            clear_graph();
            throw std::runtime_error("failed to allocate MiDashengLM-Gen audio projector graph");
        }
    }

    ~ProjectorGraph() {
        clear_graph();
    }

    int64_t batch() const noexcept {
        return batch_;
    }

    std::vector<float> run(const std::vector<float> & latent) {
        if (latent.size() != static_cast<size_t>(batch_ * config_.patch_size * config_.target_embedding_size)) {
            throw std::runtime_error("MiDashengLM-Gen audio projector input size mismatch");
        }
        core::write_tensor_f32(core::wrap_tensor(input_, core::TensorShape::from_dims({batch_, config_.patch_size, config_.target_embedding_size})), latent);
        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiDashengLM-Gen audio projector graph compute failed");
        }
        return core::read_tensor_float(output_);
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
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const MiDashengLmGenARWeights> weights_;
    MiDashengLmGenConfig config_;
    int64_t batch_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
};

class MiDashengLmGenARRuntime::StopHeadGraph {
public:
    StopHeadGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const MiDashengLmGenARWeights> weights,
        MiDashengLmGenConfig config,
        int64_t batch,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          config_(config),
          batch_(batch) {
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        ggml_init_params input_params{4ull * 1024ull * 1024ull, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (ctx_ == nullptr || input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MiDashengLM-Gen stop head graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "midashenglm_gen.ar.stop_head", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{input_ctx_.get(), "midashenglm_gen.ar.stop_head.inputs", execution_.backend_type()};
        input_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_, config_.hidden_size})).tensor;
        ggml_set_input(input_);
        auto x = core::wrap_tensor(input_, core::TensorShape::from_dims({batch_, config_.hidden_size}));
        x = modules::LinearModule({config_.hidden_size, 2, true}).build(ctx, x, weights_->stop_head);
        output_ = core::ensure_backend_addressable_layout(ctx, x).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 4096, false);
        ggml_build_forward_expand(graph_, output_);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (input_buffer_ == nullptr || gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            clear_graph();
            throw std::runtime_error("failed to allocate MiDashengLM-Gen stop head graph");
        }
    }

    ~StopHeadGraph() {
        clear_graph();
    }

    int64_t batch() const noexcept {
        return batch_;
    }

    std::vector<float> run(const std::vector<float> & hidden) {
        if (hidden.size() != static_cast<size_t>(batch_ * config_.hidden_size)) {
            throw std::runtime_error("MiDashengLM-Gen stop head input size mismatch");
        }
        core::write_tensor_f32(core::wrap_tensor(input_, core::TensorShape::from_dims({batch_, config_.hidden_size})), hidden);
        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiDashengLM-Gen stop head graph compute failed");
        }
        return core::read_tensor_float(output_);
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
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const MiDashengLmGenARWeights> weights_;
    MiDashengLmGenConfig config_;
    int64_t batch_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
};

MiDashengLmGenARRuntime::MiDashengLmGenARRuntime(
    std::shared_ptr<const MiDashengLmGenAssets> assets,
    core::ExecutionContext & execution,
    MiDashengLmGenFlowRuntime & flow,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes,
    size_t helper_graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type)
    : assets_(std::move(assets)),
      execution_(&execution),
      flow_(&flow),
      helper_graph_arena_bytes_(helper_graph_arena_bytes) {
    if (assets_ == nullptr) {
        throw std::runtime_error("MiDashengLM-Gen AR runtime requires assets");
    }
    modules::QwenCausalDecoderConfig qwen_decoder;
    weights_ = load_weights(
        *assets_,
        execution.backend(),
        execution.backend_type(),
        weight_context_bytes,
        storage_type,
        qwen_decoder);
    modules::QwenCausalDecodeRuntimeConfig qwen_config;
    qwen_config.trace_name = "midashenglm_gen.ar.qwen";
    qwen_config.decoder = qwen_decoder;
    qwen_config.prefill_graph_arena_bytes = prefill_graph_arena_bytes;
    qwen_config.decode_graph_arena_bytes = decode_graph_arena_bytes;
    qwen_config.output_mode = modules::QwenCausalDecodeOutputMode::Hidden;
    qwen_config.return_hidden = true;
    qwen_ = std::make_unique<modules::QwenCausalDecodeRuntime>(
        execution,
        qwen_config,
        weights_->qwen);
}

MiDashengLmGenARRuntime::~MiDashengLmGenARRuntime() = default;

MiDashengLmGenAROutput MiDashengLmGenARRuntime::generate(
    const MiDashengLmGenPromptEncoderOutput & prompt,
    const MiDashengLmGenGenerationOptions & options) {
    const auto & config = assets_->config;
    const int64_t batch = prompt.batch;
    const int64_t num_iter = (options.seq_len + config.patch_size - 1) / config.patch_size;
    const int64_t frames = num_iter * config.patch_size;
    MiDashengLmGenAROutput out;
    out.batch = batch;
    out.frames = frames;
    out.dims = config.target_embedding_size;
    out.latents.assign(static_cast<size_t>(batch * frames * config.target_embedding_size), 0.0F);
    out.stop_probs.assign(static_cast<size_t>(batch * num_iter), 0.0F);
    if (projector_ == nullptr || projector_->batch() != 1) {
        projector_.reset();
        projector_ = std::make_unique<ProjectorGraph>(*execution_, weights_, config, 1, helper_graph_arena_bytes_);
    }
    if (stop_head_ == nullptr || stop_head_->batch() != 1) {
        stop_head_.reset();
        stop_head_ = std::make_unique<StopHeadGraph>(*execution_, weights_, config, 1, helper_graph_arena_bytes_);
    }

    uint64_t randn_offset_blocks = 0;
    for (int64_t b = 0; b < batch; ++b) {
        int64_t valid_tokens = 0;
        for (int64_t t = 0; t < prompt.tokens; ++t) {
            valid_tokens += prompt.attention_mask[static_cast<size_t>(b * prompt.tokens + t)] != 0 ? 1 : 0;
        }
        if (valid_tokens <= 0) {
            throw std::runtime_error("MiDashengLM-Gen AR prompt has no valid tokens");
        }
        std::vector<float> prompt_embeddings(static_cast<size_t>(valid_tokens * config.hidden_size));
        for (int64_t t = 0; t < valid_tokens; ++t) {
            const size_t src = static_cast<size_t>((b * prompt.tokens + t) * config.hidden_size);
            const size_t dst = static_cast<size_t>(t * config.hidden_size);
            std::copy(
                prompt.embeddings.begin() + static_cast<std::ptrdiff_t>(src),
                prompt.embeddings.begin() + static_cast<std::ptrdiff_t>(src + config.hidden_size),
                prompt_embeddings.begin() + static_cast<std::ptrdiff_t>(dst));
        }
        auto prefill = qwen_->prefill_embeddings(prompt_embeddings, valid_tokens);
        qwen_->start_decode_embeddings(prefill.state, valid_tokens + num_iter + 1);
        auto hidden = last_hidden_row(prefill.hidden, valid_tokens, config.hidden_size);
        std::vector<float> latent_history(static_cast<size_t>(config.patch_size * config.target_embedding_size), 0.0F);
        for (int64_t step = 0; step < num_iter; ++step) {
            MiDashengLmGenFlowInput flow_input;
            flow_input.x.clear();
            flow_input.latent_history = latent_history;
            flow_input.condition = hidden;
            flow_input.batch = 1;
            const auto sampled = flow_->sample(flow_input, options.eval_cfg, options.seed, randn_offset_blocks);
            latent_history = sampled;
            const size_t dst = static_cast<size_t>((b * frames + step * config.patch_size) * config.target_embedding_size);
            std::copy(sampled.begin(), sampled.end(), out.latents.begin() + static_cast<std::ptrdiff_t>(dst));
            const auto stop_logits = stop_head_->run(hidden);
            out.stop_probs[static_cast<size_t>(b * num_iter + step)] = stop_probability(stop_logits);
            const auto audio_embedding = projector_->run(sampled);
            if (step + 1 < num_iter) {
                auto decoded = qwen_->decode_embedding(audio_embedding);
                hidden = decoded.hidden;
            }
        }
    }
    return out;
}

void MiDashengLmGenARRuntime::release_graphs() {
    if (qwen_ != nullptr) {
        qwen_->release_runtime_graphs();
    }
    projector_.reset();
    stop_head_.reset();
}

}  // namespace engine::models::midashenglm_gen
