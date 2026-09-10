#include "engine/framework/midi/sheetsage2_runtime.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::midi {
namespace {

namespace modules = engine::modules;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct SheetSage2AttentionWeights {
    modules::LinearWeights q_proj;
    modules::LinearWeights k_proj;
    modules::LinearWeights v_proj;
    modules::LinearWeights out_proj;
};

struct SheetSage2DecoderLayerWeights {
    SheetSage2AttentionWeights self_attn;
    modules::NormWeights self_attn_layer_norm;
    SheetSage2AttentionWeights encoder_attn;
    modules::NormWeights encoder_attn_layer_norm;
    modules::LinearWeights fc1;
    modules::LinearWeights fc2;
    modules::NormWeights final_layer_norm;
};

struct SheetSage2DecoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue token_embedding;
    core::TensorValue position_embedding;
    modules::NormWeights layernorm_embedding;
    modules::LinearWeights encoder_projection;
    std::vector<SheetSage2DecoderLayerWeights> layers;
};

void validate_config(const SheetSage2DecoderConfig & config) {
    if (config.vocab_size <= 0 || config.hidden_size <= 0 || config.encoder_hidden_size <= 0 ||
        config.intermediate_size <= 0 || config.decoder_layers <= 0 ||
        config.num_attention_heads <= 0 || config.max_position_embeddings <= 0) {
        throw std::runtime_error("SheetSage2 decoder config dimensions must be positive");
    }
    if (config.hidden_size % config.num_attention_heads != 0) {
        throw std::runtime_error("SheetSage2 decoder hidden size must be divisible by head count");
    }
}

modules::LinearWeights load_linear(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t out_features,
    int64_t in_features,
    bool use_bias) {
    modules::LinearWeights weights;
    weights.weight = store.load_tensor(source, prefix + ".weight", storage_type, {out_features, in_features});
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_features});
    }
    return weights;
}

modules::NormWeights load_norm(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t hidden) {
    return {
        store.load_f32_tensor(source, prefix + ".weight", {hidden}),
        store.load_f32_tensor(source, prefix + ".bias", {hidden}),
    };
}

SheetSage2AttentionWeights load_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t hidden) {
    return {
        load_linear(store, source, prefix + ".q_proj", storage_type, hidden, hidden, true),
        load_linear(store, source, prefix + ".k_proj", storage_type, hidden, hidden, true),
        load_linear(store, source, prefix + ".v_proj", storage_type, hidden, hidden, true),
        load_linear(store, source, prefix + ".out_proj", storage_type, hidden, hidden, true),
    };
}

SheetSage2DecoderWeights load_weights(
    const assets::TensorSource & source,
    const SheetSage2DecoderConfig & config,
    ggml_backend_t backend,
    core::BackendType backend_type,
    const SheetSage2DecoderRuntimeOptions & options) {
    SheetSage2DecoderWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "framework.sheetsage2.decoder.weights",
        options.weight_context_bytes);
    weights.token_embedding = weights.store->load_tensor(
        source,
        "token_embedding.weight",
        options.weight_storage_type,
        {config.vocab_size, config.hidden_size});
    weights.position_embedding = weights.store->load_tensor(
        source,
        "decoder.embed_positions.weight",
        options.weight_storage_type,
        {config.max_position_embeddings + 2, config.hidden_size});
    weights.layernorm_embedding = load_norm(*weights.store, source, "decoder.layernorm_embedding", config.hidden_size);
    weights.encoder_projection = load_linear(
        *weights.store,
        source,
        "encoder_projection",
        options.weight_storage_type,
        config.hidden_size,
        config.encoder_hidden_size,
        true);
    weights.layers.reserve(static_cast<size_t>(config.decoder_layers));
    for (int64_t i = 0; i < config.decoder_layers; ++i) {
        const std::string prefix = "decoder.layers." + std::to_string(i);
        SheetSage2DecoderLayerWeights layer;
        layer.self_attn = load_attention(*weights.store, source, prefix + ".self_attn", options.weight_storage_type, config.hidden_size);
        layer.self_attn_layer_norm = load_norm(*weights.store, source, prefix + ".self_attn_layer_norm", config.hidden_size);
        layer.encoder_attn = load_attention(*weights.store, source, prefix + ".encoder_attn", options.weight_storage_type, config.hidden_size);
        layer.encoder_attn_layer_norm = load_norm(*weights.store, source, prefix + ".encoder_attn_layer_norm", config.hidden_size);
        layer.fc1 = load_linear(*weights.store, source, prefix + ".fc1", options.weight_storage_type, config.intermediate_size, config.hidden_size, true);
        layer.fc2 = load_linear(*weights.store, source, prefix + ".fc2", options.weight_storage_type, config.hidden_size, config.intermediate_size, true);
        layer.final_layer_norm = load_norm(*weights.store, source, prefix + ".final_layer_norm", config.hidden_size);
        weights.layers.push_back(std::move(layer));
    }
    weights.store->upload();
    return weights;
}

core::TensorValue split_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t head_dim) {
    auto shaped = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, input),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, head_dim}));
    return modules::TransposeModule({{0, 2, 1, 3}, shaped.shape.rank}).build(ctx, shaped);
}

core::TensorValue merge_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t hidden) {
    return core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, input),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], hidden}));
}

core::TensorValue attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & hidden,
    const core::TensorValue & key_value,
    const SheetSage2AttentionWeights & weights,
    int64_t hidden_size,
    int64_t heads,
    bool causal) {
    const int64_t head_dim = hidden_size / heads;
    auto q = modules::LinearModule({hidden_size, hidden_size, true}).build(ctx, hidden, weights.q_proj);
    auto k = modules::LinearModule({hidden_size, hidden_size, true}).build(ctx, key_value, weights.k_proj);
    auto v = modules::LinearModule({hidden_size, hidden_size, true}).build(ctx, key_value, weights.v_proj);
    q = split_heads(ctx, q, heads, head_dim);
    k = split_heads(ctx, k, heads, head_dim);
    v = split_heads(ctx, v, heads, head_dim);
    auto context = modules::ScaledDotProductAttentionModule({
        head_dim,
        modules::ScaledDotProductAttentionLowering::Explicit,
        GGML_PREC_F32,
        causal ? modules::AttentionCausality::Causal : modules::AttentionCausality::NonCausal,
    }).build(ctx, q, k, v);
    context = merge_heads(ctx, context, hidden_size);
    return modules::LinearModule({hidden_size, hidden_size, true}).build(ctx, context, weights.out_proj);
}

core::TensorValue decoder_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & memory,
    const SheetSage2DecoderLayerWeights & weights,
    const SheetSage2DecoderConfig & config) {
    auto hidden = modules::ResidualAddModule{}.build(
        ctx,
        attention(ctx, input, input, weights.self_attn, config.hidden_size, config.num_attention_heads, true),
        input);
    hidden = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true}).build(
        ctx,
        hidden,
        weights.self_attn_layer_norm);
    auto cross = attention(ctx, hidden, memory, weights.encoder_attn, config.hidden_size, config.num_attention_heads, false);
    hidden = modules::ResidualAddModule{}.build(ctx, cross, hidden);
    hidden = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true}).build(
        ctx,
        hidden,
        weights.encoder_attn_layer_norm);
    auto ff = modules::LinearModule({config.hidden_size, config.intermediate_size, true}).build(ctx, hidden, weights.fc1);
    ff = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, ff);
    ff = modules::LinearModule({config.intermediate_size, config.hidden_size, true}).build(ctx, ff, weights.fc2);
    hidden = modules::ResidualAddModule{}.build(ctx, ff, hidden);
    return modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true}).build(
        ctx,
        hidden,
        weights.final_layer_norm);
}

}  // namespace

struct SheetSage2DecoderRuntime::Impl {
    class DecodeGraph;

    Impl(
        std::shared_ptr<const assets::TensorSource> source,
        core::ExecutionContext & execution,
        SheetSage2DecoderConfig config,
        SheetSage2DecoderRuntimeOptions options)
        : source(std::move(source)),
          execution(&execution),
          config(config),
          options(options) {
        if (!this->source) {
            throw std::runtime_error("SheetSage2 decoder runtime requires tensor source");
        }
        validate_config(this->config);
    }

    const SheetSage2DecoderWeights & require_weights() {
        if (!weights) {
            weights = std::make_unique<SheetSage2DecoderWeights>(load_weights(
                *source,
                config,
                execution->backend(),
                execution->backend_type(),
                options));
            source->release_storage();
        }
        return *weights;
    }

    std::shared_ptr<const assets::TensorSource> source;
    core::ExecutionContext * execution = nullptr;
    SheetSage2DecoderConfig config;
    SheetSage2DecoderRuntimeOptions options;
    std::unique_ptr<SheetSage2DecoderWeights> weights;
    std::unique_ptr<DecodeGraph> graph;
};

class SheetSage2DecoderRuntime::Impl::DecodeGraph {
public:
    DecodeGraph(
        core::ExecutionContext & execution,
        const SheetSage2DecoderConfig & config,
        const SheetSage2DecoderRuntimeOptions & options,
        const SheetSage2DecoderWeights & weights,
        int64_t batch,
        int64_t memory_steps,
        int64_t decoder_steps)
        : backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          config_(config),
          options_(options),
          weights_(weights),
          batch_(batch),
          memory_steps_(memory_steps),
          decoder_steps_(decoder_steps) {
        if (backend_ == nullptr || batch_ <= 0 || memory_steps_ <= 0 || decoder_steps_ <= 0) {
            throw std::runtime_error("SheetSage2 decoder graph initialization failed");
        }
        if (decoder_steps_ > config_.max_position_embeddings) {
            throw std::runtime_error("SheetSage2 decoder steps exceed max position embeddings");
        }
        build();
    }

    ~DecodeGraph() {
        if (backend_ != nullptr && graph_ != nullptr) {
            core::release_backend_graph_resources(backend_type_, backend_, graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(int64_t batch, int64_t memory_steps, int64_t decoder_steps) const noexcept {
        return batch == batch_ && memory_steps == memory_steps_ && decoder_steps == decoder_steps_;
    }

    std::vector<float> run(
        const std::vector<float> & mixed_encoder_state,
        const std::vector<int32_t> & decoder_input_ids) const {
        if (static_cast<int64_t>(mixed_encoder_state.size()) != batch_ * memory_steps_ * config_.encoder_hidden_size) {
            throw std::runtime_error("SheetSage2 mixed encoder state shape mismatch");
        }
        if (static_cast<int64_t>(decoder_input_ids.size()) != batch_ * decoder_steps_) {
            throw std::runtime_error("SheetSage2 decoder input id shape mismatch");
        }
        core::write_tensor_f32(mixed_encoder_state_, mixed_encoder_state);
        core::write_tensor_i32(decoder_input_ids_, decoder_input_ids);
        core::write_tensor_i32(position_ids_, position_ids());
        core::set_backend_threads(backend_, threads_);
        const ggml_status status = core::compute_backend_graph(backend_, graph_, nullptr, "framework.sheetsage2.decoder");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("SheetSage2 decoder graph compute failed");
        }
        return core::read_tensor_f32(logits_);
    }

private:
    std::vector<int32_t> position_ids() const {
        std::vector<int32_t> ids(static_cast<size_t>(batch_ * decoder_steps_));
        for (int64_t b = 0; b < batch_; ++b) {
            for (int64_t t = 0; t < decoder_steps_; ++t) {
                ids[static_cast<size_t>(b * decoder_steps_ + t)] = static_cast<int32_t>(t + 2);
            }
        }
        return ids;
    }

    void build() {
        ggml_init_params params{options_.graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("SheetSage2 decoder ggml context initialization failed");
        }
        core::ModuleBuildContext input_ctx{ctx_.get(), "framework.sheetsage2.decoder.inputs", backend_type_};
        mixed_encoder_state_ = core::make_tensor(
            input_ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({batch_, memory_steps_, config_.encoder_hidden_size}));
        decoder_input_ids_ = core::make_tensor(
            input_ctx,
            GGML_TYPE_I32,
            core::TensorShape::from_dims({batch_, decoder_steps_}));
        position_ids_ = core::make_tensor(
            input_ctx,
            GGML_TYPE_I32,
            core::TensorShape::from_dims({batch_, decoder_steps_}));
        ggml_set_input(mixed_encoder_state_.tensor);
        ggml_set_input(decoder_input_ids_.tensor);
        ggml_set_input(position_ids_.tensor);
        core::ModuleBuildContext build_ctx{ctx_.get(), "framework.sheetsage2.decoder", backend_type_};
        auto logits = build_graph_output(build_ctx);
        logits_ = logits.tensor;
        ggml_set_output(logits_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 524288, false);
        ggml_build_forward_expand(graph_, logits_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("SheetSage2 decoder backend buffer allocation failed");
        }
    }

    core::TensorValue build_graph_output(core::ModuleBuildContext & ctx) const {
        auto memory = modules::LinearModule({config_.encoder_hidden_size, config_.hidden_size, true}).build(
            ctx,
            mixed_encoder_state_,
            weights_.encoder_projection);
        auto hidden = modules::EmbeddingModule({config_.vocab_size, config_.hidden_size}).build(
            ctx,
            decoder_input_ids_,
            weights_.token_embedding);
        auto positions = modules::EmbeddingModule({config_.max_position_embeddings + 2, config_.hidden_size}).build(
            ctx,
            position_ids_,
            weights_.position_embedding);
        hidden = modules::AddModule{}.build(ctx, hidden, positions);
        hidden = modules::LayerNormModule({config_.hidden_size, config_.layer_norm_eps, true, true}).build(
            ctx,
            hidden,
            weights_.layernorm_embedding);
        for (const auto & layer : weights_.layers) {
            hidden = decoder_layer(ctx, hidden, memory, layer, config_);
        }
        return modules::LinearModule({config_.hidden_size, config_.vocab_size, false}).build(
            ctx,
            hidden,
            {weights_.token_embedding, std::nullopt});
    }

    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    SheetSage2DecoderConfig config_;
    SheetSage2DecoderRuntimeOptions options_;
    const SheetSage2DecoderWeights & weights_;
    int64_t batch_ = 0;
    int64_t memory_steps_ = 0;
    int64_t decoder_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue mixed_encoder_state_;
    core::TensorValue decoder_input_ids_;
    core::TensorValue position_ids_;
    ggml_tensor * logits_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

SheetSage2DecoderRuntime::SheetSage2DecoderRuntime(
    std::shared_ptr<const assets::TensorSource> source,
    core::ExecutionContext & execution,
    SheetSage2DecoderConfig config,
    SheetSage2DecoderRuntimeOptions options)
    : impl_(std::make_unique<Impl>(std::move(source), execution, config, options)) {}

SheetSage2DecoderRuntime::~SheetSage2DecoderRuntime() = default;
SheetSage2DecoderRuntime::SheetSage2DecoderRuntime(SheetSage2DecoderRuntime &&) noexcept = default;
SheetSage2DecoderRuntime & SheetSage2DecoderRuntime::operator=(SheetSage2DecoderRuntime &&) noexcept = default;

void SheetSage2DecoderRuntime::prepare(int64_t batch, int64_t memory_steps, int64_t decoder_steps) {
    const auto & weights = impl_->require_weights();
    if (!impl_->graph || !impl_->graph->matches(batch, memory_steps, decoder_steps)) {
        impl_->graph = std::make_unique<Impl::DecodeGraph>(
            *impl_->execution,
            impl_->config,
            impl_->options,
            weights,
            batch,
            memory_steps,
            decoder_steps);
    }
}

std::vector<float> SheetSage2DecoderRuntime::decode_logits(
    const std::vector<float> & mixed_encoder_state,
    int64_t memory_steps,
    const std::vector<int32_t> & decoder_input_ids) {
    if (memory_steps <= 0) {
        throw std::runtime_error("SheetSage2 memory steps must be positive");
    }
    if (decoder_input_ids.empty()) {
        throw std::runtime_error("SheetSage2 decoder input ids must not be empty");
    }
    const int64_t memory_values_per_batch = memory_steps * impl_->config.encoder_hidden_size;
    if (static_cast<int64_t>(mixed_encoder_state.size()) % memory_values_per_batch != 0) {
        throw std::runtime_error("SheetSage2 mixed encoder state does not divide into batches");
    }
    const int64_t batch = static_cast<int64_t>(mixed_encoder_state.size()) / memory_values_per_batch;
    if (static_cast<int64_t>(decoder_input_ids.size()) % batch != 0) {
        throw std::runtime_error("SheetSage2 decoder input ids do not divide by batch");
    }
    const int64_t decoder_steps = static_cast<int64_t>(decoder_input_ids.size()) / batch;
    prepare(batch, memory_steps, decoder_steps);
    return impl_->graph->run(mixed_encoder_state, decoder_input_ids);
}

void SheetSage2DecoderRuntime::release_runtime_graphs() {
    impl_->graph.reset();
}

}  // namespace engine::midi
