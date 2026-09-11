#include "engine/framework/modules/transformers/gemma_decoder.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::modules {
namespace {

void validate_config(const GemmaDecoderStackConfig & config) {
    if (config.hidden_size <= 0 || config.layers <= 0 || config.attention_heads <= 0 ||
        config.kv_heads <= 0 || config.head_dim <= 0 || config.intermediate_size <= 0 ||
        config.vocab_size <= 0) {
        throw std::runtime_error("GemmaDecoderStackConfig dimensions must be positive");
    }
    if (!(config.rope_theta > 0.0F) || !(config.local_rope_theta > 0.0F) ||
        !(config.rms_norm_eps > 0.0F) || !(config.query_pre_attn_scalar > 0.0F)) {
        throw std::runtime_error("GemmaDecoderStackConfig scalar values must be positive");
    }
}

core::TensorValue cast_f32(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    if (input.type == GGML_TYPE_F32 && input.tensor->type == GGML_TYPE_F32) {
        return input;
    }
    return core::wrap_tensor(
        ggml_cast(ctx.ggml, core::ensure_backend_addressable_layout(ctx, input).tensor, GGML_TYPE_F32),
        input.shape,
        GGML_TYPE_F32);
}

std::string join_name(std::string_view lhs, std::string_view rhs) {
    if (lhs.empty()) {
        return std::string(rhs);
    }
    if (rhs.empty()) {
        return std::string(lhs);
    }
    return std::string(lhs) + "." + std::string(rhs);
}

std::string layer_prefix(
    const GemmaDecoderWeightBinding & binding,
    int64_t layer_index) {
    return join_name(
        join_name(binding.model_prefix, binding.layers_prefix),
        std::to_string(layer_index));
}

core::TensorValue load_weight(
    core::BackendWeightStore & store,
    const GemmaDecoderTensorSourceResolver & resolver,
    const std::string & name,
    assets::TensorStorageType storage_type,
    std::initializer_list<int64_t> expected_shape) {
    return store.load_tensor(resolver(name), name, storage_type, expected_shape);
}

LinearWeights load_linear_weight(
    core::BackendWeightStore & store,
    const GemmaDecoderTensorSourceResolver & resolver,
    const std::string & name,
    assets::TensorStorageType storage_type,
    int64_t out_features,
    int64_t in_features) {
    const std::string weight_name = name + ".weight";
    return {
        store.load_tensor(resolver(weight_name), weight_name, storage_type, {out_features, in_features}),
        std::nullopt,
    };
}

GemmaDecoderLayerWeights load_layer_weights(
    core::BackendWeightStore & store,
    const GemmaDecoderStackConfig & config,
    const GemmaDecoderWeightBinding & binding,
    const GemmaDecoderTensorSourceResolver & resolver,
    int64_t layer_index) {
    const auto prefix = layer_prefix(binding, layer_index);
    const auto & names = binding.layer;
    GemmaDecoderLayerWeights weights;
    weights.input_norm = load_weight(
        store,
        resolver,
        join_name(prefix, names.input_norm),
        binding.norm_storage_type,
        {config.hidden_size});
    weights.post_attention_norm = load_weight(
        store,
        resolver,
        join_name(prefix, names.post_attention_norm),
        binding.norm_storage_type,
        {config.hidden_size});
    weights.pre_feedforward_norm = load_weight(
        store,
        resolver,
        join_name(prefix, names.pre_feedforward_norm),
        binding.norm_storage_type,
        {config.hidden_size});
    weights.post_feedforward_norm = load_weight(
        store,
        resolver,
        join_name(prefix, names.post_feedforward_norm),
        binding.norm_storage_type,
        {config.hidden_size});
    weights.self_attention.q_norm.weight = load_weight(
        store,
        resolver,
        join_name(prefix, names.q_norm),
        binding.norm_storage_type,
        {config.head_dim});
    weights.self_attention.q_norm.bias = std::nullopt;
    weights.self_attention.k_norm.weight = load_weight(
        store,
        resolver,
        join_name(prefix, names.k_norm),
        binding.norm_storage_type,
        {config.head_dim});
    weights.self_attention.k_norm.bias = std::nullopt;
    weights.self_attention.q_proj = load_linear_weight(
        store,
        resolver,
        join_name(prefix, names.q_proj),
        binding.projection_storage_type,
        config.attention_heads * config.head_dim,
        config.hidden_size);
    weights.self_attention.k_proj = load_linear_weight(
        store,
        resolver,
        join_name(prefix, names.k_proj),
        binding.projection_storage_type,
        config.kv_heads * config.head_dim,
        config.hidden_size);
    weights.self_attention.v_proj = load_linear_weight(
        store,
        resolver,
        join_name(prefix, names.v_proj),
        binding.projection_storage_type,
        config.kv_heads * config.head_dim,
        config.hidden_size);
    weights.self_attention.o_proj = load_linear_weight(
        store,
        resolver,
        join_name(prefix, names.o_proj),
        binding.projection_storage_type,
        config.hidden_size,
        config.attention_heads * config.head_dim);
    weights.feed_forward.gate_proj = load_linear_weight(
        store,
        resolver,
        join_name(prefix, names.gate_proj),
        binding.projection_storage_type,
        config.intermediate_size,
        config.hidden_size);
    weights.feed_forward.up_proj = load_linear_weight(
        store,
        resolver,
        join_name(prefix, names.up_proj),
        binding.projection_storage_type,
        config.intermediate_size,
        config.hidden_size);
    weights.feed_forward.down_proj = load_linear_weight(
        store,
        resolver,
        join_name(prefix, names.down_proj),
        binding.projection_storage_type,
        config.hidden_size,
        config.intermediate_size);
    return weights;
}

ProjectedGroupedSelfAttentionConfig attention_config(const GemmaDecoderStackConfig & config) {
    ProjectedGroupedSelfAttentionConfig out;
    out.hidden_size = config.hidden_size;
    out.attention_heads = config.attention_heads;
    out.kv_heads = config.kv_heads;
    out.head_dim = config.head_dim;
    out.use_weight_type_projection_precision = true;
    out.use_fast_cuda_projection = config.use_fast_cuda_projection;
    out.lowering = GroupedQueryAttentionLowering::FlashGrouped;
    out.qk_norm = ProjectedGroupedSelfAttentionQKNorm::GemmaRMSNorm;
    out.qk_norm_eps = config.rms_norm_eps;
    out.rope_theta = config.rope_theta;
    out.local_rope_theta = config.local_rope_theta;
    out.rope_freq_scale = config.rope_freq_scale;
    out.sliding_window_pattern = config.sliding_window_pattern;
    out.query_pre_attn_scalar = config.query_pre_attn_scalar;
    return out;
}

GatedFeedForwardConfig feed_forward_config(const GemmaDecoderStackConfig & config) {
    GatedFeedForwardConfig out;
    out.hidden_size = config.hidden_size;
    out.intermediate_size = config.intermediate_size;
    out.activation = GatedFeedForwardActivation::Gelu;
    out.gelu_approximation = GeluApproximation::Tanh;
    out.use_weight_type_projection_precision = true;
    out.use_fast_cuda_projection = config.use_fast_cuda_projection;
    return out;
}

core::TensorValue layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const core::TensorValue & additive_attention_mask,
    const GemmaDecoderLayerWeights & weights,
    const GemmaDecoderStackConfig & config,
    int64_t layer_index) {
    const GemmaRMSNormModule norm({config.hidden_size, config.rms_norm_eps, true, false});
    auto hidden = norm.build(ctx, input, {weights.input_norm, std::nullopt});
    hidden = ProjectedGroupedSelfAttentionModule(attention_config(config))
                 .build(ctx, hidden, positions, weights.self_attention, layer_index, additive_attention_mask);
    hidden = norm.build(ctx, hidden, {weights.post_attention_norm, std::nullopt});
    auto output = AddModule{}.build(ctx, input, hidden);
    hidden = norm.build(ctx, output, {weights.pre_feedforward_norm, std::nullopt});
    hidden = GatedFeedForwardModule(feed_forward_config(config)).build(ctx, hidden, weights.feed_forward);
    hidden = norm.build(ctx, hidden, {weights.post_feedforward_norm, std::nullopt});
    return AddModule{}.build(ctx, output, hidden);
}

}  // namespace

GemmaDecoderStackWeights load_gemma_decoder_stack_weights(
    core::BackendWeightStore & store,
    const GemmaDecoderStackConfig & config,
    const GemmaDecoderWeightBinding & binding,
    const GemmaDecoderTensorSourceResolver & resolver) {
    validate_config(config);
    GemmaDecoderStackWeights weights;
    const std::string embed_name = join_name(binding.model_prefix, binding.embed_tokens);
    weights.embed_tokens = store.load_tensor(
        resolver(embed_name),
        embed_name,
        binding.embedding_storage_type,
        {config.vocab_size, config.hidden_size});
    weights.layers.reserve(static_cast<size_t>(config.layers));
    for (int64_t layer_index = 0; layer_index < config.layers; ++layer_index) {
        weights.layers.push_back(load_layer_weights(store, config, binding, resolver, layer_index));
    }
    const std::string norm_name = join_name(binding.model_prefix, binding.final_norm);
    weights.norm = store.load_tensor(
        resolver(norm_name),
        norm_name,
        binding.norm_storage_type,
        {config.hidden_size});
    return weights;
}

GemmaDecoderStackModule::GemmaDecoderStackModule(GemmaDecoderStackConfig config) : config_(config) {
    validate_config(config_);
}

const GemmaDecoderStackConfig & GemmaDecoderStackModule::config() const noexcept {
    return config_;
}

GemmaDecoderStackOutputs GemmaDecoderStackModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_ids,
    const core::TensorValue & positions,
    const core::TensorValue & additive_attention_mask,
    const GemmaDecoderStackWeights & weights,
    bool capture_input_and_layers) const {
    core::validate_rank_between(input_ids, 2, 2, "GemmaDecoderStack input_ids");
    core::validate_shape(positions, core::TensorShape::from_dims({input_ids.shape.dims[1]}), "GemmaDecoderStack positions");
    if (additive_attention_mask.shape.rank != 4 ||
        additive_attention_mask.shape.dims[0] != input_ids.shape.dims[0] ||
        !(additive_attention_mask.shape.dims[1] == 1 || additive_attention_mask.shape.dims[1] == config_.attention_heads) ||
        additive_attention_mask.shape.dims[2] != input_ids.shape.dims[1] ||
        additive_attention_mask.shape.dims[3] != input_ids.shape.dims[1]) {
        throw std::runtime_error("GemmaDecoderStack additive attention mask shape mismatch");
    }
    if (static_cast<int64_t>(weights.layers.size()) != config_.layers) {
        throw std::runtime_error("GemmaDecoderStack layer count mismatch");
    }

    GemmaDecoderStackOutputs outputs;
    auto hidden = EmbeddingModule({config_.vocab_size, config_.hidden_size}).build(ctx, input_ids, weights.embed_tokens);
    if (config_.scale_embeddings) {
        hidden = core::wrap_tensor(
            ggml_scale(ctx.ggml, cast_f32(ctx, hidden).tensor, std::sqrt(static_cast<float>(config_.hidden_size))),
            hidden.shape,
            GGML_TYPE_F32);
    }
    if (capture_input_and_layers) {
        outputs.captured_hidden_states.reserve(static_cast<size_t>(config_.layers + 1));
        outputs.captured_hidden_states.push_back(hidden);
    }
    for (int64_t layer_index = 0; layer_index < config_.layers; ++layer_index) {
        hidden = layer(
            ctx,
            hidden,
            positions,
            additive_attention_mask,
            weights.layers[static_cast<size_t>(layer_index)],
            config_,
            layer_index);
        if (capture_input_and_layers) {
            outputs.captured_hidden_states.push_back(hidden);
        }
    }
    outputs.hidden = GemmaRMSNormModule({config_.hidden_size, config_.rms_norm_eps, true, false}).build(
        ctx,
        hidden,
        {weights.norm, std::nullopt});
    if (capture_input_and_layers && !outputs.captured_hidden_states.empty()) {
        outputs.captured_hidden_states.back() = outputs.hidden;
    }
    return outputs;
}

GemmaDecoderComponent GemmaDecoderComponent::load_from_tensor_source(
    std::shared_ptr<const assets::TensorSource> source,
    ggml_backend_t backend,
    core::BackendType backend_type,
    GemmaDecoderStackConfig config,
    GemmaDecoderWeightBinding binding,
    size_t weight_context_bytes) {
    if (source == nullptr) {
        throw std::runtime_error("GemmaDecoderComponent requires a tensor source");
    }
    return load_from_resolver(
        backend,
        backend_type,
        std::move(config),
        std::move(binding),
        weight_context_bytes,
        [source = std::move(source)](std::string_view) -> const assets::TensorSource & {
            return *source;
        });
}

GemmaDecoderComponent GemmaDecoderComponent::load_from_resolver(
    ggml_backend_t backend,
    core::BackendType backend_type,
    GemmaDecoderStackConfig config,
    GemmaDecoderWeightBinding binding,
    size_t weight_context_bytes,
    const GemmaDecoderTensorSourceResolver & resolver) {
    auto store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "gemma_decoder.weights",
        weight_context_bytes);
    auto weights = std::make_shared<GemmaDecoderStackWeights>(
        load_gemma_decoder_stack_weights(*store, config, binding, resolver));
    store->upload();
    return GemmaDecoderComponent(std::move(config), std::move(store), std::move(weights));
}

GemmaDecoderComponent::GemmaDecoderComponent(
    GemmaDecoderStackConfig config,
    std::shared_ptr<core::BackendWeightStore> store,
    std::shared_ptr<const GemmaDecoderStackWeights> weights)
    : config_(std::move(config)),
      store_(std::move(store)),
      weights_(std::move(weights)) {
    validate_config(config_);
    if (store_ == nullptr || weights_ == nullptr) {
        throw std::runtime_error("GemmaDecoderComponent requires loaded weights");
    }
}

const GemmaDecoderStackConfig & GemmaDecoderComponent::config() const noexcept {
    return config_;
}

const std::shared_ptr<const GemmaDecoderStackWeights> & GemmaDecoderComponent::weights() const noexcept {
    return weights_;
}

const std::shared_ptr<core::BackendWeightStore> & GemmaDecoderComponent::store() const noexcept {
    return store_;
}

GemmaDecoderStackOutputs GemmaDecoderComponent::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_ids,
    const core::TensorValue & positions,
    const core::TensorValue & additive_attention_mask,
    bool capture_input_and_layers) const {
    if (weights_ == nullptr) {
        throw std::runtime_error("GemmaDecoderComponent has no loaded weights");
    }
    return GemmaDecoderStackModule(config_).build(
        ctx,
        input_ids,
        positions,
        additive_attention_mask,
        *weights_,
        capture_input_and_layers);
}

}  // namespace engine::modules
