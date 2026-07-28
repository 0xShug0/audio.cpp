#include "engine/models/dramabox/gemma3_encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/grouped_query_attention.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_projection_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace engine::models::dramabox {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kGemmaPromptWeightContextBytes = 28ull * 1024ull * 1024ull * 1024ull;
constexpr float kFeatureRmsNormEps = 1.0e-6F;
constexpr float kMaskNegInf = -std::numeric_limits<float>::max();

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

const assets::TensorSource & source_for(
    const std::vector<std::shared_ptr<const assets::TensorSource>> & sources,
    std::string_view name) {
    for (const auto & source : sources) {
        if (source != nullptr && source->has_tensor(name)) {
            return *source;
        }
    }
    throw std::runtime_error("missing DramaBox Gemma tensor: " + std::string(name));
}

modules::LinearWeights load_gemma_linear(
    core::BackendWeightStore & store,
    const DramaBoxAssets & assets,
    const std::string & name,
    assets::TensorStorageType storage_type,
    int64_t out_features,
    int64_t in_features) {
    const std::string weight_name = name + ".weight";
    const auto & source = source_for(assets.gemma_weights, weight_name);
    return {
        store.load_tensor(source, weight_name, storage_type, {out_features, in_features}),
        std::nullopt,
    };
}

core::TensorValue load_gemma_tensor(
    core::BackendWeightStore & store,
    const DramaBoxAssets & assets,
    const std::string & name,
    assets::TensorStorageType storage_type,
    std::initializer_list<int64_t> expected_shape) {
    return store.load_tensor(source_for(assets.gemma_weights, name), name, storage_type, expected_shape);
}

std::vector<modules::LinearWeights> load_aggregate_layers(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t output_size,
    int64_t hidden_size,
    int64_t hidden_states) {
    const int64_t flat_size = hidden_size * hidden_states;
    const auto raw = source.require_f32(prefix + ".weight", {output_size, flat_size});
    std::vector<modules::LinearWeights> layers;
    layers.reserve(static_cast<size_t>(hidden_states));
    for (int64_t layer_index = 0; layer_index < hidden_states; ++layer_index) {
        std::vector<float> packed(static_cast<size_t>(output_size * hidden_size), 0.0F);
        for (int64_t out = 0; out < output_size; ++out) {
            for (int64_t dim = 0; dim < hidden_size; ++dim) {
                packed[static_cast<size_t>(out * hidden_size + dim)] =
                    raw[static_cast<size_t>(out * flat_size + dim * hidden_states + layer_index)];
            }
        }
        modules::LinearWeights weights;
        weights.weight = store.make_from_f32(
            core::TensorShape::from_dims({output_size, hidden_size}),
            storage_type,
            std::move(packed));
        weights.bias = std::nullopt;
        layers.push_back(std::move(weights));
    }
    return layers;
}

DramaBoxGemma3LayerWeights load_layer(
    core::BackendWeightStore & store,
    const DramaBoxAssets & assets,
    int64_t layer,
    assets::TensorStorageType storage_type) {
    const auto prefix = "language_model.model.layers." + std::to_string(layer);
    const auto & config = assets.config.gemma;
    DramaBoxGemma3LayerWeights weights;
    const auto & input_norm_source = source_for(assets.gemma_weights, prefix + ".input_layernorm.weight");
    weights.input_norm = input_norm_source.require_tensor_data(prefix + ".input_layernorm.weight").metadata.dtype == "BF16"
        ? store.load_tensor(input_norm_source, prefix + ".input_layernorm.weight", assets::TensorStorageType::BF16, {config.hidden_size})
        : store.load_f32_tensor(input_norm_source, prefix + ".input_layernorm.weight", {config.hidden_size});
    weights.post_attention_norm = load_gemma_tensor(
        store,
        assets,
        prefix + ".post_attention_layernorm.weight",
        assets::TensorStorageType::Native,
        {config.hidden_size});
    weights.pre_feedforward_norm = load_gemma_tensor(
        store,
        assets,
        prefix + ".pre_feedforward_layernorm.weight",
        assets::TensorStorageType::Native,
        {config.hidden_size});
    weights.post_feedforward_norm = load_gemma_tensor(
        store,
        assets,
        prefix + ".post_feedforward_layernorm.weight",
        assets::TensorStorageType::Native,
        {config.hidden_size});
    weights.q_norm = load_gemma_tensor(
        store,
        assets,
        prefix + ".self_attn.q_norm.weight",
        assets::TensorStorageType::Native,
        {config.head_dim});
    weights.k_norm = load_gemma_tensor(
        store,
        assets,
        prefix + ".self_attn.k_norm.weight",
        assets::TensorStorageType::Native,
        {config.head_dim});
    weights.q_proj = load_gemma_linear(
        store,
        assets,
        prefix + ".self_attn.q_proj",
        storage_type,
        config.num_attention_heads * config.head_dim,
        config.hidden_size);
    weights.k_proj = load_gemma_linear(
        store,
        assets,
        prefix + ".self_attn.k_proj",
        storage_type,
        config.num_key_value_heads * config.head_dim,
        config.hidden_size);
    weights.v_proj = load_gemma_linear(
        store,
        assets,
        prefix + ".self_attn.v_proj",
        storage_type,
        config.num_key_value_heads * config.head_dim,
        config.hidden_size);
    weights.o_proj = load_gemma_linear(
        store,
        assets,
        prefix + ".self_attn.o_proj",
        storage_type,
        config.hidden_size,
        config.num_attention_heads * config.head_dim);
    weights.gate_proj = load_gemma_linear(
        store,
        assets,
        prefix + ".mlp.gate_proj",
        storage_type,
        config.intermediate_size,
        config.hidden_size);
    weights.up_proj = load_gemma_linear(
        store,
        assets,
        prefix + ".mlp.up_proj",
        storage_type,
        config.intermediate_size,
        config.hidden_size);
    weights.down_proj = load_gemma_linear(
        store,
        assets,
        prefix + ".mlp.down_proj",
        storage_type,
        config.hidden_size,
        config.intermediate_size);
    return weights;
}

core::TensorValue linear_projection(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::LinearWeights & weights,
    int64_t in_features,
    int64_t out_features) {
    const bool dense_weight =
        weights.weight.type == GGML_TYPE_F32 || weights.weight.type == GGML_TYPE_F16 || weights.weight.type == GGML_TYPE_BF16;
    const ggml_prec precision = ggml_is_quantized(weights.weight.type) ? GGML_PREC_DEFAULT : GGML_PREC_F32;
    if (ctx.backend_type == core::BackendType::Cuda && dense_weight && out_features >= 128 && out_features % 4 == 0) {
        return modules::FastPackedProjection4Module({in_features, out_features, precision})
            .build(ctx, input, weights);
    }
    return modules::LinearModule({in_features, out_features, false, precision}).build(ctx, input, weights);
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

core::TensorValue grouped_attention_from_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    const core::TensorValue & additive_attention_mask,
    const DramaBoxGemma3Config & config) {
    if (config.query_pre_attn_scalar != static_cast<float>(config.head_dim)) {
        throw std::runtime_error("DramaBox Gemma grouped attention requires query_pre_attn_scalar == head_dim");
    }
    return modules::GroupedQueryAttentionModule({
        config.head_dim,
        ctx.backend_type == core::BackendType::Cuda
            ? modules::GroupedQueryAttentionLowering::FlashGrouped
            : modules::GroupedQueryAttentionLowering::ManualRepeat,
        GGML_PREC_F32,
        modules::AttentionCausality::NonCausal,
    }).build(ctx, q_heads, k_heads, v_heads, additive_attention_mask);
}

core::TensorValue self_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const core::TensorValue & additive_attention_mask,
    const DramaBoxGemma3LayerWeights & weights,
    const DramaBoxGemma3Config & config,
    int64_t layer_index) {
    auto q = linear_projection(ctx, input, weights.q_proj, config.hidden_size, config.num_attention_heads * config.head_dim);
    auto k = linear_projection(ctx, input, weights.k_proj, config.hidden_size, config.num_key_value_heads * config.head_dim);
    auto v = linear_projection(ctx, input, weights.v_proj, config.hidden_size, config.num_key_value_heads * config.head_dim);
    const modules::GemmaRMSNormModule head_norm({config.head_dim, config.rms_norm_eps, true, false});
    q = head_norm.build(
        ctx,
        core::reshape_tensor(
            ctx,
            core::ensure_backend_addressable_layout(ctx, q),
            core::TensorShape::from_dims({q.shape.dims[0], q.shape.dims[1], config.num_attention_heads, config.head_dim})),
        {weights.q_norm, std::nullopt});
    k = head_norm.build(
        ctx,
        core::reshape_tensor(
            ctx,
            core::ensure_backend_addressable_layout(ctx, k),
            core::TensorShape::from_dims({k.shape.dims[0], k.shape.dims[1], config.num_key_value_heads, config.head_dim})),
        {weights.k_norm, std::nullopt});
    v = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, v),
        core::TensorShape::from_dims({v.shape.dims[0], v.shape.dims[1], config.num_key_value_heads, config.head_dim}));
    const bool full_attention_layer =
        config.sliding_window_pattern > 0 && ((layer_index + 1) % config.sliding_window_pattern) == 0;
    const float rope_theta = full_attention_layer ? config.rope_theta : config.rope_local_base_freq;
    const float rope_freq_scale = full_attention_layer ? (1.0F / config.rope_scaling_factor) : 1.0F;
    q = modules::RoPEModule({config.head_dim, GGML_ROPE_TYPE_NEOX, rope_theta, rope_freq_scale}).build(ctx, q, positions);
    k = modules::RoPEModule({config.head_dim, GGML_ROPE_TYPE_NEOX, rope_theta, rope_freq_scale}).build(ctx, k, positions);
    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    core::TensorValue context;
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    context = grouped_attention_from_heads(ctx, q_heads, k_heads, v_heads, additive_attention_mask, config);
    context = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, context),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.num_attention_heads * config.head_dim}));
    return linear_projection(ctx, context, weights.o_proj, config.num_attention_heads * config.head_dim, config.hidden_size);
}

core::TensorValue mlp(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const DramaBoxGemma3LayerWeights & weights,
    const DramaBoxGemma3Config & config) {
    auto gate = linear_projection(ctx, input, weights.gate_proj, config.hidden_size, config.intermediate_size);
    gate = modules::GeluModule({modules::GeluApproximation::Tanh}).build(ctx, gate);
    auto up = linear_projection(ctx, input, weights.up_proj, config.hidden_size, config.intermediate_size);
    auto hidden = modules::MulModule{}.build(ctx, gate, up);
    return linear_projection(ctx, hidden, weights.down_proj, config.intermediate_size, config.hidden_size);
}

core::TensorValue layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const core::TensorValue & additive_attention_mask,
    const DramaBoxGemma3LayerWeights & weights,
    const DramaBoxGemma3Config & config,
    int64_t layer_index) {
    const modules::GemmaRMSNormModule norm({config.hidden_size, config.rms_norm_eps, true, false});
    auto hidden = norm.build(ctx, input, {weights.input_norm, std::nullopt});
    hidden = self_attention(ctx, hidden, positions, additive_attention_mask, weights, config, layer_index);
    hidden = norm.build(ctx, hidden, {weights.post_attention_norm, std::nullopt});
    auto output = modules::AddModule{}.build(ctx, input, hidden);
    hidden = norm.build(ctx, output, {weights.pre_feedforward_norm, std::nullopt});
    hidden = mlp(ctx, hidden, weights, config);
    hidden = norm.build(ctx, hidden, {weights.post_feedforward_norm, std::nullopt});
    return modules::AddModule{}.build(ctx, output, hidden);
}

core::TensorValue aggregate_hidden(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & accumulated,
    const core::TensorValue & hidden,
    const core::TensorValue & hidden_attention_mask,
    const DramaBoxGemma3PromptWeights & weights,
    const DramaBoxConfig & config,
    int64_t hidden_index) {
    const int64_t hidden_size = config.gemma.hidden_size;
    const int64_t output_size = config.transformer.cross_attention_dim;
    auto normalized = modules::RMSNormModule({hidden_size, kFeatureRmsNormEps, false, false}).build(ctx, hidden, {});
    auto mask = core::reshape_tensor(
        ctx,
        hidden_attention_mask,
        core::TensorShape::from_dims({hidden.shape.dims[0], hidden.shape.dims[1], 1}));
    mask = modules::RepeatModule({normalized.shape}).build(ctx, mask);
    normalized = modules::MulModule{}.build(ctx, normalized, mask);
    normalized = core::wrap_tensor(
        ggml_scale(
            ctx.ggml,
            core::ensure_backend_addressable_layout(ctx, normalized).tensor,
            std::sqrt(static_cast<float>(output_size) /
                      static_cast<float>(hidden_size))),
        normalized.shape,
        GGML_TYPE_F32);
    auto contribution = linear_projection(
        ctx,
        normalized,
        weights.aggregate_layers.at(static_cast<size_t>(hidden_index)),
        hidden_size,
        output_size);
    if (!accumulated.valid()) {
        return contribution;
    }
    return modules::AddModule{}.build(ctx, accumulated, contribution);
}

std::vector<int32_t> positions(int64_t tokens) {
    std::vector<int32_t> out(static_cast<size_t>(tokens));
    for (int64_t i = 0; i < tokens; ++i) {
        out[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    return out;
}

std::vector<float> make_causal_key_padding_mask(
    const DramaBoxGemmaTokenBatch & tokens,
    int64_t max_batch,
    int64_t heads,
    bool keep_padded_query_diagonal) {
    std::vector<float> out(static_cast<size_t>(max_batch * heads * tokens.tokens * tokens.tokens), kMaskNegInf);
    for (int64_t b = 0; b < tokens.batch; ++b) {
        for (int64_t h = 0; h < heads; ++h) {
            for (int64_t q = 0; q < tokens.tokens; ++q) {
                if (keep_padded_query_diagonal &&
                    tokens.attention_mask[static_cast<size_t>(b * tokens.tokens + q)] == 0) {
                    out[static_cast<size_t>(((b * heads + h) * tokens.tokens + q) * tokens.tokens + q)] = 0.0F;
                    continue;
                }
                for (int64_t k = 0; k <= q; ++k) {
                    const bool keep = tokens.attention_mask[static_cast<size_t>(b * tokens.tokens + k)] != 0;
                    out[static_cast<size_t>(((b * heads + h) * tokens.tokens + q) * tokens.tokens + k)] =
                        keep ? 0.0F : kMaskNegInf;
                }
            }
        }
    }
    return out;
}

std::vector<float> make_hidden_attention_mask(const DramaBoxGemmaTokenBatch & tokens, int64_t max_batch) {
    std::vector<float> out(static_cast<size_t>(max_batch * tokens.tokens), 0.0F);
    for (int64_t b = 0; b < tokens.batch; ++b) {
        for (int64_t t = 0; t < tokens.tokens; ++t) {
            out[static_cast<size_t>(b * tokens.tokens + t)] =
                tokens.attention_mask[static_cast<size_t>(b * tokens.tokens + t)] != 0 ? 1.0F : 0.0F;
        }
    }
    return out;
}

}  // namespace

DramaBoxGemma3PromptWeights load_dramabox_gemma3_prompt_weights(
    const DramaBoxAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType gemma_weight_storage_type,
    assets::TensorStorageType projection_weight_storage_type) {
    DramaBoxGemma3PromptWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "dramabox.gemma3_prompt.weights",
        weight_context_bytes == 0 ? kGemmaPromptWeightContextBytes : weight_context_bytes);
    const auto & config = assets.config.gemma;
    const auto & embed_source = source_for(assets.gemma_weights, "language_model.model.embed_tokens.weight");
    weights.embed_tokens = weights.store->load_tensor(
        embed_source,
        "language_model.model.embed_tokens.weight",
        assets::TensorStorageType::Native,
        {config.vocab_size, config.hidden_size});
    weights.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
    for (int64_t layer_index = 0; layer_index < config.num_hidden_layers; ++layer_index) {
        weights.layers.push_back(load_layer(*weights.store, assets, layer_index, gemma_weight_storage_type));
    }
    const auto & norm_source = source_for(assets.gemma_weights, "language_model.model.norm.weight");
    weights.norm = weights.store->load_tensor(
        norm_source,
        "language_model.model.norm.weight",
        assets::TensorStorageType::Native,
        {config.hidden_size});
    const std::string aggregate_prefix = "text_embedding_projection.audio_aggregate_embed";
    weights.aggregate_layers = load_aggregate_layers(
        *weights.store,
        *assets.audio_weights,
        aggregate_prefix,
        projection_weight_storage_type,
        assets.config.transformer.cross_attention_dim,
        config.hidden_size,
        config.num_hidden_layers + 1);
    weights.aggregate_bias = weights.store->load_tensor(
        *assets.audio_weights,
        aggregate_prefix + ".bias",
        assets::TensorStorageType::F32,
        {assets.config.transformer.cross_attention_dim});
    weights.store->upload();
    return weights;
}

core::TensorValue build_dramabox_gemma3_prompt_encoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_ids,
    const core::TensorValue & positions,
    const core::TensorValue & additive_attention_mask,
    const core::TensorValue & hidden_attention_mask,
    const DramaBoxGemma3PromptWeights & weights,
    const DramaBoxConfig & config) {
    if (static_cast<int64_t>(weights.layers.size()) != config.gemma.num_hidden_layers) {
        throw std::runtime_error("DramaBox Gemma3 layer count mismatch");
    }
    auto hidden = modules::EmbeddingModule({config.gemma.vocab_size, config.gemma.hidden_size})
        .build(ctx, input_ids, weights.embed_tokens);
    hidden = core::wrap_tensor(
        ggml_scale(ctx.ggml, cast_f32(ctx, hidden).tensor, std::sqrt(static_cast<float>(config.gemma.hidden_size))),
        hidden.shape,
        GGML_TYPE_F32);
    core::TensorValue accumulated;
    accumulated = aggregate_hidden(ctx, accumulated, hidden, hidden_attention_mask, weights, config, 0);
    for (int64_t layer_index = 0; layer_index < config.gemma.num_hidden_layers; ++layer_index) {
        hidden = layer(
            ctx,
            hidden,
            positions,
            additive_attention_mask,
            weights.layers[static_cast<size_t>(layer_index)],
            config.gemma,
            layer_index);
        auto captured = hidden;
        if (layer_index + 1 == config.gemma.num_hidden_layers) {
            captured = modules::GemmaRMSNormModule({config.gemma.hidden_size, config.gemma.rms_norm_eps, true, false}).build(
                ctx,
                hidden,
                {weights.norm, std::nullopt});
        }
        accumulated = aggregate_hidden(ctx, accumulated, captured, hidden_attention_mask, weights, config, layer_index + 1);
    }
    auto bias = core::reshape_tensor(
        ctx,
        weights.aggregate_bias,
        core::TensorShape::from_dims({1, 1, config.transformer.cross_attention_dim}));
    bias = modules::RepeatModule({accumulated.shape}).build(ctx, bias);
    return modules::AddModule{}.build(ctx, accumulated, bias);
}

class DramaBoxGemma3PromptRuntime::Graph {
public:
    Graph(
        core::ExecutionContext & execution,
        std::shared_ptr<const DramaBoxAssets> assets,
        const DramaBoxGemma3PromptWeights & weights,
        int64_t max_batch)
        : backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          assets_(std::move(assets)),
          max_batch_(max_batch),
          weights_(weights) {
        if (backend_ == nullptr) {
            throw std::runtime_error("DramaBox Gemma prompt backend initialization failed");
        }
        if (assets_ == nullptr) {
            throw std::runtime_error("DramaBox Gemma prompt runtime requires assets");
        }
        if (max_batch_ <= 0) {
            throw std::runtime_error("DramaBox Gemma prompt max_batch must be positive");
        }
        for (const auto & source : assets_->gemma_weights) {
            source->release_storage();
        }
        build();
    }

    ~Graph() {
        if (backend_ != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(backend_type_, backend_, graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(int64_t batch) const noexcept {
        return batch == max_batch_;
    }

    DramaBoxPromptEncoding encode(const DramaBoxGemmaTokenBatch & tokens) const {
        const auto total_start = Clock::now();
        const auto & config = assets_->config;
        if (tokens.batch <= 0 || tokens.batch > max_batch_) {
            throw std::runtime_error("DramaBox Gemma prompt batch exceeds prepared max_batch");
        }
        if (tokens.tokens != config.gemma.prompt_max_length) {
            throw std::runtime_error("DramaBox Gemma prompt token length mismatch");
        }
        const auto input_start = Clock::now();
        std::vector<int32_t> padded_ids(static_cast<size_t>(max_batch_ * tokens.tokens), 0);
        for (int64_t b = 0; b < tokens.batch; ++b) {
            std::copy_n(
                tokens.input_ids.data() + static_cast<std::ptrdiff_t>(b * tokens.tokens),
                static_cast<size_t>(tokens.tokens),
                padded_ids.data() + static_cast<std::ptrdiff_t>(b * tokens.tokens));
        }
        core::write_tensor_i32(input_ids_, padded_ids);
        core::write_tensor_i32(positions_, positions(tokens.tokens));
        const bool use_flash_attention = backend_type_ == core::BackendType::Cuda;
        const int64_t mask_heads = use_flash_attention ? 1 : config.gemma.num_attention_heads;
        const auto attention_mask = make_causal_key_padding_mask(tokens, max_batch_, mask_heads, use_flash_attention);
        if (use_flash_attention) {
            core::write_tensor_f16(attention_mask_, attention_mask);
        } else {
            core::write_tensor_f32(attention_mask_, attention_mask);
        }
        core::write_tensor_f32(hidden_attention_mask_, make_hidden_attention_mask(tokens, max_batch_));
        core::set_backend_threads(backend_, threads_);
        engine::debug::timing_log_scalar("dramabox.gemma_prompt.input_upload_ms", engine::debug::elapsed_ms(input_start, Clock::now()));
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_, nullptr, "dramabox.gemma_prompt");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DramaBox Gemma prompt graph compute failed");
        }
        ggml_backend_synchronize(backend_);
        engine::debug::timing_log_scalar("dramabox.gemma_prompt.graph.compute_ms", engine::debug::elapsed_ms(compute_start, Clock::now()));
        const auto output_start = Clock::now();
        const auto full = core::read_tensor_f32(output_);
        DramaBoxPromptEncoding out;
        out.batch = tokens.batch;
        out.tokens = tokens.tokens;
        out.hidden_size = config.transformer.cross_attention_dim;
        out.attention_mask = tokens.attention_mask;
        out.audio_features.resize(static_cast<size_t>(tokens.batch * tokens.tokens * out.hidden_size));
        const int64_t row_values = tokens.tokens * out.hidden_size;
        for (int64_t b = 0; b < tokens.batch; ++b) {
            std::copy_n(
                full.data() + static_cast<std::ptrdiff_t>(b * row_values),
                static_cast<size_t>(row_values),
                out.audio_features.data() + static_cast<std::ptrdiff_t>(b * row_values));
        }
        engine::debug::timing_log_scalar("dramabox.gemma_prompt.output_read_ms", engine::debug::elapsed_ms(output_start, Clock::now()));
        engine::debug::timing_log_scalar("dramabox.gemma_prompt.total_ms", engine::debug::elapsed_ms(total_start, Clock::now()));
        return out;
    }

private:
    void build() {
        const auto build_start = Clock::now();
        const auto & config = assets_->config;
        ggml_init_params params{1536ull * 1024ull * 1024ull, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("DramaBox Gemma prompt ggml context initialization failed");
        }
        const int64_t tokens = config.gemma.prompt_max_length;
        input_ids_ = core::wrap_tensor(
            ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_I32, tokens, max_batch_),
            core::TensorShape::from_dims({max_batch_, tokens}),
            GGML_TYPE_I32);
        positions_ = core::wrap_tensor(
            ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, tokens),
            core::TensorShape::from_dims({tokens}),
            GGML_TYPE_I32);
        const bool use_flash_attention = backend_type_ == core::BackendType::Cuda;
        attention_mask_ = use_flash_attention
            ? core::wrap_tensor(
                  ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, tokens, tokens, 1, max_batch_),
                  core::TensorShape::from_dims({max_batch_, 1, tokens, tokens}),
                  GGML_TYPE_F16)
            : core::wrap_tensor(
                  ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F32, tokens, tokens, config.gemma.num_attention_heads, max_batch_),
                  core::TensorShape::from_dims({max_batch_, config.gemma.num_attention_heads, tokens, tokens}),
                  GGML_TYPE_F32);
        hidden_attention_mask_ = core::wrap_tensor(
            ggml_new_tensor_3d(ctx_.get(), GGML_TYPE_F32, 1, tokens, max_batch_),
            core::TensorShape::from_dims({max_batch_, tokens, 1}),
            GGML_TYPE_F32);
        ggml_set_input(input_ids_.tensor);
        ggml_set_input(positions_.tensor);
        ggml_set_input(attention_mask_.tensor);
        ggml_set_input(hidden_attention_mask_.tensor);
        core::ModuleBuildContext build_ctx{ctx_.get(), "dramabox.gemma_prompt", backend_type_};
        auto output = build_dramabox_gemma3_prompt_encoder(
            build_ctx,
            input_ids_,
            positions_,
            attention_mask_,
            hidden_attention_mask_,
            weights_,
            config);
        output_ = output.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 1048576, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("DramaBox Gemma prompt backend buffer allocation failed");
        }
        core::write_tensor_i32(positions_, positions(tokens));
        engine::debug::timing_log_scalar(
            "dramabox.gemma_prompt.graph.build_ms",
            engine::debug::elapsed_ms(build_start, Clock::now()));
    }

    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    std::shared_ptr<const DramaBoxAssets> assets_;
    int64_t max_batch_ = 1;
    const DramaBoxGemma3PromptWeights & weights_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue input_ids_;
    core::TensorValue positions_;
    core::TensorValue attention_mask_;
    core::TensorValue hidden_attention_mask_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

DramaBoxGemma3PromptRuntime::DramaBoxGemma3PromptRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const DramaBoxAssets> assets,
    assets::TensorStorageType gemma_weight_storage_type,
    assets::TensorStorageType projection_weight_storage_type,
    int64_t max_batch)
    : execution_(&execution),
      assets_(std::move(assets)),
      gemma_weight_storage_type_(gemma_weight_storage_type),
      projection_weight_storage_type_(projection_weight_storage_type),
      max_batch_(max_batch) {
    if (execution_ == nullptr) {
        throw std::runtime_error("DramaBox Gemma prompt runtime requires execution context");
    }
    if (assets_ == nullptr) {
        throw std::runtime_error("DramaBox Gemma prompt runtime requires assets");
    }
    if (max_batch_ <= 0) {
        throw std::runtime_error("DramaBox Gemma prompt max_batch must be positive");
    }
}

DramaBoxGemma3PromptRuntime::~DramaBoxGemma3PromptRuntime() = default;

void DramaBoxGemma3PromptRuntime::prepare(int64_t batch) const {
    if (batch <= 0 || batch > max_batch_) {
        throw std::runtime_error("DramaBox Gemma prompt batch exceeds session max_batch");
    }
    if (!weights_) {
        weights_ = std::make_unique<DramaBoxGemma3PromptWeights>(load_dramabox_gemma3_prompt_weights(
            *assets_,
            execution_->backend(),
            execution_->backend_type(),
            0,
            gemma_weight_storage_type_,
            projection_weight_storage_type_));
        for (const auto & source : assets_->gemma_weights) {
            source->release_storage();
        }
    }
    if (!graph_ || !graph_->matches(batch)) {
        graph_.reset();
        graph_ = std::make_unique<Graph>(
            *execution_,
            assets_,
            *weights_,
            batch);
    }
}

DramaBoxPromptEncoding DramaBoxGemma3PromptRuntime::encode(const DramaBoxGemmaTokenBatch & tokens) const {
    prepare(tokens.batch);
    return graph_->encode(tokens);
}

}  // namespace engine::models::dramabox
