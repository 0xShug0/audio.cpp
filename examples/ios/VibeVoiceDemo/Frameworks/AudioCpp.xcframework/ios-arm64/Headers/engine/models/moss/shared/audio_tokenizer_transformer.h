#pragma once

// Shared building blocks for the MOSS-Audio-Tokenizer-v2 codec transformer
// stacks. The encoder and decoder are structural mirrors: both are stacks of
// causal ProjectedTransformers (fused qkv, interleaved RoPE, LayerScale, erf
// GELU MLP, pre-norm LayerNorm) separated by reshape-based patch transforms.
// The only differences are the module order (decoder: transformer -> upsample;
// encoder: downsample -> transformer), the per-stage specs, and the safetensors
// prefix ("decoder"/"encoder"). Everything below is prefix- and spec-agnostic so
// both stacks can share it.

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::moss::codec_detail {

namespace modules = engine::modules;
namespace binding = engine::modules::binding;

inline constexpr float kMaskedAttentionBias = std::numeric_limits<float>::lowest();
inline constexpr int64_t kCodeDim = 768;
inline constexpr int64_t kSamplesPerFrame = 3840;  // downsample_rate (per interleaved stream frame)
inline constexpr float kRopeTheta = 10000.0F;
inline constexpr float kLayerNormEps = 1.0e-5F;

// One ProjectedTransformer stage. `patch` is the reshape factor applied to the
// stage (after the transformer for the decoder, before it for the encoder);
// `context` is the local-attention window in tokens at that stage's frame rate.
struct TransformerSpec {
    int64_t input_dim;
    int64_t output_dim;
    int64_t d_model;
    int64_t num_heads;
    int64_t num_layers;
    int64_t intermediate_size;
    int64_t context;
    int64_t patch;
};

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct LayerWeights {
    core::TensorValue norm1_w;
    core::TensorValue norm1_b;
    core::TensorValue in_proj;   // fused qkv [3 * d_model, d_model]
    core::TensorValue out_proj;  // [d_model, d_model]
    core::TensorValue norm2_w;
    core::TensorValue norm2_b;
    core::TensorValue fc1;  // [intermediate_size, d_model]
    core::TensorValue fc2;  // [d_model, intermediate_size]
    core::TensorValue layer_scale1;  // [d_model]
    core::TensorValue layer_scale2;  // [d_model]
};

struct TransformerWeights {
    TransformerSpec spec;
    core::TensorValue input_proj;   // [d_model, input_dim]
    core::TensorValue output_proj;  // [output_dim, d_model]
    std::vector<LayerWeights> layers;
};

struct AttentionWindow {
    int64_t query_start;
    int64_t query_steps;
    int64_t key_start;
    int64_t key_steps;
    core::TensorValue mask;
};

class CodecWeights {
public:
    explicit CodecWeights(const assets::TensorSource & source) : source_(source) {}

    const assets::TensorSource & source_for(const std::string & name) const {
        if (source_.has_tensor(name)) {
            return source_;
        }
        throw std::runtime_error("MOSS codec tensor not found: " + name);
    }

private:
    const assets::TensorSource & source_;
};

// Loads one ProjectedTransformer's weights. `stack_prefix` is "decoder" or
// "encoder"; `module_index` is the module's position in that ModuleList.
inline TransformerWeights load_transformer(
    core::BackendWeightStore & store,
    const CodecWeights & codec_weights,
    const TransformerSpec & spec,
    const std::string & stack_prefix,
    int64_t module_index) {
    const std::string prefix = stack_prefix + "." + std::to_string(module_index);
    const auto load = [&](const std::string & name, std::initializer_list<int64_t> shape) {
        return store.load_tensor(codec_weights.source_for(name), name, assets::TensorStorageType::F32, shape);
    };
    const auto load_f32 = [&](const std::string & name, std::initializer_list<int64_t> shape) {
        return store.load_f32_tensor(codec_weights.source_for(name), name, shape);
    };

    TransformerWeights weights;
    weights.spec = spec;
    weights.input_proj = load(prefix + ".input_proj.weight", {spec.d_model, spec.input_dim});
    weights.output_proj = load(prefix + ".output_proj.weight", {spec.output_dim, spec.d_model});
    weights.layers.reserve(static_cast<size_t>(spec.num_layers));
    for (int64_t layer = 0; layer < spec.num_layers; ++layer) {
        const std::string lp = prefix + ".transformer.layers." + std::to_string(layer);
        LayerWeights w;
        w.norm1_w = load_f32(lp + ".norm1.weight", {spec.d_model});
        w.norm1_b = load_f32(lp + ".norm1.bias", {spec.d_model});
        w.in_proj = load(lp + ".self_attn.in_proj.weight", {3 * spec.d_model, spec.d_model});
        w.out_proj = load(lp + ".self_attn.out_proj.weight", {spec.d_model, spec.d_model});
        w.norm2_w = load_f32(lp + ".norm2.weight", {spec.d_model});
        w.norm2_b = load_f32(lp + ".norm2.bias", {spec.d_model});
        w.fc1 = load(lp + ".ffn.0.weight", {spec.intermediate_size, spec.d_model});
        w.fc2 = load(lp + ".ffn.2.weight", {spec.d_model, spec.intermediate_size});
        w.layer_scale1 = load_f32(lp + ".layer_scale_1.scale", {spec.d_model});
        w.layer_scale2 = load_f32(lp + ".layer_scale_2.scale", {spec.d_model});
        weights.layers.push_back(std::move(w));
    }
    return weights;
}

inline core::TensorValue attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const core::TensorValue & mask) {
    const modules::MatMulModule matmul;
    auto scores = matmul.build(
        ctx,
        q_heads,
        modules::TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads));
    scores = core::ensure_backend_addressable_layout(ctx, scores);
    auto attn = core::wrap_tensor(
        ggml_soft_max_ext(
            ctx.ggml,
            scores.tensor,
            mask.tensor,
            1.0F / std::sqrt(static_cast<float>(dim)),
            0.0F),
        scores.shape,
        GGML_TYPE_F32);
    return matmul.build(ctx, attn, v_heads);
}

inline core::TensorValue windowed_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const std::vector<AttentionWindow> & windows) {
    if (windows.empty()) {
        throw std::runtime_error("MOSS codec windowed attention requires at least one window");
    }
    core::TensorValue merged;
    for (const auto & window : windows) {
        auto q_slice = modules::SliceModule({2, window.query_start, window.query_steps}).build(ctx, q_heads);
        auto k_slice = modules::SliceModule({2, window.key_start, window.key_steps}).build(ctx, k_heads);
        auto v_slice = modules::SliceModule({2, window.key_start, window.key_steps}).build(ctx, v_heads);
        auto part = attention(ctx, q_slice, k_slice, v_slice, dim, window.mask);
        merged = merged.valid() ? modules::ConcatModule({2}).build(ctx, merged, part) : part;
    }
    return merged;
}

inline core::TensorValue transformer_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const LayerWeights & weights,
    const TransformerSpec & spec,
    const core::TensorValue & positions,
    const core::TensorValue & mask,
    const std::vector<AttentionWindow> * windows,
    int64_t steps) {
    const int64_t dim = spec.d_model / spec.num_heads;
    const modules::LayerNormModule norm({spec.d_model, kLayerNormEps, true, true});

    auto normed = norm.build(ctx, input, binding::norm_data(ctx, weights.norm1_w, weights.norm1_b));
    auto qkv = modules::LinearModule(binding::linear_config(spec.d_model, 3 * spec.d_model, false))
                   .build(ctx, normed, binding::linear_data(ctx, weights.in_proj));

    auto q = core::ensure_backend_addressable_layout(
        ctx, modules::SliceModule({2, 0, spec.d_model}).build(ctx, qkv));
    auto k = core::ensure_backend_addressable_layout(
        ctx, modules::SliceModule({2, spec.d_model, spec.d_model}).build(ctx, qkv));
    auto v = core::ensure_backend_addressable_layout(
        ctx, modules::SliceModule({2, 2 * spec.d_model, spec.d_model}).build(ctx, qkv));

    q = modules::ReshapeModule({
        core::TensorShape::from_dims({q.shape.dims[0], q.shape.dims[1], spec.num_heads, dim}),
    }).build(ctx, q);
    k = modules::ReshapeModule({
        core::TensorShape::from_dims({k.shape.dims[0], k.shape.dims[1], spec.num_heads, dim}),
    }).build(ctx, k);
    v = modules::ReshapeModule({
        core::TensorShape::from_dims({v.shape.dims[0], v.shape.dims[1], spec.num_heads, dim}),
    }).build(ctx, v);
    q = modules::RoPEModule({dim, GGML_ROPE_TYPE_NORMAL, kRopeTheta}).build(ctx, q, positions);
    k = modules::RoPEModule({dim, GGML_ROPE_TYPE_NORMAL, kRopeTheta}).build(ctx, k, positions);

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    auto context = windows == nullptr ? attention(ctx, q_heads, k_heads, v_heads, dim, mask)
                                      : windowed_attention(ctx, q_heads, k_heads, v_heads, dim, *windows);
    context = modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = modules::ReshapeModule({
        core::TensorShape::from_dims({1, steps, spec.d_model}),
    }).build(ctx, context);
    auto attn_out = modules::LinearModule(binding::linear_config(spec.d_model, spec.d_model, false))
                        .build(ctx, context, binding::linear_data(ctx, weights.out_proj));
    auto layer_scale1 = modules::ReshapeModule({
        core::TensorShape::from_dims({1, 1, spec.d_model}),
    }).build(ctx, weights.layer_scale1);
    layer_scale1 = modules::RepeatModule({attn_out.shape}).build(ctx, layer_scale1);
    attn_out = modules::MulModule{}.build(ctx, attn_out, layer_scale1);
    auto x = modules::AddModule{}.build(ctx, input, attn_out);

    auto ff_in = norm.build(ctx, x, binding::norm_data(ctx, weights.norm2_w, weights.norm2_b));
    auto ff = modules::LinearModule(binding::linear_config(spec.d_model, spec.intermediate_size, false))
                  .build(ctx, ff_in, binding::linear_data(ctx, weights.fc1));
    ff = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, ff);
    ff = modules::LinearModule(binding::linear_config(spec.intermediate_size, spec.d_model, false))
             .build(ctx, ff, binding::linear_data(ctx, weights.fc2));
    auto layer_scale2 = modules::ReshapeModule({
        core::TensorShape::from_dims({1, 1, spec.d_model}),
    }).build(ctx, weights.layer_scale2);
    layer_scale2 = modules::RepeatModule({ff.shape}).build(ctx, layer_scale2);
    ff = modules::MulModule{}.build(ctx, ff, layer_scale2);
    return modules::AddModule{}.build(ctx, x, ff);
}

// ProjectedTransformer: input projection -> transformer stack -> output
// projection. Input/output are [1, steps, channels] (feature-last).
inline core::TensorValue run_transformer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const TransformerWeights & weights,
    const core::TensorValue & positions,
    const core::TensorValue & mask,
    int64_t steps,
    const std::vector<AttentionWindow> * windows = nullptr) {
    const auto & spec = weights.spec;
    auto x = modules::LinearModule(binding::linear_config(spec.input_dim, spec.d_model, false))
                 .build(ctx, input, binding::linear_data(ctx, weights.input_proj));
    for (const auto & layer : weights.layers) {
        x = transformer_layer(ctx, x, layer, spec, positions, mask, windows, steps);
    }
    return modules::LinearModule(binding::linear_config(spec.d_model, spec.output_dim, false))
        .build(ctx, x, binding::linear_data(ctx, weights.output_proj));
}

inline std::vector<float> causal_context_mask(int64_t steps, int64_t context) {
    std::vector<float> mask(static_cast<size_t>(steps * steps), kMaskedAttentionBias);
#ifdef _OPENMP
#pragma omp parallel for if(steps * steps >= 4096)
#endif
    for (int64_t query = 0; query < steps; ++query) {
        for (int64_t key = 0; key <= query; ++key) {
            if (query - key < context) {
                mask[static_cast<size_t>(query * steps + key)] = 0.0F;
            }
        }
    }
    return mask;
}

inline std::vector<float> causal_context_mask_window(
    int64_t query_start,
    int64_t query_steps,
    int64_t key_start,
    int64_t key_steps,
    int64_t context) {
    std::vector<float> mask(static_cast<size_t>(query_steps * key_steps), kMaskedAttentionBias);
#ifdef _OPENMP
#pragma omp parallel for if(query_steps * key_steps >= 4096)
#endif
    for (int64_t q = 0; q < query_steps; ++q) {
        const int64_t query = query_start + q;
        for (int64_t k = 0; k < key_steps; ++k) {
            const int64_t key = key_start + k;
            if (key <= query && query - key < context) {
                mask[static_cast<size_t>(q * key_steps + k)] = 0.0F;
            }
        }
    }
    return mask;
}

}  // namespace engine::models::moss::codec_detail
