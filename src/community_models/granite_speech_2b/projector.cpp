// granite_speech_2b Q-Former projector: weight binding + graph execution.
//
// Math reference: transformers 4.57.0 modeling_granite_speech.py
// (GraniteSpeechEncoderProjector) + modeling_blip_2.py (Blip2QFormerModel,
// Blip2QFormerLayer with query_length=3 and no text input).
//
// Per window, with queries q [3,1024] and acoustic rows e [15,1024]:
//   q = LayerNorm(q)                                     (eps 1e-12)
//   repeat 2 layers:
//     s = LayerNorm(SelfAttn(q) + q)                    (16 heads x 64, /8)
//     c = LayerNorm(CrossAttn(s, e) + s)                 (3 queries x 15 keys)
//     q = LayerNorm(GELU(inter_q(c)) @ out_q + c)        (query-only FFN)
//   collect [W,3,1024] window-major, query-major
// projected = Linear(1024->2048, bias)                  -> [3W,2048]
// All Q/K/V/out/intermediate/output projections carry a bias; all LayerNorms
// carry weight + bias with eps 1e-12; the GELU is the exact erf form. No
// encoder attention mask is applied anywhere, so zero-padded acoustic rows
// participate in the final cross-attention window exactly as real rows do.

#include "engine/community_models/granite_speech_2b/projector.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::granite_speech_2b {
namespace {

constexpr size_t kProjectorGraphBaseNodes = 8192;
constexpr size_t kProjectorGraphNodesPerWindow = 1024;
constexpr size_t kProjectorWeightContextBytes = 256ull * 1024ull * 1024ull;

std::string shape_text(const std::vector<int64_t> & shape) {
    std::string text;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            text += "x";
        }
        text += std::to_string(shape[i]);
    }
    return text.empty() ? std::string("<scalar>") : text;
}

void require_weight_shape(
    const assets::TensorSource & source,
    const std::string & name,
    const std::vector<int64_t> & expected) {
    const auto metadata = source.require_metadata(name);
    if (metadata.shape != expected) {
        throw std::runtime_error(
            "granite_speech_2b projector weight shape mismatch for " + name + ": expected [" +
            shape_text(expected) + "], got [" + shape_text(metadata.shape) + "]");
    }
}

void require_weight_vector(const assets::TensorSource & source, const std::string & name, int64_t size) {
    require_weight_shape(source, name, {size});
}

void check_qformer_attention(
    const assets::TensorSource & source, const std::string & prefix, int64_t hidden = 1024) {
    for (const char * proj : {"query", "key", "value"}) {
        const std::string base = prefix + "attention." + proj + ".";
        require_weight_vector(source, base + "bias", hidden);
        require_weight_shape(source, base + "weight", {hidden, hidden});
    }
    require_weight_vector(source, prefix + "output.LayerNorm.bias", hidden);
    require_weight_vector(source, prefix + "output.LayerNorm.weight", hidden);
    require_weight_vector(source, prefix + "output.dense.bias", hidden);
    require_weight_shape(source, prefix + "output.dense.weight", {hidden, hidden});
}

void check_qformer_layer(const assets::TensorSource & source, int layer) {
    const std::string prefix =
        "projector.qformer.encoder.layer." + std::to_string(layer) + ".";
    check_qformer_attention(source, prefix + "attention.");
    check_qformer_attention(source, prefix + "crossattention.");
    require_weight_vector(source, prefix + "intermediate_query.dense.bias", 4096);
    require_weight_shape(source, prefix + "intermediate_query.dense.weight", {4096, 1024});
    require_weight_vector(source, prefix + "output_query.LayerNorm.bias", 1024);
    require_weight_vector(source, prefix + "output_query.LayerNorm.weight", 1024);
    require_weight_vector(source, prefix + "output_query.dense.bias", 1024);
    require_weight_shape(source, prefix + "output_query.dense.weight", {1024, 4096});
}

modules::LinearWeights load_linear_with_bias(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t in_features,
    int64_t out_features,
    assets::TensorStorageType storage_type) {
    return {
        store.load_tensor(source, prefix + ".weight", storage_type, {out_features, in_features}),
        store.load_f32_tensor(source, prefix + ".bias", {out_features}),
    };
}

modules::NormWeights load_layer_norm(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix) {
    return {
        store.load_f32_tensor(source, prefix + ".weight", {1024}),
        store.load_f32_tensor(source, prefix + ".bias", {1024}),
    };
}

struct QFormerAttentionWeights {
    modules::LinearWeights query;
    modules::LinearWeights key;
    modules::LinearWeights value;
    modules::LinearWeights output_dense;
    modules::NormWeights output_norm;
};

struct QFormerLayerWeights {
    QFormerAttentionWeights self_attn;
    QFormerAttentionWeights cross_attn;
    modules::LinearWeights intermediate_query;
    modules::LinearWeights output_query_dense;
    modules::NormWeights output_query_norm;
};

struct ProjectorWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue query;  // [1, 3, 1024] learned queries
    modules::NormWeights input_norm;
    std::vector<QFormerLayerWeights> layers;
    modules::LinearWeights final_linear;  // 1024 -> 2048
};

QFormerAttentionWeights load_qformer_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type) {
    QFormerAttentionWeights weights;
    weights.query = load_linear_with_bias(store, source, prefix + "attention.query", 1024, 1024, storage_type);
    weights.key = load_linear_with_bias(store, source, prefix + "attention.key", 1024, 1024, storage_type);
    weights.value = load_linear_with_bias(store, source, prefix + "attention.value", 1024, 1024, storage_type);
    weights.output_dense =
        load_linear_with_bias(store, source, prefix + "output.dense", 1024, 1024, storage_type);
    weights.output_norm = load_layer_norm(store, source, prefix + "output.LayerNorm");
    return weights;
}

QFormerLayerWeights load_qformer_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    int64_t layer,
    assets::TensorStorageType storage_type) {
    QFormerLayerWeights weights;
    const std::string prefix = "projector.qformer.encoder.layer." + std::to_string(layer) + ".";
    weights.self_attn = load_qformer_attention(store, source, prefix + "attention.", storage_type);
    weights.cross_attn = load_qformer_attention(store, source, prefix + "crossattention.", storage_type);
    weights.intermediate_query =
        load_linear_with_bias(store, source, prefix + "intermediate_query.dense", 1024, 4096, storage_type);
    weights.output_query_dense =
        load_linear_with_bias(store, source, prefix + "output_query.dense", 4096, 1024, storage_type);
    weights.output_query_norm = load_layer_norm(store, source, prefix + "output_query.LayerNorm");
    return weights;
}

ProjectorWeights load_projector_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    assets::TensorStorageType storage_type) {
    ProjectorWeights weights;
    weights.query = store.load_tensor(source, "projector.query", storage_type, {1, 3, 1024});
    weights.input_norm = load_layer_norm(store, source, "projector.qformer.layernorm");
    weights.layers.reserve(2);
    for (int64_t layer = 0; layer < kGraniteSpeech2BProjectorLayers; ++layer) {
        weights.layers.push_back(load_qformer_layer(store, source, layer, storage_type));
    }
    weights.final_linear = load_linear_with_bias(store, source, "projector.linear", 1024, 2048, storage_type);
    store.upload();
    return weights;
}

// 16-head x 64-dim attention block: q_in [1,Q,1024], kv_in [1,K,1024].
// Returns [1,Q,1024] WITHOUT residual/LayerNorm (the caller adds those, since
// the reference applies dense -> dropout -> LayerNorm(x + input)).
core::TensorValue build_qformer_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_in,
    const core::TensorValue & kv_in,
    int64_t num_queries,
    int64_t num_keys,
    const QFormerAttentionWeights & weights) {
    constexpr int64_t kHeads = kGraniteSpeech2BProjectorHeads;
    constexpr int64_t kHeadDim = kGraniteSpeech2BProjectorHeadDim;
    const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));

    auto q_proj = modules::LinearModule({1024, 1024, true}).build(ctx, q_in, weights.query);
    auto k_proj = modules::LinearModule({1024, 1024, true}).build(ctx, kv_in, weights.key);
    auto v_proj = modules::LinearModule({1024, 1024, true}).build(ctx, kv_in, weights.value);

    auto q_3d = core::wrap_tensor(
        ggml_reshape_3d(ctx.ggml, q_proj.tensor, kHeadDim, kHeads, num_queries),
        core::TensorShape::from_dims({num_queries, kHeads, kHeadDim}),
        GGML_TYPE_F32);
    auto k_3d = core::wrap_tensor(
        ggml_reshape_3d(ctx.ggml, k_proj.tensor, kHeadDim, kHeads, num_keys),
        core::TensorShape::from_dims({num_keys, kHeads, kHeadDim}),
        GGML_TYPE_F32);
    auto v_3d = core::wrap_tensor(
        ggml_reshape_3d(ctx.ggml, v_proj.tensor, kHeadDim, kHeads, num_keys),
        core::TensorShape::from_dims({num_keys, kHeads, kHeadDim}),
        GGML_TYPE_F32);

    auto q_perm = core::wrap_tensor(
        ggml_permute(ctx.ggml, q_3d.tensor, 0, 2, 1, 3),
        core::TensorShape::from_dims({kHeads, num_queries, kHeadDim}),
        GGML_TYPE_F32);
    auto k_perm = core::wrap_tensor(
        ggml_permute(ctx.ggml, k_3d.tensor, 0, 2, 1, 3),
        core::TensorShape::from_dims({kHeads, num_keys, kHeadDim}),
        GGML_TYPE_F32);
    auto k_cont = core::wrap_tensor(ggml_cont(ctx.ggml, k_perm.tensor), k_perm.shape, GGML_TYPE_F32);

    auto scores = core::wrap_tensor(
        ggml_mul_mat(ctx.ggml, k_cont.tensor, q_perm.tensor),
        core::TensorShape::from_dims({kHeads, num_queries, num_keys}),
        GGML_TYPE_F32);
    scores = core::wrap_tensor(ggml_scale(ctx.ggml, scores.tensor, scale), scores.shape, GGML_TYPE_F32);
    // No attention mask anywhere: the upstream passes encoder_attention_mask
    // None, so padded acoustic rows participate exactly like real rows.
    auto probs = core::wrap_tensor(ggml_soft_max(ctx.ggml, scores.tensor), scores.shape, GGML_TYPE_F32);

    auto v_perm = core::wrap_tensor(
        ggml_permute(ctx.ggml, v_3d.tensor, 1, 2, 0, 3),
        core::TensorShape::from_dims({kHeads, kHeadDim, num_keys}),
        GGML_TYPE_F32);
    auto v_cont = core::wrap_tensor(ggml_cont(ctx.ggml, v_perm.tensor), v_perm.shape, GGML_TYPE_F32);
    auto out_head = core::wrap_tensor(
        ggml_mul_mat(ctx.ggml, v_cont.tensor, probs.tensor),
        core::TensorShape::from_dims({kHeads, num_queries, kHeadDim}),
        GGML_TYPE_F32);
    auto out_perm = core::wrap_tensor(
        ggml_permute(ctx.ggml, out_head.tensor, 0, 2, 1, 3),
        core::TensorShape::from_dims({num_queries, kHeads, kHeadDim}),
        GGML_TYPE_F32);
    auto out_cont = core::wrap_tensor(
        ggml_cont(ctx.ggml, out_perm.tensor), out_perm.shape, GGML_TYPE_F32);
    auto merged = core::wrap_tensor(
        ggml_reshape_3d(ctx.ggml, out_cont.tensor, 1024, num_queries, 1),
        core::TensorShape::from_dims({1, num_queries, 1024}),
        GGML_TYPE_F32);
    return modules::LinearModule({1024, 1024, true}).build(ctx, merged, weights.output_dense);
}

core::TensorValue build_qformer_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & queries,  // [1, 3, 1024]
    const core::TensorValue & acoustic,  // [1, 15, 1024]
    const QFormerLayerWeights & weights) {
    auto self_out = build_qformer_attention(ctx, queries, queries, 3, 3, weights.self_attn);
    auto self_resid = core::wrap_tensor(
        ggml_add(ctx.ggml, self_out.tensor, queries.tensor), queries.shape, GGML_TYPE_F32);
    auto self_normed = modules::LayerNormModule({1024, kGraniteSpeech2BProjectorLayerNormEps})
                           .build(ctx, self_resid, weights.self_attn.output_norm);

    auto cross_out = build_qformer_attention(ctx, self_normed, acoustic, 3, 15, weights.cross_attn);
    auto cross_resid = core::wrap_tensor(
        ggml_add(ctx.ggml, cross_out.tensor, self_normed.tensor), self_normed.shape, GGML_TYPE_F32);
    auto cross_normed = modules::LayerNormModule({1024, kGraniteSpeech2BProjectorLayerNormEps})
                            .build(ctx, cross_resid, weights.cross_attn.output_norm);

    // Query-only FFN (use_qformer_text_input=false): intermediate_query +
    // GELU (exact erf) + output_query dense, post-residual LayerNorm.
    auto inter = modules::LinearModule({1024, 4096, true}).build(ctx, cross_normed, weights.intermediate_query);
    inter = modules::GeluModule().build(ctx, inter);
    auto ffn_out = modules::LinearModule({4096, 1024, true}).build(ctx, inter, weights.output_query_dense);
    auto ffn_resid = core::wrap_tensor(
        ggml_add(ctx.ggml, ffn_out.tensor, cross_normed.tensor), cross_normed.shape, GGML_TYPE_F32);
    return modules::LayerNormModule({1024, kGraniteSpeech2BProjectorLayerNormEps})
        .build(ctx, ffn_resid, weights.output_query_norm);
}

}  // namespace

void check_granite_speech_2b_projector_weight_shapes(const assets::TensorSource & source) {
    require_weight_shape(source, "projector.query", {1, 3, 1024});
    require_weight_vector(source, "projector.qformer.layernorm.weight", 1024);
    require_weight_vector(source, "projector.qformer.layernorm.bias", 1024);
    require_weight_shape(source, "projector.linear.weight", {2048, 1024});
    require_weight_vector(source, "projector.linear.bias", 2048);
    check_qformer_layer(source, 0);
    check_qformer_layer(source, 1);
    if (source.has_tensor("projector.qformer.encoder.layer.2.attention.attention.query.weight")) {
        throw std::runtime_error("granite_speech_2b projector has more than 2 Q-Former layers");
    }
}

struct GraniteSpeech2BProjectorRuntime::Impl {
    Impl(
        std::shared_ptr<const GraniteSpeech2BAssets> assets,
        core::ExecutionContext & execution,
        assets::TensorStorageType storage_type,
        size_t graph_arena_bytes)
        : assets_(std::move(assets)), execution_(&execution), graph_arena_bytes_(graph_arena_bytes) {
        if (assets_ == nullptr) {
            throw std::runtime_error("granite_speech_2b projector runtime requires assets");
        }
        if (assets_->source == nullptr) {
            throw std::runtime_error("granite_speech_2b projector runtime requires a tensor source");
        }
        check_granite_speech_2b_projector_weight_shapes(*assets_->source);
        store_ = std::make_shared<core::BackendWeightStore>(
            execution.backend(),
            execution.backend_type(),
            "granite_speech_2b.projector.weights",
            kProjectorWeightContextBytes);
        weights_ = load_projector_weights(*store_, *assets_->source, storage_type);
    }

    std::shared_ptr<const GraniteSpeech2BAssets> assets_;
    core::ExecutionContext * execution_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::shared_ptr<core::BackendWeightStore> store_;
    ProjectorWeights weights_;
};

GraniteSpeech2BProjectorRuntime::GraniteSpeech2BProjectorRuntime(
    std::shared_ptr<const GraniteSpeech2BAssets> assets,
    core::ExecutionContext & execution,
    assets::TensorStorageType storage_type,
    size_t graph_arena_bytes)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, storage_type, graph_arena_bytes)) {}

GraniteSpeech2BProjectorRuntime::~GraniteSpeech2BProjectorRuntime() = default;

GraniteSpeech2BProjectorOutputs GraniteSpeech2BProjectorRuntime::project(
    const std::vector<float> & encoder_last, int64_t steps) const {
    const auto plan = granite_speech_2b_projector_plan_for_steps(steps);
    if (static_cast<int64_t>(encoder_last.size()) != steps * kGraniteSpeech2BProjectorHidden) {
        throw std::runtime_error(
            "granite_speech_2b projector encoder input size " + std::to_string(encoder_last.size()) +
            " does not match " + std::to_string(steps) + " steps of width 1024");
    }

    // Right-zero-pad to whole windows; the padding participates in the final
    // cross-attention window exactly like real rows (no mask upstream).
    std::vector<float> padded(static_cast<size_t>(plan.padded_steps * kGraniteSpeech2BProjectorHidden), 0.0F);
    std::memcpy(
        padded.data(), encoder_last.data(), encoder_last.size() * sizeof(float));

    ggml_init_params params{};
    params.mem_size = impl_->graph_arena_bytes_;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ggml_ctx = ggml_init(params);
    if (ggml_ctx == nullptr) {
        throw std::runtime_error("granite_speech_2b projector failed to initialize GGML context");
    }
    ggml_gallocr * galloc =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl_->execution_->backend()));
    if (galloc == nullptr) {
        ggml_free(ggml_ctx);
        throw std::runtime_error("granite_speech_2b projector failed to initialize GGML allocator");
    }

    GraniteSpeech2BProjectorOutputs outputs;
    try {
        core::ModuleBuildContext ctx{ggml_ctx, "granite_speech_2b_projector", impl_->execution_->backend_type()};

        auto acoustic = core::wrap_tensor(
            ggml_new_tensor_2d(
                ggml_ctx, GGML_TYPE_F32, kGraniteSpeech2BProjectorHidden, plan.padded_steps),
            core::TensorShape::from_dims({1, plan.padded_steps, kGraniteSpeech2BProjectorHidden}),
            GGML_TYPE_F32);

        auto queries = core::wrap_tensor(
            ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_F32, kGraniteSpeech2BProjectorHidden, 3),
            core::TensorShape::from_dims({1, 3, kGraniteSpeech2BProjectorHidden}),
            GGML_TYPE_F32);
        auto queries_normed = modules::LayerNormModule({1024, kGraniteSpeech2BProjectorLayerNormEps})
                                  .build(ctx, queries, impl_->weights_.input_norm);

        std::vector<core::TensorValue> window_outputs;
        window_outputs.reserve(static_cast<size_t>(plan.windows));
        for (int64_t w = 0; w < plan.windows; ++w) {
            auto window = modules::SliceModule({1, w * kGraniteSpeech2BProjectorWindow, kGraniteSpeech2BProjectorWindow})
                              .build(ctx, acoustic);
            core::TensorValue q = queries_normed;
            for (const auto & layer : impl_->weights_.layers) {
                q = build_qformer_layer(ctx, q, window, layer);
            }
            window_outputs.push_back(q);
        }
        core::TensorValue stacked = window_outputs[0];
        for (size_t i = 1; i < window_outputs.size(); ++i) {
            stacked = modules::ConcatModule({1}).build(ctx, stacked, window_outputs[i]);
        }
        auto projected = modules::LinearModule({1024, 2048, true}).build(ctx, stacked, impl_->weights_.final_linear);

        // The per-window Q-Former loop adds O(100+) graph nodes per window,
        // so the node budget scales with the plan instead of assuming a
        // short clip (a fixed 64k budget aborted on spans above ~60 s).
        const size_t graph_nodes = kProjectorGraphBaseNodes +
            kProjectorGraphNodesPerWindow * static_cast<size_t>(plan.windows);
        ggml_cgraph * graph = ggml_new_graph_custom(ggml_ctx, graph_nodes, false);
        ggml_build_forward_expand(graph, projected.tensor);
        ggml_set_output(projected.tensor);

        if (!ggml_gallocr_alloc_graph(galloc, graph)) {
            throw std::runtime_error("granite_speech_2b projector failed to allocate GGML graph");
        }

        ggml_backend_tensor_set(
            acoustic.tensor, padded.data(), 0, padded.size() * sizeof(float));
        // Same 3 learned queries seed every window (broadcast of
        // projector.query [1,3,1024]).
        const std::vector<float> query_values =
            impl_->assets_->source->require_f32("projector.query", {1, 3, 1024});
        ggml_backend_tensor_set(queries.tensor, query_values.data(), 0, query_values.size() * sizeof(float));

        if (ggml_backend_graph_compute(impl_->execution_->backend(), graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("granite_speech_2b projector graph compute failed");
        }

        outputs.audio_placeholders = plan.audio_placeholders;
        outputs.projected.resize(static_cast<size_t>(plan.audio_placeholders * kGraniteSpeech2BProjectorOutputDim));
        ggml_backend_tensor_get(
            projected.tensor, outputs.projected.data(), 0, outputs.projected.size() * sizeof(float));
    } catch (...) {
        ggml_gallocr_free(galloc);
        ggml_free(ggml_ctx);
        throw;
    }
    ggml_gallocr_free(galloc);
    ggml_free(ggml_ctx);
    return outputs;
}

}  // namespace engine::community_models::granite_speech_2b
