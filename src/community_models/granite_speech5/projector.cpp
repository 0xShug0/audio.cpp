// granite_speech5 Q-Former projector: weight binding + graph execution.
//
// Math reference: transformers 4.57.0 modeling_granite_speech5.py
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

#include "engine/community_models/granite_speech5/projector.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/cross_attention.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::granite_speech5 {
namespace {

constexpr size_t kProjectorGraphNodes = 8192;
constexpr size_t kProjectorWeightContextBytes = 4ull * 1024ull * 1024ull;

struct QFormerAttentionWeights {
    modules::AttentionWeights attention;
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
    core::TensorValue query;  // [windows, 3, 1024] learned queries
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
    weights.attention.q_weight = store.load_tensor(source, prefix + "attention.query.weight", storage_type, {1024, 1024});
    weights.attention.q_bias = store.load_f32_tensor(source, prefix + "attention.query.bias", {1024});
    weights.attention.k_weight = store.load_tensor(source, prefix + "attention.key.weight", storage_type, {1024, 1024});
    weights.attention.k_bias = store.load_f32_tensor(source, prefix + "attention.key.bias", {1024});
    weights.attention.v_weight = store.load_tensor(source, prefix + "attention.value.weight", storage_type, {1024, 1024});
    weights.attention.v_bias = store.load_f32_tensor(source, prefix + "attention.value.bias", {1024});
    weights.attention.out_weight = store.load_tensor(source, prefix + "output.dense.weight", storage_type, {1024, 1024});
    weights.attention.out_bias = store.load_f32_tensor(source, prefix + "output.dense.bias", {1024});
    weights.output_norm = modules::binding::norm_from_source(store, source, prefix + "output.LayerNorm", 1024);
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
        modules::binding::linear_from_source(store, source, prefix + "intermediate_query.dense", storage_type, 4096, 1024, true);
    weights.output_query_dense =
        modules::binding::linear_from_source(store, source, prefix + "output_query.dense", storage_type, 1024, 4096, true);
    weights.output_query_norm = modules::binding::norm_from_source(store, source, prefix + "output_query.LayerNorm", 1024);
    return weights;
}

ProjectorWeights load_projector_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    assets::TensorStorageType storage_type) {
    ProjectorWeights weights;
    weights.query = store.load_f32_tensor(source, "projector.query", {1, 3, 1024});
    weights.input_norm = modules::binding::norm_from_source(store, source, "projector.qformer.layernorm", 1024);
    weights.layers.reserve(2);
    for (int64_t layer = 0; layer < kGraniteSpeech5ProjectorLayers; ++layer) {
        weights.layers.push_back(load_qformer_layer(store, source, layer, storage_type));
    }
    weights.final_linear = modules::binding::linear_from_source(store, source, "projector.linear", storage_type, 2048, 1024, true);
    store.upload();
    return weights;
}

core::TensorValue build_qformer_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & queries,  // [windows, 3, 1024]
    const core::TensorValue & acoustic,  // [windows, 15, 1024]
    const QFormerLayerWeights & weights) {
    auto self_out = modules::CrossAttentionModule({1024, 16, true}).build(
        ctx, queries, queries, weights.self_attn.attention);
    auto self_resid = core::wrap_tensor(
        ggml_add(ctx.ggml, self_out.tensor, queries.tensor), queries.shape, GGML_TYPE_F32);
    auto self_normed = modules::LayerNormModule({1024, kGraniteSpeech5ProjectorLayerNormEps})
                           .build(ctx, self_resid, weights.self_attn.output_norm);

    auto cross_out = modules::CrossAttentionModule({1024, 16, true}).build(
        ctx, self_normed, acoustic, weights.cross_attn.attention);
    auto cross_resid = core::wrap_tensor(
        ggml_add(ctx.ggml, cross_out.tensor, self_normed.tensor), self_normed.shape, GGML_TYPE_F32);
    auto cross_normed = modules::LayerNormModule({1024, kGraniteSpeech5ProjectorLayerNormEps})
                            .build(ctx, cross_resid, weights.cross_attn.output_norm);

    // Query-only FFN (use_qformer_text_input=false): intermediate_query +
    // GELU (exact erf) + output_query dense, post-residual LayerNorm.
    auto ffn_out = modules::FeedForwardModule({1024, 4096, true}).build(
        ctx, cross_normed,
        {weights.intermediate_query.weight, weights.intermediate_query.bias,
         weights.output_query_dense.weight, weights.output_query_dense.bias});
    auto ffn_resid = core::wrap_tensor(
        ggml_add(ctx.ggml, ffn_out.tensor, cross_normed.tensor), cross_normed.shape, GGML_TYPE_F32);
    return modules::LayerNormModule({1024, kGraniteSpeech5ProjectorLayerNormEps})
        .build(ctx, ffn_resid, weights.output_query_norm);
}

}  // namespace



struct GraniteSpeech5ProjectorRuntime::Impl {
    Impl(
        std::shared_ptr<const GraniteSpeech5Assets> assets,
        core::ExecutionContext & execution,
        assets::TensorStorageType storage_type,
        size_t graph_arena_bytes)
        : assets_(std::move(assets)), execution_(&execution), graph_arena_bytes_(graph_arena_bytes) {
        if (assets_ == nullptr) {
            throw std::runtime_error("granite_speech5 projector runtime requires assets");
        }
        if (assets_->source == nullptr) {
            throw std::runtime_error("granite_speech5 projector runtime requires a tensor source");
        }
        check_granite_speech5_projector_weight_shapes(*assets_->source);
        store_ = std::make_shared<core::BackendWeightStore>(
            execution.backend(),
            execution.backend_type(),
            "granite_speech5.projector.weights",
            kProjectorWeightContextBytes);
        weights_ = load_projector_weights(*store_, *assets_->source, storage_type);
    }

    std::shared_ptr<const GraniteSpeech5Assets> assets_;
    core::ExecutionContext * execution_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::shared_ptr<core::BackendWeightStore> store_;
    ProjectorWeights weights_;
    std::unique_ptr<core::PreparedGraph> graph_;
    int64_t steps_ = 0;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
};

GraniteSpeech5ProjectorRuntime::GraniteSpeech5ProjectorRuntime(
    std::shared_ptr<const GraniteSpeech5Assets> assets,
    core::ExecutionContext & execution,
    assets::TensorStorageType storage_type,
    size_t graph_arena_bytes)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, storage_type, graph_arena_bytes)) {}

GraniteSpeech5ProjectorRuntime::~GraniteSpeech5ProjectorRuntime() = default;

GraniteSpeech5ProjectorOutputs GraniteSpeech5ProjectorRuntime::project(
    const std::vector<float> & encoder_last, int64_t steps) const {
    const auto plan = granite_speech5_projector_plan_for_steps(steps);
    if (static_cast<int64_t>(encoder_last.size()) != steps * kGraniteSpeech5ProjectorHidden) {
        throw std::runtime_error(
            "granite_speech5 projector encoder input size " + std::to_string(encoder_last.size()) +
            " does not match " + std::to_string(steps) + " steps of width 1024");
    }

    // Right-zero-pad to whole windows; the padding participates in the final
    // cross-attention window exactly like real rows (no mask upstream).
    std::vector<float> padded(static_cast<size_t>(plan.padded_steps * kGraniteSpeech5ProjectorHidden), 0.0F);
    std::memcpy(
        padded.data(), encoder_last.data(), encoder_last.size() * sizeof(float));

    if (!impl_->graph_ || impl_->steps_ != steps) {
        impl_->graph_.reset();
        impl_->steps_ = 0;
        const size_t graph_nodes = kProjectorGraphNodes;
        const size_t metadata_bytes = std::min(impl_->graph_arena_bytes_,
            ggml_graph_overhead_custom(graph_nodes, false) + 2 * graph_nodes * ggml_tensor_overhead());
        auto prepared = std::make_unique<core::PreparedGraph>(
            *impl_->execution_, core::PreparedGraphConfig{
                "granite_speech5.projector", metadata_bytes, graph_nodes});
        auto & ctx = prepared->build_context();
        auto * ggml_ctx = ctx.ggml;
        auto acoustic = core::wrap_tensor(
            ggml_new_tensor_2d(
                ggml_ctx, GGML_TYPE_F32, kGraniteSpeech5ProjectorHidden, plan.padded_steps),
            core::TensorShape::from_dims({1, plan.padded_steps, kGraniteSpeech5ProjectorHidden}),
            GGML_TYPE_F32);

        const auto & queries = impl_->weights_.query;
        auto queries_normed = modules::LayerNormModule({1024, kGraniteSpeech5ProjectorLayerNormEps})
                                  .build(ctx, queries, impl_->weights_.input_norm);

        // Windows are independent batch items. The padded last window still
        // participates in attention exactly as in the reference.
        auto windows = core::reshape_tensor(ctx, acoustic,
            core::TensorShape::from_dims({plan.windows, kGraniteSpeech5ProjectorWindow, 1024}));
        auto q = modules::RepeatModule({core::TensorShape::from_dims({plan.windows, 3, 1024})})
                     .build(ctx, queries_normed);
        for (const auto & layer : impl_->weights_.layers) {
            q = build_qformer_layer(ctx, q, windows, layer);
        }
        auto stacked = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, q),
            core::TensorShape::from_dims({1, plan.audio_placeholders, 1024}));
        auto projected = modules::LinearModule({1024, 2048, true}).build(ctx, stacked, impl_->weights_.final_linear);

        impl_->input_ = acoustic.tensor;
        impl_->output_ = projected.tensor;
        ggml_set_input(acoustic.tensor);
        ggml_set_output(projected.tensor);
        ggml_build_forward_expand(prepared->graph(), projected.tensor);
        prepared->allocate();
        impl_->steps_ = steps;
        impl_->graph_ = std::move(prepared);
    }
    ggml_backend_tensor_set(impl_->input_, padded.data(), 0, padded.size() * sizeof(float));
    impl_->graph_->compute();
    GraniteSpeech5ProjectorOutputs outputs;
    outputs.audio_placeholders = plan.audio_placeholders;
    outputs.projected.resize(static_cast<size_t>(plan.audio_placeholders * kGraniteSpeech5ProjectorOutputDim));
    ggml_backend_tensor_get(
        impl_->output_, outputs.projected.data(), 0, outputs.projected.size() * sizeof(float));
    return outputs;
}

}  // namespace engine::community_models::granite_speech5
