#include "engine/models/meanvc2/flow_sampler_runtime.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/flow_sampler_runtime.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/runtime/flow_kv_cache.h"
#include "engine/framework/sampling/torch_random.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::meanvc2 {
namespace {

constexpr int64_t kSpeakerDim = 256;
constexpr int64_t kMemorySlots = 32;
constexpr int64_t kMemoryValues = kMemorySlots * kSpeakerDim;
constexpr int64_t kMelDim = 80;
constexpr int64_t kConditionFrames = 16;
constexpr int64_t kChunkFrames = 12;
constexpr int64_t kBlockFrames = 4;
constexpr int64_t kHiddenDim = 512;
constexpr int64_t kTimeEmbedDim = 256;
constexpr size_t kDepth = 4;
constexpr int64_t kHeads = 2;
constexpr int64_t kHeadDim = 64;
constexpr int64_t kInnerDim = kHeads * kHeadDim;
constexpr int64_t kFeedForwardDim = 1024;
constexpr int64_t kTemporalHeads = 4;
constexpr int64_t kTemporalHeadDim = 32;
constexpr int64_t kTemporalInnerDim = kTemporalHeads * kTemporalHeadDim;
constexpr std::array<int64_t, kDepth> kLayerPastChunks = {2, 2, 1, 1};
constexpr std::array<int64_t, kDepth> kLayerFutureChunks = {1, 0, 0, 0};
constexpr size_t kGraphNodes = 16384;
constexpr const char * kFlowCacheName = "meanvc2.dit_kv";
constexpr const char * kFlowCacheMode = "rolling";

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

std::vector<float> tanh_prior(const assets::TensorSource & source, const std::string & name) {
    auto values = source.require_f32(name, {kMemorySlots, kSpeakerDim});
    for (float & value : values) {
        value = std::tanh(value);
    }
    return values;
}

}  // namespace

struct MeanVC2GtmWeights {
    modules::LinearWeights mlp0;
    modules::LinearWeights mlp2;
    modules::NormWeights norm;
    core::TensorValue prior;
};

struct MeanVC2TimeEmbeddingWeights {
    modules::LinearWeights mlp0;
    modules::LinearWeights mlp2;
};

struct MeanVC2TemporalTimbreWeights {
    modules::LinearWeights q_proj;
    modules::LinearWeights k_proj;
    modules::LinearWeights v_proj;
    modules::LinearWeights out_proj;
    modules::NormWeights norm;
};

struct MeanVC2BlockWeights {
    modules::LinearWeights to_q;
    modules::LinearWeights to_k;
    modules::LinearWeights to_v;
    modules::LinearWeights to_out;
    modules::NormWeights q_norm;
    modules::NormWeights k_norm;
    modules::LinearWeights attn_norm;
    modules::LinearWeights ff0;
    modules::LinearWeights ff2;
};

struct MeanVC2FlowWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    MeanVC2GtmWeights key;
    MeanVC2GtmWeights value;
    MeanVC2TimeEmbeddingWeights t_time;
    MeanVC2TimeEmbeddingWeights r_time;
    MeanVC2TemporalTimbreWeights temporal_timbre;
    modules::LinearWeights input_proj;
    std::array<MeanVC2BlockWeights, kDepth> blocks;
    modules::LinearWeights norm_out;
    modules::LinearWeights proj_out;
};

namespace {

MeanVC2GtmWeights load_gtm_side(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & mlp_prefix,
    const std::string & norm_prefix,
    const std::string & prior_name,
    assets::TensorStorageType storage_type) {
    MeanVC2GtmWeights out;
    out.mlp0 = modules::binding::linear_from_source(
        store,
        source,
        mlp_prefix + ".0",
        storage_type,
        kSpeakerDim,
        kSpeakerDim,
        true);
    out.mlp2 = modules::binding::linear_from_source(
        store,
        source,
        mlp_prefix + ".2",
        storage_type,
        kMemoryValues,
        kSpeakerDim,
        true);
    out.norm = modules::binding::norm_from_source(store, source, norm_prefix, kSpeakerDim);
    out.prior = store.make_f32(
        core::TensorShape::from_dims({1, kMemorySlots, kSpeakerDim}),
        tanh_prior(source, prior_name));
    return out;
}

MeanVC2TimeEmbeddingWeights load_time_embedding(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type) {
    return {
        modules::binding::linear_from_source(
            store,
            source,
            prefix + ".time_mlp.0",
            storage_type,
            kHiddenDim,
            kTimeEmbedDim,
            true),
        modules::binding::linear_from_source(
            store,
            source,
            prefix + ".time_mlp.2",
            storage_type,
            kHiddenDim,
            kHiddenDim,
            true),
    };
}

MeanVC2TemporalTimbreWeights load_temporal_timbre(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    assets::TensorStorageType storage_type) {
    return {
        modules::binding::linear_from_source(
            store,
            source,
            "temporal_timbre.q_proj",
            storage_type,
            kTemporalInnerDim,
            kSpeakerDim,
            true),
        modules::binding::linear_from_source(
            store,
            source,
            "temporal_timbre.k_proj",
            storage_type,
            kTemporalInnerDim,
            kSpeakerDim,
            true),
        modules::binding::linear_from_source(
            store,
            source,
            "temporal_timbre.v_proj",
            storage_type,
            kTemporalInnerDim,
            kSpeakerDim,
            true),
        modules::binding::linear_from_source(
            store,
            source,
            "temporal_timbre.out_proj",
            storage_type,
            kSpeakerDim,
            kTemporalInnerDim,
            true),
        modules::binding::norm_from_source(store, source, "temporal_timbre.attn_norm", kSpeakerDim),
    };
}

MeanVC2BlockWeights load_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    size_t index,
    assets::TensorStorageType storage_type) {
    const std::string prefix = "transformer_blocks." + std::to_string(index);
    return {
        modules::binding::linear_from_source(
            store,
            source,
            prefix + ".attn.to_q",
            storage_type,
            kInnerDim,
            kHiddenDim,
            true),
        modules::binding::linear_from_source(
            store,
            source,
            prefix + ".attn.to_k",
            storage_type,
            kInnerDim,
            kHiddenDim,
            true),
        modules::binding::linear_from_source(
            store,
            source,
            prefix + ".attn.to_v",
            storage_type,
            kInnerDim,
            kHiddenDim,
            true),
        modules::binding::linear_from_source(
            store,
            source,
            prefix + ".attn.to_out.0",
            storage_type,
            kHiddenDim,
            kInnerDim,
            true),
        modules::binding::norm_weight_from_source(store, source, prefix + ".attn.q_norm", kHeadDim),
        modules::binding::norm_weight_from_source(store, source, prefix + ".attn.k_norm", kHeadDim),
        modules::binding::linear_from_source(
            store,
            source,
            prefix + ".attn_norm.linear",
            storage_type,
            6 * kHiddenDim,
            kHiddenDim,
            true),
        modules::binding::linear_from_source(
            store,
            source,
            prefix + ".ff.ff.0.0",
            storage_type,
            kFeedForwardDim,
            kHiddenDim,
            true),
        modules::binding::linear_from_source(
            store,
            source,
            prefix + ".ff.ff.2",
            storage_type,
            kHiddenDim,
            kFeedForwardDim,
            true),
    };
}

std::shared_ptr<const MeanVC2FlowWeights> load_flow_weights(
    ggml_backend_t backend,
    core::BackendType backend_type,
    const assets::TensorSource & source,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    auto weights = std::make_shared<MeanVC2FlowWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "meanvc2.vc.weights",
        weight_context_bytes);
    weights->key = load_gtm_side(
        *weights->store,
        source,
        "gtm.mlp_k",
        "gtm.norm_k",
        "gtm.k_prior",
        storage_type);
    weights->value = load_gtm_side(
        *weights->store,
        source,
        "gtm.mlp_v",
        "gtm.norm_v",
        "gtm.v_prior",
        storage_type);
    weights->t_time = load_time_embedding(*weights->store, source, "t_time_embed", storage_type);
    weights->r_time = load_time_embedding(*weights->store, source, "r_time_embed", storage_type);
    weights->temporal_timbre = load_temporal_timbre(*weights->store, source, storage_type);
    weights->input_proj = modules::binding::linear_from_source(
        *weights->store,
        source,
        "input_embed.proj",
        storage_type,
        kHiddenDim,
        kMelDim + 2 * kSpeakerDim,
        true);
    for (size_t i = 0; i < kDepth; ++i) {
        weights->blocks[i] = load_block(*weights->store, source, i, storage_type);
    }
    weights->norm_out = modules::binding::linear_from_source(
        *weights->store,
        source,
        "norm_out.linear",
        storage_type,
        2 * kHiddenDim,
        kHiddenDim,
        true);
    weights->proj_out = modules::binding::linear_from_source(
        *weights->store,
        source,
        "proj_out",
        storage_type,
        kMelDim,
        kHiddenDim,
        true);
    weights->store->upload();
    return weights;
}

core::TensorValue build_gtm_side(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & speaker,
    const MeanVC2GtmWeights & weights) {
    auto x = modules::LinearModule({kSpeakerDim, kSpeakerDim, true}).build(ctx, speaker, weights.mlp0);
    x = modules::SiluModule{}.build(ctx, x);
    x = modules::LinearModule({kSpeakerDim, kMemoryValues, true}).build(ctx, x, weights.mlp2);
    x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, kMemorySlots, kSpeakerDim}));
    x = modules::AddModule{}.build(ctx, x, weights.prior);
    return modules::LayerNormModule({kSpeakerDim, 1.0e-5F, true, true}).build(ctx, x, weights.norm);
}

int64_t layer_cache_capacity(size_t layer) {
    return (kLayerPastChunks[layer] + 1) * kChunkFrames;
}

std::vector<float> sinus_time_embedding(float value) {
    constexpr int64_t kHalf = kTimeEmbedDim / 2;
    std::vector<float> out(static_cast<size_t>(kTimeEmbedDim), 0.0F);
    const float log_base = std::log(10000.0F) / static_cast<float>(kHalf - 1);
    const float scaled_value = 1000.0F * value;
    for (int64_t i = 0; i < kHalf; ++i) {
        const float frequency = std::exp(static_cast<float>(i) * -log_base);
        const float phase = scaled_value * frequency;
        out[static_cast<size_t>(i)] = std::sin(phase);
        out[static_cast<size_t>(kHalf + i)] = std::cos(phase);
    }
    return out;
}

core::TensorValue modulate(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & normalized,
    const core::TensorValue & shift,
    const core::TensorValue & scale) {
    const auto scale_broadcast = modules::RepeatModule({normalized.shape}).build(
        ctx,
        core::reshape_tensor(ctx, scale, core::TensorShape::from_dims({1, 1, scale.shape.last_dim()})));
    const auto scale_rep_input = core::ensure_backend_addressable_layout(ctx, scale_broadcast);
    ggml_tensor * scale_rep_raw = ggml_scale_bias(ctx.ggml, scale_rep_input.tensor, 1.0F, 1.0F);
    const auto scale_rep = core::wrap_tensor(scale_rep_raw, scale_broadcast.shape, scale_rep_raw->type);
    const auto shift_broadcast = modules::RepeatModule({normalized.shape}).build(
        ctx,
        core::reshape_tensor(ctx, shift, core::TensorShape::from_dims({1, 1, shift.shape.last_dim()})));
    const auto shifted = modules::MulModule{}.build(ctx, normalized, scale_rep);
    return modules::AddModule{}.build(ctx, shifted, shift_broadcast);
}

core::TensorValue gated_residual(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & residual,
    const core::TensorValue & update,
    const core::TensorValue & gate) {
    const auto gated = modules::MulModule{}.build(
        ctx,
        update,
        modules::RepeatModule({update.shape}).build(
            ctx,
            core::reshape_tensor(ctx, gate, core::TensorShape::from_dims({1, 1, gate.shape.last_dim()}))));
    return modules::AddModule{}.build(ctx, residual, gated);
}

core::TensorValue reshape_heads_bthd(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value,
    int64_t steps,
    int64_t heads,
    int64_t head_dim) {
    return core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, value),
        core::TensorShape::from_dims({1, steps, heads, head_dim}));
}

core::TensorValue bthd_to_bhtd(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    return modules::TransposeModule({{0, 2, 1, 3}, value.shape.rank}).build(ctx, value);
}

core::TensorValue build_time_embedding(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const MeanVC2TimeEmbeddingWeights & weights) {
    auto x = modules::LinearModule({kTimeEmbedDim, kHiddenDim, true}).build(ctx, input, weights.mlp0);
    x = modules::SiluModule{}.build(ctx, x);
    return modules::LinearModule({kHiddenDim, kHiddenDim, true}).build(ctx, x, weights.mlp2);
}

core::TensorValue build_temporal_timbre(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & condition,
    const core::TensorValue & memory_key,
    const core::TensorValue & memory_value,
    const MeanVC2TemporalTimbreWeights & weights) {
    auto q = modules::LinearModule({kSpeakerDim, kTemporalInnerDim, true}).build(ctx, condition, weights.q_proj);
    auto k = modules::LinearModule({kSpeakerDim, kTemporalInnerDim, true}).build(ctx, memory_key, weights.k_proj);
    auto v = modules::LinearModule({kSpeakerDim, kTemporalInnerDim, true}).build(ctx, memory_value, weights.v_proj);
    q = bthd_to_bhtd(ctx, reshape_heads_bthd(ctx, q, kConditionFrames, kTemporalHeads, kTemporalHeadDim));
    k = bthd_to_bhtd(ctx, reshape_heads_bthd(ctx, k, kMemorySlots, kTemporalHeads, kTemporalHeadDim));
    v = bthd_to_bhtd(ctx, reshape_heads_bthd(ctx, v, kMemorySlots, kTemporalHeads, kTemporalHeadDim));
    auto attended = modules::ScaledDotProductAttentionModule({
        kTemporalHeadDim,
        modules::ScaledDotProductAttentionLowering::Explicit,
        GGML_PREC_DEFAULT,
        modules::AttentionCausality::NonCausal,
    }).build(ctx, q, k, v);
    attended = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, attended),
        core::TensorShape::from_dims({1, kConditionFrames, kTemporalInnerDim}));
    attended = modules::LinearModule({kTemporalInnerDim, kSpeakerDim, true}).build(ctx, attended, weights.out_proj);
    return modules::LayerNormModule({kSpeakerDim, 1.0e-5F, true, true}).build(ctx, attended, weights.norm);
}

struct MeanVC2BlockBuildResult {
    core::TensorValue hidden;
    core::TensorValue current_key;
    core::TensorValue current_value;
};

MeanVC2BlockBuildResult build_transformer_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & hidden,
    const core::TensorValue & time_embedding,
    const core::TensorValue & cache_key,
    const core::TensorValue & cache_value,
    const core::TensorValue & attention_mask,
    const core::TensorValue & query_positions,
    const core::TensorValue & key_positions,
    const MeanVC2BlockWeights & weights) {
    auto params = modules::LinearModule({kHiddenDim, 6 * kHiddenDim, true}).build(
        ctx,
        modules::SiluModule{}.build(ctx, time_embedding),
        weights.attn_norm);
    const auto shift_msa = modules::SliceModule({1, 0 * kHiddenDim, kHiddenDim}).build(ctx, params);
    const auto scale_msa = modules::SliceModule({1, 1 * kHiddenDim, kHiddenDim}).build(ctx, params);
    const auto gate_msa = modules::SliceModule({1, 2 * kHiddenDim, kHiddenDim}).build(ctx, params);
    const auto shift_mlp = modules::SliceModule({1, 3 * kHiddenDim, kHiddenDim}).build(ctx, params);
    const auto scale_mlp = modules::SliceModule({1, 4 * kHiddenDim, kHiddenDim}).build(ctx, params);
    const auto gate_mlp = modules::SliceModule({1, 5 * kHiddenDim, kHiddenDim}).build(ctx, params);

    auto norm = modules::LayerNormModule({kHiddenDim, 1.0e-6F, false, false}).build(ctx, hidden, {});
    norm = modulate(ctx, norm, shift_msa, scale_msa);

    auto q = modules::LinearModule({kHiddenDim, kInnerDim, true}).build(ctx, norm, weights.to_q);
    auto k = modules::LinearModule({kHiddenDim, kInnerDim, true}).build(ctx, norm, weights.to_k);
    auto v = modules::LinearModule({kHiddenDim, kInnerDim, true}).build(ctx, norm, weights.to_v);
    q = reshape_heads_bthd(ctx, q, kConditionFrames, kHeads, kHeadDim);
    k = reshape_heads_bthd(ctx, k, kConditionFrames, kHeads, kHeadDim);
    v = reshape_heads_bthd(ctx, v, kConditionFrames, kHeads, kHeadDim);
    q = modules::RMSNormModule({kHeadDim, 1.0e-6F, true, false}).build(ctx, q, weights.q_norm);
    k = modules::RMSNormModule({kHeadDim, 1.0e-6F, true, false}).build(ctx, k, weights.k_norm);
    v = core::wrap_tensor(ggml_cont(ctx.ggml, v.tensor), v.shape, v.type);

    auto key_cat = modules::ConcatModule({1}).build(ctx, cache_key, k);
    auto value_cat = modules::ConcatModule({1}).build(ctx, cache_value, v);
    q = modules::RoPEModule({kHeadDim, GGML_ROPE_TYPE_NORMAL}).build(ctx, q, query_positions);
    key_cat = modules::RoPEModule({kHeadDim, GGML_ROPE_TYPE_NORMAL}).build(ctx, key_cat, key_positions);
    auto context = modules::ScaledDotProductAttentionModule({
        kHeadDim,
        modules::ScaledDotProductAttentionLowering::Explicit,
        GGML_PREC_DEFAULT,
        modules::AttentionCausality::NonCausal,
    }).build(ctx, bthd_to_bhtd(ctx, q), bthd_to_bhtd(ctx, key_cat), bthd_to_bhtd(ctx, value_cat), attention_mask);
    context = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, context),
        core::TensorShape::from_dims({1, kConditionFrames, kInnerDim}));
    auto attn_out = modules::LinearModule({kInnerDim, kHiddenDim, true}).build(ctx, context, weights.to_out);
    auto x = gated_residual(ctx, hidden, attn_out, gate_msa);

    auto ff_norm = modules::LayerNormModule({kHiddenDim, 1.0e-6F, false, false}).build(ctx, x, {});
    ff_norm = modulate(ctx, ff_norm, shift_mlp, scale_mlp);
    auto ff = modules::LinearModule({kHiddenDim, kFeedForwardDim, true}).build(ctx, ff_norm, weights.ff0);
    ff = modules::GeluModule({modules::GeluApproximation::Tanh}).build(ctx, ff);
    ff = modules::LinearModule({kFeedForwardDim, kHiddenDim, true}).build(ctx, ff, weights.ff2);
    return {gated_residual(ctx, x, ff, gate_mlp), k, v};
}

core::TensorValue build_dit_velocity(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const core::TensorValue & condition,
    const core::TensorValue & speaker,
    const core::TensorValue & memory_key,
    const core::TensorValue & memory_value,
    const core::TensorValue & t_hidden,
    const core::TensorValue & r_hidden,
    const std::array<core::TensorValue, kDepth> & cache_keys,
    const std::array<core::TensorValue, kDepth> & cache_values,
    const std::array<core::TensorValue, kDepth> & attention_masks,
    const core::TensorValue & query_positions,
    const std::array<core::TensorValue, kDepth> & key_positions,
    const MeanVC2FlowWeights & weights,
    std::array<core::TensorValue, kDepth> & current_keys,
    std::array<core::TensorValue, kDepth> & current_values) {
    const auto t_emb = build_time_embedding(ctx, t_hidden, weights.t_time);
    const auto r_emb = build_time_embedding(ctx, r_hidden, weights.r_time);
    auto time_emb = modules::AddModule{}.build(ctx, t_emb, r_emb);

    const auto timbre = build_temporal_timbre(ctx, condition, memory_key, memory_value, weights.temporal_timbre);
    const auto speaker_frame = core::reshape_tensor(ctx, speaker, core::TensorShape::from_dims({1, 1, kSpeakerDim}));
    const auto speaker_expanded = modules::RepeatModule({condition.shape}).build(ctx, speaker_frame);
    auto input = modules::ConcatModule({2}).build(ctx, x, timbre);
    input = modules::ConcatModule({2}).build(ctx, input, speaker_expanded);
    auto hidden = modules::LinearModule({kMelDim + 2 * kSpeakerDim, kHiddenDim, true}).build(ctx, input, weights.input_proj);

    for (size_t i = 0; i < kDepth; ++i) {
        auto block = build_transformer_block(
            ctx,
            hidden,
            time_emb,
            cache_keys[i],
            cache_values[i],
            attention_masks[i],
            query_positions,
            key_positions[i],
            weights.blocks[i]);
        hidden = block.hidden;
        current_keys[i] = block.current_key;
        current_values[i] = block.current_value;
    }

    auto final_params = modules::LinearModule({kHiddenDim, 2 * kHiddenDim, true}).build(
        ctx,
        modules::SiluModule{}.build(ctx, time_emb),
        weights.norm_out);
    const auto scale = modules::SliceModule({1, 0, kHiddenDim}).build(ctx, final_params);
    const auto shift = modules::SliceModule({1, kHiddenDim, kHiddenDim}).build(ctx, final_params);
    hidden = modules::LayerNormModule({kHiddenDim, 1.0e-6F, false, false}).build(ctx, hidden, {});
    hidden = modulate(ctx, hidden, shift, scale);
    return modules::LinearModule({kHiddenDim, kMelDim, true}).build(ctx, hidden, weights.proj_out);
}

}  // namespace

struct MeanVC2GtmGraph {
    MeanVC2GtmGraph(
        ggml_backend_t backend,
        core::BackendType backend_type,
        size_t graph_context_bytes,
        std::shared_ptr<const MeanVC2FlowWeights> weights)
        : backend(backend),
          weights(std::move(weights)) {
        if (backend == nullptr || this->weights == nullptr) {
            throw std::runtime_error("MeanVC2 GTM graph requires backend and weights");
        }
        ggml_init_params params{graph_context_bytes, nullptr, true};
        ctx.reset(ggml_init(params));
        if (ctx == nullptr) {
            throw std::runtime_error("failed to initialize MeanVC2 GTM graph context");
        }
        core::ModuleBuildContext build_ctx{ctx.get(), "meanvc2.vc.gtm", backend_type};
        speaker = core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, kSpeakerDim}));
        ggml_set_input(speaker.tensor);
        key_output = build_gtm_side(build_ctx, speaker, this->weights->key);
        value_output = build_gtm_side(build_ctx, speaker, this->weights->value);
        key_output = core::ensure_backend_addressable_layout(build_ctx, key_output);
        value_output = core::ensure_backend_addressable_layout(build_ctx, value_output);
        ggml_set_output(key_output.tensor);
        ggml_set_output(value_output.tensor);

        graph = ggml_new_graph_custom(ctx.get(), kGraphNodes, false);
        ggml_build_forward_expand(graph, key_output.tensor);
        ggml_build_forward_expand(graph, value_output.tensor);
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
            throw std::runtime_error("failed to allocate MeanVC2 GTM graph");
        }
    }

    ~MeanVC2GtmGraph() {
        if (backend != nullptr) {
            core::release_backend_graph_resources(backend, graph);
        }
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
            gallocr = nullptr;
        }
    }

    MeanVC2GtmMemory run(const std::vector<float> & speaker_embedding) {
        if (static_cast<int64_t>(speaker_embedding.size()) != kSpeakerDim) {
            throw std::runtime_error("MeanVC2 GTM speaker embedding shape mismatch");
        }
        core::write_tensor_float(speaker, speaker_embedding);
        const ggml_status status = core::compute_backend_graph(backend, graph, nullptr, "MeanVC2 GTM");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MeanVC2 GTM graph compute failed");
        }
        return {
            core::read_tensor_float(key_output.tensor),
            core::read_tensor_float(value_output.tensor),
        };
    }

    ggml_backend_t backend = nullptr;
    std::shared_ptr<const MeanVC2FlowWeights> weights;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    core::TensorValue speaker;
    core::TensorValue key_output;
    core::TensorValue value_output;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t gallocr = nullptr;
};

struct MeanVC2DitStepOutput {
    std::vector<float> velocity;
    std::array<std::vector<std::byte>, kDepth> current_keys;
    std::array<std::vector<std::byte>, kDepth> current_values;
};

struct MeanVC2DitStepGraph {
    MeanVC2DitStepGraph(
        ggml_backend_t backend,
        core::BackendType backend_type,
        size_t graph_context_bytes,
        std::shared_ptr<const MeanVC2FlowWeights> weights)
        : backend(backend),
          weights(std::move(weights)) {
        if (backend == nullptr || this->weights == nullptr) {
            throw std::runtime_error("MeanVC2 DiT graph requires backend and weights");
        }
        ggml_init_params params{graph_context_bytes, nullptr, true};
        ctx.reset(ggml_init(params));
        if (ctx == nullptr) {
            throw std::runtime_error("failed to initialize MeanVC2 DiT graph context");
        }
        core::ModuleBuildContext build_ctx{ctx.get(), "meanvc2.vc.dit", backend_type};
        x = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, kConditionFrames, kMelDim}));
        condition = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, kConditionFrames, kSpeakerDim}));
        speaker = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, kSpeakerDim}));
        memory_key = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, kMemorySlots, kSpeakerDim}));
        memory_value = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, kMemorySlots, kSpeakerDim}));
        t_hidden = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, kTimeEmbedDim}));
        r_hidden = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, kTimeEmbedDim}));
        query_positions = core::make_tensor(build_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({kConditionFrames}));
        for (size_t i = 0; i < kDepth; ++i) {
            const int64_t cap = layer_cache_capacity(i);
            cache_keys[i] = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, cap, kHeads, kHeadDim}));
            cache_values[i] = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, cap, kHeads, kHeadDim}));
            attention_masks[i] = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, kConditionFrames, cap + kConditionFrames}));
            key_positions[i] = core::make_tensor(build_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({cap + kConditionFrames}));
        }
        ggml_set_input(x.tensor);
        ggml_set_input(condition.tensor);
        ggml_set_input(speaker.tensor);
        ggml_set_input(memory_key.tensor);
        ggml_set_input(memory_value.tensor);
        ggml_set_input(t_hidden.tensor);
        ggml_set_input(r_hidden.tensor);
        ggml_set_input(query_positions.tensor);
        for (size_t i = 0; i < kDepth; ++i) {
            ggml_set_input(cache_keys[i].tensor);
            ggml_set_input(cache_values[i].tensor);
            ggml_set_input(attention_masks[i].tensor);
            ggml_set_input(key_positions[i].tensor);
        }

        velocity = build_dit_velocity(
            build_ctx,
            x,
            condition,
            speaker,
            memory_key,
            memory_value,
            t_hidden,
            r_hidden,
            cache_keys,
            cache_values,
            attention_masks,
            query_positions,
            key_positions,
            *this->weights,
            current_keys,
            current_values);
        velocity = core::ensure_backend_addressable_layout(build_ctx, velocity);
        ggml_set_output(velocity.tensor);
        for (size_t i = 0; i < kDepth; ++i) {
            current_keys[i] = core::ensure_backend_addressable_layout(build_ctx, current_keys[i]);
            current_values[i] = core::ensure_backend_addressable_layout(build_ctx, current_values[i]);
            ggml_set_output(current_keys[i].tensor);
            ggml_set_output(current_values[i].tensor);
        }

        graph = ggml_new_graph_custom(ctx.get(), kGraphNodes, false);
        ggml_build_forward_expand(graph, velocity.tensor);
        for (size_t i = 0; i < kDepth; ++i) {
            ggml_build_forward_expand(graph, current_keys[i].tensor);
            ggml_build_forward_expand(graph, current_values[i].tensor);
        }
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
            throw std::runtime_error("failed to allocate MeanVC2 DiT graph");
        }
    }

    ~MeanVC2DitStepGraph() {
        if (backend != nullptr) {
            core::release_backend_graph_resources(backend, graph);
        }
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
            gallocr = nullptr;
        }
    }

    MeanVC2DitStepOutput run(
        const std::vector<float> & x_values,
        const std::vector<float> & condition_values,
        const std::vector<float> & speaker_values,
        const MeanVC2GtmMemory & memory,
        float t,
        float r,
        int64_t offset,
        const runtime::RollingFlowKVCache & kv_cache) {
        if (x_values.size() != static_cast<size_t>(kConditionFrames * kMelDim) ||
            condition_values.size() != static_cast<size_t>(kConditionFrames * kSpeakerDim) ||
            speaker_values.size() != static_cast<size_t>(kSpeakerDim) ||
            memory.keys.size() != static_cast<size_t>(kMemoryValues) ||
            memory.values.size() != static_cast<size_t>(kMemoryValues) ||
            kv_cache.layers() != kDepth) {
            throw std::runtime_error("MeanVC2 DiT input shape mismatch");
        }
        core::write_tensor_float(x, x_values);
        core::write_tensor_float(condition, condition_values);
        core::write_tensor_float(speaker, speaker_values);
        core::write_tensor_float(memory_key, memory.keys);
        core::write_tensor_float(memory_value, memory.values);
        core::write_tensor_float(t_hidden, sinus_time_embedding(t));
        core::write_tensor_float(r_hidden, sinus_time_embedding(r));
        core::write_tensor_i32(query_positions, query_position_values(offset));
        for (size_t i = 0; i < kDepth; ++i) {
            const auto & layer = kv_cache.layer(i);
            core::write_tensor_bytes(cache_keys[i], layer.key_bytes);
            core::write_tensor_bytes(cache_values[i], layer.value_bytes);
            core::write_tensor_float(attention_masks[i], attention_mask_values(i, offset, layer.valid_steps));
            core::write_tensor_i32(key_positions[i], key_position_values(i, offset, layer.valid_steps));
        }
        const ggml_status status = core::compute_backend_graph(backend, graph, nullptr, "MeanVC2 DiT");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MeanVC2 DiT graph compute failed");
        }
        MeanVC2DitStepOutput output;
        output.velocity = core::read_tensor_float(velocity.tensor);
        for (size_t i = 0; i < kDepth; ++i) {
            output.current_keys[i] = core::read_tensor_bytes(current_keys[i].tensor);
            output.current_values[i] = core::read_tensor_bytes(current_values[i].tensor);
        }
        return output;
    }

    static std::vector<int32_t> query_position_values(int64_t offset) {
        std::vector<int32_t> out(static_cast<size_t>(kConditionFrames), 0);
        for (int64_t i = 0; i < kConditionFrames; ++i) {
            out[static_cast<size_t>(i)] = static_cast<int32_t>(offset + i);
        }
        return out;
    }

    static std::vector<int32_t> key_position_values(size_t layer, int64_t offset, int64_t valid_frames) {
        const int64_t cap = layer_cache_capacity(layer);
        std::vector<int32_t> out(static_cast<size_t>(cap + kConditionFrames), 0);
        const int64_t valid_start = cap - valid_frames;
        for (int64_t i = 0; i < cap; ++i) {
            if (i >= valid_start) {
                out[static_cast<size_t>(i)] = static_cast<int32_t>(offset - valid_frames + (i - valid_start));
            }
        }
        for (int64_t i = 0; i < kConditionFrames; ++i) {
            out[static_cast<size_t>(cap + i)] = static_cast<int32_t>(offset + i);
        }
        return out;
    }

    static bool attention_visible(int64_t query_abs, int64_t key_abs, size_t layer) {
        const int64_t q_chunk = query_abs / kChunkFrames;
        const int64_t k_chunk = key_abs / kChunkFrames;
        if (q_chunk == k_chunk) {
            return true;
        }
        if (k_chunk >= q_chunk - kLayerPastChunks[layer] && k_chunk < q_chunk) {
            return true;
        }
        if (kLayerFutureChunks[layer] > 0 && k_chunk == q_chunk + 1) {
            return key_abs % kChunkFrames < kBlockFrames * kLayerFutureChunks[layer];
        }
        return false;
    }

    static std::vector<float> attention_mask_values(size_t layer, int64_t offset, int64_t valid_frames) {
        const int64_t cap = layer_cache_capacity(layer);
        std::vector<float> out(static_cast<size_t>(kConditionFrames * (cap + kConditionFrames)), -INFINITY);
        const int64_t valid_start = cap - valid_frames;
        for (int64_t q = 0; q < kConditionFrames; ++q) {
            const int64_t q_abs = offset + q;
            for (int64_t k = 0; k < cap + kConditionFrames; ++k) {
                bool valid = false;
                int64_t k_abs = 0;
                if (k < cap) {
                    valid = k >= valid_start;
                    k_abs = offset - valid_frames + (k - valid_start);
                } else {
                    valid = true;
                    k_abs = offset + (k - cap);
                }
                if (valid && attention_visible(q_abs, k_abs, layer)) {
                    out[static_cast<size_t>(q * (cap + kConditionFrames) + k)] = 0.0F;
                }
            }
        }
        return out;
    }

    ggml_backend_t backend = nullptr;
    std::shared_ptr<const MeanVC2FlowWeights> weights;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    core::TensorValue x;
    core::TensorValue condition;
    core::TensorValue speaker;
    core::TensorValue memory_key;
    core::TensorValue memory_value;
    core::TensorValue t_hidden;
    core::TensorValue r_hidden;
    core::TensorValue query_positions;
    std::array<core::TensorValue, kDepth> cache_keys;
    std::array<core::TensorValue, kDepth> cache_values;
    std::array<core::TensorValue, kDepth> attention_masks;
    std::array<core::TensorValue, kDepth> key_positions;
    core::TensorValue velocity;
    std::array<core::TensorValue, kDepth> current_keys;
    std::array<core::TensorValue, kDepth> current_values;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t gallocr = nullptr;
};

class MeanVC2DenoiserRuntime final : public modules::FlowSamplerDenoiserRuntime {
public:
    MeanVC2DenoiserRuntime(
        ggml_backend_t backend,
        core::BackendType backend_type,
        size_t graph_context_bytes,
        std::shared_ptr<const MeanVC2FlowWeights> weights)
        : backend_(backend),
          backend_type_(backend_type),
          graph_context_bytes_(graph_context_bytes),
          weights_(std::move(weights)) {
        if (backend_ == nullptr || weights_ == nullptr) {
            throw std::runtime_error("MeanVC2 denoiser runtime requires backend and weights");
        }
        reset_kv_cache();
    }

    void set_window_inputs(
        const std::vector<float> & condition,
        const std::vector<float> & speaker,
        const MeanVC2GtmMemory & memory) {
        if (condition.size() != static_cast<size_t>(kConditionFrames * kSpeakerDim) ||
            speaker.size() != static_cast<size_t>(kSpeakerDim) ||
            memory.keys.size() != static_cast<size_t>(kMemoryValues) ||
            memory.values.size() != static_cast<size_t>(kMemoryValues)) {
            throw std::runtime_error("MeanVC2 denoiser window input shape mismatch");
        }
        condition_ = condition;
        speaker_ = speaker;
        memory_ = memory;
        has_window_inputs_ = true;
    }

    void reset_sampler_caches(const std::vector<modules::FlowSamplerCacheState> & caches) override {
        if (caches.size() != 1 || caches.front().name != kFlowCacheName) {
            throw std::runtime_error("MeanVC2 denoiser received unexpected flow cache state");
        }
        offset_ = 0;
        kv_cache_.clear_valid_steps();
    }

    std::vector<modules::FlowSamplerCacheUpdate> begin_sampler_sequence(const modules::FlowSamplerSequenceState & state) override {
        if (state.caches.size() != 1 || state.caches.front().name != kFlowCacheName) {
            throw std::runtime_error("MeanVC2 denoiser received unexpected flow cache state");
        }
        if (!has_window_inputs_) {
            throw std::runtime_error("MeanVC2 denoiser requires window inputs before sequence");
        }
        if (offset_ >= kMaxCachedOffset) {
            offset_ = 0;
            kv_cache_.clear_valid_steps();
            return {{kFlowCacheName, modules::FlowSamplerCacheUpdateKind::Reset}};
        }
        return {};
    }

    modules::FlowSamplerGraphKey sampler_graph_key(const modules::FlowSamplerStepState & state) override {
        modules::FlowSamplerGraphKey key;
        key.latent_shape = {kConditionFrames * kMelDim};
        key.branch_count = static_cast<int64_t>(state.branches.size());
        key.schedule_steps = 2;
        key.sampler_mode = "velocity_euler";
        key.caches = {{kFlowCacheName, kFlowCacheMode}};
        key.modulation_revision = 0;
        return key;
    }

    void rebuild_sampler_graph(
        const modules::FlowSamplerGraphKey &,
        const modules::FlowSamplerStepState &) override {
        graph_ = std::make_unique<MeanVC2DitStepGraph>(
            backend_,
            backend_type_,
            graph_context_bytes_,
            weights_);
    }

    modules::FlowSamplerDenoiserOutput run_sampler_denoiser(const modules::FlowSamplerDenoiserInput & input) override {
        if (graph_ == nullptr) {
            throw std::runtime_error("MeanVC2 denoiser graph is not built");
        }
        if (!has_window_inputs_) {
            throw std::runtime_error("MeanVC2 denoiser window inputs are not set");
        }
        const auto step = graph_->run(
            input.latent,
            condition_,
            speaker_,
            memory_,
            input.state.schedule.t,
            input.state.schedule.t_next,
            offset_,
            kv_cache_);
        if (step.velocity.size() != input.latent.size()) {
            throw std::runtime_error("MeanVC2 DiT velocity shape mismatch");
        }

        modules::FlowSamplerDenoiserOutput output;
        output.predictions.push_back({"main", step.velocity});
        if (input.state.sequence_index + 1 == input.state.graph_key.schedule_steps) {
            for (size_t layer = 0; layer < kDepth; ++layer) {
                kv_cache_.append_tail_bytes(
                    layer,
                    step.current_keys[layer],
                    step.current_values[layer],
                    kConditionFrames,
                    kChunkFrames,
                    "MeanVC2 DiT");
            }
            offset_ += kChunkFrames;
            output.cache_updates.push_back({kFlowCacheName, modules::FlowSamplerCacheUpdateKind::Updated});
        }
        return output;
    }

    void release_sampler_graphs() override {
        graph_.reset();
    }

private:
    void reset_kv_cache() {
        std::vector<int64_t> capacities;
        capacities.reserve(kDepth);
        for (size_t i = 0; i < kDepth; ++i) {
            capacities.push_back(layer_cache_capacity(i));
        }
        kv_cache_.reset(capacities, kHeads * kHeadDim);
    }

    static constexpr int64_t kMaxCachedOffset = 4000;

    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    size_t graph_context_bytes_ = 0;
    std::shared_ptr<const MeanVC2FlowWeights> weights_;
    runtime::RollingFlowKVCache kv_cache_;
    int64_t offset_ = 0;
    bool has_window_inputs_ = false;
    std::vector<float> condition_;
    std::vector<float> speaker_;
    MeanVC2GtmMemory memory_;
    std::unique_ptr<MeanVC2DitStepGraph> graph_;
};

modules::FlowSamplerRuntimeConfig make_meanvc2_sampler_config() {
    modules::FlowSamplerRuntimeConfig config;
    config.label = "MeanVC2 DiT";
    config.latent_shape = {kConditionFrames * kMelDim};
    config.schedule = {
        {0, 1.0F, 0.5F, 0.0F, 0.0F},
        {1, 0.5F, 0.0F, 0.0F, 0.0F},
    };
    config.branches = {{"main", 1.0F}};
    config.caches = {{kFlowCacheName, kFlowCacheMode, false}};
    config.prediction_type = modules::FlowSamplerPredictionType::Velocity;
    config.update_rule = modules::FlowSamplerUpdateRule::Euler;
    return config;
}

MeanVC2FlowSamplerRuntime::MeanVC2FlowSamplerRuntime(
    std::shared_ptr<const assets::TensorSource> source,
    core::ExecutionContext & execution_context,
    size_t weight_context_bytes,
    size_t graph_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : execution_context_(execution_context),
      graph_context_bytes_(graph_context_bytes),
      source_(std::move(source)) {
    if (source_ == nullptr) {
        throw std::runtime_error("MeanVC2 VC runtime requires tensor source");
    }
    rng_policy_ = sampling::resolve_torch_cuda_sampling_policy(
        execution_context_.backend_type(),
        execution_context_.config().device,
        "meanvc2.vc.cuda_sampling_policy",
        "MeanVC2",
        sampling::TorchCudaSamplingPolicyFailureMode::StrictCuda);
    weights_ = load_flow_weights(
        execution_context_.backend(),
        execution_context_.backend_type(),
        *source_,
        weight_context_bytes,
        weight_storage_type);
    source_->release_storage();
}

MeanVC2FlowSamplerRuntime::~MeanVC2FlowSamplerRuntime() = default;

MeanVC2GtmMemory MeanVC2FlowSamplerRuntime::encode_speaker_memory(const std::vector<float> & speaker_embedding) const {
    if (gtm_graph_ == nullptr) {
        gtm_graph_ = std::make_unique<MeanVC2GtmGraph>(
            execution_context_.backend(),
            execution_context_.backend_type(),
            graph_context_bytes_,
            weights_);
    }
    const auto start = std::chrono::steady_clock::now();
    auto memory = gtm_graph_->run(speaker_embedding);
    debug::timing_log_scalar("meanvc2.vc.gtm_ms", debug::elapsed_ms(start, std::chrono::steady_clock::now()));
    return memory;
}

void MeanVC2FlowSamplerRuntime::start_streaming(
    const std::vector<float> & speaker_embedding,
    const MeanVC2GtmMemory & speaker_memory,
    uint64_t seed) {
    if (speaker_embedding.size() != static_cast<size_t>(kSpeakerDim) ||
        speaker_memory.keys.size() != static_cast<size_t>(kMemoryValues) ||
        speaker_memory.values.size() != static_cast<size_t>(kMemoryValues)) {
        throw std::runtime_error("MeanVC2 streaming VC speaker state shape mismatch");
    }
    if (sampler_runtime_ == nullptr) {
        auto denoiser = std::make_unique<MeanVC2DenoiserRuntime>(
            execution_context_.backend(),
            execution_context_.backend_type(),
            graph_context_bytes_,
            weights_);
        denoiser_ = denoiser.get();
        sampler_runtime_ = std::make_unique<modules::FlowSamplerRuntime>(
            make_meanvc2_sampler_config(),
            std::move(denoiser));
    }
    sampler_runtime_->reset_runtime_caches();
    stream_noise_cache_.assign(static_cast<size_t>(kBlockFrames * kMelDim), 0.0F);
    stream_speaker_embedding_ = speaker_embedding;
    stream_speaker_memory_ = speaker_memory;
    stream_seed_ = seed;
    stream_rng_offset_blocks_ = 0;
    stream_has_noise_cache_ = false;
    stream_started_ = true;
}

void MeanVC2FlowSamplerRuntime::reset_streaming() {
    if (sampler_runtime_ != nullptr) {
        sampler_runtime_->reset_runtime_caches();
    }
    stream_noise_cache_.clear();
    stream_speaker_embedding_.clear();
    stream_speaker_memory_ = {};
    stream_rng_offset_blocks_ = 0;
    stream_has_noise_cache_ = false;
    stream_started_ = false;
}

std::vector<float> MeanVC2FlowSamplerRuntime::synthesize_streaming_chunk(
    const std::vector<float> & condition_frames) {
    if (!stream_started_ || sampler_runtime_ == nullptr || denoiser_ == nullptr) {
        throw std::runtime_error("MeanVC2 streaming VC requires start_streaming()");
    }
    if (condition_frames.size() != static_cast<size_t>(kConditionFrames * kSpeakerDim)) {
        throw std::runtime_error("MeanVC2 streaming VC condition frame shape mismatch");
    }
    const uint64_t noise_elements = static_cast<uint64_t>(kConditionFrames * kMelDim);
    std::vector<float> x = sampling::generate_torch_cuda_tensor_iterator_randn(
        static_cast<size_t>(noise_elements),
        stream_seed_,
        stream_rng_offset_blocks_,
        rng_policy_,
        sampling::TorchRandnPrecision::Float32);
    stream_rng_offset_blocks_ += sampling::torch_cuda_tensor_iterator_offset_blocks(noise_elements, rng_policy_);
    if (stream_has_noise_cache_) {
        std::copy(stream_noise_cache_.begin(), stream_noise_cache_.end(), x.begin());
    }
    std::copy(
        x.end() - static_cast<std::ptrdiff_t>(kBlockFrames * kMelDim),
        x.end(),
        stream_noise_cache_.begin());
    stream_has_noise_cache_ = true;

    denoiser_->set_window_inputs(condition_frames, stream_speaker_embedding_, stream_speaker_memory_);
    sampler_runtime_->run_sequence(x);
    const auto & latent = sampler_runtime_->latent();
    return std::vector<float>(
        latent.begin(),
        latent.begin() + static_cast<std::ptrdiff_t>(kChunkFrames * kMelDim));
}

std::vector<float> MeanVC2FlowSamplerRuntime::synthesize_mel(
    const std::vector<float> & condition_frames,
    int64_t condition_frame_count,
    const std::vector<float> & speaker_embedding,
    const MeanVC2GtmMemory & speaker_memory,
    uint64_t seed) {
    if (condition_frame_count <= 0 ||
        condition_frames.size() != static_cast<size_t>(condition_frame_count * kSpeakerDim)) {
        throw std::runtime_error("MeanVC2 VC condition frame shape mismatch");
    }
    if (speaker_embedding.size() != static_cast<size_t>(kSpeakerDim)) {
        throw std::runtime_error("MeanVC2 VC speaker embedding shape mismatch");
    }
    start_streaming(speaker_embedding, speaker_memory, seed);
    std::vector<float> bn_buffer;
    std::vector<float> mel_frames;
    bn_buffer.reserve(static_cast<size_t>((condition_frame_count + kBlockFrames) * kSpeakerDim));
    mel_frames.reserve(static_cast<size_t>(condition_frame_count * kMelDim));

    const auto run_condition = [&](const float * condition_ptr) {
        std::vector<float> condition(
            condition_ptr,
            condition_ptr + static_cast<std::ptrdiff_t>(kConditionFrames * kSpeakerDim));
        const auto latent = synthesize_streaming_chunk(condition);
        mel_frames.insert(
            mel_frames.end(),
            latent.begin(),
            latent.end());
    };

    const auto append_condition_frames = [&](const float * frames, int64_t count) {
        bn_buffer.insert(
            bn_buffer.end(),
            frames,
            frames + static_cast<std::ptrdiff_t>(count * kSpeakerDim));
    };

    for (int64_t frame = 0; frame < condition_frame_count; frame += kConditionFrames) {
        const int64_t available = std::min<int64_t>(kConditionFrames, condition_frame_count - frame);
        append_condition_frames(
            condition_frames.data() + static_cast<std::ptrdiff_t>(frame * kSpeakerDim),
            available);
        int steps = 0;
        while (static_cast<int64_t>(bn_buffer.size() / static_cast<size_t>(kSpeakerDim)) >= kConditionFrames && steps < 4) {
            run_condition(bn_buffer.data());
            bn_buffer.erase(
                bn_buffer.begin(),
                bn_buffer.begin() + static_cast<std::ptrdiff_t>(kChunkFrames * kSpeakerDim));
            ++steps;
        }
    }

    while (static_cast<int64_t>(bn_buffer.size() / static_cast<size_t>(kSpeakerDim)) >= kChunkFrames) {
        std::vector<float> condition(static_cast<size_t>(kConditionFrames * kSpeakerDim), 0.0F);
        const int64_t buffered = static_cast<int64_t>(bn_buffer.size() / static_cast<size_t>(kSpeakerDim));
        const int64_t current = std::min<int64_t>(kChunkFrames, buffered);
        std::copy(
            bn_buffer.begin(),
            bn_buffer.begin() + static_cast<std::ptrdiff_t>(current * kSpeakerDim),
            condition.begin());
        int64_t future = std::min<int64_t>(kBlockFrames, std::max<int64_t>(0, buffered - kChunkFrames));
        if (future > 0) {
            std::copy(
                bn_buffer.begin() + static_cast<std::ptrdiff_t>(kChunkFrames * kSpeakerDim),
                bn_buffer.begin() + static_cast<std::ptrdiff_t>((kChunkFrames + future) * kSpeakerDim),
                condition.begin() + static_cast<std::ptrdiff_t>(kChunkFrames * kSpeakerDim));
        }
        const float * repeat_frame = buffered > 0
            ? bn_buffer.data() + static_cast<std::ptrdiff_t>((std::min<int64_t>(buffered, kChunkFrames) - 1) * kSpeakerDim)
            : condition.data();
        for (int64_t frame = current + future; frame < kConditionFrames; ++frame) {
            std::copy(
                repeat_frame,
                repeat_frame + static_cast<std::ptrdiff_t>(kSpeakerDim),
                condition.begin() + static_cast<std::ptrdiff_t>(frame * kSpeakerDim));
        }
        run_condition(condition.data());
        const int64_t erase_frames = std::min<int64_t>(kChunkFrames, buffered);
        bn_buffer.erase(
            bn_buffer.begin(),
            bn_buffer.begin() + static_cast<std::ptrdiff_t>(erase_frames * kSpeakerDim));
    }
    if (!bn_buffer.empty()) {
        std::vector<float> condition(static_cast<size_t>(kConditionFrames * kSpeakerDim), 0.0F);
        const int64_t buffered = static_cast<int64_t>(bn_buffer.size() / static_cast<size_t>(kSpeakerDim));
        const float * repeat_frame = bn_buffer.data() + static_cast<std::ptrdiff_t>((buffered - 1) * kSpeakerDim);
        for (int64_t frame = 0; frame < kConditionFrames; ++frame) {
            std::copy(
                repeat_frame,
                repeat_frame + static_cast<std::ptrdiff_t>(kSpeakerDim),
                condition.begin() + static_cast<std::ptrdiff_t>(frame * kSpeakerDim));
        }
        std::copy(bn_buffer.begin(), bn_buffer.end(), condition.begin());
        run_condition(condition.data());
        bn_buffer.clear();
    }

    debug::timing_log_scalar("meanvc2.vc.mel_frames", static_cast<int64_t>(mel_frames.size() / static_cast<size_t>(kMelDim)));
    reset_streaming();
    return mel_frames;
}

}  // namespace engine::models::meanvc2
