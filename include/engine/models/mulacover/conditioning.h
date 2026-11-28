#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/modules/transformers/decoder.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/mulacover/symbolic.h"

#include <array>
#include <memory>
#include <optional>
#include <vector>
#include <string>

namespace engine::models::mulacover {

struct SymbolicAdaptorWeights {
    modules::LinearWeights pianoroll_proj, drum_pianoroll_proj, chord_proj, fusion_proj;
    std::array<modules::DecoderLayerWeights, 2> decoder;
    modules::NormWeights decoder_norm;
    modules::LinearWeights out_proj;
};

struct CrossAttentionRoPEWeights {
    modules::LinearWeights q_proj, k_proj, v_proj, output_proj;
};

struct CrossAttentionBlockWeights {
    modules::NormWeights norm;
    CrossAttentionRoPEWeights attn, gate;
    modules::LinearWeights x_gate_proj, context_gate_proj, gate_proj;
};

struct ConditioningLayerWeights {
    SymbolicAdaptorWeights adaptor;
    CrossAttentionBlockWeights cross_attn;
    modules::LinearWeights style_mlp;
};

struct ConditioningWeights {
    std::array<ConditioningLayerWeights, 8> layers;
    core::TensorValue adaptor_rope_factors, attention_rope_factors, gate_rope_factors;

    // Source names are upstream names under an optional model prefix (including
    // its trailing dot when nonempty). PEFT/style indices are backbone indices (20..27 for the upstream 28-layer backbone).
    static ConditioningWeights bind(
        core::BackendWeightStore & store, const assets::TensorSource & source,
        int64_t first_peft_layer = 20, int64_t embed_dim = 3072,
        assets::TensorStorageType storage_type = assets::TensorStorageType::Native,
        const std::string & prefix = "");
};

struct ProjectedCrossAttention {
    // [batch, heads, context_steps, head_dim], key already rotated.
    // Projected K/V use the runtime cache type; style affine vectors stay F32.
    core::TensorValue key, value;
};

struct PreparedConditioningLayer {
    core::TensorValue style_scale, style_shift;
    ProjectedCrossAttention attention, gate;
};

struct PreparedConditioning {
    std::array<PreparedConditioningLayer, 8> layers;
};

class SymbolicAdaptorModule {
public:
    explicit SymbolicAdaptorModule(int64_t embed_dim = 3072, ggml_type compute_type = GGML_TYPE_F32)
        : embed_dim_(embed_dim), compute_type_(compute_type) {}
    // Features [batch, steps, 256/128/12]; positions I32 [steps], 0..S-1.
    // Caller enforces upstream context limit 5000; query positions are <8192.
    // Required additive F32/F16 mask [batch or 1, 1, steps, steps]:
    // 0 valid, -inf padding. Shared batch masks avoid redundant allocation.
    // It masks keys only, and must contain no causal triangle.
    core::TensorValue build(core::ModuleBuildContext & ctx,
        const core::TensorValue & pianoroll, const core::TensorValue & drum_pianoroll,
        const core::TensorValue & chord, const core::TensorValue & positions,
        const core::TensorValue & attention_mask, const SymbolicAdaptorWeights & weights) const;
private:
    int64_t embed_dim_;
    ggml_type compute_type_;
};

class CrossAttentionRoPEModule {
public:
    CrossAttentionRoPEModule(int64_t embed_dim, int64_t heads, ggml_type compute_type = GGML_TYPE_F32);
    ProjectedCrossAttention build_context(core::ModuleBuildContext & ctx,
        const core::TensorValue & context, const core::TensorValue & positions,
        const core::TensorValue & rope_factors, const CrossAttentionRoPEWeights & weights) const;
    core::TensorValue build(core::ModuleBuildContext & ctx,
        const core::TensorValue & query, const core::TensorValue & positions,
        const std::optional<core::TensorValue> & attention_mask, const ProjectedCrossAttention & context,
        const core::TensorValue & rope_factors, const CrossAttentionRoPEWeights & weights) const;
private:
    int64_t embed_dim_, heads_;
    ggml_type compute_type_;
};

class ConditioningModule {
public:
    explicit ConditioningModule(int64_t embed_dim = 3072, int64_t style_dim = 1024, ggml_type compute_type = GGML_TYPE_F32);
    // Build once per request in a caller-owned, session-cached graph. Caller
    // retains output storage and passes detached cached tensor inputs to the
    // query graph; feeding these graph nodes directly would recompute them.
    // Style [batch, 1, style_dim] is already CFG-masked by the caller. Apply
    // style affine before backbone only when the model train_tag flag is true.
    PreparedConditioning build_context(core::ModuleBuildContext & ctx,
        const core::TensorValue & pianoroll, const core::TensorValue & drum_pianoroll,
        const core::TensorValue & chord, const core::TensorValue & context_positions,
        const core::TensorValue & adaptor_mask, const core::TensorValue & style,
        const ConditioningWeights & weights) const;
    // Query [batch, query_steps, embed_dim]; positions I32 [query_steps],
    // shared across batch. Apply this block AFTER the backbone layer.
    // Optional additive F32/F16 mask [batch or 1, 1, query_steps, context_steps].
    // nullopt means all keys valid and NONCAUSAL attention (including the gate).
    // Returns CrossAttentionBlock output; style affine is exposed separately
    // because upstream applies it before the corresponding backbone layer.
    core::TensorValue build_cross_attention(core::ModuleBuildContext & ctx,
        const core::TensorValue & query, const core::TensorValue & positions,
        const std::optional<core::TensorValue> & attention_mask, const PreparedConditioningLayer & prepared,
        const CrossAttentionBlockWeights & weights, const core::TensorValue & attention_rope_factors,
        const core::TensorValue & gate_rope_factors) const;
private:
    int64_t embed_dim_, style_dim_;
    ggml_type compute_type_;
};


// Session-owned; execution/backend must outlive this adapter and every backbone
// graph referring to it. prepare and backbone execution must be serialized.
// Separate CUDA backend instances must select the same physical device;
// prepare synchronizes its backend before detached outputs are consumed.
class ConditioningRuntime final {
public:
    ConditioningRuntime(core::ExecutionContext & execution,
        std::shared_ptr<const assets::TensorSource> source, int64_t first_peft_layer,
        int64_t style_dim, bool train_tag,
        assets::TensorStorageType storage_type = assets::TensorStorageType::Native,
        ggml_type cache_type = GGML_TYPE_F16);
    ~ConditioningRuntime();
    ConditioningRuntime(const ConditioningRuntime &) = delete;
    ConditioningRuntime & operator=(const ConditioningRuntime &) = delete;

    // One conditional example; qwen_embedding is raw Qwen style (1024 values
    // for the released checkpoint). CFG adds an unconditional (-1) row.
    // Reuses the precompute graph and detached output storage at the same shape.
    // True means shape/storage changed: clear backbone graph caches
    // before their next use. Revision changes only on successful shape changes.
    bool prepare(const SymbolicInputs & symbolic, const std::vector<float> & qwen_embedding,
        bool cfg_enabled);
    uint64_t revision() const noexcept;
    int64_t context_length() const noexcept;
    int64_t batch_size() const noexcept;
    const PreparedConditioning & prepared() const;

    core::TensorValue before_layer(core::ModuleBuildContext & ctx,
        const core::TensorValue & input, const core::TensorValue & positions,
        size_t layer_index);
    core::TensorValue after_layer(core::ModuleBuildContext & ctx,
        const core::TensorValue & output, const core::TensorValue & positions,
        size_t layer_index);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::mulacover
