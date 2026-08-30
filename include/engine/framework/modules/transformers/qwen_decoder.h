#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

#include <optional>
#include <vector>

struct ggml_cgraph;

namespace engine::modules {

// Attention lowerings for the Qwen decoder block.
//
//   ManualRepeat                 explicit QK^T -> scale/mask -> softmax -> V, with the
//                                grouped-query K/V heads materialised at full query width.
//                                Quadratic working set in sequence length; the numerical
//                                reference path, and the only one that does not require the
//                                backend to implement GGML_OP_FLASH_ATTN_EXT for this head size.
//   FlashGrouped                 ggml_flash_attn_ext, with the permuted K/V cache views
//                                copied into packed tensors (ggml_cont) beforehand.
//   FlashGroupedViewKV           the same ggml_flash_attn_ext call on the K/V views directly.
//                                The flash kernels address K/V through nb[1..3] on both the CPU
//                                and Metal backends, so this is the same arithmetic in the same
//                                order on the same values as FlashGrouped -- only the copy is
//                                gone. See tests/unittests/test_qwen_attention_modes.cpp.
//   ManualRepeatThenGroupedQuery per-head slice/matmul/softmax loop. No model selects it.
enum class QwenDecoderAttentionMode {
    ManualRepeat,
    FlashGrouped,
    FlashGroupedViewKV,
    ManualRepeatThenGroupedQuery,
};

enum class QwenDecoderStaticCacheUpdateMode {
    ScratchTail,
    DirectSetRows,
};

enum class QwenDecoderStaticCacheSetRowsMode {
    Exact,
    BackendViewOptimized,
};

enum class QwenDecoderQKVLayout {
    Separate,
    PackedQKV,
};

enum class QwenDecoderMLPMode {
    Exact,
    FusedSwiGLU,
    PackedGateUp,
};

enum class QwenDecoderPrefixAttentionMode {
    Exact,
    FlashWithPrefix,
};

enum class QwenDecoderPositionEncoding {
    Rotary,
    None,
};

struct QwenDecoderActivationCastPolicy {
    bool enabled = false;
    ggml_type type = GGML_TYPE_BF16;
    bool after_input_norm = false;
    bool after_qkv_projection = false;
    bool after_qk_norm = false;
    bool after_rope = false;
    bool after_static_cache_update = false;
    bool after_attention = false;
    bool after_attention_output = false;
    bool after_residual = false;
    bool after_ffn_norm = false;
    bool after_mlp_projection = false;
    bool after_mlp_silu = false;
    bool after_mlp_mul = false;
    bool after_output = false;
};

struct QwenDecoderAttentionPolicy {
    // prefill_mode stays on ManualRepeat deliberately. Switching prefill to flash is not a
    // pure copy elision the way static_mode is: it replaces a materialised softmax with a
    // fused one (different floating-point reassociation) *and* it introduces a hard backend
    // requirement -- ggml_flash_attn_ext supports only a fixed whitelist of head sizes on
    // Metal, and the decode/prefill graphs run on a single backend with no CPU fallback, so an
    // unsupported head_dim becomes a hard failure rather than a slowdown. Models whose head
    // size is known-good opt in; minimax_music3's depth decoder already restricts flash prefill
    // to CUDA/HIP for exactly this reason (src/community_models/minimax_music3/depth_decoder.cpp).
    QwenDecoderAttentionMode prefill_mode = QwenDecoderAttentionMode::ManualRepeat;
    // static_mode defaults to the zero-copy flash lowering. FlashGrouped issues the identical
    // ggml_flash_attn_ext call but copies the whole K/V cache per layer per token first
    // (~100 MB/token at 24 layers / 2048 steps / f32). Set ENGINE_QWEN_ATTENTION_CONT_KV=1 to
    // restore the copy at runtime without a rebuild, or select FlashGrouped explicitly.
    QwenDecoderAttentionMode static_mode = QwenDecoderAttentionMode::FlashGroupedViewKV;
    QwenDecoderPrefixAttentionMode prefix_mode = QwenDecoderPrefixAttentionMode::Exact;
    int64_t grouped_query_min_steps = 0;
};

struct QwenDecoderStaticCachePolicy {
    QwenDecoderStaticCacheUpdateMode update_mode = QwenDecoderStaticCacheUpdateMode::ScratchTail;
    QwenDecoderStaticCacheSetRowsMode set_rows_mode = QwenDecoderStaticCacheSetRowsMode::Exact;
};

struct QwenDecoderMLPPolicy {
    QwenDecoderMLPMode mode = QwenDecoderMLPMode::Exact;
};

struct QwenDecoderRuntimePolicy {
    QwenDecoderAttentionPolicy attention;
    QwenDecoderStaticCachePolicy static_cache;
    QwenDecoderMLPPolicy mlp;
};

struct QwenDecoderLayerConfig {
    int64_t hidden_size = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t intermediate_size = 0;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    int rope_type = GGML_ROPE_TYPE_NEOX;
    QwenDecoderPositionEncoding position_encoding = QwenDecoderPositionEncoding::Rotary;
    ggml_prec attention_precision = GGML_PREC_F32;
    ggml_prec projection_precision = GGML_PREC_DEFAULT;
    QwenDecoderQKVLayout qkv_layout = QwenDecoderQKVLayout::Separate;
    bool use_qk_norm = true;
    QwenDecoderActivationCastPolicy activation_cast;
    QwenDecoderRuntimePolicy runtime;
};

struct QwenMLPWeights {
    LinearWeights gate_proj;
    LinearWeights up_proj;
    std::optional<LinearWeights> gate_up_proj;
    LinearWeights down_proj;
};

struct QwenDecoderLayerWeights {
    NormWeights input_norm;
    AttentionWeights self_attention;
    NormWeights q_norm;
    NormWeights k_norm;
    NormWeights post_norm;
    QwenMLPWeights mlp;
    // Optional per-frequency RoPE divisors (head_dim / 2), used by Llama-3
    // scaling and compatible checkpoints.
    std::optional<core::TensorValue> rope_frequency_factors;
};

struct QwenDecoderLayerOutputs {
    core::TensorValue output;
    core::TensorValue key;
    core::TensorValue value;
};

class QwenDecoderLayerModule {
public:
    explicit QwenDecoderLayerModule(QwenDecoderLayerConfig config);

    const QwenDecoderLayerConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    QwenDecoderLayerOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const QwenDecoderLayerWeights & weights,
        const std::optional<core::TensorValue> & prefix_key = std::nullopt,
        const std::optional<core::TensorValue> & prefix_value = std::nullopt,
        const std::optional<core::TensorValue> & attention_mask = std::nullopt) const;

    QwenDecoderLayerOutputs build_with_static_cache_tail(
        core::ModuleBuildContext & ctx,
        ggml_cgraph * graph,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const QwenDecoderLayerWeights & weights,
        const core::TensorValue & cache_key,
        const core::TensorValue & cache_value,
        const std::optional<core::TensorValue> & cache_slot,
        const core::TensorValue & attention_mask) const;

    QwenDecoderLayerOutputs build_with_static_cache_tail_batched(
        core::ModuleBuildContext & ctx,
        ggml_cgraph * graph,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const QwenDecoderLayerWeights & weights,
        const core::TensorValue & cache_key,
        const core::TensorValue & cache_value,
        const core::TensorValue & cache_slot,
        const core::TensorValue & attention_mask) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    QwenDecoderLayerConfig config_;
};

struct QwenDecoderStackConfig {
    int64_t hidden_size = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t intermediate_size = 0;
    int64_t layers = 0;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    int rope_type = GGML_ROPE_TYPE_NEOX;
    QwenDecoderPositionEncoding position_encoding = QwenDecoderPositionEncoding::Rotary;
    ggml_prec attention_precision = GGML_PREC_F32;
    ggml_prec projection_precision = GGML_PREC_DEFAULT;
    QwenDecoderQKVLayout qkv_layout = QwenDecoderQKVLayout::Separate;
    bool use_qk_norm = true;
    QwenDecoderActivationCastPolicy activation_cast;
    QwenDecoderRuntimePolicy runtime;
};

QwenDecoderLayerConfig qwen_decoder_layer_config_from_stack(const QwenDecoderStackConfig & config);

struct QwenDecoderStackWeights {
    std::vector<QwenDecoderLayerWeights> layers;
};

struct QwenDecoderStackLayerState {
    std::optional<core::TensorValue> key;
    std::optional<core::TensorValue> value;
};

struct QwenDecoderStackState {
    std::vector<QwenDecoderStackLayerState> layers;
};

struct QwenDecoderStackOutputs {
    core::TensorValue output;
    QwenDecoderStackState state;
};

class QwenDecoderStackModule {
public:
    explicit QwenDecoderStackModule(QwenDecoderStackConfig config);

    const QwenDecoderStackConfig & config() const noexcept;

    QwenDecoderStackOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const QwenDecoderStackWeights & weights,
        const std::optional<QwenDecoderStackState> & prefix_state = std::nullopt,
        const std::optional<core::TensorValue> & attention_mask = std::nullopt) const;

private:
    QwenDecoderStackConfig config_;
};

}  // namespace engine::modules
