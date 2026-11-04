#pragma once

#include "engine/framework/modules/attention/grouped_query_attention.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/norm_modules.h"

#include "engine/framework/core/execution_context.h"

#include <cstddef>
#include <memory>
#include <cstdint>
#include <optional>
#include <vector>

namespace engine::modules {

enum class LfmDecoderLayerKind { ShortConv, FullAttention, SlidingAttention };

struct LfmDecoderRopeConfig {
    float theta = 1000000.0f;
    // -1 rotates the full head. Otherwise an even number in [0, head_dim].
    // Partial rotation retains the full-head frequency progression.
    int64_t rotary_dims = -1;
};

struct LfmDecoderConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t conv_kernel_size = 0;
    float norm_eps = 1e-5f;
    std::vector<LfmDecoderLayerKind> layers;
    bool attention_gate = true;
    bool sandwich_norm = true;
    bool conv_bias = false;
    bool final_norm = true;
    LfmDecoderRopeConfig full_rope;
    LfmDecoderRopeConfig sliding_rope;
    ggml_prec projection_precision = GGML_PREC_DEFAULT;
    ggml_prec attention_precision = GGML_PREC_DEFAULT;
    GroupedQueryAttentionLowering attention_lowering = GroupedQueryAttentionLowering::ManualRepeat;
    // Optional rounding before the RMSNorm weight multiplication, as in upstream.
    // This alone does not emulate all BF16 activation rounding.
    std::optional<ggml_type> normalized_round_type;
};

struct LfmDecoderAttentionWeights {
    LinearWeights q_proj, k_proj, v_proj, out_proj;
    NormWeights q_norm, k_norm;
    std::optional<LinearWeights> gate_proj; // [heads, hidden_size], one scalar gate per head
};

struct LfmDecoderConvWeights {
    LinearWeights in_proj, out_proj;
    core::TensorValue kernel; // F32 logical [hidden_size, 1, kernel_size], oldest tap first
    std::optional<core::TensorValue> bias; // F32 [hidden_size]
};

struct LfmDecoderLayerWeights {
    NormWeights operator_norm, ffn_norm;
    std::optional<NormWeights> post_operator_norm, post_ffn_norm;
    LfmDecoderAttentionWeights attention;
    LfmDecoderConvWeights conv;
    GatedFeedForwardWeights feed_forward;
};

struct LfmDecoderWeights {
    std::vector<LfmDecoderLayerWeights> layers;
    NormWeights final_norm;
    // Optional F32 [head_dim / 2] RoPE divisors, independently selected by layer kind.
    // When supplied, these define all frequency scaling/partial rotation and take
    // precedence over the corresponding rotary_dims. Theta still comes from config.
    std::optional<core::TensorValue> full_freq_factors, sliding_freq_factors;
};

struct LfmDecoderInputs {
    core::TensorValue hidden; // F32 [batch, steps, hidden_size]
    core::TensorValue positions; // I32 [steps], shared by the batch
    // Additive F32/F16 [query_steps, key_steps] masks. No implicit causal/window mask.
    // Each attention kind uses only its own mask; absence means unmasked attention.
    std::optional<core::TensorValue> full_mask, sliding_mask;
    // Optional F32 [batch, steps] keep mask for short-conv padding, applied both
    // before in_proj and to B*x after it (important when projections have bias).
    std::optional<core::TensorValue> conv_padding_mask;
    // Zero-based layer indices, after both residuals and before final_norm.
    // Captures are returned in this order, including repeated indices.
    std::vector<int64_t> capture_layers;
};

struct LfmDecoderOutputs {
    core::TensorValue hidden;
    std::vector<core::TensorValue> captured_hidden;
};

// Graph composition entry for encoders using the same hybrid stack. No allocation or state ownership.
LfmDecoderOutputs build_lfm_decoder(
    core::ModuleBuildContext & ctx, const LfmDecoderInputs & inputs,
    const LfmDecoderWeights & weights, const LfmDecoderConfig & config);

struct LfmDecoderRuntimeWeights {
    LfmDecoderWeights decoder;
    std::optional<core::TensorValue> token_embedding;
    std::optional<LinearWeights> output_projection;
};

enum class LfmDecoderReadout { Hidden, Logits };

struct LfmDecoderForward {
    // Exactly one input: token IDs, host hidden rows, or a borrowed device tensor.
    std::vector<int32_t> token_ids;
    std::vector<float> hidden;
    const core::TensorValue * hidden_tensor = nullptr;
    // Optional replacements of token-embedding rows, independent of their modality.
    std::vector<int32_t> replacement_positions;
    std::vector<float> replacement_values;
    const core::TensorValue * replacement_tensor = nullptr;
    std::vector<int32_t> positions;
    // Query-major additive masks; empty means unmasked. No mask is inferred.
    std::vector<ggml_fp16_t> full_mask, sliding_mask;
    // Empty selects every hidden row / every output-projection row respectively.
    std::vector<int32_t> readout_positions, output_tokens;
    LfmDecoderReadout readout = LfmDecoderReadout::Hidden;
};

struct LfmDecoderForwardResult {
    std::vector<float> values; // row-major [rows, columns]
    int64_t rows = 0;
    int64_t columns = 0;
};

struct LfmDecoderDeviceResult {
    core::TensorValue output; // borrowed [rows, columns], valid until next execution or release_workspace
    int64_t rows = 0;
    int64_t columns = 0;
};

// Resolved weights and execution remain caller-owned. The runtime owns only
// independent full-sequence execution; each DecodeSession owns a separate causal
// stream, persistent hybrid state, cached graphs and scratch allocation.
class LfmDecoderRuntime {
    struct Impl;
public:
    LfmDecoderRuntime(LfmDecoderConfig config, LfmDecoderRuntimeWeights weights,
                      core::ExecutionContext & execution, size_t graph_cache_slots = 4);
    ~LfmDecoderRuntime();
    LfmDecoderRuntime(const LfmDecoderRuntime &) = delete;
    LfmDecoderRuntime & operator=(const LfmDecoderRuntime &) = delete;

    // Explicit positions and masks, with no implicit causal or window policy.
    LfmDecoderForwardResult forward(const LfmDecoderForward & input);
    LfmDecoderDeviceResult forward_device(const LfmDecoderForward & input);
    void release_workspace();
    size_t graphs_built() const noexcept;

    class DecodeSession {
    public:
        ~DecodeSession();
        DecodeSession(const DecodeSession &) = delete;
        DecodeSession & operator=(const DecodeSession &) = delete;

        // Append real rows only. Positions and causal masks are generated from
        // this stream's cursor; custom positions/masks belong to forward().
        LfmDecoderForwardResult prefill(const LfmDecoderForward & input);
        LfmDecoderDeviceResult prefill_device(const LfmDecoderForward & input);
        LfmDecoderForwardResult decode_step(const LfmDecoderForward & input);
        // Explicit backend-only capacity growth preserves KV and short-conv state.
        void reserve(int64_t cache_steps);
        void reset();
        // Releases graph scratch without resetting the stream or its topology cache.
        void release_workspace();
        int64_t valid_steps() const noexcept;
        int64_t cache_steps() const noexcept;
        size_t graphs_built() const noexcept;

    private:
        friend class LfmDecoderRuntime;
        explicit DecodeSession(std::unique_ptr<Impl> impl);
        std::unique_ptr<Impl> impl_;
    };

    // Sliding layers require their architectural window length. Streams borrow
    // the same weights/context but own independent state; no hidden shared cursor.
    std::unique_ptr<DecodeSession> create_decode_session(int64_t cache_steps, int64_t sliding_window = 0) const;

private:
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::modules
