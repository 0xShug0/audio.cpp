#pragma once

// granite_speech5 Q-Former projector: weight binding, graph
// execution, and probe access.
//
// Reference: transformers 4.57.0 modeling_granite_speech5.py
// (GraniteSpeechEncoderProjector) over a Blip2QFormerModel with
// cross_attention_frequency=1, use_qformer_text_input=false,
// hidden_act="gelu" (exact erf), layer_norm_eps=1e-12.
//
// - Right-zero-pad encoder states (width 1024) to W*15 rows, reshape into
//   independent 15-row windows. The upstream passes NO encoder attention
//   mask, so the zero-padded rows PARTICIPATE in the final cross-attention
//   window exactly like real rows; the padding is confined to this stage.
// - The same 3 learned queries [1,3,1024] seed every window: query input
//   LayerNorm (eps 1e-12), then 2 layers of query self-attention,
//   cross-attention (3 queries x 15 acoustic rows, 16 heads x 64 dim, scale
//   1/8), and query-only GELU FFN (1024->4096->1024). All attention outputs
//   use dense + bias, post-residual LayerNorm placement (eps 1e-12).
// - Flatten window-major, query-major; final 1024->2048 projection + bias.
//   Output A=3W rows of width 2048.
//
// Errors are std::runtime_error with family/component/file/expected-vs-actual
// detail.

#include "engine/community_models/granite_speech5/assets.h"
#include "engine/community_models/granite_speech5/frontend.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::granite_speech5 {

inline constexpr int64_t kGraniteSpeech5ProjectorHidden = 1024;
// Window size, queries per window, and the length contract live in the frontend
// frontend header; reuse them here (no redefinition).
inline constexpr int64_t kGraniteSpeech5ProjectorLayers = 2;
inline constexpr int64_t kGraniteSpeech5ProjectorHeads = 16;
inline constexpr int64_t kGraniteSpeech5ProjectorHeadDim = 64;
inline constexpr int64_t kGraniteSpeech5ProjectorIntermediate = 4096;
inline constexpr int64_t kGraniteSpeech5ProjectorOutputDim = 2048;
inline constexpr float kGraniteSpeech5ProjectorLayerNormEps = 1e-12F;

// Window decomposition of T encoder steps for the projector: ceil(T / 15)
// windows, right-zero-padded to full windows.
struct GraniteSpeech5ProjectorPlan {
    int64_t windows = 0;
    int64_t padded_steps = 0;
    int64_t pad_steps = 0;
    int64_t audio_placeholders = 0;
};

inline GraniteSpeech5ProjectorPlan granite_speech5_projector_plan_for_steps(int64_t steps) {
    if (steps <= 0) {
        throw std::runtime_error(
            "granite_speech5 projector plan requires a positive step count");
    }
    GraniteSpeech5ProjectorPlan plan;
    plan.windows = (steps + kGraniteSpeech5ProjectorWindow - 1) / kGraniteSpeech5ProjectorWindow;
    plan.padded_steps = plan.windows * kGraniteSpeech5ProjectorWindow;
    plan.pad_steps = plan.padded_steps - steps;
    plan.audio_placeholders = kGraniteSpeech5QueriesPerWindow * plan.windows;
    return plan;
}

struct GraniteSpeech5ProjectorOutputs {
    int64_t audio_placeholders = 0;
    std::vector<float> projected;  // [A, 2048] window-major, query-major
};

// No-weight shape gate: validates the query, Q-Former input norm, both
// Q-Former layers, and the final projection. Throws std::runtime_error
// naming family/component/tensor/expected-vs-actual on any mismatch.
void check_granite_speech5_projector_weight_shapes(const assets::TensorSource & source);

class GraniteSpeech5ProjectorRuntime {
public:
    GraniteSpeech5ProjectorRuntime(
        std::shared_ptr<const GraniteSpeech5Assets> assets,
        core::ExecutionContext & execution,
        assets::TensorStorageType storage_type = assets::TensorStorageType::F32,
        size_t graph_arena_bytes = 1024ull * 1024ull * 1024ull);
    ~GraniteSpeech5ProjectorRuntime();

    // encoder_last: time-major [steps, 1024] rows (RAW block-16 states).
    // steps must be positive and match encoder_last.size() / 1024.
    GraniteSpeech5ProjectorOutputs project(const std::vector<float> & encoder_last, int64_t steps) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::granite_speech5
