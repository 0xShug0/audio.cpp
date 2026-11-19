#pragma once

// granite_speech5 Conformer encoder: graph construction, weight
// binding, CPU/GPU execution, and stage-probe access.
//
// Reference: transformers 4.57.0 modeling_granite_speech5.py at pinned
// revision ec49ca6b (GraniteSpeechCTCEncoder + GraniteSpeechConformerBlock).
// Per block: pre-norm Macaron FFN (0.5 residual) -> Shaw relative attention
// with block-local scores + residual -> convolution (full time sequence,
// across attention block boundaries) + residual -> second FFN (0.5) ->
// post-layer norm. After layer index 7 (the 8th block):
// h += out_mid(softmax(out(h))) independently per real frame, reusing the
// same out/out_mid projections. RAW block-16 states feed the projector.
//
// Layouts are time-major rows: input [steps, 160], all states [steps, 1024].
// Errors are std::runtime_error with family/component/file/expected-vs-actual
// detail. BN is NOT folded into the convolution weights: the
// inference-time affine (BatchNorm1dEvalModule) is evaluated as its own op
// from the checkpoint running statistics.

#include "engine/community_models/granite_speech5/assets.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::granite_speech5 {

// Model constants shared by the encoder implementation and its tests.
inline constexpr int64_t kGraniteSpeech5EncoderHidden = 1024;
inline constexpr int64_t kGraniteSpeech5EncoderInputDim = 160;
inline constexpr int64_t kGraniteSpeech5EncoderLayers = 16;
inline constexpr int64_t kGraniteSpeech5EncoderHeads = 8;
inline constexpr int64_t kGraniteSpeech5EncoderHeadDim = 128;
inline constexpr int64_t kGraniteSpeech5EncoderContext = 200;
inline constexpr int64_t kGraniteSpeech5EncoderFfnMult = 4;
inline constexpr int64_t kGraniteSpeech5EncoderConvExpansion = 2;
inline constexpr int64_t kGraniteSpeech5EncoderConvKernel = 15;
inline constexpr int64_t kGraniteSpeech5MidFeedbackAfter = 8;  // 1-based layer index
inline constexpr float kGraniteSpeech5EncoderLayerNormEps = 1e-5F;
inline constexpr float kGraniteSpeech5EncoderBatchNormEps = 1e-5F;  // nn.BatchNorm1d default
inline constexpr float kGraniteSpeech5MacaronScale = 0.5F;

// Block decomposition of T encoder steps for the block-local attention:
// ceil(T / 200) blocks; the final block keeps its real length (no zero pad is
// materialized: with relative offsets depending only on i - j, the valid
// r x r score submatrix is identical with or without padding, the mask over
// the missing keys is vacuous, and the padded query rows would be cropped).
struct GraniteSpeech5EncoderBlockPlan {
    int64_t num_blocks = 0;
    int64_t last_block_len = 0;
};

inline GraniteSpeech5EncoderBlockPlan granite_speech5_encoder_blocks_for_steps(int64_t steps) {
    if (steps <= 0) {
        throw std::runtime_error(
            "granite_speech5 encoder block plan requires a positive step count");
    }
    GraniteSpeech5EncoderBlockPlan plan;
    plan.num_blocks = (steps + kGraniteSpeech5EncoderContext - 1) / kGraniteSpeech5EncoderContext;
    plan.last_block_len = steps - (plan.num_blocks - 1) * kGraniteSpeech5EncoderContext;
    return plan;
}

// Probe stages through the projector cutoff. Names match reference.py
// PIPELINE dump names exactly; all tensors are time-major rows.
struct GraniteSpeech5EncoderOutputs {
    int64_t steps = 0;
    std::vector<float> input_proj;              // [T, 1024]
    std::vector<float> block1;                  // [T, 1024]
    std::vector<float> layer8_pre_feedback;     // [T, 1024]
    std::vector<float> layer8_post_feedback;    // [T, 1024]
    std::vector<float> last;                    // [T, 1024] (RAW block-16 states)
};

// No-weight shape gate: validates layer 0 first, then clones the checks
// across all 16 layers plus the shared projections. Throws std::runtime_error
// naming family/component/tensor/expected-vs-actual on any mismatch.
void check_granite_speech5_encoder_weight_shapes(const assets::TensorSource & source);

class GraniteSpeech5EncoderRuntime {
public:
    GraniteSpeech5EncoderRuntime(
        std::shared_ptr<const GraniteSpeech5Assets> assets,
        core::ExecutionContext & execution,
        assets::TensorStorageType storage_type = assets::TensorStorageType::F32,
        size_t graph_arena_bytes = 1024ull * 1024ull * 1024ull);
    ~GraniteSpeech5EncoderRuntime();

    // stacked: time-major [steps, 160] rows (frontend output). steps must be
    // positive and match stacked.size() / 160.
    // Intermediate captures are only requested by the parity probe.
    GraniteSpeech5EncoderOutputs encode(
        const std::vector<float> & stacked, int64_t steps, bool capture_stages = false) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::granite_speech5
