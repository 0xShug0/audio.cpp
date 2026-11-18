#pragma once

// granite_speech_2b Conformer encoder (Task B2): graph construction, weight
// binding, CPU/GPU execution, and stage-probe access.
//
// Reference: transformers 4.57.0 modeling_granite_speech.py at pinned
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
// detail (B1 pattern). BN is NOT folded into the convolution weights: the
// inference-time affine (BatchNorm1dEvalModule) is evaluated as its own op
// from the checkpoint running statistics.

#include "engine/community_models/granite_speech_2b/assets.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::granite_speech_2b {

// Model constants shared by the encoder implementation and its tests.
inline constexpr int64_t kGraniteSpeech2BEncoderHidden = 1024;
inline constexpr int64_t kGraniteSpeech2BEncoderInputDim = 160;
inline constexpr int64_t kGraniteSpeech2BEncoderLayers = 16;
inline constexpr int64_t kGraniteSpeech2BEncoderHeads = 8;
inline constexpr int64_t kGraniteSpeech2BEncoderHeadDim = 128;
inline constexpr int64_t kGraniteSpeech2BEncoderContext = 200;
inline constexpr int64_t kGraniteSpeech2BEncoderFfnMult = 4;
inline constexpr int64_t kGraniteSpeech2BEncoderConvExpansion = 2;
inline constexpr int64_t kGraniteSpeech2BEncoderConvKernel = 15;
inline constexpr int64_t kGraniteSpeech2BMidFeedbackAfter = 8;  // 1-based layer index
inline constexpr float kGraniteSpeech2BEncoderLayerNormEps = 1e-5F;
inline constexpr float kGraniteSpeech2BEncoderBatchNormEps = 1e-5F;  // nn.BatchNorm1d default
inline constexpr float kGraniteSpeech2BMacaronScale = 0.5F;

// Block decomposition of T encoder steps for the block-local attention:
// ceil(T / 200) blocks; the final block keeps its real length (no zero pad is
// materialized: with relative offsets depending only on i - j, the valid
// r x r score submatrix is identical with or without padding, the mask over
// the missing keys is vacuous, and the padded query rows would be cropped).
struct GraniteSpeech2BEncoderBlockPlan {
    int64_t num_blocks = 0;
    int64_t last_block_len = 0;
};

inline GraniteSpeech2BEncoderBlockPlan granite_speech_2b_encoder_blocks_for_steps(int64_t steps) {
    if (steps <= 0) {
        throw std::runtime_error(
            "granite_speech_2b encoder block plan requires a positive step count");
    }
    GraniteSpeech2BEncoderBlockPlan plan;
    plan.num_blocks = (steps + kGraniteSpeech2BEncoderContext - 1) / kGraniteSpeech2BEncoderContext;
    plan.last_block_len = steps - (plan.num_blocks - 1) * kGraniteSpeech2BEncoderContext;
    return plan;
}

// Probe stages through the projector cutoff. Names match reference.py
// PIPELINE dump names exactly; all tensors are time-major rows.
struct GraniteSpeech2BEncoderOutputs {
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
void check_granite_speech_2b_encoder_weight_shapes(const assets::TensorSource & source);

class GraniteSpeech2BEncoderRuntime {
public:
    GraniteSpeech2BEncoderRuntime(
        std::shared_ptr<const GraniteSpeech2BAssets> assets,
        core::ExecutionContext & execution,
        assets::TensorStorageType storage_type = assets::TensorStorageType::F32,
        size_t graph_arena_bytes = 1024ull * 1024ull * 1024ull);
    ~GraniteSpeech2BEncoderRuntime();

    // stacked: time-major [steps, 160] rows (frontend output). steps must be
    // positive and match stacked.size() / 160.
    GraniteSpeech2BEncoderOutputs encode(const std::vector<float> & stacked, int64_t steps) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::granite_speech_2b
