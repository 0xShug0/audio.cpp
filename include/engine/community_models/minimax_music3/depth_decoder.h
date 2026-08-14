#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/community_models/minimax_music3/assets.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::minimax_music3 {

// The local language model: per frame it autoregressively samples the seven residual RVQ
// codebooks from the global LM's last hidden state and the frame's semantic code, and
// returns the per-step hidden states that condition the flow-matching stage, plus the
// frame feedback embedding for the global LM.
//
// The seven codebook steps run as one unrolled graph with on-device sampling: each step
// applies classifier-free guidance, keeps the top-k logits, adds caller-provided Gumbel
// noise, and takes the argmax, which draws exactly from the reference's renormalized
// top-k distribution. Zero noise reduces to greedy decoding (used by the parity probe).
class MiniMaxMusic3DepthDecoderRuntime final {
public:
    struct FrameOutput {
        std::array<int32_t, 8> codes{};          // semantic + 7 residual codes
        std::vector<float> depth_hidden;         // [7 * hidden], conditional row only
        std::vector<float> feedback_embedding;   // [hidden], scaled frame embedding
    };

    MiniMaxMusic3DepthDecoderRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const assets::TensorSource> source,
        core::TensorValue lm_token_embedding,
        const MiniMaxMusic3Config & config,
        size_t weight_context_bytes);
    ~MiniMaxMusic3DepthDecoderRuntime();

    // last_hidden: [2 * hidden] (conditional row then unconditional row).
    // gumbel_noise: [(codebooks - 1) * audio_vocab] Gumbel(0, 1) samples, or empty for
    // greedy decoding.
    FrameOutput decode_frame(
        const std::vector<float> & last_hidden,
        int32_t semantic_code,
        const std::vector<float> & gumbel_noise);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::minimax_music3
