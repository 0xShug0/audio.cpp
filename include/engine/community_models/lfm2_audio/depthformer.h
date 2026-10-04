#pragma once

// LFM2.5-Audio depthformer: from the backbone's hidden state it predicts the
// codes of one audio frame, one codebook after another. Step i reads slice i
// of depth_linear(hidden) plus the embedding of the code picked at step
// i - 1, and attends over the frame's earlier steps
// (LFM2AudioModel._sample_audio_frame, model/lfm2_audio.py).

#include "engine/community_models/lfm2_audio/assets.h"
#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace engine::community_models::lfm2_audio {

class Lfm2DepthformerRuntime {
public:
    // Gets codebook i's logits (audio_vocab_size of them; they may be
    // modified) and returns the code to keep.
    using PickCode = std::function<int32_t(int64_t codebook, std::vector<float> & logits)>;

    Lfm2DepthformerRuntime(
        std::shared_ptr<const assets::TensorSource> vocoder,
        const Lfm2DepthformerConfig & config,
        core::ExecutionContext & execution);
    ~Lfm2DepthformerRuntime();

    Lfm2DepthformerRuntime(const Lfm2DepthformerRuntime &) = delete;
    Lfm2DepthformerRuntime & operator=(const Lfm2DepthformerRuntime &) = delete;

    // `hidden` is the backbone's final-norm output for the step.
    std::vector<int32_t> frame(const std::vector<float> & hidden, const PickCode & pick);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::lfm2_audio
