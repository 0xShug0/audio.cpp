#pragma once

// LFM2 hybrid backbone of LFM2.5-Audio: gated short-conv blocks and GQA
// attention blocks, prefilled with text and audio embeddings and decoded
// greedily. The short-conv blocks keep a rolling conv state and the attention
// blocks a KV cache between decode steps.
//
// Reference: transformers Lfm2Model, used by liquid-audio's LFM2AudioModel.

#include "engine/community_models/lfm2_audio/assets.h"
#include "engine/community_models/lfm2_audio/audio_encoder.h"
#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::lfm2_audio {

struct Lfm2Prompt {
    std::vector<int32_t> input_ids;
    // Positions in input_ids taken by audio embeddings, in order. Their ids
    // are placeholders and never looked up.
    std::vector<int32_t> audio_positions;
};

struct Lfm2GenerationOptions {
    int64_t max_new_tokens = 0;
    std::vector<int32_t> stop_token_ids;
};

struct Lfm2GenerationResult {
    std::vector<int32_t> tokens;  // without the stop token
    std::vector<float> prefill_logits;
};

class Lfm2BackboneRuntime {
public:
    Lfm2BackboneRuntime(
        std::shared_ptr<const assets::TensorSource> source,
        const Lfm2BackboneConfig & config,
        core::ExecutionContext & execution);
    ~Lfm2BackboneRuntime();

    Lfm2BackboneRuntime(const Lfm2BackboneRuntime &) = delete;
    Lfm2BackboneRuntime & operator=(const Lfm2BackboneRuntime &) = delete;

    Lfm2GenerationResult generate(
        const Lfm2Prompt & prompt,
        const Lfm2AudioEmbeddings & audio,
        const Lfm2GenerationOptions & options);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::lfm2_audio
