#pragma once

// LFM2 hybrid backbone of LFM2.5-Audio: gated short-conv blocks and GQA
// attention blocks, prefilled with text and audio embeddings, then run one
// step at a time. The short-conv blocks keep a rolling conv state and the
// attention blocks a KV cache between steps.
//
// Reference: Lfm2Model in transformers 4.56 models/lfm2/modeling_lfm2.py,
// which liquid-audio's LFM2AudioModel (model/lfm2_audio.py) wraps.

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

// What a step returns: the text logits, or the final-norm hidden state that
// the depthformer turns into an audio frame (LFM2AudioModel.generate_sequential
// uses one or the other depending on the modality it is in).
enum class Lfm2StepOutput { Logits, Hidden };

struct Lfm2GenerationResult {
    std::vector<int32_t> tokens;  // without the stop token
    // False when max_new_tokens ran out before a stop token.
    bool stopped = false;
    std::vector<float> prefill_logits;
};

class Lfm2BackboneRuntime {
public:
    // With `audio_embedding` (the mmproj file, whose a.position_embd holds one
    // table of `audio_vocab_size` rows per codebook), steps can also feed
    // generated audio frames back.
    Lfm2BackboneRuntime(
        std::shared_ptr<const assets::TensorSource> source,
        const Lfm2BackboneConfig & config,
        core::ExecutionContext & execution,
        std::shared_ptr<const assets::TensorSource> audio_embedding = nullptr,
        int64_t codebooks = 0,
        int64_t audio_vocab_size = 0);
    ~Lfm2BackboneRuntime();

    Lfm2BackboneRuntime(const Lfm2BackboneRuntime &) = delete;
    Lfm2BackboneRuntime & operator=(const Lfm2BackboneRuntime &) = delete;

    // Greedy text generation until a stop token or max_new_tokens.
    Lfm2GenerationResult generate(
        const Lfm2Prompt & prompt,
        const Lfm2AudioEmbeddings & audio,
        const Lfm2GenerationOptions & options);

    // Prefills the prompt, leaving room for `max_steps` more steps, and
    // returns the text logits after it.
    std::vector<float> start(const Lfm2Prompt & prompt, const Lfm2AudioEmbeddings & audio, int64_t max_steps);

    // One step after start(): a text token, or the codes of an audio frame
    // (the sum of their audio embeddings goes in).
    std::vector<float> step_text(int32_t token, Lfm2StepOutput output);
    std::vector<float> step_audio(const std::vector<int32_t> & codes, Lfm2StepOutput output);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::lfm2_audio
