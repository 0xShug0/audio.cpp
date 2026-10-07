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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
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

// How a request sizes the backbone's decode cache. Every step attends over the
// whole cache, and on some backends the logits change in their last bits with
// its length (ggml's CPU flash attention splits a long cache into one piece
// per thread), enough to flip a near-tie. So the length decides what a
// request's output can depend on, and each task picks its own.
enum class Lfm2DecodeCache {
    // ASR. Exactly the prompt plus the step budget, and a cache an earlier
    // Transcript request left is kept while it holds this one and is at most
    // twice its size: the sizing ASR has always had, so transcripts stay as
    // they were. A cache sized for Speech is never reused.
    Transcript,
    // TTS and S2S. The prompt plus the step budget rounded up to 256, whatever
    // ran before, so seeded speech does not depend on earlier requests.
    Speech,
};

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
    // generated audio frames back. With `cpu_repack` (lfm2_audio.cpu_repack),
    // the CPU multiplies the matrices with the kernels of ggml's CPU extra
    // buffer types (repacked, or AMX) where they take the weight's type and
    // shape.
    Lfm2BackboneRuntime(
        std::shared_ptr<const assets::TensorSource> source,
        const Lfm2BackboneConfig & config,
        core::ExecutionContext & execution,
        std::shared_ptr<const assets::TensorSource> audio_embedding = nullptr,
        int64_t codebooks = 0,
        int64_t audio_vocab_size = 0,
        bool cpu_repack = true);
    ~Lfm2BackboneRuntime();

    Lfm2BackboneRuntime(const Lfm2BackboneRuntime &) = delete;
    Lfm2BackboneRuntime & operator=(const Lfm2BackboneRuntime &) = delete;

    // Greedy text generation until a stop token or max_new_tokens: the ASR
    // decoder, so its cache is always sized as Lfm2DecodeCache::Transcript.
    Lfm2GenerationResult generate(
        const Lfm2Prompt & prompt,
        const Lfm2AudioEmbeddings & audio,
        const Lfm2GenerationOptions & options);

    // Prefills the prompt, leaving room for `max_steps` more steps, and
    // returns the text logits after it. `cache` is the task's sizing policy:
    // Transcript for a text decoder that transcribes, Speech for TTS and S2S.
    std::vector<float> start(
        const Lfm2Prompt & prompt, const Lfm2AudioEmbeddings & audio, int64_t max_steps, Lfm2DecodeCache cache);

    // One step after start(): a text token, or the codes of an audio frame
    // (the sum of their audio embeddings goes in).
    std::vector<float> step_text(int32_t token, Lfm2StepOutput output);
    std::vector<float> step_audio(const std::vector<int32_t> & codes, Lfm2StepOutput output);

    // The decode cache length of the last request, 0 before the first. Steps
    // attend over the whole cache, and on some backends the logits change in
    // their last bits with its length.
    [[nodiscard]] int64_t decode_cache_steps() const noexcept;

    // The weight buffers, by buffer type ("CPU", "CPU_REPACK", ...), and how
    // many tensors each holds.
    [[nodiscard]] std::vector<std::pair<std::string, size_t>> weight_buffers() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::lfm2_audio
