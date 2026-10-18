#pragma once

#include "engine/community_models/sopro_tts/acoustic.h"
#include "engine/community_models/sopro_tts/reference.h"
#include "engine/community_models/sopro_tts/vocoder.h"

#include <cstdint>
#include <random>
#include <vector>

namespace engine::community_models::sopro_tts {

// sopro/streaming.py. The reference prompt is trimmed to a whole number of
// semantic tokens so the canvas keeps exactly hop_ratio mel frames per token
// as it grows, and its complete chunks are solved once per voice; every text
// segment then streams from a copy of that state.
struct SoproPromptState {
    int64_t prompt_frames = 0;  // prompt mel frames after the token alignment
    std::vector<int32_t> semantic_tokens;
    std::vector<float> mel;     // [n_mels, prompt_frames], normalised
    std::vector<float> noise;   // solver noise of the prompt span, [n_mels, prompt_frames]
    SoproChunkedSolveState state;
};

SoproPromptState build_prompt_state(
    const SoproAcousticRuntime & acoustic,
    const SoproReference & reference,
    int64_t steps,
    int64_t chunk_frames,
    int64_t hop_ratio,
    int64_t lookahead_tokens,
    std::mt19937_64 & rng);

// StreamSession: one text segment. push() adds semantic tokens and returns the
// audio whose frames can no longer change (a lookahead margin behind the last
// token, rounded down to whole chunks); finish() solves the rest and flushes
// the vocoder.
class SoproStreamSession final {
public:
    SoproStreamSession(
        const SoproAcousticRuntime & acoustic,
        const SoproVocosRuntime & vocoder,
        const SoproReference & reference,
        const SoproPromptState & prompt,
        const std::vector<float> & mel_mean,
        const std::vector<float> & mel_std,
        int64_t steps,
        int64_t chunk_frames,
        int64_t hop_ratio,
        int64_t lookahead_tokens,
        int64_t warmup_frames,
        std::mt19937_64 & rng);

    std::vector<float> push(const std::vector<int32_t> & tokens);
    std::vector<float> finish();
    const std::vector<int32_t> & tokens() const noexcept { return tokens_; }

private:
    void extend_noise(int64_t frames);
    std::vector<float> render(int64_t keep_end);
    std::vector<float> feed(const std::vector<float> & mel, bool flush);

    const SoproAcousticRuntime & acoustic_;
    const std::vector<float> & mel_mean_;
    const std::vector<float> & mel_std_;
    SoproVocoderStream vocoder_;
    std::mt19937_64 & rng_;
    SoproChunkedSolveState state_;
    SoproChunkedRequest canvas_;
    std::vector<int32_t> tokens_;
    int64_t n_mels_ = 0;
    int64_t hop_ratio_ = 0;
    int64_t finality_margin_ = 0;
    int64_t kept_ = 0;
    int64_t skip_samples_ = 0;
};

}  // namespace engine::community_models::sopro_tts
