#pragma once

// LFM2.5-Audio speech output: the TTS prompt from liquid-audio's README (one
// system prompt per built-in voice), sequential generation of audio frames
// (LFM2AudioModel.generate_sequential, model/lfm2_audio.py) and the sampling
// it uses.

#include "engine/community_models/lfm2_audio/backbone.h"
#include "engine/community_models/lfm2_audio/depthformer.h"
#include "engine/community_models/lfm2_audio/tokenizer.h"
#include "engine/framework/sampling/hf_sampler.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace engine::community_models::lfm2_audio {

// The voices of a checkpoint: us_male, us_female, uk_male and uk_female for
// "en"; the "ja" checkpoint speaks with one voice and lists none.
std::vector<std::string> lfm2_tts_voices(const std::string & language);

// "Perform TTS. Use the US male voice." and the like; "Perform TTS in
// japanese." for "ja". An empty voice means the first one. Throws on a voice
// the checkpoint does not have.
std::string lfm2_tts_system_prompt(const std::string & language, const std::string & voice);

// <|startoftext|><|im_start|>system\n{system prompt}<|im_end|>\n
// <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
Lfm2Prompt make_lfm2_tts_prompt(const Lfm2TextTokenizer & tokenizer, const std::string & system_prompt, const std::string & text);

// Audio codes are sampled with a temperature and top-k like
// LFM2AudioModel._sample_audio_frame: greedy when temperature <= 0 or
// top_k == 1. The README uses 0.8 and 64 for TTS.
struct Lfm2AudioSampling {
    float temperature = 0.8f;
    int64_t top_k = 64;
    uint64_t seed = 0;
};

// S2S text tokens, sampled like LFM2AudioModel._sample_text_token by the same
// rule as the audio codes. Greedy by default, as in liquid-audio's README and
// demo.
struct Lfm2TextSampling {
    float temperature = 0.0f;
    int64_t top_k = 0;
};

// Picks the codes of audio frames, or text tokens, with one random stream per
// sampler.
class Lfm2CodeSampler {
public:
    explicit Lfm2CodeSampler(const Lfm2AudioSampling & sampling);
    // Text draws come from a stream of their own, seeded from a fixed mix of
    // the request seed, so they leave the audio codes' draws as they were.
    Lfm2CodeSampler(const Lfm2TextSampling & sampling, uint64_t seed);

    // May change `logits`. Throws on non-finite logits, greedy or not.
    int32_t pick(std::vector<float> & logits);

private:
    Lfm2CodeSampler(float temperature, int64_t top_k, uint64_t seed, const char * context);

    bool greedy_;
    const char * context_;
    sampling::HfSamplingOptions options_;
    sampling::HfSampler sampler_;
    sampling::HfSamplerScratch scratch_;
    std::mt19937 rng_;
};

// The first of the largest logits, like torch.argmax. Throws on non-finite
// logits.
int32_t lfm2_greedy(const std::vector<float> & logits);

// Whether a frame has sound: no codebook picked end-of-audio. The frame that
// ends the audio has it first; liquid-audio's demo also skips a frame that
// picked it for another codebook, which the detokenizer has no code for.
bool lfm2_speaks(const std::vector<int32_t> & codes, int32_t end_of_audio);

struct Lfm2SpeechOptions {
    int64_t max_frames = 0;
    Lfm2AudioSampling sampling;
};

struct Lfm2Speech {
    std::vector<std::vector<int32_t>> frames;  // the frames that speak (lfm2_speaks)
    std::vector<int32_t> text_tokens;          // any text before <|audio_start|>
    bool ended = false;                        // false when max_frames ran out
};

// generate_sequential for speech, one audio frame at a time, which is how
// liquid-audio's demo streams: text tokens are greedy, as in the reference,
// until <|audio_start|>; then the depthformer picks a frame per step until a
// frame opens with the end-of-audio code. The backbone and depthformer are
// borrowed and must outlive the generator; one generator runs at a time.
class Lfm2SpeechGenerator {
public:
    Lfm2SpeechGenerator(
        Lfm2BackboneRuntime & backbone,
        Lfm2DepthformerRuntime & depthformer,
        const Lfm2TextTokenizer & tokenizer,
        Lfm2Prompt prompt,
        int32_t end_of_audio,
        const Lfm2SpeechOptions & options);
    ~Lfm2SpeechGenerator();

    Lfm2SpeechGenerator(const Lfm2SpeechGenerator &) = delete;
    Lfm2SpeechGenerator & operator=(const Lfm2SpeechGenerator &) = delete;

    // The next frame that speaks, or nothing once the speech has ended or
    // max_frames ran out (ended() tells which). A frame with end-of-audio for
    // another codebook still goes back into the backbone and counts against
    // max_frames, but is not returned. The first call runs the prompt, and
    // throws if the turn ends with no speech.
    std::optional<std::vector<int32_t>> next_frame();

    [[nodiscard]] bool ended() const;
    [[nodiscard]] const std::vector<int32_t> & text_tokens() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// All frames of one turn.
Lfm2Speech generate_lfm2_speech(
    Lfm2BackboneRuntime & backbone,
    Lfm2DepthformerRuntime & depthformer,
    const Lfm2TextTokenizer & tokenizer,
    const Lfm2Prompt & prompt,
    int32_t end_of_audio,
    const Lfm2SpeechOptions & options);

}  // namespace engine::community_models::lfm2_audio
