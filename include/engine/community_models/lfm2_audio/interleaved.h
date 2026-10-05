#pragma once

// LFM2.5-Audio interleaved generation, the speech-to-speech chat mode of
// liquid-audio's README (LFM2AudioModel.generate_interleaved,
// model/lfm2_audio.py). The reply alternates a fixed number of text tokens
// with a fixed number of audio frames. After <|text_end|> only audio
// follows, until a frame opens with end-of-audio, and the reply ends at
// <|im_end|>.

#include "engine/community_models/lfm2_audio/audio_encoder.h"
#include "engine/community_models/lfm2_audio/backbone.h"
#include "engine/community_models/lfm2_audio/depthformer.h"
#include "engine/community_models/lfm2_audio/tokenizer.h"
#include "engine/community_models/lfm2_audio/tts.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace engine::community_models::lfm2_audio {

// The README's system prompt for interleaved generation, for both checkpoints.
inline constexpr const char * kLfm2ChatSystemPrompt = "Respond with interleaved text and audio.";

struct Lfm2InterleavedOptions {
    int64_t text_steps = 0;   // text tokens per block
    int64_t audio_steps = 0;  // audio frames per block
    // Text tokens and audio frames together, the reference's max_new_tokens.
    int64_t max_steps = 512;
    // The README samples audio at temperature 1.0 with top-k 4; text is greedy.
    Lfm2AudioSampling sampling{1.0f, 4, 0};
};

// One step of a reply: a text token, or the codes of an audio frame.
struct Lfm2ReplyStep {
    int32_t token = -1;          // -1 for an audio frame
    std::vector<int32_t> codes;  // empty for a text token
};

// The backbone and depthformer are borrowed and must outlive the generator;
// one generator runs at a time.
class Lfm2InterleavedGenerator {
public:
    Lfm2InterleavedGenerator(
        Lfm2BackboneRuntime & backbone,
        Lfm2DepthformerRuntime & depthformer,
        const Lfm2TextTokenizer & tokenizer,
        Lfm2Prompt prompt,
        Lfm2AudioEmbeddings audio,
        int32_t end_of_audio,
        const Lfm2InterleavedOptions & options);
    ~Lfm2InterleavedGenerator();

    Lfm2InterleavedGenerator(const Lfm2InterleavedGenerator &) = delete;
    Lfm2InterleavedGenerator & operator=(const Lfm2InterleavedGenerator &) = delete;

    // The next step, or nothing once the reply has ended or max_steps ran out
    // (ended() tells which). The first call runs the prompt. As in the
    // reference, the frame that ends the audio comes out too, with every code
    // set to end-of-audio; it has no sound.
    std::optional<Lfm2ReplyStep> next();

    [[nodiscard]] bool ended() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::lfm2_audio
