#pragma once

#include "engine/community_models/moss_voicegen/assets.h"
#include "engine/community_models/moss_voicegen/heads.h"

#include <cstdint>
#include <random>
#include <vector>

namespace engine::models::moss_voicegen {

// Defaults from MossTTSDelayModel.generate. The model card warns that this family is
// sensitive to them, and at a generic TTS preset it collapses into an immediate
// end-of-speech, so these travel with the checkpoint rather than with the caller.
struct MossVoiceGenSamplingOptions {
    float text_temperature = 1.5F;
    float text_top_p = 1.0F;
    int text_top_k = 50;
    float audio_temperature = 1.5F;
    float audio_top_p = 0.6F;
    int audio_top_k = 50;
    float audio_repetition_penalty = 1.1F;
    // Greedy decoding, used by the parity tests: it removes the RNG from the comparison so
    // any divergence is a real divergence.
    bool do_sample = true;
};

// MOSS-VoiceGenerator has no reference recording to anchor duration against, so left
// unbounded it either retires the codebooks on the first frame or keeps talking well past
// the text. These bounds gate the two decisions the model would otherwise make freely:
// starting the flush, and ending the turn. They do not touch pauses inside an utterance.
struct MossVoiceGenLengthBounds {
    int64_t min_frames = 0;  // 0 disables the floor
    int64_t max_frames = 0;  // 0 disables the ceiling
};

struct MossVoiceGenDelayRow {
    int32_t text_token = 0;
    std::vector<int32_t> codes;  // n_vq entries, audio_pad_code where nothing was sampled
};

// The delay-pattern state machine. Codebook i is delayed by i steps: it stays padded until
// the audio has been running for more than i steps, and after the text ends there is an
// n_vq-step flush window in which the codebooks retire one by one. Ported from
// MossTTSDelayModel.generate with batch size one.
class MossVoiceGenDelayDecoder {
public:
    MossVoiceGenDelayDecoder(
        MossVoiceGenConfig config,
        MossVoiceGenSamplingOptions sampling,
        uint32_t seed,
        MossVoiceGenLengthBounds bounds = {});

    // Consumes one step's logits (modified in place while masking) and returns the row to
    // feed back into the model.
    MossVoiceGenDelayRow step(MossVoiceGenStepLogits & logits);

    bool stopped() const noexcept { return stopped_; }
    int64_t steps() const noexcept { return step_index_; }

    // Strips the delay pattern and the padding rows, yielding [n_vq, frames] row-major.
    std::vector<int32_t> extract_audio_codes(int64_t & codebooks_out, int64_t & frames_out) const;

private:
    int32_t sample_text(std::vector<float> & logits);
    int32_t sample_code(std::vector<float> & logits, int64_t codebook);

    MossVoiceGenConfig config_;
    MossVoiceGenSamplingOptions sampling_;
    MossVoiceGenLengthBounds bounds_;
    uint32_t seed_ = 0;
    std::mt19937 rng_;
    uint64_t sample_call_index_ = 0;

    int64_t step_index_ = 0;
    bool stopped_ = false;
    bool in_audio_ = false;
    int64_t audio_length_ = 0;
    // Counts down the flush window once the text side has finished. The sentinel means
    // "not flushing"; the reference uses INT64_MAX for the same purpose.
    static constexpr int64_t kNotDelaying = -1;
    int64_t delayed_length_ = kNotDelaying;

    std::vector<MossVoiceGenDelayRow> history_;
};

}  // namespace engine::models::moss_voicegen
