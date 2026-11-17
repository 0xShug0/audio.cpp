#pragma once

// granite_speech_2b frontend: PCM validation, NemoMel wiring, reference
// pointwise ops, explicit pairwise stacking, and the M/T/W/A length contract.
//
// Reference pipeline (transformers GraniteSpeechFeatureExtractor, pinned
// revision ec49ca6b): 16 kHz mono -> STFT (512 FFT / 400 Hann-periodic window
// / 160 hop, centered, reflect-padded) -> power spectrum -> HTK mel (80) ->
// log10 (1e-10 floor) -> per-clip max floor over ALL raw frames including a
// trailing odd frame, then (value / 4) + 1 -> pair adjacent frames into
// 160-wide rows, DROPPING the final row when the frame count is odd.
//
// The NemoMelFrontend DensePowerLog10 path (max_log_drop 8.0, scale 0.25,
// bias 1.0) already performs exactly the max-floor/scale/bias ops, so wiring
// it with stack_frames=1/no-deltas yields the reference mel directly. Pairwise
// stacking is done explicitly here -- never through the shared stack path,
// whose even-padding waveform extension would change the trailing frame.
//
// B1 -> B2 interface:
//   - granite_speech_2b_lengths_for_samples(n): pure checked arithmetic on the
//     post-resample sample count. Throws std::runtime_error for n <= 256
//     (empty n == 0 is NOT valid input here; use the predicate below).
//   - granite_speech_2b_clip_too_short(n): true for 0 < n <= 256. Task D's
//     chunk planner must merge such tails into a neighbour; the frontend
//     rejects them with a clear input error.
//   - GraniteSpeech2BFrontend::extract(audio): validates the interleaved
//     AudioBuffer (shape, positive rate/channel count, finite amplitudes),
//     downmixes/resamples to 16 kHz mono via the framework path, then returns
//     time-major mel [M, 80] and stacked [T, 160] rows. A zero-sample buffer
//     yields empty features (no error). Errors are std::runtime_error.

#include "engine/community_models/granite_speech_2b/assets.h"
#include "engine/framework/audio/nemo_mel_frontend.h"
#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace engine::community_models::granite_speech_2b {

// Mel frames, encoder steps, projector windows, audio placeholders.
struct GraniteSpeech2BLengths {
    int64_t mel_frames = 0;
    int64_t encoder_steps = 0;
    int64_t projector_windows = 0;
    int64_t audio_placeholders = 0;
};

// Minimum post-resample samples accepted by the frontend. Clips at or below
// this length carry no full encoder step pair the reference would keep.
inline constexpr int64_t kGraniteSpeech2BMinSamples = 256;
inline constexpr int64_t kGraniteSpeech2BStackWidth = 160;
inline constexpr int64_t kGraniteSpeech2BProjectorWindow = 15;
inline constexpr int64_t kGraniteSpeech2BQueriesPerWindow = 3;

// True for nonempty clips Task D must merge into a neighbour chunk.
inline bool granite_speech_2b_clip_too_short(int64_t post_resample_samples) {
    return post_resample_samples > 0 && post_resample_samples <= kGraniteSpeech2BMinSamples;
}

// M = floor(N/160)+1 (raw mel frames); T = floor(M/2) (encoder steps, width
// 160); W = ceil(T/15) (projector windows); A = 3*W (audio placeholders).
inline GraniteSpeech2BLengths granite_speech_2b_lengths_for_samples(int64_t post_resample_samples) {
    if (post_resample_samples <= 0) {
        throw std::runtime_error("granite_speech_2b lengths require a positive sample count");
    }
    if (granite_speech_2b_clip_too_short(post_resample_samples)) {
        throw std::runtime_error(
            "granite_speech_2b clip too short after resample; Task D must merge tails of <=256 samples");
    }
    GraniteSpeech2BLengths lengths;
    lengths.mel_frames = post_resample_samples / kGraniteSpeech2BStackWidth + 1;
    lengths.encoder_steps = lengths.mel_frames / 2;
    if (lengths.encoder_steps <= 0) {
        throw std::runtime_error("granite_speech_2b lengths produced no encoder step");
    }
    lengths.projector_windows =
        (lengths.encoder_steps + kGraniteSpeech2BProjectorWindow - 1) / kGraniteSpeech2BProjectorWindow;
    lengths.audio_placeholders = kGraniteSpeech2BQueriesPerWindow * lengths.projector_windows;
    return lengths;
}

struct GraniteSpeech2BFrontendFeatures {
    // Time-major mel rows: mel_frames x 80, reference `mel` stage.
    std::vector<float> mel;
    int64_t mel_frames = 0;
    // Time-major stacked rows: encoder_steps x 160, reference `stacked` stage.
    std::vector<float> stacked;
    int64_t encoder_steps = 0;
    GraniteSpeech2BLengths lengths;
    int64_t post_resample_samples = 0;
};

class GraniteSpeech2BFrontend {
public:
    explicit GraniteSpeech2BFrontend(std::shared_ptr<const GraniteSpeech2BAssets> assets);

    GraniteSpeech2BFrontendFeatures extract(const runtime::AudioBuffer & audio) const;

private:
    audio::NemoMelFrontend frontend_;
};

}  // namespace engine::community_models::granite_speech_2b
