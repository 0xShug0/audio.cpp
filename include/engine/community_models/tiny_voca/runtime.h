#pragma once

#include "engine/community_models/tiny_voca/assets.h"
#include "engine/framework/core/backend.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace engine::models::tiny_voca {

struct TinyVocaGenerationOptions {
    int64_t voice = 0;                  // index into the model's voices
    float speed = 1.0F;                 // larger is faster
    float pitch_shift = 0.0F;           // added to log F0
    float pitch_range = 1.0F;           // scales pitch movement around the mean
    uint32_t seed = 0;
};

// The utterance needs more frames than the model takes at once: split the text.
struct TinyVocaTooLong : std::runtime_error {
    TinyVocaTooLong() : std::runtime_error("tiny-voca: the utterance is too long for one pass") {}
};

struct TinyVocaOutput {
    std::vector<float> samples;         // 24 kHz mono
    std::vector<int64_t> durations;     // frames for each token
    std::vector<float> f0;              // pitch of each frame in Hz
};

// Phoneme ids -> waveform. The three transformer stacks run as ggml graphs;
// the small steps between them (durations, frame expansion, pitch rules) and
// the source-filter waveform head run on the host.
class TinyVocaRuntime {
public:
    TinyVocaRuntime(std::shared_ptr<const TinyVocaAssets> assets, core::BackendConfig backend_config);
    ~TinyVocaRuntime();

    // `noise`, when given, is the white noise of the voice (one value for each
    // sample); otherwise it is drawn from `options.seed`.
    TinyVocaOutput synthesize(
        const std::vector<int32_t> & ids,
        const std::vector<int32_t> & stress,
        const TinyVocaGenerationOptions & options,
        const std::vector<float> * noise = nullptr);

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace engine::models::tiny_voca
