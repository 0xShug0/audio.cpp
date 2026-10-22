#pragma once

#include "engine/framework/audio/harmonic_vocoder.h"
#include "engine/framework/neon/execution.h"

namespace engine::modules {
struct HiftVocoderWeights;
}

namespace engine::community_models::kitten_tts2 {

class NeonVocoder {
public:
    explicit NeonVocoder(const modules::HiftVocoderWeights & weights);
    std::vector<float> synthesize(const std::vector<float> & mel, int64_t frames,
        const std::vector<float> & f0, int threads, uint64_t seed, uint64_t prior_noise_values);

private:
    neon::ExecutionPlan plan_;
    int input_channels_ = 0;
    int source_channels_ = 0;
    int64_t scale_ = 0;
    int64_t n_fft_ = 0;
    int64_t hop_ = 0;
    float audio_limit_ = 0.99F;
    audio::HarmonicSourceConfig source_config_;
    std::vector<float> source_weight_;
    float source_bias_ = 0.0F;
};

} // namespace engine::community_models::kitten_tts2
