#pragma once

#include "engine/framework/modules/vocoders/hift_vocoder.h"
#include "engine/framework/neon/execution.h"

namespace engine::community_models::kitten_tts2 {

class NeonVocoder final : public modules::HiftNeuralDecoder {
public:
    explicit NeonVocoder(const modules::HiftVocoderWeights & weights);
    std::vector<float> run(const std::vector<float> & mel, int64_t frames,
        const std::vector<float> & source_stft, int64_t stft_frames, int threads) override;

private:
    neon::ExecutionPlan plan_;
    int input_channels_ = 0;
    int source_channels_ = 0;
};

} // namespace engine::community_models::kitten_tts2
