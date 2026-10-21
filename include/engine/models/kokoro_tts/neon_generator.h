#pragma once

#include "engine/framework/neon/execution.h"
#include "engine/models/kokoro_tts/assets.h"

namespace kokoro_ggml {

// Model-local weight binding and generator topology. Execution uses only NEON
// tensor inputs; harmonic conditioning and iSTFT remain separate components.
class KokoroNeonGenerator {
public:
    KokoroNeonGenerator(const KokoroWeights::GeneratorWeights &weights, int style_channels);
    const engine::neon::Tensor &run(const std::vector<float> &latent, int frames,
        const std::vector<float> &conditioning, int conditioning_frames,
        const std::vector<float> &style, int threads);

private:
    engine::neon::ExecutionPlan plan_;
    int style_channels_;
};

} // namespace kokoro_ggml
