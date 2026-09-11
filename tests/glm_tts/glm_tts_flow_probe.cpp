#include "engine/community_models/glm_tts/assets.h"
#include "engine/community_models/glm_tts/flow.h"

#include <cmath>
#include <exception>
#include <iostream>

int main(int argc, char ** argv) try {
    if (argc != 2) {
        std::cerr << "usage: glm_tts_flow_probe <model-path>\n";
        return 2;
    }
    auto assets = engine::models::glm_tts::load_glm_tts_assets(argv[1]);
    const auto config = assets->config.flow;
    engine::models::glm_tts::GlmTTSFlowRuntime flow(
        assets->flow_weights,
        {engine::core::BackendType::Cpu, 0, 8},
        engine::assets::TensorStorageType::Native,
        config);
    engine::models::glm_tts::GlmTTSFlowInput input;
    input.speech_tokens = {29252, 4906, 833, 15564, 12971};
    input.speaker_embedding.resize(192);
    for (size_t i = 0; i < input.speaker_embedding.size(); ++i) {
        input.speaker_embedding[i] =
            std::sin(static_cast<float>(i + 1) * 0.03125F);
    }
    const int64_t frames = static_cast<int64_t>(
        static_cast<double>(input.speech_tokens.size()) /
        static_cast<double>(config.input_frame_rate) *
        static_cast<double>(config.mel_framerate));
    input.initial_noise.resize(
        static_cast<size_t>(frames * config.mel_dim));
    for (size_t i = 0; i < input.initial_noise.size(); ++i) {
        input.initial_noise[i] =
            std::sin(static_cast<float>(i + 1) * 0.017F);
    }
    input.inference_steps = 1;
    const auto output = flow.generate(input);
    std::cout << "{\"frames\":" << output.frames << ",\"values\":[";
    const size_t count = std::min<size_t>(output.mel.size(), 16);
    for (size_t i = 0; i < count; ++i) {
        if (i != 0) {
            std::cout << ',';
        }
        std::cout << output.mel[i];
    }
    std::cout << "]}\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << "glm_tts_flow_probe failed: " << error.what() << '\n';
    return 1;
}
