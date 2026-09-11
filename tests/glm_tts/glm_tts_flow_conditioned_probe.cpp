#include "engine/community_models/glm_tts/assets.h"
#include "engine/community_models/glm_tts/flow.h"

#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>

int main(int argc, char ** argv) try {
    if (argc < 2 || argc > 3) {
        std::cerr
            << "usage: glm_tts_flow_conditioned_probe <model-path> "
               "[steps]\n";
        return 2;
    }
    const int inference_steps = argc == 3 ? std::stoi(argv[2]) : 10;
    auto assets =
        engine::models::glm_tts::load_glm_tts_assets(argv[1]);
    const auto config = assets->config.flow;
    engine::models::glm_tts::GlmTTSFlowRuntime flow(
        assets->flow_weights,
        {engine::core::BackendType::Cpu, 0, 8},
        engine::assets::TensorStorageType::Native,
        config);

    engine::models::glm_tts::GlmTTSFlowInput input;
    input.speech_tokens = {
        29252, 4906, 833, 15564, 12971, 13808, 8457, 20137,
        7313, 30766, 5123, 2098, 891, 2464, 16935, 26908,
        20338, 286, 32542, 10092, 13792, 18489, 16200, 9895,
        3221, 29590, 5542, 29513, 25025, 9777, 29029, 21176,
        9781, 15068, 30891, 5324, 28699, 28591, 20197, 4599,
        11932, 4759, 19157, 10647, 4938, 8754, 30267, 31575,
        4731, 6326, 30991, 5040, 17687, 17687, 19635, 10780,
        21045, 5387, 11503, 13228, 13228, 6164, 12240,
        25568, 26211, 20216, 29473, 21992, 30971, 14699,
        25619, 20563, 20602, 22277, 9540, 25669, 21373,
        22055, 13228, 6387, 16809, 13228, 4503, 15102, 32001,
        31164, 26617, 6852, 32401, 27451, 5997, 5590, 18014,
        780, 26030, 17460, 15143, 18324, 5413, 23684, 28226,
        16826, 31440, 19490, 15302, 3693, 17870, 515, 32307,
        12991, 23343, 10209, 1118, 10442, 9769, 6294, 12860,
        10055, 3886, 7692,
    };
    input.prompt_frames = 126;
    input.prompt_mel.resize(
        static_cast<size_t>(input.prompt_frames * config.mel_dim));
    for (size_t index = 0; index < input.prompt_mel.size(); ++index) {
        input.prompt_mel[index] =
            std::sin(static_cast<float>(index + 1) * 0.011F) * 0.5F;
    }
    input.speaker_embedding.resize(192);
    for (size_t index = 0;
         index < input.speaker_embedding.size();
         ++index) {
        input.speaker_embedding[index] =
            std::sin(static_cast<float>(index + 1) * 0.03125F);
    }
    const int64_t frames = static_cast<int64_t>(
        static_cast<double>(input.speech_tokens.size()) /
        static_cast<double>(config.input_frame_rate) *
        static_cast<double>(config.mel_framerate));
    input.initial_noise.resize(
        static_cast<size_t>(frames * config.mel_dim));
    for (size_t index = 0; index < input.initial_noise.size(); ++index) {
        input.initial_noise[index] =
            std::sin(static_cast<float>(index + 1) * 0.017F);
    }
    input.inference_steps = inference_steps;
    input.cfg_rate = 0.7F;

    const auto output = flow.generate(input);
    std::cout << std::setprecision(9);
    std::cout << "{\"frames\":" << output.frames << ",\"values\":[";
    for (size_t index = 0; index < output.mel.size(); ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        std::cout << output.mel[index];
    }
    std::cout << "]}\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr
        << "glm_tts_flow_conditioned_probe failed: "
        << error.what() << '\n';
    return 1;
}
