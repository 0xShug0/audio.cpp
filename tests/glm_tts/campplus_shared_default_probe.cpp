#include "engine/framework/modules/speech_encoders/campplus_encoder.h"

#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char ** argv) try {
    if (argc != 2) {
        std::cerr
            << "usage: campplus_shared_default_probe "
               "<campplus.safetensors>\n";
        return 2;
    }

    constexpr int64_t kFrames = 250;
    constexpr int64_t kDims = 80;
    std::vector<float> features(
        static_cast<size_t>(kFrames * kDims));
    for (size_t index = 0; index < features.size(); ++index) {
        features[index] =
            std::sin(static_cast<float>(index + 1) * 0.013F) *
            0.5F;
    }

    const auto encoder =
        engine::modules::CampplusEncoderComponent::
            load_from_safetensors(
                argv[1],
                {engine::core::BackendType::Cpu, 0, 8});
    const auto output =
        encoder.embed_from_features(features, kFrames, kDims);

    std::cout << std::setprecision(9);
    std::cout << "{\"embedding\":[";
    for (size_t index = 0;
         index < output.embedding.size();
         ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        std::cout << output.embedding[index];
    }
    std::cout << "]}\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr
        << "campplus_shared_default_probe failed: "
        << error.what() << '\n';
    return 1;
}
