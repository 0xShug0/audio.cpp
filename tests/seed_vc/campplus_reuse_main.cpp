#include "engine/models/seed_vc/assets.h"
#include "engine/framework/modules/speech_encoders/campplus_encoder.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) try {
    if (argc < 2 || argc > 4) {
        std::cerr << "usage: campplus_graph_reuse_probe <Seed-VC-model> [cpu|cuda|vulkan] [device]\n";
        return 2;
    }
    engine::core::BackendConfig backend{engine::core::BackendType::Cpu, 0, 8};
    const std::string name = argc > 2 ? argv[2] : "cpu";
    if (name == "cuda") backend.type = engine::core::BackendType::Cuda;
    else if (name == "vulkan") backend.type = engine::core::BackendType::Vulkan;
    else if (name != "cpu") throw std::runtime_error("unknown backend");
    if (argc > 3) backend.device = std::stoi(argv[3]);
    auto assets = engine::models::seed_vc::load_seed_vc_assets(argv[1]);
    auto encoder = engine::modules::CampplusEncoderComponent::load_from_tensor_source(
        assets->campplus_weights, backend);
    auto features = [](int frames, float phase) {
        std::vector<float> values(static_cast<size_t>(frames * 80));
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = 0.5F * std::sin(float(i + 1) * 0.013F + phase);
        return values;
    };
    const auto first = features(257, 0.0F);
    const auto changed = features(257, 0.7F);
    const auto shorter = features(143, 0.2F);
    const auto reference = encoder.embed_from_features(first, 257, 80).embedding;
    auto check = [&](const std::vector<float> & actual) {
        if (actual.empty() || actual.size() != reference.size() ||
            std::memcmp(actual.data(), reference.data(), reference.size() * sizeof(float)) != 0)
            throw std::runtime_error("cached CAMPPlus embedding changed");
        for (float value : actual)
            if (!std::isfinite(value)) throw std::runtime_error("nonfinite embedding");
    };
    for (int repeat = 0; repeat < 3; ++repeat) {
        check(encoder.embed_from_features(first, 257, 80).embedding);
        encoder.embed_from_features(changed, 257, 80);
        check(encoder.embed_from_features(first, 257, 80).embedding);
        encoder.embed_from_features(shorter, 143, 80);
        check(encoder.embed_from_features(first, 257, 80).embedding);
    }
    std::cout << "PASS repeated, changed-input and changed-length CAMPPlus graph reuse on " << name << '\n';
    return 0;
} catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
}
