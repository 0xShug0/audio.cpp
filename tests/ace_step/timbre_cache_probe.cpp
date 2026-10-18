#include "engine/models/ace_step/condition_encoder.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace engine;
using namespace engine::models::ace_step;

int main(int argc, char ** argv) {
    try {
        if (argc < 4 || argc > 5) throw std::runtime_error("usage: ace_step_timbre_cache_probe GGUF DiT-variant cpu|cuda|vulkan [device]");
        AceStepModelSelection selection;
        selection.dit_model_path = argv[2];
        auto assets = load_ace_step_assets(argv[1], selection);
        core::BackendConfig backend;
        const std::string type = argv[3];
        if (type == "cuda") backend.type = core::BackendType::Cuda;
        else if (type == "vulkan") backend.type = core::BackendType::Vulkan;
        else if (type == "cpu") backend.type = core::BackendType::Cpu;
        else throw std::runtime_error("unknown backend: " + type);
        backend.device = argc == 5 ? std::stoi(argv[4]) : 0;
        backend.threads = 8;
        core::ExecutionContext execution(backend);
        auto weights = std::make_shared<AceStepDitWeightsRuntime>(assets, execution, assets::TensorStorageType::Q8_0);
        AceStepConditionEncoderRuntime cached(execution, assets, weights);
        AceStepConditionEncoderRuntime computed(execution, assets, weights);
        const auto & config = assets->config.encoder;
        AceStepTextConditioning text, lyrics;
        text.tokens = 16;
        lyrics.tokens = 11;
        text.hidden_size = lyrics.hidden_size = config.text_hidden_dim;
        text.values.resize(text.tokens * text.hidden_size);
        lyrics.values.resize(lyrics.tokens * lyrics.hidden_size);
        for (size_t i = 0; i < text.values.size(); ++i) text.values[i] = 0.02F * std::sin(float(i % 113));
        for (size_t i = 0; i < lyrics.values.size(); ++i) lyrics.values[i] = 0.02F * std::cos(float(i % 97));
        const int64_t frames[] = {4, 4, 4, 12, 12, 6, 6, 12, 4, 4};
        const int content[] = {1, 1, 2, 2, 2, 2, 2, 1, 1, 1};
        for (int test = 0; test < 10; ++test) {
            if (test == 4 || test == 6) {
                cached.release_runtime_graphs();
                computed.release_runtime_graphs();
            }
            std::vector<float> reference(frames[test] * config.timbre_hidden_dim);
            for (size_t i = 0; i < reference.size(); ++i) reference[i] = 0.03F * std::sin(float((i + content[test] * 31) % 127));
            auto actual = cached.encode(text, lyrics, reference, 1, frames[test], {0});
            auto duplicated = reference;
            duplicated.insert(duplicated.end(), reference.begin(), reference.end());
            // Two references bypass the one-entry memoization. Both execute
            // the same graph each time, exercising warm constants and views.
            auto expected = computed.encode(text, lyrics, duplicated, 2, frames[test], {0, 0});
            const auto second_timbre = lyrics.tokens + 1;
            expected.values.erase(expected.values.begin() + second_timbre * expected.hidden_size,
                                  expected.values.begin() + (second_timbre + 1) * expected.hidden_size);
            expected.attention_mask.erase(expected.attention_mask.begin() + second_timbre);
            --expected.tokens;
            if (actual.tokens != expected.tokens || actual.hidden_size != expected.hidden_size ||
                actual.attention_mask != expected.attention_mask || actual.values.size() != expected.values.size() ||
                std::memcmp(actual.values.data(), expected.values.data(), actual.values.size() * sizeof(float)) != 0) {
                throw std::runtime_error("cached/computed mismatch at case " + std::to_string(test));
            }
            bool rejected = false;
            try { (void) cached.encode(text, lyrics, reference, 1, frames[test], {1}); }
            catch (const std::runtime_error &) { rejected = true; }
            if (!rejected) throw std::runtime_error("cached request accepted invalid ordering");
            std::cout << "case=" << test << " frames=" << frames[test] << " exact=1 invalid_order_rejected=1\n";
        }
        std::cout << "PASS: changed references, repeats, resize, graph release, multi-reference warm execution\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
