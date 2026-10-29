#include "engine/framework/conditioners/musiccoca_runtime.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void compare(const std::string & name, const std::vector<float> & actual, const std::vector<float> & expected) {
    if (actual.size() != expected.size()) throw std::runtime_error(name + " shape mismatch");
    double dot = 0, aa = 0, bb = 0;
    float error = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) throw std::runtime_error(name + " non-finite output");
        dot += static_cast<double>(actual[i]) * expected[i];
        aa += static_cast<double>(actual[i]) * actual[i];
        bb += static_cast<double>(expected[i]) * expected[i];
        error = std::max(error, std::abs(actual[i] - expected[i]));
    }
    const double cosine = dot / std::sqrt(aa * bb);
    engine::debug::trace_log_f32(name, {static_cast<int64_t>(actual.size())}, actual);
    std::cout << name << " cosine=" << cosine << " max_abs_error=" << error << '\n';
    if (cosine < 0.99999) throw std::runtime_error(name + " parity failed");
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 6 || std::string(argv[4]) != "--log") {
            throw std::runtime_error("Usage: magenta_musiccoca_parity MODEL.gguf REFERENCE.safetensors cpu|cuda|vulkan --log FILE");
        }
        engine::debug::configure_logging({true, argv[5]});
        const auto start = std::chrono::steady_clock::now();
        engine::core::BackendConfig config;
        const std::string backend = argv[3];
        if (backend == "cpu") config.type = engine::core::BackendType::Cpu;
        else if (backend == "cuda") config.type = engine::core::BackendType::Cuda;
        else if (backend == "vulkan") config.type = engine::core::BackendType::Vulkan;
        else throw std::runtime_error("Unknown backend: " + backend);
        config.threads = 8;
        engine::core::ExecutionContext execution(config);
        const auto resources = engine::model_spec::load_resource_bundle_for_family(argv[1], "magenta_realtime_2");
        engine::conditioners::MusicCoCaRuntime model(execution, resources.open_tensor_source("musiccoca"),
            resources.parse_json("musiccoca_config"), resources.require_file("musiccoca_tokenizer"));
        const auto reference = engine::assets::open_tensor_source(argv[2]);
        const std::string prompts[] = {"disco funk", "A gentle acoustic guitar with warm piano", "Driving electronic drums and deep bass"};
        for (int i : {0, 1, 2, 0}) {
            const auto suffix = std::to_string(i);
            const auto text = model.embed_text(prompts[i]);
            compare("text_" + suffix, text, reference->require_f32("text_" + suffix));
            const auto mapped = model.map_text(reference->require_f32("text_" + suffix), reference->require_f32("mapper_noise"));
            compare("mapped_" + suffix, mapped, reference->require_f32("mapped_" + suffix));
            compare("mapped_seed_" + suffix, model.map_text(reference->require_f32("text_" + suffix), 0), mapped);
            const auto codes = model.quantize(reference->require_f32("mapped_" + suffix));
            const auto raw = reference->require_tensor_data("codes_" + suffix);
            std::array<int32_t, 12> expected{};
            if (raw.metadata.dtype != "I32" || raw.bytes.size() != sizeof(expected))
                throw std::runtime_error("Invalid MusicCoCa code fixture");
            std::memcpy(expected.data(), raw.bytes.data(), sizeof(expected));
            for (size_t k = 0; k < codes.size(); ++k) {
                if (codes[k] != static_cast<int32_t>(expected[k])) throw std::runtime_error("MusicCoCa quantizer code mismatch");
            }
        }
        compare("audio_embedding", model.embed_features(reference->require_f32("audio_features")),
                reference->require_f32("audio_embedding"));
        compare("audio_features", model.preprocess_audio(reference->require_f32("audio_input")), reference->require_f32("audio_features"));
        compare("audio_waveform_embedding", model.embed_audio(reference->require_f32("audio_input")), reference->require_f32("audio_embedding"));
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
