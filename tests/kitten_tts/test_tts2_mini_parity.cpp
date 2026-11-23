#include "engine/community_models/kitten_tts/assets.h"
#include "engine/community_models/kitten_tts/predictor.h"
#include "engine/community_models/kitten_tts/decoder.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/binary.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void compare(const std::string & name, const std::vector<float> & actual, const std::vector<float> & expected) {
    if (actual.size() != expected.size()) throw std::runtime_error(name + " shape mismatch: " +
        std::to_string(actual.size()) + " vs " + std::to_string(expected.size()));
    double dot = 0, aa = 0, bb = 0, maximum = 0, error_energy = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) throw std::runtime_error(name + " non-finite output");
        dot += static_cast<double>(actual[i]) * expected[i];
        aa += static_cast<double>(actual[i]) * actual[i];
        bb += static_cast<double>(expected[i]) * expected[i];
        maximum = std::max(maximum, std::abs(static_cast<double>(actual[i]) - expected[i]));
        const double error = static_cast<double>(actual[i]) - expected[i];
        error_energy += error * error;
    }
    const double cosine = dot / std::sqrt(aa * bb);
    const double relative_error = std::sqrt(error_energy / bb);
    engine::debug::timing_log_scalar(name + ".cosine", cosine);
    engine::debug::timing_log_scalar(name + ".max_error", maximum);
    engine::debug::timing_log_scalar(name + ".relative_error", relative_error);
    std::cout << name << " cosine=" << cosine << " max_error=" << maximum << '\n';
    if (!(cosine > 0.99999) || !(relative_error < 0.005)) throw std::runtime_error(name + " failed FP32 parity gate");
}
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 5 || std::string(argv[4]) != "--log")
            throw std::runtime_error("usage: kitten_tts_mini_parity_test MODEL REFERENCE_DIR cpu|cuda|vulkan --log");
        engine::debug::configure_logging({true, std::nullopt});
        namespace kitten = engine::models::kitten_tts;
        namespace core = engine::core;
        const std::string backend_name = argv[3];
        core::BackendConfig config;
        if (backend_name == "cuda") config.type = core::BackendType::Cuda;
        else if (backend_name == "vulkan") config.type = core::BackendType::Vulkan;
        else if (backend_name != "cpu") throw std::runtime_error("unknown test backend");
        config.threads = 8;
        core::ExecutionContext execution(config);
        auto assets = kitten::load_kitten_assets(argv[1]);
        auto original = kitten::load_kitten_backend_weights(*assets, execution.backend(), execution.backend_type(),
            engine::assets::TensorStorageType::Native, engine::assets::TensorStorageType::Native, 8ull * 1024 * 1024);
        auto weights = std::make_shared<kitten::KittenWeights>(*original);
        // Test-only excitation control; production weights remain unchanged.
        std::fill(weights->decoder.generator.source_linear.weight.begin(), weights->decoder.generator.source_linear.weight.end(), 0.0f);
        std::fill(weights->decoder.generator.source_linear.bias.begin(), weights->decoder.generator.source_linear.bias.end(), 0.0f);
        const std::filesystem::path root = argv[2];
        const auto cases = engine::io::json::parse_file(root / "cases.json");
        auto checks = cases.as_array();
        if (checks.empty()) throw std::runtime_error("empty parity reference");
        checks.push_back(checks.front()); // Return to A after different languages/shapes.
        for (const int64_t capacity : {int64_t{0}, int64_t{512}}) {
        kitten::KittenPredictorRuntime predictor(weights, execution.backend(), 8, !execution.uses_host_graph_plan(), capacity, capacity);
        std::cout << "capacity=" << capacity << '\n';
        for (const auto & item : checks) {
            const auto name = item.require("name").as_string();
            std::vector<int32_t> tokens;
            std::vector<float> style;
            for (const auto & token : item.require("tokens").as_array()) tokens.push_back(static_cast<int32_t>(token.as_i64()));
            for (const auto & value : item.require("style").as_array()) style.push_back(value.as_f32());
            auto output = predictor.predict(tokens, style, 1.0f);
            const auto & durations = item.require("durations").as_array();
            if (output.durations.size() != durations.size()) throw std::runtime_error(name + " duration shape mismatch");
            for (size_t i = 0; i < durations.size(); ++i)
                if (output.durations[i] != durations[i].as_i64()) throw std::runtime_error(name + " duration mismatch at " + std::to_string(i) +
                    ": actual=" + std::to_string(output.durations[i]) + " expected=" + std::to_string(durations[i].as_i64()));
            compare(name + ".f0", output.f0_curve, engine::io::read_f32_file(root / (name + "-f0.f32")));
            const auto x = output.decoder_x_on_backend ? core::read_tensor_f32(output.borrowed_decoder_x_tensor) : output.decoder_x;
            compare(name + ".decoder_input", x, engine::io::read_f32_file(root / (name + "-decoder_x.f32")));
            kitten::PredictorOutputs isolated;
            isolated.decoder_x = engine::io::read_f32_file(root / (name + "-decoder_x.f32"));
            isolated.f0_curve = engine::io::read_f32_file(root / (name + "-f0.f32"));
            isolated.decoder_x_rows = 512;
            isolated.decoder_x_cols = static_cast<int64_t>(isolated.f0_curve.size());
            kitten::KittenStyleTTS2DecoderRuntime decoder(weights, execution.backend(), 8, !execution.uses_host_graph_plan(), 42,
                {isolated.decoder_x_cols, isolated.decoder_x_cols * 60 + 1});
            compare(name + ".decoder", decoder.decode(isolated, style), engine::io::read_f32_file(root / (name + "-decoder.f32")));
            std::cout << "PASS " << name << '\n';
        }
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
