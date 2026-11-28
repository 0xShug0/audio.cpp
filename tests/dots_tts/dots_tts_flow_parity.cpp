#include "engine/models/dots_tts/assets.h"
#include "engine/models/dots_tts/flow.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/safetensors.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>

// Real-package component replay: fixed inputs isolate estimator math from AR
// feedback. This model test is deliberately not part of the generic unit suite.
int main(int argc, char ** argv) try {
    using namespace engine;
    using namespace models::dots_tts;
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) throw std::invalid_argument("Expected --option value");
        args.emplace(argv[i], argv[i + 1]);
    }
    debug::configure_logging({true, args.at("--log")});
    const auto started = std::chrono::steady_clock::now();
    const auto assets = load_dots_assets(args.at("--model"));
    core::BackendConfig backend;
    const std::map<std::string, core::BackendType> backends{
        {"cpu", core::BackendType::Cpu}, {"cuda", core::BackendType::Cuda},
        {"vulkan", core::BackendType::Vulkan}, {"metal", core::BackendType::Metal}};
    backend.type = backends.at(args.at("--backend"));
    backend.threads = 8;
    const std::map<std::string, assets::TensorStorageType> storages{
        {"native", assets::TensorStorageType::Native}, {"f32", assets::TensorStorageType::F32}};
    auto flow = DotsFlowComponent::load_from_tensor_source(assets->core_weights, backend,
        assets->config, storages.at(args.count("--weight-type") ? args.at("--weight-type") : "native"));
    const auto & config = assets->config;
    const auto pattern = [](int64_t count, float phase) {
        std::vector<float> data(static_cast<size_t>(count));
        for (int64_t i = 0; i < count; ++i)
            data[static_cast<size_t>(i)] = 0.15f * std::sin(static_cast<float>(i) * 0.013f + phase);
        return data;
    };
    std::vector<io::SafeTensorWriteEntry> outputs;
    std::map<std::string, std::vector<float>> first_outputs;
    const auto reference = args.count("--reference") ? assets::open_tensor_source(args.at("--reference")) : nullptr;
    const auto record = [&](std::string name, const std::vector<float> & values) {
        for (const auto value : values) if (!std::isfinite(value)) throw std::runtime_error(name + " is non-finite");
        if (reference && reference->require_f32(name, {static_cast<int64_t>(values.size())}) != values)
            throw std::runtime_error(name + " differs from the reference fixture");
        const auto previous = first_outputs.find(name);
        if (previous != first_outputs.end()) {
            if (previous->second != values) throw std::runtime_error(name + " changed after phase buffer release/reuse");
            return;
        }
        first_outputs.emplace(name, values);
        io::SafeTensorWriteEntry tensor{std::move(name), "F32", {static_cast<int64_t>(values.size())}, {}};
        tensor.data.resize(values.size() * sizeof(float));
        std::memcpy(tensor.data.data(), values.data(), tensor.data.size());
        outputs.push_back(std::move(tensor));
    };
    for (const int64_t batch : {int64_t{1}, int64_t{2}}) {
        const int64_t frames = 11;
        const auto tokens = pattern(batch * frames * config.dit.hidden_size, 0.3f);
        const auto speaker = pattern(batch * config.dit.hidden_size, 0.7f);
        std::vector<float> times(static_cast<size_t>(batch), 0.375f), durations;
        if (config.meanflow && config.meanflow->use_duration_embedding) durations.assign(static_cast<size_t>(batch), 0.25f);
        std::vector<int32_t> positions(static_cast<size_t>(frames));
        std::vector<uint8_t> mask(static_cast<size_t>(batch * frames * frames));
        for (int64_t row = 0; row < frames; ++row) {
            positions[static_cast<size_t>(row)] = static_cast<int32_t>(row + 7);
            for (int64_t b = 0; b < batch; ++b)
                for (int64_t col = 0; col <= row; ++col)
                    mask[static_cast<size_t>((b * frames + row) * frames + col)] = 1;
        }
        record("velocity.batch" + std::to_string(batch),
            flow.predict_velocity(tokens, frames, times, durations, speaker, batch, positions, mask).values);
    }
    const bool meanflow = config.meanflow && config.meanflow->enabled;
    const auto speaker = pattern(config.dit.hidden_size, 0.7f);
    const auto unit = config.patch_size + 1;
    for (int cycle = 0; cycle < 3; ++cycle) {
        flow.release_decode_cache();
        for (const bool cached : {false, true}) {
            DotsFlowDecodeState state;
            for (int64_t history = 1; history <= 3; ++history) {
                const auto frames = history * unit + 1;
                const auto tokens = pattern(frames * config.dit.hidden_size, 0.3f);
                const auto negative = pattern(frames * config.dit.hidden_size, -0.2f);
                DotsFlowDecodeRequest request;
                request.sequence = &tokens; request.cfg_sequence = &negative;
                request.speaker_condition = &speaker; request.fm_seq_len = frames;
                request.num_inference_steps = 4; request.seed = 42;
                request.guidance_scale = 1.2f;
                request.cache_capacity = 64 * unit; request.use_cached_dit = cached;
                request.decode_state = &state;
                const auto latent = meanflow ? flow.decode_next_meanflow(request) : flow.decode_next_soar(request);
                record(std::string(cached ? "cached." : "uncached.") + std::to_string(history), latent.values);
            }
        }
    }
    io::write_safetensors_file(args.at("--out"), outputs);
    debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
    return 0;
} catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
}
