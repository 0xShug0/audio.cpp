#include "engine/models/magenta_realtime_2/depthformer.h"
#include "engine/models/magenta_realtime_2/spectrostream.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>

int main(int argc, char ** argv) {
    try {
        if (argc != 6 || std::string(argv[4]) != "--log") throw std::runtime_error(
            "Usage: magenta_greedy_parity MODEL.gguf REFERENCE.safetensors cpu|cuda|vulkan --log FILE");
        engine::debug::configure_logging({true, argv[5]});
        const auto start = std::chrono::steady_clock::now();
        engine::core::BackendConfig config;
        const std::string backend = argv[3];
        if (backend == "cuda") config.type = engine::core::BackendType::Cuda;
        else if (backend == "vulkan") config.type = engine::core::BackendType::Vulkan;
        else if (backend != "cpu") throw std::runtime_error("Unknown backend");
        config.threads = 8;
        engine::core::ExecutionContext execution(config);
        auto assets = engine::models::magenta_realtime_2::load_magenta_realtime_2_assets(argv[1]);
        engine::models::magenta_realtime_2::DepthformerRuntime decoder(execution, assets, engine::assets::TensorStorageType::Native);
        engine::models::magenta_realtime_2::SpectroStreamRuntime codec(execution, assets);
        const auto reference = engine::assets::open_tensor_source(argv[2]);
        const auto raw = reference->require_tensor_data("codes");
        if (raw.metadata.dtype != "I32" || raw.metadata.shape.size() != 2 || raw.metadata.shape[1] != 12)
            throw std::runtime_error("Invalid reference codes");
        std::vector<std::array<int32_t, 12>> expected_codes(static_cast<size_t>(raw.metadata.shape[0]));
        std::memcpy(expected_codes.data(), raw.bytes.data(), raw.bytes.size());
        const auto style_data = reference->require_tensor_data("style_codes");
        const auto controls_data = reference->require_tensor_data("control_tokens");
        std::array<int32_t, 12> style;
        std::array<int32_t, 132> controls;
        if (style_data.bytes.size() != sizeof(style) || controls_data.bytes.size() != sizeof(controls))
            throw std::runtime_error("Invalid reference conditioning shapes");
        std::memcpy(style.data(), style_data.bytes.data(), sizeof(style));
        std::memcpy(controls.data(), controls_data.bytes.data(), sizeof(controls));
        for (size_t i = 0; i < style.size(); ++i) style[i] += 7 + static_cast<int32_t>(i) * 1031;
        for (size_t i = 0; i < 128; ++i) controls[i] += 7 + static_cast<int32_t>(i) * 11;
        const int offsets[] = {1408, 1417, 1464, 1511};
        for (size_t i = 0; i < 4; ++i) controls[128 + i] += 7 + offsets[i];
        decoder.reset({0, 40, 0});
        decoder.condition(style, controls);
        codec.reset();
        std::vector<std::array<int32_t, 12>> generated;
        for (size_t i = 0; i < expected_codes.size(); ++i) generated.push_back(decoder.step());
        size_t mismatched = 0;
        for (size_t i = 0; i < generated.size(); ++i)
            for (size_t j = 0; j < 12; ++j) mismatched += generated[i][j] != expected_codes[i][j];
        std::vector<float> audio;
        for (size_t start_frame = 0; start_frame < generated.size(); start_frame += 4) {
            const auto end = std::min(start_frame + 4, generated.size());
            auto chunk = codec.decode({generated.begin() + static_cast<std::ptrdiff_t>(start_frame), generated.begin() + static_cast<std::ptrdiff_t>(end)});
            audio.insert(audio.end(), chunk.begin(), chunk.end());
        }
        const auto expected = reference->require_f32("waveform");
        if (audio.size() != expected.size()) throw std::runtime_error("Greedy waveform length mismatch");
        double dot = 0, aa = 0, bb = 0;
        for (size_t i = 0; i < audio.size(); ++i) {
            dot += static_cast<double>(audio[i]) * expected[i];
            aa += static_cast<double>(audio[i]) * audio[i];
            bb += static_cast<double>(expected[i]) * expected[i];
        }
        const double cosine = dot / std::sqrt(aa * bb);
        std::cout << std::setprecision(12) << "codes_mismatched=" << mismatched << "/" << generated.size()*12
                  << " waveform_cosine=" << cosine << '\n';
        engine::audio::WavWriteOptions wav;
        wav.format = engine::audio::WavSampleFormat::Float32;
        engine::audio::write_wav(std::filesystem::path(argv[5]).replace_extension("wav"), 48000, 2, audio, wav);
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
        if (cosine <= .99999) throw std::runtime_error("Greedy waveform parity failed");
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
