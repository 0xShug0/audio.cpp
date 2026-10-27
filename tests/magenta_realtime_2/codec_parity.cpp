#include "engine/models/magenta_realtime_2/spectrostream.h"

#include "engine/framework/assets/tensor_source.h"
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
            "Usage: magenta_codec_parity MODEL.gguf REFERENCE.safetensors cpu|cuda|vulkan --log FILE");
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
        engine::models::magenta_realtime_2::SpectroStreamRuntime codec(execution, assets);
        const auto reference = engine::assets::open_tensor_source(argv[2]);
        const auto raw = reference->require_tensor_data("codes");
        if (raw.metadata.dtype != "I32" || raw.metadata.shape.size() != 2 || raw.metadata.shape[1] != 12)
            throw std::runtime_error("Invalid codec tokens fixture");
        std::vector<std::array<int32_t, 12>> codes(static_cast<size_t>(raw.metadata.shape[0]));
        std::memcpy(codes.data(), raw.bytes.data(), raw.bytes.size());
        const auto expected = reference->require_f32("waveform");
        for (size_t mode : {size_t{1}, size_t{4}, size_t{0}, size_t{0}}) {
            const size_t block_size = mode == 0 ? 4 : mode;
            codec.reset();
            std::vector<float> output;
            for (size_t offset = 0; offset < codes.size(); offset += block_size) {
                const auto end = std::min(codes.size(), offset + block_size);
                std::vector<std::array<int32_t, 12>> block{codes.begin() + static_cast<std::ptrdiff_t>(offset), codes.begin() + static_cast<std::ptrdiff_t>(end)};
                if (mode == 0) block.resize(4);
                auto decoded = codec.decode(block);
                decoded.resize((end - offset) * 1920 * 2);
                output.insert(output.end(), decoded.begin(), decoded.end());
            }
            if (output.size() != expected.size()) throw std::runtime_error("Codec waveform length mismatch");
            double dot = 0, aa = 0, bb = 0, error = 0;
            for (size_t i = 0; i < output.size(); ++i) {
                if (!std::isfinite(output[i])) throw std::runtime_error("Non-finite codec waveform");
                dot += static_cast<double>(output[i])*expected[i];
                aa += static_cast<double>(output[i])*output[i];
                bb += static_cast<double>(expected[i])*expected[i];
                error = std::max(error, static_cast<double>(std::abs(output[i]-expected[i])));
            }
            const auto cosine = dot / std::sqrt(aa*bb);
            std::cout << std::setprecision(12) << "block=" << block_size << " pad_final=" << (mode == 0)
                      << " cosine=" << cosine << " max_abs_error=" << error << '\n';
            engine::debug::trace_log_f32("spectrostream.waveform", {static_cast<int64_t>(output.size()/2), 2}, output);
            if (cosine <= .99999) throw std::runtime_error("Codec final-waveform parity failed");
        }
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
