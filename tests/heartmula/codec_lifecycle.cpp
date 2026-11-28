#include "engine/framework/codecs/heartcodec_runtime.h"
#include "engine/models/heartmula/assets.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <cmath>
#include <iostream>
#include <iomanip>

int main(int argc, char ** argv) {
    try {
        if (argc != 6 || std::string(argv[4]) != "--log") throw std::runtime_error(
            "Usage: heartmula_codec_lifecycle MODEL REFERENCE.safetensors cpu|cuda|vulkan --log FILE");
        engine::debug::configure_logging({true, argv[5]});
        const auto start = std::chrono::steady_clock::now();
        engine::core::BackendConfig config;
        const std::string backend = argv[3];
        if (backend == "cuda") config.type = engine::core::BackendType::Cuda;
        else if (backend == "vulkan") config.type = engine::core::BackendType::Vulkan;
        else if (backend != "cpu") throw std::runtime_error("Unknown backend");
        config.threads = 8;
        engine::core::ExecutionContext execution(config);
        auto assets = engine::models::heartmula::load_heartmula_assets(argv[1]);
        engine::codecs::HeartCodecRuntime codec(assets->codec_weights, execution, assets->codec_config,
            {32ull << 20, 2048ull << 20, 512ull << 20, 1536ull << 20,
             engine::assets::TensorStorageType::F32});
        const auto reference = engine::assets::open_tensor_source(argv[2]);
        const auto raw = reference->require_tensor_data("scalar_latents");
        if (raw.metadata.shape != std::vector<int64_t>{2, 128, 744})
            throw std::runtime_error("Expected stereo scalar latent fixture [2,128,744]");
        const auto latents = reference->require_f32("scalar_latents");
        const auto expected = reference->require_f32("scalar_waveform");
        std::vector<float> first;
        for (int pass = 0; pass < 3; ++pass) {
            const int64_t batch = pass == 1 ? 1 : 2;
            const int64_t frames = pass == 1 ? 300 : 744;
            std::vector<float> input(static_cast<size_t>(batch * 128 * frames));
            for (int64_t row = 0; row < batch * 128; ++row)
                std::copy_n(latents.begin() + row * 744, frames, input.begin() + row * frames);
            const auto output = codec.decode_scalar_latents(input, batch, frames);
            if (output.samples != frames * 1920 || output.channels != 1 || output.batch_size != batch)
                throw std::runtime_error("Scalar waveform shape mismatch");
            const int64_t compared = (frames == 744 ? frames : frames - 2) * 1920;
            double dot = 0, aa = 0, bb = 0, error = 0;
            for (int64_t b = 0; b < batch; ++b) for (int64_t i = 0; i < compared; ++i) {
                const float actual = output.values[b * output.samples + i];
                const float wanted = expected[b * 744 * 1920 + i];
                if (!std::isfinite(actual)) throw std::runtime_error("Non-finite scalar waveform");
                dot += static_cast<double>(actual) * wanted;
                aa += static_cast<double>(actual) * actual;
                bb += static_cast<double>(wanted) * wanted;
                error = std::max(error, static_cast<double>(std::abs(actual - wanted)));
            }
            const double cosine = dot / std::sqrt(aa * bb);
            std::cout << std::setprecision(12) << "pass=" << pass << " batch=" << batch
                      << " frames=" << frames << " cosine=" << cosine << " max_abs=" << error << '\n';
            if (cosine <= .99999) throw std::runtime_error("Scalar waveform parity failed");
            if (pass == 0) first = output.values;
            if (pass == 2 && first != output.values) throw std::runtime_error("Repeated scalar decode changed");
            codec.release_graph_workspaces();
        }
        const auto raw_codes = reference->require_tensor_data("codec_codes");
        if (raw_codes.metadata.dtype != "I32" ||
            raw_codes.metadata.shape != std::vector<int64_t>{1, 8, 372} ||
            raw_codes.bytes.size() != 8 * 372 * sizeof(int32_t))
            throw std::runtime_error("Expected codec code fixture I32[1,8,372]");
        std::vector<int32_t> codebook_major(8 * 372), codes(8 * 372);
        std::memcpy(codebook_major.data(), raw_codes.bytes.data(), raw_codes.bytes.size());
        for (size_t t = 0; t < 372; ++t) for (size_t q = 0; q < 8; ++q)
            codes[t * 8 + q] = codebook_major[q * 372 + t];
        const std::array<uint64_t, 3> seeds{1234, 2026, 9001};
        std::array<std::vector<float>, 3> outputs;
        for (size_t pass = 0; pass < 6; ++pass) {
            const size_t index = pass % seeds.size();
            const auto decode_start = std::chrono::steady_clock::now();
            const auto output = codec.detokenize_codes(codes, 372, 8, {}, seeds[index]);
            if (output.batch_size != 1 || output.channels != 2 || output.samples <= 0 ||
                output.values.size() != static_cast<size_t>(2 * output.samples) ||
                !std::all_of(output.values.begin(), output.values.end(),
                    [](float value) { return std::isfinite(value); }))
                throw std::runtime_error("Invalid detokenized waveform");
            if (pass < seeds.size()) {
                outputs[index] = output.values;
                if (index > 0 && outputs[index] == outputs[0])
                    throw std::runtime_error("Different codec seeds produced identical output");
            } else if (outputs[index] != output.values) {
                throw std::runtime_error("Repeated codec seed changed output");
            }
            if (pass == 2) codec.clear_graph_cache();
            else codec.release_graph_workspaces();
            engine::debug::trace_log_scalar("validation.seed", static_cast<int64_t>(seeds[index]));
            engine::debug::timing_log_scalar("validation.decode.wall_ms", engine::debug::elapsed_ms(decode_start));
            std::cout << "seed_pass=" << pass << " seed=" << seeds[index]
                      << " samples=" << output.samples << " repeated_exact=" << (pass >= seeds.size()) << '\n';
        }
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
