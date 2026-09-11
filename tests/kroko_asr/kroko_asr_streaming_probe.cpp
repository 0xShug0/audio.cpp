#include "engine/community_models/kroko_asr/assets.h"
#include "engine/community_models/kroko_asr/encoder.h"
#include "engine/community_models/kroko_asr/zipformer.h"
#include "engine/framework/core/execution_context.h"

#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char ** argv) try {
    if (argc != 3 && argc != 4) {
        std::cerr
            << "usage: kroko_asr_streaming_probe <model-directory> "
               "<output-f32.bin> [embedded-f32.bin]\n";
        return 2;
    }
    const auto assets =
        engine::models::kroko_asr::load_kroko_asr_assets(
            std::filesystem::path(argv[1]));
    engine::core::ExecutionContext execution(
        engine::core::BackendConfig{
            engine::core::BackendType::Cpu, 0, 4});
    engine::models::kroko_asr::KrokoEncoderRuntime subsampling(
        assets, execution);
    engine::models::kroko_asr::KrokoZipformerRuntime zipformer(
        assets, execution);

    const int64_t chunk_size = assets->config.chunk_size;
    const int64_t chunk_shift = assets->config.chunk_shift;
    const int64_t total_frames = chunk_shift + chunk_size;
    constexpr int64_t feature_dim = 80;
    std::vector<float> features(
        static_cast<size_t>(total_frames * feature_dim));
    for (size_t index = 0; index < features.size(); ++index) {
        features[index] =
            std::sin(static_cast<float>(index) * 0.0013F) * 0.7F +
            std::cos(static_cast<float>(index) * 0.0007F) * 0.2F;
    }

    std::ofstream output(
        std::filesystem::path(argv[2]),
        std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "failed to open Kroko streaming probe output");
    }
    std::ofstream embedded_output;
    if (argc == 4) {
        embedded_output.open(
            std::filesystem::path(argv[3]),
            std::ios::binary | std::ios::trunc);
        if (!embedded_output) {
            throw std::runtime_error(
                "failed to open Kroko embedding probe output");
        }
    }
    for (int64_t chunk = 0; chunk < 2; ++chunk) {
        const int64_t offset = chunk * chunk_shift;
        std::vector<float> input(
            features.begin() + offset * feature_dim,
            features.begin() + (offset + chunk_size) * feature_dim);
        const auto embedded =
            subsampling.encode_subsampled_chunk(input);
        if (embedded_output) {
            embedded_output.write(
                reinterpret_cast<const char *>(
                    embedded.values.data()),
                static_cast<std::streamsize>(
                    embedded.values.size() * sizeof(float)));
        }
        const auto encoded =
            zipformer.encode_chunk(embedded.values);
        output.write(
            reinterpret_cast<const char *>(encoded.values.data()),
            static_cast<std::streamsize>(
                encoded.values.size() * sizeof(float)));
    }
    std::cout << "{\"chunks\":2,\"frames_per_chunk\":"
              << chunk_shift / 4 << ','
              << "\"dims\":512}\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << "kroko_asr_streaming_probe failed: "
              << error.what() << '\n';
    return 1;
}
