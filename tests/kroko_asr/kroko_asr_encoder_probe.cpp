#include "engine/community_models/kroko_asr/assets.h"
#include "engine/community_models/kroko_asr/encoder.h"
#include "engine/framework/core/execution_context.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

int main(int argc, char ** argv) try {
    if (argc != 2 && argc != 3) {
        std::cerr
            << "usage: kroko_asr_encoder_probe <model-directory> "
               "[output-f32.bin]\n";
        return 2;
    }
    const auto assets =
        engine::models::kroko_asr::load_kroko_asr_assets(
            std::filesystem::path(argv[1]));
    engine::core::ExecutionContext execution(
        engine::core::BackendConfig{
            engine::core::BackendType::Cpu, 0, 4});
    engine::models::kroko_asr::KrokoEncoderRuntime encoder(
        assets, execution);
    std::vector<float> input(
        static_cast<size_t>(
            assets->config.chunk_size *
            assets->config.feature_dim),
        0.0F);
    for (size_t index = 0; index < input.size(); ++index) {
        input[index] =
            std::sin(static_cast<float>(index) * 0.0013F) * 0.7F +
            std::cos(static_cast<float>(index) * 0.0007F) * 0.2F;
    }
    const auto output = encoder.encode_subsampled_chunk(input);
    if (argc == 3) {
        std::ofstream stream(
            std::filesystem::path(argv[2]),
            std::ios::binary | std::ios::trunc);
        if (!stream) {
            throw std::runtime_error(
                "failed to open Kroko probe output");
        }
        stream.write(
            reinterpret_cast<const char *>(output.values.data()),
            static_cast<std::streamsize>(
                output.values.size() * sizeof(float)));
    }
    const size_t count = std::min<size_t>(output.values.size(), 32);
    std::cout << std::setprecision(9)
              << "{\"frames\":" << output.frames
              << ",\"dims\":" << output.channels
              << ",\"sum\":"
              << std::accumulate(
                     output.values.begin(),
                     output.values.end(),
                     0.0)
              << ",\"first\":[";
    for (size_t index = 0; index < count; ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        std::cout << output.values[index];
    }
    std::cout << "]}\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << "kroko_asr_encoder_probe failed: "
              << error.what() << '\n';
    return 1;
}
