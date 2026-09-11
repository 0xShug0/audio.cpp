#include "engine/community_models/kroko_asr/assets.h"
#include "engine/community_models/kroko_asr/zipformer.h"
#include "engine/framework/core/execution_context.h"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

int main(int argc, char ** argv) try {
    if (argc != 3 && argc != 4) {
        std::cerr
            << "usage: kroko_asr_zipformer_probe <model-directory> "
               "<subsampled-f32.bin> [output-f32.bin]\n";
        return 2;
    }
    const auto assets =
        engine::models::kroko_asr::load_kroko_asr_assets(
            std::filesystem::path(argv[1]));
    std::vector<float> input(128 * 192, 0.0F);
    std::ifstream stream(
        std::filesystem::path(argv[2]), std::ios::binary);
    if (!stream.read(
            reinterpret_cast<char *>(input.data()),
            static_cast<std::streamsize>(
                input.size() * sizeof(float)))) {
        throw std::runtime_error(
            "failed to read 128x192 Kroko probe input");
    }
    engine::core::ExecutionContext execution(
        engine::core::BackendConfig{
            engine::core::BackendType::Cpu, 0, 4});
    engine::models::kroko_asr::KrokoZipformerRuntime encoder(
        assets, execution);
    const auto output = encoder.encode_chunk(input);
    if (argc == 4) {
        std::ofstream output_stream(
            std::filesystem::path(argv[3]),
            std::ios::binary | std::ios::trunc);
        if (!output_stream) {
            throw std::runtime_error(
                "failed to open Kroko Zipformer probe output");
        }
        output_stream.write(
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
    std::cerr << "kroko_asr_zipformer_probe failed: "
              << error.what() << '\n';
    return 1;
}
