// The audio tower against its PyTorch tap: log-mel -> 100 tokens of 2048 dims.

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/models/auk/audio_tower.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {
std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) if (argv[i] == name) return argv[i + 1];
    return fallback;
}
std::vector<float> read_f32(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    const auto bytes = static_cast<std::streamoff>(input.tellg());
    std::vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(values.data()), bytes);
    return values;
}
}  // namespace

int main(int argc, char ** argv) try {
    const std::filesystem::path model = arg_value(argc, argv, "--model", "");
    const std::filesystem::path dir = arg_value(argc, argv, "--fixtures", "");
    const int64_t valid_frames = std::stoll(arg_value(argc, argv, "--frames", "400"));
    if (model.empty() || dir.empty()) {
        std::cerr << "usage: auk_audio_tower_parity --model <thinker.gguf> --fixtures <dir> [--frames 400]\n";
        return 2;
    }
    engine::core::BackendConfig backend;
    // ⚠ Selectable, and it matters: a CUDA BUILD does not mean a CUDA RUN. Hardcoding
    // Cpu here made a GPU build reproduce the CPU numbers exactly, which reads as
    // "the backends agree" when it actually means the GPU never ran.
    const std::string backend_name = arg_value(argc, argv, "--backend", "cpu");
    backend.type = backend_name == "cuda"  ? engine::core::BackendType::Cuda
                 : backend_name == "best"  ? engine::core::BackendType::BestAvailable
                                           : engine::core::BackendType::Cpu;
    backend.threads = std::stoi(arg_value(argc, argv, "--threads", "1"));
    engine::core::ExecutionContext execution(backend);

    engine::models::auk::AukAudioTowerConfig config;
    auto source = engine::assets::open_tensor_source(model);
    engine::models::auk::AukAudioTower tower(config, *source, execution);

    // The reference's mel is padded to 30000 frames; only the valid ones are encoded,
    // which is what feature_attention_mask means.
    const auto padded = read_f32(dir / "stage6_mel.f32");
    const int64_t padded_frames = static_cast<int64_t>(padded.size()) / config.mel_bins;
    std::vector<float> mel(static_cast<size_t>(config.mel_bins * valid_frames));
    for (int64_t bin = 0; bin < config.mel_bins; ++bin) {
        std::copy(padded.begin() + bin * padded_frames,
                  padded.begin() + bin * padded_frames + valid_frames,
                  mel.begin() + bin * valid_frames);
    }

    const auto embeddings = tower.encode(mel, valid_frames);
    const auto reference = read_f32(dir / "stage6_tower.f32");
    std::cout << "loaded tensors=" << tower.loaded_tensor_count()
              << " mel_frames=" << valid_frames << " -> tokens=" << embeddings.tokens
              << " dim=" << embeddings.dim << "\n";
    if (embeddings.values.size() != reference.size()) {
        std::cerr << "FAIL: produced " << embeddings.values.size() << " against " << reference.size() << "\n";
        return 1;
    }
    const std::string dump = arg_value(argc, argv, "--dump", "");
    if (!dump.empty()) {
        std::ofstream out(dump, std::ios::binary);
        out.write(reinterpret_cast<const char *>(embeddings.values.data()),
                  static_cast<std::streamsize>(embeddings.values.size() * sizeof(float)));
    }
    double dot = 0.0, ln = 0.0, rn = 0.0, worst = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        dot += double(embeddings.values[i]) * double(reference[i]);
        ln += double(embeddings.values[i]) * double(embeddings.values[i]);
        rn += double(reference[i]) * double(reference[i]);
        worst = std::max(worst, std::fabs(double(embeddings.values[i]) - double(reference[i])));
    }
    const double cos = dot / (std::sqrt(ln) * std::sqrt(rn));
    std::cout << std::fixed << std::setprecision(8) << "cosine=" << cos << " max_abs_diff=" << worst << "\n";
    if (cos < std::stod(arg_value(argc, argv, "--min-cosine", "0.999"))) {
        std::cerr << "FAIL: cosine below gate\n";
        return 1;
    }
    std::cout << "auk audio tower parity OK\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << "FAIL: " << error.what() << "\n";
    return 1;
}
