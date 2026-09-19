// Stage 3: AuK's conditioner -- Qwen2.5-Omni's Thinker forward plus the ELMo-style
// fusion over all 36 layers -- against a PyTorch dump.
//
// Takes the reference's OWN input_ids rather than tokenizing here, so this measures the
// decoder and the fusion and not a chat template or a BPE merge table. Those are worth
// testing; they are not what this test is for.
//
//   auk_conditioner_parity --model <thinker.gguf> --tokens ids.i32 --fusion fusion.f32
//                          --reference fused.f32 [--backend cpu] [--min-cosine 0.9999]
//
// --fusion carries the learned layer_weights (one per layer, PRE-softmax) followed by
// layer_scale, all f32. Pre-softmax on purpose: the port should own the softmax.

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/models/auk/conditioner.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) return argv[i + 1];
    }
    return fallback;
}

double double_arg(int argc, char ** argv, const std::string & name, double fallback) {
    return std::stod(arg_value(argc, argv, name, std::to_string(fallback)));
}

int int_arg(int argc, char ** argv, const std::string & name, int fallback) {
    return std::stoi(arg_value(argc, argv, name, std::to_string(fallback)));
}

engine::core::BackendType parse_backend(const std::string & value) {
    if (value == "cpu") return engine::core::BackendType::Cpu;
    if (value == "cuda") return engine::core::BackendType::Cuda;
    if (value == "best") return engine::core::BackendType::BestAvailable;
    throw std::runtime_error("unknown backend: " + value);
}

template <typename T>
std::vector<T> read_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    const auto bytes = static_cast<std::streamoff>(input.tellg());
    if (bytes % static_cast<std::streamoff>(sizeof(T)) != 0) {
        throw std::runtime_error(path.string() + " is not a whole number of elements");
    }
    std::vector<T> values(static_cast<size_t>(bytes) / sizeof(T));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(values.data()), bytes);
    return values;
}

double cosine(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    const size_t count = std::min(lhs.size(), rhs.size());
    double dot = 0.0, ln = 0.0, rn = 0.0;
    for (size_t i = 0; i < count; ++i) {
        dot += double(lhs[i]) * double(rhs[i]);
        ln += double(lhs[i]) * double(lhs[i]);
        rn += double(rhs[i]) * double(rhs[i]);
    }
    return (ln == 0.0 || rn == 0.0) ? 0.0 : dot / (std::sqrt(ln) * std::sqrt(rn));
}

double max_abs_diff(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    const size_t count = std::min(lhs.size(), rhs.size());
    double worst = 0.0;
    for (size_t i = 0; i < count; ++i) {
        worst = std::max(worst, std::fabs(double(lhs[i]) - double(rhs[i])));
    }
    return worst;
}

}  // namespace

int main(int argc, char ** argv) try {
    const std::filesystem::path model_path = arg_value(argc, argv, "--model", "");
    const std::filesystem::path tokens_path = arg_value(argc, argv, "--tokens", "");
    const std::filesystem::path fusion_path = arg_value(argc, argv, "--fusion", "");
    const std::filesystem::path reference_path = arg_value(argc, argv, "--reference", "");
    const double min_cosine = double_arg(argc, argv, "--min-cosine", 0.9999);

    if (model_path.empty() || tokens_path.empty() || fusion_path.empty() || reference_path.empty()) {
        std::cerr << "usage: auk_conditioner_parity --model <gguf> --tokens <i32> "
                     "--fusion <f32> --reference <f32> [--backend cpu] [--threads N]\n";
        return 2;
    }

    engine::core::BackendConfig backend;
    backend.type = parse_backend(arg_value(argc, argv, "--backend", "cpu"));
    backend.device = 0;
    backend.threads = int_arg(argc, argv, "--threads", 1);
    engine::core::ExecutionContext execution(backend);

    engine::models::auk::AukConditionerConfig config;
    if (arg_value(argc, argv, "--weights", "native") == "f32") {
        config.weight_storage = engine::assets::TensorStorageType::F32;
    }
    auto fusion_values = read_file<float>(fusion_path);
    if (static_cast<int64_t>(fusion_values.size()) != config.layers + 1) {
        throw std::runtime_error(
            "fusion fixture should hold one weight per layer plus the scale");
    }
    engine::models::auk::AukFusionParameters fusion;
    fusion.layer_weights.assign(fusion_values.begin(), fusion_values.end() - 1);
    fusion.layer_scale = fusion_values.back();

    auto source = engine::assets::open_tensor_source(model_path);
    engine::models::auk::AukConditioner conditioner(config, fusion, *source, execution);

    const auto tokens = read_file<int32_t>(tokens_path);

    // With --audio, the tower's embeddings replace the <|AUDIO|> placeholder rows.
    std::vector<float> audio;
    std::vector<int64_t> audio_positions;
    const std::filesystem::path audio_path = arg_value(argc, argv, "--audio", "");
    if (!audio_path.empty()) {
        audio = read_file<float>(audio_path);
        const int32_t audio_token = static_cast<int32_t>(int_arg(argc, argv, "--audio-token", 151646));
        for (size_t index = 0; index < tokens.size(); ++index) {
            if (tokens[index] == audio_token) audio_positions.push_back(static_cast<int64_t>(index));
        }
        std::cout << "audio: " << audio_positions.size() << " placeholder tokens at "
                  << (audio_positions.empty() ? -1 : audio_positions.front()) << "..\n";
    }
    const auto conditioning = conditioner.encode(tokens, audio, audio_positions);
    const auto reference = read_file<float>(reference_path);

    std::cout << "loaded tensors=" << conditioner.loaded_tensor_count()
              << " tokens=" << conditioning.tokens
              << " hidden=" << conditioning.hidden_size << "\n";
    if (conditioning.values.size() != reference.size()) {
        std::cerr << "FAIL: produced " << conditioning.values.size()
                  << " values against the reference " << reference.size() << "\n";
        return 1;
    }
    const double cos = cosine(conditioning.values, reference);
    const double worst = max_abs_diff(conditioning.values, reference);
    std::cout << std::fixed << std::setprecision(8)
              << "cosine=" << cos << " max_abs_diff=" << worst << "\n";
    if (cos < min_cosine) {
        std::cerr << "FAIL: cosine " << cos << " is below " << min_cosine << "\n";
        return 1;
    }
    std::cout << "auk conditioner parity OK\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << "FAIL: " << error.what() << "\n";
    return 1;
}
