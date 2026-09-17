// Standalone YuE2 NAR acoustic-flow parity probe.
//
// The probe drives Yue2ArRuntime/Yue2NarRuntime directly instead of adding
// reference-only inputs to the request surface: everything the acoustic flow
// consumes is read from a directory produced by
// tests/yue2/yue2_nar_reference_dump.py, and the latents it produces are
// compared against the reference dump. Feeding the exact prefix, codec tokens
// and noise the Python reference used isolates the port from sampling RNG.
//
// Reference directory contract (little-endian raw tensors):
//   prefix.i32        token prefix: EOD, tags/lyrics text, ABC block, MUSIC_START
//   codec.i32         codec tokens, one per acoustic frame
//   noise.f32         [frames, 64] acoustic noise the reference integrated
//   latents_ref.f32   [frames, 64] reference latents
//   metadata.json     {"seed": .., "ode_steps": .., "context": .., "latent_dim": ..}
//
// Example:
//   yue2_nar_parity_probe --model models/Yue2-3B-GGUF --reference /tmp/yue2nar \
//                         --backend cuda --threads 8

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/options.h"
#include "engine/models/yue2/assets.h"
#include "engine/models/yue2/ar_runtime.h"
#include "engine/models/yue2/nar_runtime.h"
#include "engine/models/yue2/types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback = {}) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

int64_t int_arg(int argc, char ** argv, const std::string & name, int64_t fallback) {
    return std::stoll(arg_value(argc, argv, name, std::to_string(fallback)));
}

double float_arg(int argc, char ** argv, const std::string & name, double fallback) {
    return std::stod(arg_value(argc, argv, name, std::to_string(fallback)));
}

engine::core::BackendType parse_backend(const std::string & value) {
    if (value == "cpu") return engine::core::BackendType::Cpu;
    if (value == "cuda") return engine::core::BackendType::Cuda;
    if (value == "vulkan") return engine::core::BackendType::Vulkan;
    throw std::runtime_error("unsupported backend: " + value);
}

std::vector<char> read_bytes(const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open " + path.string());
    }
    in.seekg(0, std::ios::end);
    const auto bytes = in.tellg();
    if (bytes < 0) {
        throw std::runtime_error("invalid file size: " + path.string());
    }
    in.seekg(0, std::ios::beg);
    std::vector<char> data(static_cast<size_t>(bytes));
    in.read(data.data(), bytes);
    return data;
}

std::vector<int32_t> read_i32_file(const std::filesystem::path & path) {
    const auto bytes = read_bytes(path);
    if (bytes.size() % sizeof(int32_t) != 0) {
        throw std::runtime_error("invalid i32 byte size: " + path.string());
    }
    std::vector<int32_t> values(bytes.size() / sizeof(int32_t));
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
}

std::vector<float> read_f32_file(const std::filesystem::path & path) {
    const auto bytes = read_bytes(path);
    if (bytes.size() % sizeof(float) != 0) {
        throw std::runtime_error("invalid f32 byte size: " + path.string());
    }
    std::vector<float> values(bytes.size() / sizeof(float));
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
}

void write_f32_file(const std::filesystem::path & path, const std::vector<float> & values) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to write " + path.string());
    }
    out.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
}

struct CompareMetrics {
    double max_abs = 0.0;
    double rmse = 0.0;
    double cosine = 0.0;
};

CompareMetrics compare(const std::vector<float> & got, const std::vector<float> & ref) {
    if (got.size() != ref.size()) {
        throw std::runtime_error("output/reference size mismatch");
    }
    double sum_sq = 0.0;
    double dot = 0.0;
    double got_sq = 0.0;
    double ref_sq = 0.0;
    double max_abs = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double g = got[i];
        const double r = ref[i];
        const double d = g - r;
        max_abs = std::max(max_abs, std::abs(d));
        sum_sq += d * d;
        dot += g * r;
        got_sq += g * g;
        ref_sq += r * r;
    }
    return {
        max_abs,
        std::sqrt(sum_sq / static_cast<double>(got.size())),
        dot / std::sqrt(std::max(got_sq * ref_sq, std::numeric_limits<double>::min())),
    };
}

engine::assets::TensorStorageType parse_weight_type(const std::string & value) {
    const std::unordered_map<std::string, std::string> options{{"weight_type", value}};
    return engine::runtime::parse_tensor_storage_option(
        options, "weight_type", engine::assets::TensorStorageType::Native,
        {engine::assets::TensorStorageType::Native,
         engine::assets::TensorStorageType::F32,
         engine::assets::TensorStorageType::F16,
         engine::assets::TensorStorageType::BF16,
         engine::assets::TensorStorageType::Q8_0,
         engine::assets::TensorStorageType::Q4_0,
         engine::assets::TensorStorageType::Q4_K});
}

constexpr size_t kMib = 1024ull * 1024ull;

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto model = std::filesystem::path(arg_value(argc, argv, "--model"));
        const auto reference = std::filesystem::path(arg_value(argc, argv, "--reference"));
        if (model.empty() || reference.empty()) {
            std::cerr << "usage: yue2_nar_parity_probe --model <dir> --reference <dir>"
                         " [--backend cpu|cuda|vulkan] [--threads N] [--model-gguf name]"
                         " [--weight-type native|f32|f16|bf16|q8_0|q4_0|q4_k]"
                         " [--min-cosine X] [--max-rmse X] [--dump-latents file]\n";
            return 2;
        }
        const int threads = static_cast<int>(int_arg(argc, argv, "--threads", 8));
        const auto model_gguf = arg_value(argc, argv, "--model-gguf", "yue2-3b-q8_0.gguf");
        const auto min_cosine = float_arg(argc, argv, "--min-cosine", 0.995);
        const auto max_rmse = float_arg(argc, argv, "--max-rmse", 0.1);
        const auto metadata_path = reference / "metadata.json";
        const auto metadata = std::filesystem::is_regular_file(metadata_path)
                                  ? engine::io::json::parse_file(metadata_path)
                                  : engine::io::json::Value{};
        const int64_t seed = engine::io::json::optional_i64(metadata, "seed", 1234);
        const int64_t ode_steps = engine::io::json::optional_i64(metadata, "ode_steps", 8);
        const int64_t context = engine::io::json::optional_i64(metadata, "context", engine::models::yue2::kContextTokens);

        engine::core::BackendConfig backend_config;
        backend_config.type = parse_backend(arg_value(argc, argv, "--backend", "cpu"));
        backend_config.threads = threads;
        engine::core::ExecutionContext execution(backend_config);

        auto base = engine::models::yue2::load_yue2_assets(model);
        auto assets = std::make_shared<engine::models::yue2::Yue2Assets>(*base);
        assets->model_weights = engine::assets::open_tensor_source(model / model_gguf, "model_weights");

        const auto prefix = read_i32_file(reference / "prefix.i32");
        const auto codec = read_i32_file(reference / "codec.i32");
        const auto noise = read_f32_file(reference / "noise.f32");
        const auto latents_ref = read_f32_file(reference / "latents_ref.f32");
        const int64_t latent_dim = engine::io::json::optional_i64(metadata, "latent_dim", 64);
        if (latent_dim <= 0 ||
            codec.empty() ||
            noise.size() != static_cast<size_t>(codec.size()) * static_cast<size_t>(latent_dim) ||
            latents_ref.size() != noise.size()) {
            throw std::runtime_error("reference tensors do not agree with each other");
        }

        const auto weight_type = parse_weight_type(arg_value(argc, argv, "--weight-type", "native"));
        const size_t weight_context_bytes = static_cast<size_t>(int_arg(argc, argv, "--weight-context-mb", 6144)) * kMib;
        const size_t ar_prefill_arena_bytes = static_cast<size_t>(int_arg(argc, argv, "--ar-prefill-arena-mb", 4096)) * kMib;
        const size_t ar_decode_arena_bytes = static_cast<size_t>(int_arg(argc, argv, "--ar-decode-arena-mb", 1536)) * kMib;
        const size_t nar_arena_bytes = static_cast<size_t>(int_arg(argc, argv, "--nar-arena-mb", 6144)) * kMib;

        engine::models::yue2::Yue2ArRuntime ar(
            execution, assets, weight_type, weight_context_bytes, ar_prefill_arena_bytes, ar_decode_arena_bytes);
        engine::models::yue2::Yue2NarRuntime nar(
            execution, assets, weight_type, weight_context_bytes, nar_arena_bytes);

        std::cout << "prefix_tokens=" << prefix.size() << " codec_frames=" << codec.size()
                  << " ode_steps=" << ode_steps << " context=" << context << " seed=" << seed << "\n";
        const auto latents = nar.synthesize(
            prefix, codec,
            [&ar](const std::vector<int32_t> & tokens) { return ar.prefill_device_state(tokens); },
            noise, static_cast<uint64_t>(seed), ode_steps, context);
        if (latents.size() != latents_ref.size()) {
            throw std::runtime_error("latent count does not match the reference");
        }
        if (const auto dump_latents = arg_value(argc, argv, "--dump-latents"); !dump_latents.empty()) {
            write_f32_file(dump_latents, latents);
        }

        const auto metrics = compare(latents, latents_ref);
        std::cout << "latents=" << latents.size() << "\n";
        std::cout << "max_abs=" << metrics.max_abs << "\n";
        std::cout << "rmse=" << metrics.rmse << "\n";
        std::cout << "cosine=" << metrics.cosine << "\n";
        if (metrics.cosine < min_cosine || metrics.rmse > max_rmse) {
            throw std::runtime_error("YuE2 NAR parity check failed");
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "yue2_nar_parity_probe failed: " << error.what() << "\n";
        return 1;
    }
}
