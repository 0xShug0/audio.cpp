#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/midi/sheetsage2_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string require_arg(int argc, char ** argv, const std::string & name) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    throw std::runtime_error("missing required argument " + name);
}

std::string optional_arg(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

int64_t require_i64(int argc, char ** argv, const std::string & name) {
    return std::stoll(require_arg(argc, argv, name));
}

engine::core::BackendType parse_backend(const std::string & value) {
    if (value == "cpu") return engine::core::BackendType::Cpu;
    if (value == "cuda") return engine::core::BackendType::Cuda;
    if (value == "vulkan") return engine::core::BackendType::Vulkan;
    if (value == "metal") return engine::core::BackendType::Metal;
    throw std::runtime_error("unsupported backend: " + value);
}

template <typename T>
std::vector<T> read_binary(const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open input file: " + path.string());
    }
    in.seekg(0, std::ios::end);
    const std::streamoff bytes = in.tellg();
    in.seekg(0, std::ios::beg);
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(T)) != 0) {
        throw std::runtime_error("input byte size is not aligned: " + path.string());
    }
    std::vector<T> values(static_cast<size_t>(bytes / static_cast<std::streamoff>(sizeof(T))));
    in.read(reinterpret_cast<char *>(values.data()), bytes);
    if (!in) {
        throw std::runtime_error("failed to read input file: " + path.string());
    }
    return values;
}

void write_f32(const std::filesystem::path & path, const std::vector<float> & values) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open output file: " + path.string());
    }
    out.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
}

std::filesystem::path resolve_tensor_file(const std::filesystem::path & model) {
    if (std::filesystem::is_directory(model)) {
        const auto direct = model / "model.safetensors";
        if (std::filesystem::is_regular_file(direct)) {
            return direct;
        }
    }
    return model;
}

void compare(const std::vector<float> & got, const std::vector<float> & ref) {
    if (got.size() != ref.size()) {
        throw std::runtime_error("output/reference size mismatch");
    }
    double dot = 0.0;
    double got_norm = 0.0;
    double ref_norm = 0.0;
    double mse = 0.0;
    float max_abs = 0.0F;
    for (size_t i = 0; i < got.size(); ++i) {
        const double g = got[i];
        const double r = ref[i];
        const double d = g - r;
        dot += g * r;
        got_norm += g * g;
        ref_norm += r * r;
        mse += d * d;
        max_abs = std::max(max_abs, static_cast<float>(std::abs(d)));
    }
    const double rmse = std::sqrt(mse / std::max<size_t>(got.size(), 1));
    const double cosine = dot / std::sqrt(std::max(got_norm * ref_norm, 1.0e-30));
    std::cout << "output_values=" << got.size() << "\n";
    std::cout << "max_abs=" << max_abs << "\n";
    std::cout << "rmse=" << rmse << "\n";
    std::cout << "cosine=" << cosine << "\n";
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto model = std::filesystem::path(require_arg(argc, argv, "--model"));
        const auto mixed_path = std::filesystem::path(require_arg(argc, argv, "--mixed"));
        const auto ids_path = std::filesystem::path(require_arg(argc, argv, "--ids"));
        const auto output_path = std::filesystem::path(optional_arg(argc, argv, "--output", ""));
        const auto reference_path = std::filesystem::path(optional_arg(argc, argv, "--reference", ""));
        const int64_t memory_steps = require_i64(argc, argv, "--memory-steps");
        const int threads = static_cast<int>(std::stoll(optional_arg(argc, argv, "--threads", "8")));
        engine::core::ExecutionContext execution({parse_backend(optional_arg(argc, argv, "--backend", "cpu")), 0, threads});
        auto source = engine::assets::open_tensor_source(resolve_tensor_file(model));
        engine::midi::SheetSage2DecoderRuntime runtime(source, execution);
        const auto mixed = read_binary<float>(mixed_path);
        const auto ids = read_binary<int32_t>(ids_path);
        const auto logits = runtime.decode_logits(mixed, memory_steps, ids);
        if (!output_path.empty()) {
            write_f32(output_path, logits);
        }
        if (!reference_path.empty()) {
            compare(logits, read_binary<float>(reference_path));
        } else {
            std::cout << "output_values=" << logits.size() << "\n";
        }
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "sheetsage2_decoder_parity_probe failed: " << e.what() << "\n";
        return 1;
    }
}
