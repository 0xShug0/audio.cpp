#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/codecs/oobleck_audio_vae_runtime.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
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

engine::core::BackendType parse_backend(const std::string & value) {
    if (value == "cpu") return engine::core::BackendType::Cpu;
    if (value == "cuda") return engine::core::BackendType::Cuda;
    if (value == "vulkan") return engine::core::BackendType::Vulkan;
    throw std::runtime_error("unsupported backend: " + value);
}

std::vector<float> read_f32_file(const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open " + path.string());
    }
    in.seekg(0, std::ios::end);
    const auto bytes = in.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("invalid f32 byte size: " + path.string());
    }
    in.seekg(0, std::ios::beg);
    std::vector<float> values(static_cast<size_t>(bytes / static_cast<std::streamoff>(sizeof(float))));
    in.read(reinterpret_cast<char *>(values.data()), bytes);
    return values;
}

void write_f32_file(const std::filesystem::path & path, const std::vector<float> & values) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
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

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto mode = arg_value(argc, argv, "--mode", "decode");
        const auto model = std::filesystem::path(arg_value(argc, argv, "--model", "/home/leo/Desktop/YuE2/YuE2-Vae"));
        const auto input_path = std::filesystem::path(arg_value(argc, argv, "--input"));
        const auto output_path = std::filesystem::path(arg_value(argc, argv, "--output"));
        const auto reference_path = std::filesystem::path(arg_value(argc, argv, "--reference"));
        const int threads = static_cast<int>(int_arg(argc, argv, "--threads", 8));
        const int64_t frames = int_arg(argc, argv, "--frames", 32);
        const int64_t batch = int_arg(argc, argv, "--batch", 1);

        engine::core::BackendConfig backend_config;
        backend_config.type = parse_backend(arg_value(argc, argv, "--backend", "cpu"));
        backend_config.threads = threads;
        engine::core::ExecutionContext execution(backend_config);

        const auto tensor_source_path = std::filesystem::is_directory(model) ? model / "model.safetensors" : model;
        auto source = engine::assets::open_tensor_source(tensor_source_path);
        engine::codecs::OobleckAudioVaeRuntime runtime(source, execution);
        const auto input = read_f32_file(input_path);
        std::vector<float> output;
        if (mode == "decode") {
            output = runtime.decode_planar(input, batch, frames);
        } else if (mode == "encode") {
            output = runtime.encode_planar(input, frames);
        } else {
            throw std::runtime_error("unsupported mode: " + mode);
        }
        if (!output_path.empty()) {
            write_f32_file(output_path, output);
        }
        std::cout << "mode=" << mode << " output_values=" << output.size() << "\n";
        if (!reference_path.empty()) {
            const auto ref = read_f32_file(reference_path);
            const auto m = compare(output, ref);
            std::cout << "max_abs=" << m.max_abs << "\n";
            std::cout << "rmse=" << m.rmse << "\n";
            std::cout << "cosine=" << m.cosine << "\n";
            const double max_abs_limit = mode == "decode" ? 8.0e-4 : 1.5e-3;
            const double cosine_limit = 0.999999;
            if (m.max_abs > max_abs_limit || m.cosine < cosine_limit) {
                throw std::runtime_error("YuE2 VAE parity check failed");
            }
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "yue2_vae_parity_probe failed: " << error.what() << "\n";
        return 1;
    }
}
