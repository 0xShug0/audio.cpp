#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/models/sheetsage/runtime.h"

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

void compare(const std::vector<float> & got, const std::vector<float> & ref, double max_abs_error) {
    if (got.empty() || got.size() != ref.size()) {
        throw std::runtime_error("output/reference size mismatch");
    }
    double dot = 0.0;
    double got_norm = 0.0;
    double ref_norm = 0.0;
    double mse = 0.0;
    double max_abs = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double g = got[i];
        const double r = ref[i];
        if (!std::isfinite(g) || !std::isfinite(r)) {
            throw std::runtime_error("non-finite output/reference value");
        }
        const double d = g - r;
        dot += g * r;
        got_norm += g * g;
        ref_norm += r * r;
        mse += d * d;
        max_abs = std::max(max_abs, std::abs(d));
    }
    if (!(got_norm > 0.0) || !(ref_norm > 0.0)) {
        throw std::runtime_error("zero-energy output/reference makes cosine undefined");
    }
    const double rmse = std::sqrt(mse / got.size());
    const double cosine = dot / std::sqrt(got_norm * ref_norm);
    std::cout << "output_values=" << got.size() << "\n";
    std::cout << "max_abs=" << max_abs << "\n";
    std::cout << "rmse=" << rmse << "\n";
    std::cout << "cosine=" << cosine << "\n";
    if (!std::isfinite(rmse) || !std::isfinite(cosine) || max_abs > max_abs_error) {
        throw std::runtime_error("SheetSage2 parity failed: invalid metrics or max_abs exceeds --max-abs-error");
    }
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        engine::debug::configure_logging({std::any_of(argv + 1, argv + argc,
            [](const char * arg) { return std::string(arg) == "--log"; })});
        const auto model = std::filesystem::path(require_arg(argc, argv, "--model"));
        const auto output_path = std::filesystem::path(optional_arg(argc, argv, "--output", ""));
        const auto reference_path = std::filesystem::path(optional_arg(argc, argv, "--reference", ""));
        // No independently established error bound exists for this probe.
        double max_abs_error = 0.0;
        if (!reference_path.empty()) {
            const auto threshold = require_arg(argc, argv, "--max-abs-error");
            size_t parsed = 0;
            max_abs_error = std::stod(threshold, &parsed);
            if (parsed != threshold.size() || !std::isfinite(max_abs_error) || max_abs_error < 0.0)
                throw std::runtime_error("--max-abs-error must be a finite nonnegative number");
        } else if (std::any_of(argv + 1, argv + argc,
                       [](const char * arg) { return std::string(arg) == "--max-abs-error"; })) {
            throw std::runtime_error("--max-abs-error requires --reference");
        }
        const engine::models::sheetsage::SheetSage2DecoderConfig config;
        size_t expected_values = 0;
        const int threads = static_cast<int>(std::stoll(optional_arg(argc, argv, "--threads", "8")));
        engine::core::ExecutionContext execution({parse_backend(optional_arg(argc, argv, "--backend", "cpu")), 0, threads});
        auto source = engine::assets::open_tensor_source(resolve_tensor_file(model));
        std::vector<float> logits;
        const auto component = optional_arg(argc, argv, "--component", "decoder");
        const auto started = std::chrono::steady_clock::now();
        if (component == "encoder") {
            const auto mel = read_binary<float>(require_arg(argc, argv, "--mel"));
            const int64_t mel_frames = require_i64(argc, argv, "--mel-frames");
            const int64_t runs = std::stoll(optional_arg(argc, argv, "--runs", "1"));
            if (mel_frames < 4 || mel.size() / static_cast<size_t>(config.mel_bins) != static_cast<size_t>(mel_frames) ||
                mel.size() % static_cast<size_t>(config.mel_bins) != 0)
                throw std::runtime_error("mel must have shape [1, mel_frames >= 4, mel_bins]");
            if (!std::all_of(mel.begin(), mel.end(), [](float value) { return std::isfinite(value); }))
                throw std::runtime_error("non-finite mel input");
            // Two kernel-2, stride-2 subsampling convolutions in MERT2.
            const int64_t encoded_frames = ((mel_frames - 2) / 2 + 1 - 2) / 2 + 1;
            expected_values = static_cast<size_t>(encoded_frames) * static_cast<size_t>(config.encoder_hidden_size);
            if (runs <= 0) throw std::runtime_error("--runs must be positive");
            engine::models::sheetsage::SheetSage2DecoderRuntimeOptions options;
            options.attention_tile_rows = std::stoll(optional_arg(argc, argv, "--attention-tile-rows", "0"));
            engine::models::sheetsage::Mert2EncoderRuntime runtime(source, execution, {}, options);
            runtime.prepare(1, mel_frames);
            const auto reference = reference_path.empty() ? std::vector<float>{} : read_binary<float>(reference_path);
            for (int64_t run = 0; run < runs; ++run) {
                const auto begin = std::chrono::steady_clock::now();
                logits = runtime.encode_mel(mel, mel_frames);
                engine::debug::timing_log_scalar("sheetsage2.encoder.total_ms", engine::debug::elapsed_ms(begin));
                if (logits.size() != expected_values ||
                    !std::all_of(logits.begin(), logits.end(), [](float value) { return std::isfinite(value); }))
                    throw std::runtime_error("invalid encoder output shape or non-finite values");
                if (!reference_path.empty()) compare(logits, reference, max_abs_error);
            }
        } else if (component == "decoder") {
            engine::models::sheetsage::SheetSage2DecoderRuntime runtime(source, execution);
            const auto mixed = read_binary<float>(require_arg(argc, argv, "--mixed"));
            const auto ids = read_binary<int32_t>(require_arg(argc, argv, "--ids"));
            const int64_t memory_steps = require_i64(argc, argv, "--memory-steps");
            if (memory_steps <= 0 || mixed.size() % static_cast<size_t>(config.encoder_hidden_size) != 0 ||
                mixed.size() / static_cast<size_t>(config.encoder_hidden_size) != static_cast<size_t>(memory_steps) || ids.empty())
                throw std::runtime_error("expected mixed [1, memory_steps, encoder_hidden_size] and nonempty ids");
            if (!std::all_of(mixed.begin(), mixed.end(), [](float value) { return std::isfinite(value); }) ||
                !std::all_of(ids.begin(), ids.end(), [&](int32_t id) { return id >= 0 && id < config.vocab_size; }))
                throw std::runtime_error("non-finite mixed input or out-of-vocabulary decoder id");
            expected_values = static_cast<size_t>(config.vocab_size);
            logits = runtime.decode_logits(mixed, memory_steps, ids);
        } else {
            throw std::runtime_error("--component must be encoder or decoder");
        }
        if (logits.empty() || logits.size() != expected_values ||
            !std::all_of(logits.begin(), logits.end(), [](float value) { return std::isfinite(value); }))
            throw std::runtime_error("invalid output shape or non-finite values");
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
        if (!output_path.empty()) {
            write_f32(output_path, logits);
        }
        if (!reference_path.empty() && component == "decoder") {
            compare(logits, read_binary<float>(reference_path), max_abs_error);
        } else {
            std::cout << "output_values=" << logits.size() << "\n";
        }
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "sheetsage2_decoder_parity_probe failed: " << e.what() << "\n";
        return 1;
    }
}
