#include "engine/framework/modules/speech_encoders/wavlm_encoder.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> read_f32_file(const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open input file: " + path.string());
    }
    in.seekg(0, std::ios::end);
    const std::streamoff bytes = in.tellg();
    if (bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("input file does not contain packed f32 values: " + path.string());
    }
    in.seekg(0, std::ios::beg);
    std::vector<float> values(static_cast<size_t>(bytes / static_cast<std::streamoff>(sizeof(float))));
    in.read(reinterpret_cast<char *>(values.data()), bytes);
    if (!in) {
        throw std::runtime_error("failed to read input file: " + path.string());
    }
    return values;
}

void write_u64(std::ofstream & out, uint64_t value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void write_output(
    const std::filesystem::path & path,
    const engine::modules::WavlmEncoderOutput & output) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open output file: " + path.string());
    }
    write_u64(out, static_cast<uint64_t>(output.batch));
    write_u64(out, static_cast<uint64_t>(output.tokens));
    write_u64(out, static_cast<uint64_t>(output.hidden_size));
    out.write(
        reinterpret_cast<const char *>(output.hidden_states.data()),
        static_cast<std::streamsize>(output.hidden_states.size() * sizeof(float)));
    if (!out) {
        throw std::runtime_error("failed to write output file: " + path.string());
    }
}

engine::core::BackendType parse_backend(const std::string & value) {
    if (value == "cpu") {
        return engine::core::BackendType::Cpu;
    }
    if (value == "cuda") {
        return engine::core::BackendType::Cuda;
    }
    if (value == "vulkan") {
        return engine::core::BackendType::Vulkan;
    }
    throw std::runtime_error("unsupported backend: " + value);
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 5) {
            throw std::runtime_error(
                "usage: miocodec_wavlm_parity <weights.safetensors> <input.f32> <output.bin> <cpu|cuda|vulkan>");
        }

        engine::core::BackendConfig backend;
        backend.type = parse_backend(argv[4]);
        backend.device = 0;
        backend.threads = 8;

        const auto input = read_f32_file(argv[2]);
        auto wavlm = engine::modules::WavlmEncoderComponent::load_from_safetensors(argv[1], backend);
        const auto output = wavlm.encode(input, 1, static_cast<int64_t>(input.size()));
        write_output(argv[3], output);

        std::cout << "wavlm.batch " << output.batch << "\n";
        std::cout << "wavlm.tokens " << output.tokens << "\n";
        std::cout << "wavlm.hidden_size " << output.hidden_size << "\n";
        std::cout << "wavlm.values " << output.hidden_states.size() << "\n";
        std::cout << "wavlm.loaded_tensors " << wavlm.loaded_tensor_count() << "\n";
        std::cout << "wavlm.parameters " << wavlm.parameter_count() << "\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "miocodec_wavlm_parity failed: " << ex.what() << "\n";
        return 1;
    }
}
