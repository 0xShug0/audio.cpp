// Parity check for the MOSS-Audio-Tokenizer-v2 encoder (voice-clone front end).
// Loads the prepared reference waveform captured by scratchpad/moss_encode_ref.py
// (enc_prepared.f32, [2, samples] channel-major @ 48 kHz), runs the C++ encoder,
// and compares the resulting [num_quantizers, frames] codes against the Python
// reference (enc_codes.csv, first `num_quantizers` rows). Exact code match is the
// gate: the RLFQ quantizer is robust to small latent differences, so matching
// codes confirms both the encoder stack and the quantize path.

#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/moss/shared/audio_tokenizer_encoder.h"

#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

int int_arg(int argc, char ** argv, const std::string & name, int fallback) {
    return std::stoi(arg_value(argc, argv, name, std::to_string(fallback)));
}

engine::core::BackendType parse_backend(const std::string & name) {
    if (name == "cpu") {
        return engine::core::BackendType::Cpu;
    }
    if (name == "cuda") {
        return engine::core::BackendType::Cuda;
    }
    throw std::runtime_error("unsupported codec_encode_parity backend: " + name);
}

std::vector<float> read_f32(const std::string & path, size_t expected) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open " + path);
    }
    std::vector<float> data(expected);
    file.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(expected * sizeof(float)));
    if (static_cast<size_t>(file.gcount()) != expected * sizeof(float)) {
        throw std::runtime_error("short read from " + path);
    }
    return data;
}

std::vector<std::vector<int32_t>> read_codes_csv(const std::string & path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open " + path);
    }
    std::vector<std::vector<int32_t>> rows;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<int32_t> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stoi(cell));
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::string codec_dir = arg_value(
            argc, argv, "--codec",
            "C:/Users/justi/.cache/huggingface/hub/models--OpenMOSS-Team--MOSS-Audio-Tokenizer-v2/"
            "snapshots/f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169");
        const std::string prepared = arg_value(argc, argv, "--prepared", "enc_prepared.f32");
        const std::string ref_codes_path = arg_value(argc, argv, "--ref-codes", "enc_codes.csv");
        const int channels = int_arg(argc, argv, "--channels", 2);
        const int samples = int_arg(argc, argv, "--samples", 96000);
        const int num_quantizers = int_arg(argc, argv, "--quantizers", 12);
        const int threads = int_arg(argc, argv, "--threads", 16);
        const std::string backend_name = arg_value(argc, argv, "--backend", "cpu");
        const int device = int_arg(argc, argv, "--device", 0);

        std::cout << "loading prepared waveform [" << channels << "," << samples << "]...\n" << std::flush;
        const auto wav = read_f32(prepared, static_cast<size_t>(channels) * static_cast<size_t>(samples));
        std::vector<std::vector<float>> stereo(2, std::vector<float>(static_cast<size_t>(samples)));
        for (int c = 0; c < 2; ++c) {
            for (int i = 0; i < samples; ++i) {
                stereo[static_cast<size_t>(c)][static_cast<size_t>(i)] =
                    wav[static_cast<size_t>(c) * static_cast<size_t>(samples) + static_cast<size_t>(i)];
            }
        }

        constexpr size_t kWeightContextBytes = 256ull * 1024 * 1024;
        constexpr size_t kGraphArenaBytes = 2048ull * 1024 * 1024;

        engine::core::BackendConfig backend_config;
        backend_config.type = parse_backend(backend_name);
        backend_config.device = device;
        backend_config.threads = threads;
        engine::core::ExecutionContext execution_context(backend_config);

        std::cout << "loading codec encoder weights...\n" << std::flush;
        engine::models::moss::MossAudioTokenizerEncoder encoder(
            codec_dir, execution_context, num_quantizers, kWeightContextBytes, kGraphArenaBytes);

        std::cout << "encoding...\n" << std::flush;
        const auto codes = encoder.encode(stereo);
        const int64_t frames = codes.empty() ? 0 : static_cast<int64_t>(codes.front().size());
        std::cout << "produced codes [" << codes.size() << "," << frames << "]\n";

        const auto ref = read_codes_csv(ref_codes_path);
        std::cout << "reference codes rows=" << ref.size()
                  << " cols=" << (ref.empty() ? 0 : ref.front().size()) << "\n";

        int64_t total = 0;
        int64_t matched = 0;
        int mismatched_rows = 0;
        for (size_t q = 0; q < codes.size() && q < ref.size(); ++q) {
            int64_t row_match = 0;
            for (int64_t t = 0; t < frames && t < static_cast<int64_t>(ref[q].size()); ++t) {
                ++total;
                if (codes[q][static_cast<size_t>(t)] == ref[q][static_cast<size_t>(t)]) {
                    ++matched;
                    ++row_match;
                }
            }
            if (row_match != frames) {
                ++mismatched_rows;
                if (mismatched_rows <= 4) {
                    std::cout << "  cb" << q << ": " << row_match << "/" << frames << " match; cpp[0:8]=[";
                    for (int64_t t = 0; t < 8 && t < frames; ++t) {
                        std::cout << (t ? "," : "") << codes[q][static_cast<size_t>(t)];
                    }
                    std::cout << "] ref[0:8]=[";
                    for (int64_t t = 0; t < 8 && t < static_cast<int64_t>(ref[q].size()); ++t) {
                        std::cout << (t ? "," : "") << ref[q][static_cast<size_t>(t)];
                    }
                    std::cout << "]\n";
                }
            }
        }

        std::cout << "\n=== ENCODE PARITY: " << matched << "/" << total << " codes match ===\n";
        const bool ok = (total > 0 && matched == total);
        std::cout << (ok ? "PASS" : "FAIL") << "\n";
        return ok ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << "codec_encode_parity failed: " << error.what() << "\n";
        return 1;
    }
}
