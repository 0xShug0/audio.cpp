// Standalone verification for the MOSS-TTS-Local backbone + depth transformer + generator.
// It loads the real weights, builds a text-only generation prefix, and runs a short greedy
// generation so we can confirm the pipeline executes and emits in-range RVQ codes. No audio
// is produced here: the MOSS-Audio-Tokenizer-v2 decoder is a separate phase.

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/moss/moss_tts_local/assets.h"
#include "engine/models/moss/moss_tts_local/backbone.h"
#include "engine/models/moss/moss_tts_local/depth_transformer.h"
#include "engine/models/moss/moss_tts_local/generator.h"
#include "engine/models/moss/moss_tts_local/tokenizer_text.h"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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

engine::assets::TensorStorageType parse_weight_type(const std::string & value) {
    if (value == "native") {
        return engine::assets::TensorStorageType::Native;
    }
    if (value == "f32") {
        return engine::assets::TensorStorageType::F32;
    }
    if (value == "f16") {
        return engine::assets::TensorStorageType::F16;
    }
    if (value == "bf16") {
        return engine::assets::TensorStorageType::BF16;
    }
    if (value == "q8_0") {
        return engine::assets::TensorStorageType::Q8_0;
    }
    throw std::runtime_error("unsupported weight type: " + value);
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::filesystem::path model_path =
            arg_value(argc, argv, "--model", "models/MOSS-TTS-Local-Transformer-v1.5");
        const std::string text = arg_value(
            argc, argv, "--text", "Hello, this is a test of the MOSS text to speech system.");
        const int threads = int_arg(argc, argv, "--threads", 8);
        const int max_frames = int_arg(argc, argv, "--frames", 12);
        const auto weight_type = parse_weight_type(arg_value(argc, argv, "--weight-type", "native"));
        const std::string dump_path = arg_value(argc, argv, "--dump", "");

        constexpr size_t kBackboneWeightContextBytes = 64ull * 1024 * 1024;
        constexpr size_t kBackboneGraphArenaBytes = 512ull * 1024 * 1024;
        constexpr size_t kDepthWeightContextBytes = 16ull * 1024 * 1024;
        constexpr size_t kDepthGraphArenaBytes = 64ull * 1024 * 1024;
        constexpr size_t kProjectionWeightContextBytes = 16ull * 1024 * 1024;
        constexpr size_t kProjectionGraphArenaBytes = 16ull * 1024 * 1024;

        std::cout << "model=" << model_path.string() << "\n";
        std::cout << "loading assets...\n" << std::flush;
        auto assets = engine::models::moss_tts_local::load_moss_tts_local_assets(model_path);

        engine::core::BackendConfig backend_config;
        backend_config.type = engine::core::BackendType::Cpu;
        backend_config.device = 0;
        backend_config.threads = threads;
        engine::core::ExecutionContext execution_context(backend_config);

        std::cout << "loading backbone weights (36 layers)...\n" << std::flush;
        engine::models::moss_tts_local::MossBackboneRuntime backbone(
            assets, execution_context, kBackboneGraphArenaBytes, kBackboneWeightContextBytes, weight_type);

        std::cout << "loading depth transformer weights...\n" << std::flush;
        engine::models::moss_tts_local::MossDepthTransformer depth(
            assets, execution_context, kDepthGraphArenaBytes, kDepthWeightContextBytes);

        engine::models::moss_tts_local::MossGenerator generator(
            assets,
            execution_context,
            kProjectionGraphArenaBytes,
            kProjectionWeightContextBytes,
            backbone,
            depth);
        engine::models::moss_tts_local::MossTextProcessor processor(assets);

        // Optional: build a voice-clone prefix from a reference-codes CSV ([nq, frames],
        // e.g. the encoder's enc_codes.csv) and dump the [seq, 1+nq] input_ids so it can be
        // diffed against the Python processor. Isolates the clone-prefix assembly.
        const std::string clone_ref = arg_value(argc, argv, "--clone-ref", "");
        if (!clone_ref.empty()) {
            const int64_t n_vq = assets->config.num_codebooks;
            std::ifstream csv(clone_ref);
            if (!csv) {
                throw std::runtime_error("cannot open clone-ref csv: " + clone_ref);
            }
            std::vector<std::vector<int32_t>> ref_rows;
            std::string line;
            while (std::getline(csv, line)) {
                if (line.empty()) {
                    continue;
                }
                std::vector<int32_t> row;
                std::stringstream ss(line);
                std::string cell;
                while (std::getline(ss, cell, ',')) {
                    row.push_back(std::stoi(cell));
                }
                ref_rows.push_back(std::move(row));
            }
            ref_rows.resize(static_cast<size_t>(n_vq));  // first n_vq codebooks
            const auto clone_prefix = processor.build_clone_prefix(text, ref_rows);
            const std::string dump = arg_value(argc, argv, "--dump", "clone_prefix_cpp.csv");
            std::ofstream out(dump);
            const size_t seq = clone_prefix.text_tokens.size();
            for (size_t row = 0; row < seq; ++row) {
                out << clone_prefix.text_tokens[row];
                for (int64_t cb = 0; cb < n_vq; ++cb) {
                    out << "," << clone_prefix.audio_codes[row * static_cast<size_t>(n_vq) + static_cast<size_t>(cb)];
                }
                out << "\n";
            }
            std::cout << "clone_prefix rows=" << seq << " -> " << dump << "\n";
            return 0;
        }

        const auto prefix = processor.build_generation_prefix(text);
        std::cout << "prompt_tokens=" << prefix.text_tokens.size() << "\n";
        std::cout << "first_token_ids=[";
        const size_t preview = std::min<size_t>(20, prefix.text_tokens.size());
        for (size_t i = 0; i < preview; ++i) {
            std::cout << (i == 0 ? "" : ",") << prefix.text_tokens[i];
        }
        std::cout << "]\n" << std::flush;

        engine::models::moss_tts_local::MossGenerationOptions options;
        options.do_sample = false;
        options.max_new_frames = max_frames;

        std::cout << "generating up to " << max_frames << " frames (greedy)...\n" << std::flush;
        const auto frames = generator.generate(prefix.text_tokens, prefix.audio_codes, options);

        const int64_t n_vq = assets->config.num_codebooks;
        std::vector<int32_t> min_code(static_cast<size_t>(n_vq), std::numeric_limits<int32_t>::max());
        std::vector<int32_t> max_code(static_cast<size_t>(n_vq), std::numeric_limits<int32_t>::min());
        bool all_in_range = true;
        for (const auto & frame : frames) {
            for (int64_t codebook = 0; codebook < n_vq; ++codebook) {
                const int32_t code = frame[static_cast<size_t>(codebook)];
                min_code[static_cast<size_t>(codebook)] = std::min(min_code[static_cast<size_t>(codebook)], code);
                max_code[static_cast<size_t>(codebook)] = std::max(max_code[static_cast<size_t>(codebook)], code);
                if (code < 0 || code >= 1024) {
                    all_in_range = false;
                }
            }
        }

        std::cout << "\n=== RESULT ===\n";
        std::cout << "ran_without_crash=true\n";
        std::cout << "frames_produced=" << frames.size() << "\n";
        std::cout << "stopped_on_audio_end=" << ((int)frames.size() < max_frames ? "true" : "false")
                  << " (false means it hit the frame cap)\n";
        std::cout << "all_codes_in_[0,1023]=" << (all_in_range ? "true" : "false") << "\n";
        if (!frames.empty()) {
            std::cout << "per_codebook_min_max=\n";
            for (int64_t codebook = 0; codebook < n_vq; ++codebook) {
                std::cout << "  cb" << codebook << ": min=" << min_code[static_cast<size_t>(codebook)]
                          << " max=" << max_code[static_cast<size_t>(codebook)] << "\n";
            }
            std::cout << "first_frame_codes=[";
            for (int64_t codebook = 0; codebook < n_vq; ++codebook) {
                std::cout << (codebook == 0 ? "" : ",") << frames.front()[static_cast<size_t>(codebook)];
            }
            std::cout << "]\n";
        }

        if (!dump_path.empty()) {
            std::ofstream dump(dump_path);
            if (!dump) {
                throw std::runtime_error("failed to open dump file: " + dump_path);
            }
            for (const auto & frame : frames) {
                for (int64_t codebook = 0; codebook < n_vq; ++codebook) {
                    dump << (codebook == 0 ? "" : ",") << frame[static_cast<size_t>(codebook)];
                }
                dump << "\n";
            }
            std::cout << "dumped " << frames.size() << " frames x " << n_vq << " codes to " << dump_path << "\n";
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "moss_tts_local_smoke failed: " << error.what() << "\n";
        return 1;
    }
}
