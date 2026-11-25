// Capture the production text processor's clone prefix for independent Python
// comparison. This does not construct or run a parallel generation pipeline.
#include "engine/models/moss/moss_tts_local/assets.h"
#include "engine/models/moss/moss_tts_local/tokenizer_text.h"

#include <fstream>
#include <iostream>
#include <sstream>
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
}

int main(int argc, char ** argv) {
    try {
        const auto model_path = arg_value(argc, argv, "--model", "");
        if (model_path.empty()) throw std::runtime_error("--model is required");
        const auto text = arg_value(argc, argv, "--text", "Hello, this is a test of the MOSS text to speech system.");
        auto assets = engine::models::moss_tts_local::load_moss_tts_local_assets(model_path);
        engine::models::moss_tts_local::MossTextProcessor processor(assets);
        const std::string clone_ref = arg_value(argc, argv, "--clone-ref", "");
        if (clone_ref.empty()) {
            throw std::runtime_error("--clone-ref is required for prompt capture");
        }
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
        if (ref_rows.size() < static_cast<size_t>(n_vq)) {
            throw std::runtime_error("clone-ref has too few codebooks");
        }
        ref_rows.resize(static_cast<size_t>(n_vq));  // first n_vq codebooks
        const auto clone_prefix = processor.build_clone_prefix(text, ref_rows);
        const std::string dump = arg_value(argc, argv, "--dump", "clone_prefix_cpp.csv");
        std::ofstream out(dump);
        if (!out) throw std::runtime_error("cannot open prefix output: " + dump);
        const size_t seq = clone_prefix.text_tokens.size();
        for (size_t row = 0; row < seq; ++row) {
            out << clone_prefix.text_tokens[row];
            for (int64_t cb = 0; cb < n_vq; ++cb) {
                out << "," << clone_prefix.audio_codes[row * static_cast<size_t>(n_vq) + static_cast<size_t>(cb)];
            }
            out << "\n";
        }
        if (!out) throw std::runtime_error("cannot write prefix output: " + dump);
        std::cout << "clone_prefix rows=" << seq << " -> " << dump << "\n";
        return 0;

    } catch (const std::exception & error) {
        std::cerr << "moss_tts_local_prompt_probe: " << error.what() << '\n';
        return 1;
    }
}
