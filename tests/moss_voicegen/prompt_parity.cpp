// Prompt parity for MOSS-VoiceGenerator: builds the <user_inst> voice-design prompt with
// the audio.cpp text processor and compares it row-for-row against a reference dump taken
// from the checkpoint's own MossTTSDelayProcessor (see tools/community_models/
// moss_voicegen_reference_prompt.py). No weights are loaded beyond the tokenizer files.
//
//   moss_voicegen_prompt_parity --model <model-dir> --reference <ref_prompt.json>

#include "engine/community_models/moss_voicegen/assets.h"
#include "engine/community_models/moss_voicegen/tokenizer_text.h"
#include "engine/framework/io/json.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace json = engine::io::json;

std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

// The reference dump stores input_ids as rows of [text_token, code_0 .. code_n_vq-1].
std::vector<std::vector<int64_t>> require_rows(const json::Value & object, const std::string & key) {
    std::vector<std::vector<int64_t>> rows;
    for (const auto & row : object.require(key).as_array()) {
        std::vector<int64_t> values;
        for (const auto & value : row.as_array()) {
            values.push_back(value.as_i64());
        }
        rows.push_back(std::move(values));
    }
    return rows;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::string model_dir = arg_value(argc, argv, "--model", "");
        const std::string reference_path = arg_value(argc, argv, "--reference", "");
        if (model_dir.empty() || reference_path.empty()) {
            std::cerr << "usage: moss_voicegen_prompt_parity --model <dir> --reference <ref_prompt.json>\n";
            return 2;
        }

        const auto reference = json::parse_file(reference_path);
        const auto text = json::require_string(reference, "text");
        const auto instruction = json::require_string(reference, "instruction");
        const auto language = json::require_string(reference, "language");
        const auto expected_rows = require_rows(reference, "input_ids");

        const auto assets = engine::models::moss_voicegen::load_moss_voicegen_assets(model_dir);
        const engine::models::moss_voicegen::MossVoiceGenTextProcessor processor(assets);
        const auto rows = processor.build_generation_prefix(text, instruction, language);

        const int64_t n_vq = assets->config.num_codebooks;
        const auto actual_rows = static_cast<int64_t>(rows.text_tokens.size());
        const auto reference_row_count = static_cast<int64_t>(expected_rows.size());
        std::cout << "reference rows=" << reference_row_count << " width=" << (n_vq + 1) << "\n";
        std::cout << "actual    rows=" << actual_rows << "\n";
        if (actual_rows != reference_row_count) {
            const auto shared = std::min(actual_rows, reference_row_count);
            for (int64_t row = 0; row < shared; ++row) {
                const auto actual_text = static_cast<int64_t>(rows.text_tokens[static_cast<size_t>(row)]);
                if (actual_text != expected_rows[static_cast<size_t>(row)][0]) {
                    std::cerr << "first text divergence at row " << row << ": expected "
                              << expected_rows[static_cast<size_t>(row)][0] << ", got " << actual_text << "\n";
                    const auto from = row > 3 ? row - 3 : 0;
                    for (int64_t i = from; i < std::min(shared, row + 4); ++i) {
                        std::cerr << "  row " << i << ": expected " << expected_rows[static_cast<size_t>(i)][0]
                                  << ", got " << rows.text_tokens[static_cast<size_t>(i)] << "\n";
                    }
                    break;
                }
            }
            std::cerr << "FAIL: row count mismatch\n";
            return 1;
        }

        int64_t mismatches = 0;
        for (int64_t row = 0; row < actual_rows; ++row) {
            const auto & expected = expected_rows[static_cast<size_t>(row)];
            if (static_cast<int64_t>(expected.size()) != n_vq + 1) {
                std::cerr << "FAIL: reference row " << row << " has width " << expected.size() << "\n";
                return 1;
            }
            const auto actual_text = static_cast<int64_t>(rows.text_tokens[static_cast<size_t>(row)]);
            if (actual_text != expected[0]) {
                if (++mismatches <= 10) {
                    std::cerr << "text token mismatch at row " << row
                              << ": expected " << expected[0] << ", got " << actual_text << "\n";
                }
                continue;
            }
            for (int64_t codebook = 0; codebook < n_vq; ++codebook) {
                const auto index = static_cast<size_t>(row * n_vq + codebook);
                const auto actual_code = static_cast<int64_t>(rows.audio_codes[index]);
                if (actual_code != expected[static_cast<size_t>(codebook + 1)]) {
                    if (++mismatches <= 10) {
                        std::cerr << "audio code mismatch at row " << row << " codebook " << codebook
                                  << ": expected " << expected[static_cast<size_t>(codebook + 1)]
                                  << ", got " << actual_code << "\n";
                    }
                }
            }
        }

        if (mismatches != 0) {
            std::cerr << "FAIL: " << mismatches << " mismatching values\n";
            return 1;
        }
        std::cout << "PASS: prompt matches the reference processor token for token\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
