/*
 * Prompt parity for MOSS-TTS-v1.5.
 *
 * Renders the <user_inst> turn with the audio.cpp builder and compares the encoded
 * rows against a dump from the reference MossTTSDelayProcessor. Every channel is
 * compared, not just the text one: an instruction-only prompt should be audio-pad
 * throughout, and a cloning prompt carries the reference's delay-patterned codes,
 * so a mistake in either the template or the delay layout shows up here rather
 * than as bad audio later.
 *
 *   moss_tts_v15_prompt_parity --model <dir> --prompt <ref_prompt.json>
 */

#include "engine/community_models/moss_tts_v15/prompt.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/json.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <exception>
#include <iostream>
#include <optional>
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

std::optional<std::string> optional_field(const json::Value & object, const std::string & key) {
    const auto * value = object.find(key);
    if (value == nullptr || value->is_null()) {
        return std::nullopt;
    }
    return value->as_string();
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::string model_dir = arg_value(argc, argv, "--model", "");
        const std::string prompt_path = arg_value(argc, argv, "--prompt", "");
        if (model_dir.empty() || prompt_path.empty()) {
            std::cerr << "usage: moss_tts_v15_prompt_parity --model <dir> --prompt <ref_prompt.json>\n";
            return 2;
        }

        const auto reference = json::parse_file(prompt_path);
        const auto config_json = json::parse_file(model_dir + "/config.json");
        const auto & language_config = config_json.require("language_config");

        engine::decoders::MossTtsDelayConfig config;
        config.backbone.vocab_size = json::require_i64(language_config, "vocab_size");
        config.num_codebooks = json::require_i64(config_json, "n_vq");
        config.audio_vocab_size = json::require_i64(config_json, "audio_vocab_size");
        config.audio_pad_code = json::require_i64(config_json, "audio_pad_code");
        config.audio_start_token_id = json::require_i64(config_json, "audio_start_token_id");
        config.audio_end_token_id = json::require_i64(config_json, "audio_end_token_id");
        config.audio_user_slot_token_id = json::require_i64(config_json, "audio_user_slot_token_id");

        // Same spec moss_voicegen uses: the checkpoint ships tokenizer.json and
        // merges.txt but no vocab.json, so vocab and merge ranks come from
        // tokenizer.json rather than from a vocab/merges pair.
        engine::tokenizers::LlamaBpeTokenizerSpec spec;
        spec.tokenizer_config_path = model_dir + "/tokenizer_config.json";
        spec.tokenizer_json_path = model_dir + "/tokenizer.json";
        spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen2;
        const auto tokenizer = engine::tokenizers::load_llama_bpe_tokenizer(spec);

        engine::models::moss_tts_v15::PromptFields fields;
        fields.text = json::require_string(reference, "text");
        fields.instruction = optional_field(reference, "instruction");
        fields.language = optional_field(reference, "language");
        fields.tokens = optional_field(reference, "tokens");

        // A cloning fixture carries the reference's codec codes, [n_vq][frames].
        if (const auto * codes = reference.find("reference_codes"); codes != nullptr && codes->is_array()) {
            engine::models::moss_tts_v15::ReferenceAudio audio;
            for (const auto & codebook : codes->as_array()) {
                std::vector<int32_t> row;
                for (const auto & code : codebook.as_array()) {
                    row.push_back(static_cast<int32_t>(code.as_number()));
                }
                audio.frames = static_cast<int64_t>(row.size());
                audio.codes.push_back(std::move(row));
            }
            std::cout << "reference: " << audio.codes.size() << " codebooks x " << audio.frames
                      << " frames\n";
            fields.references.push_back(std::move(audio));
        }

        // The rendered block is compared before tokenisation: when it differs, the
        // diff is readable, where a token-id mismatch is not.
        const auto rendered = engine::models::moss_tts_v15::render_user_inst(fields);
        const auto expected_content = json::require_string(reference, "content");
        if (rendered != expected_content) {
            std::cout << "FAIL: rendered <user_inst> differs from the reference\n--- ours ---\n"
                      << rendered << "\n--- reference ---\n" << expected_content << "\n";
            return 1;
        }
        std::cout << "rendered <user_inst> matches the reference\n";

        const auto rows = engine::models::moss_tts_v15::build_generation_prefix(fields, config, *tokenizer);

        const auto & expected_rows = reference.require("input_ids").as_array();
        const auto steps = static_cast<int64_t>(rows.text_tokens.size());
        std::cout << "rows: ours=" << steps << " reference=" << expected_rows.size() << "\n";
        if (steps != static_cast<int64_t>(expected_rows.size())) {
            std::cout << "FAIL: row count differs\n";
            return 1;
        }

        int64_t text_mismatch = 0;
        int64_t code_mismatch = 0;
        for (int64_t row = 0; row < steps; ++row) {
            const auto & channels = expected_rows[static_cast<size_t>(row)].as_array();
            const auto expected_text = static_cast<int32_t>(channels[0].as_number());
            if (rows.text_tokens[static_cast<size_t>(row)] != expected_text) {
                if (text_mismatch < 4) {
                    std::cout << "  row " << row << " text: ours="
                              << rows.text_tokens[static_cast<size_t>(row)]
                              << " reference=" << expected_text << "\n";
                }
                ++text_mismatch;
            }
            for (int64_t cb = 0; cb < config.num_codebooks; ++cb) {
                const auto expected_code = static_cast<int32_t>(channels[static_cast<size_t>(cb + 1)].as_number());
                const auto ours = rows.audio_codes[static_cast<size_t>(row * config.num_codebooks + cb)];
                if (ours != expected_code) {
                    if (code_mismatch < 4) {
                        std::cout << "  row " << row << " cb" << cb << ": ours=" << ours
                                  << " reference=" << expected_code << "\n";
                    }
                    ++code_mismatch;
                }
            }
        }

        std::cout << "\n=== PROMPT PARITY: text mismatches=" << text_mismatch
                  << " code mismatches=" << code_mismatch << " ===\n";
        const bool ok = text_mismatch == 0 && code_mismatch == 0;
        std::cout << (ok ? "PASS" : "FAIL") << "\n";
        return ok ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << "moss_tts_v15_prompt_parity failed: " << error.what() << "\n";
        return 1;
    }
}
