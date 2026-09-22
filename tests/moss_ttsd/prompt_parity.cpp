/*
 * Prompt parity for MOSS-TTSD.
 *
 * Renders the <user_inst> turn with the audio.cpp builder and compares the encoded
 * rows against a dump from the reference MossTTSDelayProcessor. Every channel is
 * compared, not just the text one.
 *
 * The cases that matter here are the ones MOSS-TTS-v1.5 cannot reach: a speaker
 * list where one speaker brought audio and another did not, which renders two
 * differently shaped lines in the same block, and a prompt with no speaker list
 * at all. A mistake in either branch shifts every token after it, so it shows up
 * as a row-count or token mismatch rather than as bad audio later.
 *
 *   moss_ttsd_prompt_parity --model <dir> --prompt <ref_prompt.json>
 */

#include "engine/community_models/moss_ttsd/prompt.h"
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

// [n_vq][frames], the codec layout ReferenceAudio takes.
engine::models::moss_ttsd::ReferenceAudio read_codes(const json::Value & speaker) {
    engine::models::moss_ttsd::ReferenceAudio audio;
    for (const auto & codebook : speaker.as_array()) {
        std::vector<int32_t> row;
        for (const auto & code : codebook.as_array()) {
            row.push_back(static_cast<int32_t>(code.as_number()));
        }
        audio.frames = static_cast<int64_t>(row.size());
        audio.codes.push_back(std::move(row));
    }
    return audio;
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
            std::cerr << "usage: moss_ttsd_prompt_parity --model <dir> --prompt <ref_prompt.json>\n";
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

        engine::tokenizers::LlamaBpeTokenizerSpec spec;
        spec.tokenizer_config_path = model_dir + "/tokenizer_config.json";
        spec.tokenizer_json_path = model_dir + "/tokenizer.json";
        spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen2;
        const auto tokenizer = engine::tokenizers::load_llama_bpe_tokenizer(spec);

        engine::models::moss_ttsd::PromptFields fields;
        fields.text = json::require_string(reference, "text");
        fields.instruction = optional_field(reference, "instruction");
        fields.language = optional_field(reference, "language");
        fields.quality = optional_field(reference, "quality");
        fields.sound_event = optional_field(reference, "sound_event");
        fields.ambient_sound = optional_field(reference, "ambient_sound");

        // Positional: entry i is speaker [S(i+1)], null where that speaker
        // brought no audio. Absent entirely means no speaker list at all, which
        // is a different prompt from a list of nulls.
        if (const auto * codes = reference.find("reference_codes"); codes != nullptr && codes->is_array()) {
            for (const auto & speaker : codes->as_array()) {
                if (speaker.is_null()) {
                    fields.references.emplace_back();
                    continue;
                }
                auto audio = read_codes(speaker);
                std::cout << "  speaker " << (fields.references.size() + 1) << ": "
                          << audio.codes.size() << " codebooks x " << audio.frames << " frames\n";
                fields.references.emplace_back(std::move(audio));
            }
        }

        // The rendered block is compared before tokenisation: when it differs, the
        // diff is readable, where a token-id mismatch is not.
        const auto rendered = engine::models::moss_ttsd::render_user_inst(fields);
        const auto expected_content = json::require_string(reference, "content");
        if (rendered != expected_content) {
            std::cout << "FAIL: rendered <user_inst> differs from the reference\n--- ours ---\n"
                      << rendered << "\n--- reference ---\n" << expected_content << "\n";
            return 1;
        }
        std::cout << "rendered <user_inst> matches the reference\n";

        // A fixture carrying assistant_codes is a continuation prompt: the model is
        // handed audio to carry on from rather than asked to start fresh.
        const auto * assistant = reference.find("assistant_codes");
        const bool continuation = assistant != nullptr && assistant->is_array();
        if (continuation) {
            config.audio_assistant_gen_slot_token_id =
                json::require_i64(config_json, "audio_assistant_gen_slot_token_id");
        }
        const auto rows = continuation
            ? engine::models::moss_ttsd::build_continuation_prefix(
                  fields, read_codes(*assistant), config, *tokenizer)
            : engine::models::moss_ttsd::build_generation_prefix(fields, config, *tokenizer);
        std::cout << "mode: " << (continuation ? "continuation" : "generation") << "\n";

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
        if (text_mismatch != 0 || code_mismatch != 0) {
            std::cout << "FAIL\n";
            return 1;
        }
        std::cout << "PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "moss_ttsd_prompt_parity failed: " << error.what() << "\n";
        return 1;
    }
}
