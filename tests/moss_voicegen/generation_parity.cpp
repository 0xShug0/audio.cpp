// Generation parity for MOSS-VoiceGenerator: runs the backbone, the 1 + n_vq heads and the
// delay-pattern state machine greedily, and compares the emitted rows against a dump from
// the reference PyTorch generate(). Greedy with the repetition penalty off, so a mismatch
// is a real divergence rather than an RNG difference.
//
//   moss_voicegen_generation_parity --model <dir> --prompt <ref_prompt.json> \
//       --generation <ref_generation.json> [--weight-type bf16] [--threads N]

#include "engine/community_models/moss_voicegen/assets.h"
#include "engine/community_models/moss_voicegen/backbone.h"
#include "engine/community_models/moss_voicegen/delay_decoder.h"
#include "engine/community_models/moss_voicegen/heads.h"
#include "engine/community_models/moss_voicegen/tokenizer_text.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/io/json.h"
#include "engine/models/moss/shared/token_rows.h"

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

engine::assets::TensorStorageType parse_weight_type(const std::string & value) {
    if (value == "f32") {
        return engine::assets::TensorStorageType::F32;
    }
    if (value == "bf16") {
        return engine::assets::TensorStorageType::BF16;
    }
    if (value == "q8_0") {
        return engine::assets::TensorStorageType::Q8_0;
    }
    if (value == "native") {
        return engine::assets::TensorStorageType::Native;
    }
    throw std::runtime_error("unsupported --weight-type: " + value);
}

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
        const std::string prompt_path = arg_value(argc, argv, "--prompt", "");
        const std::string generation_path = arg_value(argc, argv, "--generation", "");
        const auto weight_type = parse_weight_type(arg_value(argc, argv, "--weight-type", "bf16"));
        if (model_dir.empty() || prompt_path.empty() || generation_path.empty()) {
            std::cerr << "usage: moss_voicegen_generation_parity --model <dir> --prompt <json>"
                         " --generation <json>\n";
            return 2;
        }

        const auto prompt_reference = json::parse_file(prompt_path);
        const auto generation_reference = json::parse_file(generation_path);
        const auto expected = require_rows(generation_reference, "generated_rows");

        const auto assets = engine::models::moss_voicegen::load_moss_voicegen_assets(model_dir);
        const auto & config = assets->config;
        const int64_t n_vq = config.num_codebooks;
        const int64_t hidden_size = config.backbone.hidden_size;

        const engine::models::moss_voicegen::MossVoiceGenTextProcessor processor(assets);
        const auto prompt = processor.build_generation_prefix(
            json::require_string(prompt_reference, "text"),
            json::require_string(prompt_reference, "instruction"),
            json::require_string(prompt_reference, "language"));
        const int64_t prompt_rows = static_cast<int64_t>(prompt.text_tokens.size());
        const auto steps = static_cast<int64_t>(expected.size());

        engine::models::moss::AudioCodebookSpec codebook_spec;
        codebook_spec.hidden_size = hidden_size;
        codebook_spec.num_codebooks = n_vq;
        codebook_spec.audio_vocab_size = config.audio_vocab_size + 1;
        codebook_spec.audio_pad_token_id = config.audio_pad_code;
        codebook_spec.tensor_prefix = "emb_ext";
        const engine::models::moss::AudioCodebookEmbeddings codebooks(*assets->model_weights, codebook_spec);

        std::vector<float> prompt_bias(static_cast<size_t>(prompt_rows * hidden_size), 0.0F);
        for (int64_t row = 0; row < prompt_rows; ++row) {
            codebooks.add_bias(
                prompt.audio_codes.data() + static_cast<size_t>(row * n_vq),
                prompt_bias.data() + static_cast<size_t>(row * hidden_size));
        }

        engine::core::BackendConfig backend_config;
        backend_config.type = engine::core::BackendType::Cpu;
        backend_config.device = 0;
        backend_config.threads = std::stoi(arg_value(argc, argv, "--threads", "8"));
        engine::core::ExecutionContext execution_context(backend_config);

        const engine::models::moss_voicegen::MossVoiceGenBackboneRuntime backbone(
            assets, execution_context, 512ull * 1024ull * 1024ull, 8192ull * 1024ull * 1024ull, weight_type);
        const engine::models::moss_voicegen::MossVoiceGenHeadsRuntime heads(
            assets, execution_context, 256ull * 1024ull * 1024ull, 4096ull * 1024ull * 1024ull, weight_type);

        engine::models::moss_voicegen::MossVoiceGenSamplingOptions sampling;
        sampling.do_sample = false;
        sampling.text_temperature = 1.0F;
        sampling.audio_temperature = 1.0F;
        sampling.audio_repetition_penalty = 1.0F;
        engine::models::moss_voicegen::MossVoiceGenDelayDecoder decoder(config, sampling, 0);

        backbone.begin_generation(prompt_rows + steps + 8);
        auto hidden = backbone.prefill(prompt.text_tokens, prompt_bias);

        engine::models::moss_voicegen::MossVoiceGenStepLogits logits;
        std::vector<float> row_bias(static_cast<size_t>(hidden_size), 0.0F);
        int64_t mismatching_rows = 0;
        int64_t produced = 0;
        int64_t first_divergent_row = -1;
        // Keeping the pre-decode logits lets a mismatch report *how close* the call was.
        // Greedy decoding is a step function: a rounding difference of a thousandth flips
        // the argmax and the trajectories part company for good.
        std::vector<std::vector<float>> logits_before;
        for (int64_t step = 0; step < steps; ++step) {
            heads.evaluate(hidden, logits);
            logits_before = logits.audio;
            const auto row = decoder.step(logits);
            ++produced;

            const auto & expected_row = expected[static_cast<size_t>(step)];
            bool row_matches = static_cast<int64_t>(expected_row.size()) == n_vq + 1
                && expected_row[0] == static_cast<int64_t>(row.text_token);
            if (row_matches) {
                for (int64_t codebook = 0; codebook < n_vq; ++codebook) {
                    if (expected_row[static_cast<size_t>(codebook + 1)]
                        != static_cast<int64_t>(row.codes[static_cast<size_t>(codebook)])) {
                        row_matches = false;
                        break;
                    }
                }
            }
            if (!row_matches) {
                if (first_divergent_row < 0) {
                    first_divergent_row = step;
                    for (int64_t codebook = 0; codebook < n_vq; ++codebook) {
                        const auto expected_code = expected_row[static_cast<size_t>(codebook + 1)];
                        const auto actual_code = static_cast<int64_t>(row.codes[static_cast<size_t>(codebook)]);
                        if (expected_code == actual_code) {
                            continue;
                        }
                        const auto & before = logits_before[static_cast<size_t>(codebook)];
                        if (expected_code < static_cast<int64_t>(before.size())
                            && actual_code < static_cast<int64_t>(before.size())) {
                            std::cout << "first divergence at row " << step << " codebook " << codebook
                                      << ": logit(expected " << expected_code << ")="
                                      << before[static_cast<size_t>(expected_code)]
                                      << " vs logit(chosen " << actual_code << ")="
                                      << before[static_cast<size_t>(actual_code)]
                                      << " gap=" << (before[static_cast<size_t>(actual_code)]
                                                     - before[static_cast<size_t>(expected_code)])
                                      << "\n";
                        }
                        break;
                    }
                }
                if (++mismatching_rows <= 5) {
                    std::cerr << "row " << step << " mismatch\n  expected text="
                              << expected_row[0] << " got text=" << row.text_token << "\n";
                    std::cerr << "  expected codes:";
                    for (int64_t codebook = 0; codebook < n_vq; ++codebook) {
                        std::cerr << " " << expected_row[static_cast<size_t>(codebook + 1)];
                    }
                    std::cerr << "\n  actual   codes:";
                    for (int64_t codebook = 0; codebook < n_vq; ++codebook) {
                        std::cerr << " " << row.codes[static_cast<size_t>(codebook)];
                    }
                    std::cerr << "\n";
                }
            }

            if (decoder.stopped()) {
                break;
            }
            std::fill(row_bias.begin(), row_bias.end(), 0.0F);
            codebooks.add_bias(row.codes.data(), row_bias.data());
            hidden = backbone.step(row.text_token, row_bias);
        }

        int64_t codebooks_out = 0;
        int64_t frames_out = 0;
        const auto codes = decoder.extract_audio_codes(codebooks_out, frames_out);
        const int64_t matching_prefix = first_divergent_row < 0 ? produced : first_divergent_row;
        std::cout << "generated rows=" << produced << "/" << steps
                  << " mismatching=" << mismatching_rows
                  << " matching_prefix=" << matching_prefix << "\n";
        std::cout << "de-delayed codes: " << codebooks_out << " codebooks x " << frames_out
                  << " frames (" << codes.size() << " values)\n";
        const int64_t required_prefix =
            std::stoll(arg_value(argc, argv, "--min-matching-prefix", std::to_string(steps)));
        const bool passed = matching_prefix >= required_prefix;
        std::cout << (passed ? "PASS" : "FAIL") << ": required matching prefix " << required_prefix << "\n";
        return passed ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
