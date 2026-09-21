/*
 * Backbone parity for the moss_tts_delay family at n_vq 32 -- MOSS-TTS-v1.5, the 8B
 * sibling of MOSS-VoiceGenerator. Same architecture, same tensor names, four times the
 * codebooks and a Qwen3-8B backbone in place of 1.7B.
 *
 * Unlike the moss_voicegen test this takes the prompt rows straight from the reference
 * dump rather than building them, because the v1.5 prompt builder does not exist yet.
 * That is deliberate: it isolates the backbone -- the geometry, the summed multi-codebook
 * embedding, rope, and the KV cache -- from prompt construction, which gets its own
 * parity test later. It is the same split the moss_voicegen tests already use between
 * prompt_parity and backbone_parity.
 *
 * Assets are assembled by hand from the raw checkpoint: v1.5's safetensors carry the
 * same tensor names the backbone already looks up (language_model.*, emb_ext.N.weight),
 * so no conversion and no model spec are needed to run this.
 *
 *   moss_tts_delay_backbone_parity --weights <model.safetensors.index.json> \
 *       --config <config.json> --prompt <ref_prompt.json> --hidden <ref_hidden.json> \
 *       [--weight-type bf16] [--tolerance 0.02]
 */

#include "engine/community_models/moss_voicegen/assets.h"
#include "engine/community_models/moss_voicegen/backbone.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/json.h"
#include "engine/framework/modules/multi_codebook_embedding.h"

#include <algorithm>
#include <cmath>
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
    throw std::runtime_error("unsupported --weight-type: " + value);
}

std::vector<float> require_floats(const json::Value & object, const std::string & key) {
    std::vector<float> out;
    for (const auto & value : object.require(key).as_array()) {
        out.push_back(static_cast<float>(value.as_number()));
    }
    return out;
}

struct Deviation {
    float max_abs = 0.0F;
    float mean_abs = 0.0F;
    float max_relative = 0.0F;
    int64_t argmax = -1;
    // NaN compares false against everything, so a run that produces NaN would otherwise
    // leave max_abs at zero and report a pass. f16 weights do exactly that on this model.
    int64_t non_finite = 0;
};

Deviation compare(const std::vector<float> & actual, const std::vector<float> & expected) {
    Deviation out;
    double total = 0.0;
    const auto count = std::min(actual.size(), expected.size());
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(actual[i])) {
            ++out.non_finite;
            continue;
        }
        const float delta = std::fabs(actual[i] - expected[i]);
        total += delta;
        if (delta > out.max_abs) {
            out.max_abs = delta;
            out.argmax = static_cast<int64_t>(i);
        }
    }
    out.mean_abs = count == 0 ? 0.0F : static_cast<float>(total / static_cast<double>(count));
    float scale = 0.0F;
    for (size_t i = 0; i < count; ++i) {
        scale = std::max(scale, std::fabs(expected[i]));
    }
    out.max_relative = scale > 0.0F ? out.max_abs / scale : out.max_abs;
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::string weights_path = arg_value(argc, argv, "--weights", "");
        const std::string config_path = arg_value(argc, argv, "--config", "");
        const std::string prompt_path = arg_value(argc, argv, "--prompt", "");
        const std::string hidden_path = arg_value(argc, argv, "--hidden", "");
        const auto weight_type = parse_weight_type(arg_value(argc, argv, "--weight-type", "bf16"));
        const float tolerance = std::stof(arg_value(argc, argv, "--tolerance", "0.001"));
        if (weights_path.empty() || config_path.empty() || prompt_path.empty() || hidden_path.empty()) {
            std::cerr << "usage: moss_tts_delay_backbone_parity --weights <index.json> "
                         "--config <config.json> --prompt <json> --hidden <json>\n";
            return 2;
        }

        const auto prompt_reference = json::parse_file(prompt_path);
        const auto hidden_reference = json::parse_file(hidden_path);
        const auto expected_last = require_floats(hidden_reference, "last_hidden");

        // Read the checkpoint's own config rather than a model spec: this test runs
        // against the unconverted HF download.
        const auto config_json = json::parse_file(config_path);
        const auto & language_config = config_json.require("language_config");
        engine::models::moss_voicegen::MossVoiceGenConfig config;
        config.backbone.hidden_size = json::require_i64(language_config, "hidden_size");
        config.backbone.intermediate_size = json::require_i64(language_config, "intermediate_size");
        config.backbone.num_hidden_layers = json::require_i64(language_config, "num_hidden_layers");
        config.backbone.num_attention_heads = json::require_i64(language_config, "num_attention_heads");
        config.backbone.num_key_value_heads = json::require_i64(language_config, "num_key_value_heads");
        config.backbone.head_dim = json::require_i64(language_config, "head_dim");
        config.backbone.max_position_embeddings =
            json::require_i64(language_config, "max_position_embeddings");
        config.backbone.vocab_size = json::require_i64(language_config, "vocab_size");
        config.backbone.rms_norm_eps =
            json::optional_f32(language_config, "rms_norm_eps", 1.0e-6F);
        config.backbone.rope_theta =
            json::optional_f32(language_config, "rope_theta", 1000000.0F);
        config.backbone.tie_word_embeddings =
            json::optional_bool(language_config, "tie_word_embeddings", true);
        config.num_codebooks = json::require_i64(config_json, "n_vq");
        config.audio_vocab_size = json::require_i64(config_json, "audio_vocab_size");
        config.audio_pad_code = json::require_i64(config_json, "audio_pad_code");

        auto assets_owned = std::make_shared<engine::models::moss_voicegen::MossVoiceGenAssets>();
        assets_owned->config = config;
        assets_owned->model_weights = engine::assets::open_tensor_source(weights_path);
        const std::shared_ptr<const engine::models::moss_voicegen::MossVoiceGenAssets> assets =
            assets_owned;

        // input_ids is [rows][1 + n_vq]: channel 0 is the text id, the rest are codes.
        const auto & id_rows = prompt_reference.require("input_ids").as_array();
        const int64_t steps = static_cast<int64_t>(id_rows.size());
        struct { std::vector<int32_t> text_tokens; std::vector<int32_t> audio_codes; } rows;
        rows.text_tokens.reserve(static_cast<size_t>(steps));
        rows.audio_codes.reserve(static_cast<size_t>(steps * config.num_codebooks));
        for (const auto & row : id_rows) {
            const auto & channels = row.as_array();
            if (static_cast<int64_t>(channels.size()) != config.num_codebooks + 1) {
                std::cerr << "FAIL: fixture row has " << channels.size() << " channels, expected "
                          << (config.num_codebooks + 1) << "\n";
                return 1;
            }
            rows.text_tokens.push_back(static_cast<int32_t>(channels[0].as_number()));
            for (size_t c = 1; c < channels.size(); ++c) {
                rows.audio_codes.push_back(static_cast<int32_t>(channels[c].as_number()));
            }
        }
        std::cout << "prompt rows=" << steps << " codebooks=" << config.num_codebooks
                  << " hidden=" << config.backbone.hidden_size << "\n";
        const int64_t hidden_size = config.backbone.hidden_size;
        if (steps != json::require_i64(hidden_reference, "rows")) {
            std::cerr << "FAIL: prompt has " << steps << " rows, reference hidden dump has "
                      << json::require_i64(hidden_reference, "rows") << "\n";
            return 1;
        }

        // Text embedding happens inside the graph; the audio codebooks are summed here and
        // handed in as a per-position bias. The shared helper skips pad codes, which is
        // exact for this checkpoint: every emb_ext table's pad row (index audio_vocab_size)
        // is all zeros, so adding it the way the reference does changes nothing.
        engine::modules::MultiCodebookEmbeddingSpec codebook_spec;
        codebook_spec.hidden_size = hidden_size;
        codebook_spec.num_codebooks = config.num_codebooks;
        codebook_spec.vocab_size = config.audio_vocab_size + 1;
        codebook_spec.pad_token_id = config.audio_pad_code;
        codebook_spec.tensor_prefix = "emb_ext";
        const engine::modules::MultiCodebookEmbedding codebooks(*assets->model_weights, codebook_spec);

        std::vector<float> audio_bias(static_cast<size_t>(steps * hidden_size), 0.0F);
        for (int64_t row = 0; row < steps; ++row) {
            codebooks.add_bias(
                rows.audio_codes.data() + static_cast<size_t>(row * config.num_codebooks),
                audio_bias.data() + static_cast<size_t>(row * hidden_size));
        }

        engine::core::BackendConfig backend_config;
        backend_config.type = engine::core::BackendType::Cpu;
        backend_config.device = 0;
        backend_config.threads = std::stoi(arg_value(argc, argv, "--threads", "8"));
        engine::core::ExecutionContext execution_context(backend_config);
        const engine::models::moss_voicegen::MossVoiceGenBackboneRuntime backbone(
            assets, execution_context, 512ull * 1024ull * 1024ull, 8192ull * 1024ull * 1024ull, weight_type);

        backbone.begin_generation(steps + 8);
        const auto prefill_hidden = backbone.prefill(rows.text_tokens, audio_bias);
        const auto prefill_deviation = compare(prefill_hidden, expected_last);
        std::cout << "prefill  last-hidden: max_abs=" << prefill_deviation.max_abs
                  << " mean_abs=" << prefill_deviation.mean_abs
                  << " max_rel=" << prefill_deviation.max_relative
                  << " non_finite=" << prefill_deviation.non_finite
                  << " at=" << prefill_deviation.argmax << "\n";

        // Same final position, reached through the cached step path instead.
        backbone.begin_generation(steps + 8);
        const std::vector<int32_t> head(rows.text_tokens.begin(), rows.text_tokens.end() - 1);
        const std::vector<float> head_bias(
            audio_bias.begin(), audio_bias.begin() + static_cast<int64_t>((steps - 1) * hidden_size));
        backbone.prefill(head, head_bias);
        const std::vector<float> last_row_bias(
            audio_bias.begin() + static_cast<int64_t>((steps - 1) * hidden_size), audio_bias.end());
        const auto step_hidden = backbone.step(rows.text_tokens.back(), last_row_bias);
        const auto step_deviation = compare(step_hidden, expected_last);
        std::cout << "stepwise last-hidden: max_abs=" << step_deviation.max_abs
                  << " mean_abs=" << step_deviation.mean_abs
                  << " max_rel=" << step_deviation.max_relative
                  << " non_finite=" << step_deviation.non_finite
                  << " at=" << step_deviation.argmax << "\n";

        const bool passed =
            prefill_deviation.non_finite == 0 && step_deviation.non_finite == 0
            && prefill_deviation.max_relative <= tolerance && step_deviation.max_relative <= tolerance;
        std::cout << (passed ? "PASS" : "FAIL") << ": relative tolerance " << tolerance << "\n";
        return passed ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
