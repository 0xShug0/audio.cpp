#include "engine/community_models/tiny_voca/assets.h"

#include "engine/framework/io/config.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>

namespace engine::models::tiny_voca {
namespace {

namespace json = engine::io::json;

TinyVocaConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    const auto & model = root.require("model");
    TinyVocaConfig out;
    out.sample_rate = json::require_i64(model, "sample_rate");
    out.hop = json::require_i64(model, "hop");
    out.f0_min = model.require("f0_min").as_f32();
    out.f0_max = model.require("f0_max").as_f32();
    for (const auto & voice : model.require("speakers").as_array()) {
        out.voices.push_back(voice.as_string());
    }
    out.vocab_size = json::require_i64(model, "vocab_size");
    out.channels = json::require_i64(model, "channels");
    out.heads = json::require_i64(model, "heads");
    out.feedforward = json::require_i64(model, "feedforward");
    out.kernel = json::require_i64(model, "kernel");
    out.encoder_layers = json::require_i64(model, "encoder_layers");
    out.prosody_layers = json::require_i64(model, "prosody_layers");
    out.acoustic_layers = json::require_i64(model, "acoustic_layers");
    out.predictor_channels = json::require_i64(model, "predictor_channels");
    out.synth_fft = json::require_i64(model, "synth_fft");
    out.synth_hop = json::require_i64(model, "synth_hop");
    out.noise_fft = json::require_i64(model, "noise_fft");
    out.noise_hop = json::require_i64(model, "noise_hop");
    out.cep_causal = json::require_i64(model, "cep_causal");
    out.cep_anti = json::require_i64(model, "cep_anti");
    out.noise_cep = json::require_i64(model, "noise_cep");
    out.harmonic_limit = model.require("harmonic_limit").as_f32();
    out.max_tokens = json::require_i64(model, "max_tokens");
    out.max_frames = json::require_i64(model, "max_frames");
    out.pitch_reference_hz = model.require("pitch_reference_hz").as_f32();
    out.phonemizer_version = json::require_i64(root.require("phonemizer"), "version");

    // The phoneme inventory, in id order.
    out.symbols.assign(static_cast<size_t>(out.vocab_size), std::string());
    for (const auto & [symbol, id] : root.require("vocab").as_object()) {
        const int64_t index = id.as_i64();
        if (index < 0 || index >= out.vocab_size) {
            throw std::runtime_error("tiny-voca config has a phoneme id outside the vocabulary");
        }
        out.symbols[static_cast<size_t>(index)] = symbol;
    }

    for (const int64_t value : {out.sample_rate, out.hop, out.vocab_size, out.channels, out.heads, out.feedforward, out.kernel,
                                out.encoder_layers, out.prosody_layers, out.acoustic_layers, out.predictor_channels, out.synth_fft,
                                out.synth_hop, out.noise_fft, out.noise_hop, out.cep_causal, out.cep_anti, out.noise_cep,
                                out.max_tokens, out.max_frames}) {
        engine::io::require_positive(value, "tiny-voca model setting");
    }
    engine::io::require_divisible(out.channels, out.heads, "tiny-voca channels");
    if ((out.channels / out.heads) % 2 != 0 || out.kernel % 2 == 0 || out.voices.empty() ||
        out.synth_fft != 4 * out.synth_hop || out.noise_fft != 4 * out.noise_hop || out.hop % out.synth_hop != 0 ||
        out.hop % out.noise_hop != 0 || out.cep_causal + out.cep_anti >= out.synth_fft / 2 || out.noise_cep >= out.noise_fft / 2 ||
        !(out.harmonic_limit > 0.0F && out.harmonic_limit < 0.5F)) {
        throw std::runtime_error("tiny-voca config uses an unsupported architecture");
    }
    return out;
}

}  // namespace

std::shared_ptr<const TinyVocaAssets> load_tiny_voca_assets(const std::filesystem::path & model_path) {
    auto resources = engine::model_spec::load_resource_bundle_for_family(model_path, "tiny_voca");
    TinyVocaAssets out;
    out.config = parse_config(resources);
    out.weights = resources.open_tensor_source("weights");
    out.g2p_weights = resources.open_tensor_source("g2p_weights");
    out.resources = std::move(resources);
    return std::make_shared<TinyVocaAssets>(std::move(out));
}

}  // namespace engine::models::tiny_voca
