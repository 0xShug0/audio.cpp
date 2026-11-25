#include "engine/community_models/kale_tts/assets.h"

#include "engine/framework/io/config.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>

namespace engine::models::kale_tts {
namespace {

namespace json = engine::io::json;

KaleConfig parse_config(const engine::io::json::Value & root) {
    const auto & model = root.require("model");
    KaleConfig out;
    out.sample_rate = json::require_i64(model, "sample_rate");
    out.ettin = model.require("text_side").as_string() == "ettin";
    out.n_token = json::require_i64(model, "n_token");
    out.hidden_dim = json::require_i64(model, "hidden_dim");
    out.style_dim = json::require_i64(model, "style_dim");
    out.n_layer = json::require_i64(model, "n_layer");
    out.max_dur = json::require_i64(model, "max_dur");
    out.decoder_width = json::require_i64(model, "decoder_width");
    out.decoder_up_channels = json::require_i64(model, "decoder_up_channels");
    out.voice_sound_dim = json::require_i64(model, "voice_sound_dim");
    out.head_dim = json::require_i64(model, "head_dim");
    out.head_blocks = json::require_i64(model, "head_blocks");
    out.head_points = json::require_i64(model, "head_points");
    out.head_burst_points = json::require_i64(model, "head_burst_points");
    out.head_harmonics = json::require_i64(model, "head_harmonics");
    out.head_mod = json::require_i64(model, "head_mod");
    out.max_tokens = json::require_i64(model, "max_tokens");
    if (out.ettin) {
        const auto & e = model.require("ettin");
        out.ettin_hidden = json::require_i64(e, "hidden_size");
        out.ettin_intermediate = json::require_i64(e, "intermediate_size");
        out.ettin_layers = json::require_i64(e, "num_hidden_layers");
        out.ettin_heads = json::require_i64(e, "num_attention_heads");
        out.ettin_global_every = json::require_i64(e, "global_attn_every_n_layers");
        out.ettin_local_window = json::require_i64(e, "local_attention");
        out.ettin_rope_theta = e.require("rope_theta").as_f32();
        const auto & s = model.require("student");
        out.student_channels = json::require_i64(s, "channels");
        out.student_heads = json::require_i64(s, "heads");
        out.student_conv_blocks = json::require_i64(s, "conv_blocks");
        out.student_transformer_blocks = json::require_i64(s, "transformer_blocks");
        engine::io::require_divisible(out.ettin_hidden, out.ettin_heads, "kale-tts Ettin hidden size");
        engine::io::require_divisible(out.student_channels, out.student_heads, "kale-tts student channels");
    } else {
        const auto & a = model.require("albert");
        out.albert_hidden = json::require_i64(a, "hidden_size");
        out.albert_embedding = json::require_i64(a, "embedding_size");
        out.albert_heads = json::require_i64(a, "num_attention_heads");
        out.albert_intermediate = json::require_i64(a, "intermediate_size");
        out.albert_layers = json::require_i64(a, "num_hidden_layers");
        out.albert_positions = json::require_i64(a, "max_position_embeddings");
        engine::io::require_divisible(out.albert_hidden, out.albert_heads, "kale-tts ALBERT hidden size");
    }
    out.kokoro_native = root.require("phonemes").as_string() == "kokoro-native";
    for (const auto & [symbol, id] : root.require("vocab").as_object()) {
        out.vocab.emplace(symbol, static_cast<int32_t>(id.as_i64()));
    }
    for (const int64_t value : {out.sample_rate, out.n_token, out.hidden_dim, out.style_dim, out.n_layer, out.max_dur, out.decoder_width,
                                out.decoder_up_channels, out.voice_sound_dim, out.head_dim, out.head_blocks, out.head_points,
                                out.head_burst_points, out.head_harmonics, out.max_tokens}) {
        engine::io::require_positive(value, "kale-tts model setting");
    }
    if (out.sample_rate != 24000 || out.hidden_dim % 2 != 0) {
        throw std::runtime_error("kale-tts config uses an unsupported architecture");
    }
    return out;
}

float scalar(const assets::TensorSource & source, const std::string & name, float fallback) {
    for (const auto & tensor : source.tensors()) {
        if (tensor.name == name) {
            return source.require_f32(name).at(0);
        }
    }
    return fallback;
}

std::vector<KaleVoice> load_voices(const engine::io::json::Value & root, const KaleConfig & config, const assets::TensorSource & source) {
    std::vector<KaleVoice> out;
    for (const auto & entry : root.require("voices").as_array()) {
        KaleVoice voice;
        voice.id = entry.require("id").as_string();
        std::string name = entry.require("name").as_string();
        for (char & c : name) {
            c = (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
        }
        voice.name = name;
        voice.accent = json::optional_string(entry, "accent", "en-us");
        voice.style = source.require_f32(voice.id + ".style");
        voice.style_rows = static_cast<int64_t>(voice.style.size()) / 256;
        if (voice.style_rows < 1 || static_cast<int64_t>(voice.style.size()) != voice.style_rows * 256) {
            throw std::runtime_error("kale-tts voice " + voice.name + " has a style table of the wrong size");
        }
        voice.sound = source.require_f32(voice.id + ".sound", {config.voice_sound_dim});
        if (!config.ettin) {
            voice.prosody = source.require_f32(voice.id + ".prosody", {config.style_dim});
        }
        voice.pitch_floor = scalar(source, voice.id + ".pitch_floor", 0.0F);
        voice.clamp_tail = scalar(source, voice.id + ".clamp_tail", 36.0F);
        voice.clamp_threshold = scalar(source, voice.id + ".clamp_thr", 50.0F);
        voice.unvoice_hz = scalar(source, voice.id + ".unvoice_hz", 80.0F);
        out.push_back(std::move(voice));
    }
    if (out.empty()) {
        throw std::runtime_error("kale-tts config has no voices");
    }
    return out;
}

}  // namespace

std::shared_ptr<const KaleAssets> load_kale_assets(const std::filesystem::path & model_path) {
    auto resources = engine::model_spec::load_resource_bundle_for_family(model_path, "kale_tts");
    KaleAssets out;
    const auto root = resources.parse_json("config");
    out.config = parse_config(root);
    out.weights = resources.open_tensor_source("weights");
    const auto voices = resources.open_tensor_source("voices");
    out.voices = load_voices(root, out.config, *voices);
    out.resources = std::move(resources);
    return std::make_shared<KaleAssets>(std::move(out));
}

}  // namespace engine::models::kale_tts
