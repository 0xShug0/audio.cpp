#include "engine/models/hviske_asr_v6/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <utility>

namespace engine::models::hviske_asr_v6 {

std::shared_ptr<const HviskeV6Assets> load_hviske_v6_assets(const std::filesystem::path & path) {
    auto result = std::make_shared<HviskeV6Assets>();
    result->resources = model_spec::load_resource_bundle_for_family(path, "hviske_asr_v6");
    const auto root = result->resources.parse_json("config");
    if (root.require("model_type").as_string() != "whisper_qwen") {
        throw std::runtime_error("Hviske v6 requires a whisper_qwen checkpoint");
    }
    auto & config = result->config;
    const auto & whisper = root.require("whisper");
    auto & enc = config.encoder;
    enc.mel_bins = whisper.require("num_mel_bins").as_i64();
    enc.hidden_size = whisper.require("d_model").as_i64();
    enc.intermediate_size = whisper.require("encoder_ffn_dim").as_i64();
    enc.layers = whisper.require("encoder_layers").as_i64();
    enc.heads = whisper.require("encoder_attention_heads").as_i64();
    enc.rotary_dim = whisper.require("encoder_rotary_dim").as_i64();
    enc.rope_theta = io::json::require_f32(root, "rope_theta");
    const auto & qwen = root.require("qwen");
    auto & dec = config.decoder;
    dec.vocab_size = qwen.require("vocab_size").as_i64();
    dec.hidden_size = qwen.require("hidden_size").as_i64();
    dec.intermediate_size = qwen.require("intermediate_size").as_i64();
    dec.layers = qwen.require("num_hidden_layers").as_i64();
    dec.heads = qwen.require("num_attention_heads").as_i64();
    dec.kv_heads = qwen.require("num_key_value_heads").as_i64();
    dec.head_dim = qwen.require("head_dim").as_i64();
    dec.max_positions = qwen.require("max_position_embeddings").as_i64();
    dec.rms_norm_eps = io::json::require_f32(qwen, "rms_norm_eps");
    dec.rope_theta = io::json::require_f32(qwen, "rope_theta");
    config.frame_stack = root.require("stack").as_i64();
    config.bos_token_id = static_cast<int32_t>(root.require("bos_token_id").as_i64());
    config.eos_token_id = static_cast<int32_t>(root.require("eos_token_id").as_i64());
    config.pad_token_id = static_cast<int32_t>(root.require("pad_token_id").as_i64());
    const auto & style = root.require("style_tokens");
    config.cased_token_id = static_cast<int32_t>(style.require("<|cased|>").as_i64());
    config.nocase_token_id = static_cast<int32_t>(style.require("<|nocase|>").as_i64());
    config.punctuation_token_id = static_cast<int32_t>(style.require("<|punc|>").as_i64());
    config.no_punctuation_token_id = static_cast<int32_t>(style.require("<|nopunc|>").as_i64());
    result->weights = result->resources.open_tensor_source("weights");
    return result;
}

}  // namespace engine::models::hviske_asr_v6
