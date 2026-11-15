#include "engine/models/m5_tts/assets.h"

#include "engine/framework/io/json.h"

#include <stdexcept>

namespace engine::models::m5_tts {
namespace {

DecoderConfig decoder_config(const io::json::Value & value) {
    DecoderConfig result;
    result.layers = value.require("layers").as_i64();
    result.hidden = value.require("hidden_size").as_i64();
    result.intermediate = value.require("intermediate_size").as_i64();
    result.heads = value.require("num_attention_heads").as_i64();
    result.kv_heads = value.require("num_key_value_heads").as_i64();
    result.head_dim = value.require("head_dim").as_i64();
    result.norm_eps = value.require("rms_norm_eps").as_f32();
    result.rope_theta = value.require("rope_theta").as_f32();
    if (result.layers <= 0 || result.hidden <= 0 || result.intermediate <= 0 ||
        result.heads <= 0 || result.kv_heads <= 0 || result.head_dim <= 0 ||
        result.hidden != result.heads * result.head_dim || result.heads % result.kv_heads != 0) {
        throw std::runtime_error("M5-TTS decoder dimensions are inconsistent");
    }
    return result;
}

}  // namespace

std::shared_ptr<const ModelAssets> load_assets(const std::filesystem::path & path) {
    auto result = std::make_shared<ModelAssets>();
    result->resources = model_spec::load_resource_bundle_for_family(path, "m5_tts");
    const auto config = result->resources.parse_json("config");
    result->variant = config.require("variant").as_string();
    result->backbone = decoder_config(config.require("backbone"));
    result->decoder = decoder_config(config.require("decoder"));
    result->text_vocabulary = config.require("text_vocab_size").as_i64();
    result->audio_vocabulary = config.require("audio_vocab_size").as_i64();
    result->codebooks = config.require("audio_num_codebooks").as_i64();
    result->context_length = config.require("max_position_embeddings").as_i64();
    const auto conditioning = result->resources.parse_json("conditioning");
    result->bos_token_id = conditioning.require("bos_token_id").as_i64();
    result->eos_token_id = conditioning.require("eos_token_id").as_i64();
    result->codec_config = codecs::parse_qwen3_audio_tokenizer_decoder_config(
        result->resources.parse_json("codec_config"));
    if ((result->variant != "voice_clone" && result->variant != "custom_voice") ||
        result->text_vocabulary <= 0 || result->audio_vocabulary != 2051 ||
        result->codebooks != 16 || result->codec_config.num_quantizers != result->codebooks ||
        result->backbone.hidden != result->decoder.hidden || result->context_length <= 0 ||
        result->bos_token_id < 0 || result->bos_token_id >= result->text_vocabulary ||
        result->eos_token_id < 0 || result->eos_token_id >= result->text_vocabulary) {
        throw std::runtime_error("M5-TTS package has an incompatible model/codec contract");
    }
    result->model = result->resources.open_tensor_source("model_weights");
    result->codec = result->resources.open_tensor_source("codec_weights");
    return result;
}

}  // namespace engine::models::m5_tts
