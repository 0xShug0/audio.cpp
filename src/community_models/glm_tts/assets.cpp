#include "engine/community_models/glm_tts/assets.h"

#include "engine/framework/io/config.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"
#include "engine/community_models/glm_tts/tokenizer_text.h"

#include <stdexcept>

namespace engine::models::glm_tts {
namespace json = engine::io::json;
namespace {

GlmTTSLlamaConfig parse_llama(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("llama_config");
    if (json::require_string(root, "model_type") != "llama") {
        throw std::runtime_error(
            "GLM-TTS expects a LlamaForCausalLM checkpoint");
    }

    GlmTTSLlamaConfig out;
    out.bos_token_id = json::require_i64(root, "bos_token_id");
    out.eos_token_id = json::require_i64(root, "eos_token_id");
    out.hidden_size = json::require_i64(root, "hidden_size");
    out.intermediate_size = json::require_i64(root, "intermediate_size");
    out.max_position_embeddings =
        json::require_i64(root, "max_position_embeddings");
    out.num_attention_heads = json::require_i64(root, "num_attention_heads");
    out.num_hidden_layers = json::require_i64(root, "num_hidden_layers");
    out.num_key_value_heads = json::require_i64(root, "num_key_value_heads");
    out.head_dim = json::optional_i64(
        root, "head_dim", out.hidden_size / out.num_attention_heads);
    out.vocab_size = json::require_i64(root, "vocab_size");
    out.rms_norm_eps =
        json::optional_f32(root, "rms_norm_eps", out.rms_norm_eps);
    out.rope_theta = json::optional_f32(root, "rope_theta", out.rope_theta);

    engine::io::require_positive(out.hidden_size, "GLM-TTS hidden_size");
    engine::io::require_positive(
        out.intermediate_size, "GLM-TTS intermediate_size");
    engine::io::require_positive(
        out.num_hidden_layers, "GLM-TTS layer count");
    engine::io::require_positive(
        out.num_attention_heads, "GLM-TTS attention heads");
    engine::io::require_positive(
        out.num_key_value_heads, "GLM-TTS KV heads");
    engine::io::require_divisible(
        out.num_attention_heads,
        out.num_key_value_heads,
        "GLM-TTS attention heads");
    if (out.num_attention_heads * out.head_dim != out.hidden_size) {
        throw std::runtime_error(
            "GLM-TTS hidden size does not match heads times head_dim");
    }
    return out;
}

GlmTTSSpeechTokenizerConfig parse_speech_tokenizer(
    const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("speech_tokenizer_config");
    if (json::require_string(root, "model_type") != "whisper") {
        throw std::runtime_error(
            "GLM-TTS expects a Whisper-VQ speech tokenizer");
    }

    GlmTTSSpeechTokenizerConfig out;
    out.d_model = json::require_i64(root, "d_model");
    out.encoder_attention_heads =
        json::require_i64(root, "encoder_attention_heads");
    out.encoder_ffn_dim = json::require_i64(root, "encoder_ffn_dim");
    out.encoder_layers = json::require_i64(root, "encoder_layers");
    out.max_source_positions =
        json::require_i64(root, "max_source_positions");
    out.pooling_kernel_size =
        json::require_i64(root, "pooling_kernel_size");
    out.pooling_position = json::require_i64(root, "pooling_position");
    out.quantize_position = json::require_i64(root, "quantize_position");
    out.quantize_vocab_size =
        json::require_i64(root, "quantize_vocab_size");

    const auto preprocessor =
        resources.parse_json("speech_tokenizer_preprocessor");
    out.feature_size = json::require_i64(preprocessor, "feature_size");
    out.sampling_rate = json::require_i64(preprocessor, "sampling_rate");
    if (out.pooling_position != out.quantize_position) {
        throw std::runtime_error(
            "GLM-TTS currently expects pooling and quantization at the same "
            "Whisper encoder layer");
    }
    return out;
}

GlmTTSFlowConfig parse_flow(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("audio_cpp_config");
    const auto & flow = root.require("flow");
    GlmTTSFlowConfig out;
    out.speech_token_dim =
        json::optional_i64(flow, "speech_token_dim", out.speech_token_dim);
    out.vocab_size =
        json::optional_i64(flow, "vocab_size", out.vocab_size);
    out.mel_dim = json::optional_i64(flow, "mel_dim", out.mel_dim);
    out.trans_dim = json::optional_i64(flow, "trans_dim", out.trans_dim);
    out.depth = json::optional_i64(flow, "depth", out.depth);
    out.heads = json::optional_i64(flow, "heads", out.heads);
    out.dim_head = json::optional_i64(flow, "dim_head", out.dim_head);
    out.conv_layers =
        json::optional_i64(flow, "conv_layers", out.conv_layers);
    out.mel_framerate =
        json::optional_i64(flow, "mel_framerate", out.mel_framerate);
    out.input_frame_rate =
        json::optional_f32(flow, "input_frame_rate", out.input_frame_rate);
    out.inference_cfg_rate =
        json::optional_f32(flow, "inference_cfg_rate", out.inference_cfg_rate);
    out.inference_steps =
        json::optional_i64(flow, "inference_steps", out.inference_steps);
    return out;
}

void validate_anchors(const GlmTTSAssets & model_assets) {
    const auto & llama = model_assets.config.llama;
    engine::assets::require_tensor_shape(
        *model_assets.llama_weights,
        "model.embed_tokens.weight",
        {llama.vocab_size, llama.hidden_size});
    engine::assets::require_tensor_shape(
        *model_assets.llama_weights, "model.norm.weight", {llama.hidden_size});
    engine::assets::require_tensor_shape(
        *model_assets.llama_weights,
        "model.layers.0.self_attn.q_proj.weight",
        {llama.num_attention_heads * llama.head_dim, llama.hidden_size});

    const auto & speech = model_assets.config.speech_tokenizer;
    engine::assets::require_tensor_shape(
        *model_assets.speech_tokenizer_weights,
        "model.encoder.codebook.weight",
        {speech.quantize_vocab_size, speech.d_model});
    engine::assets::require_tensor_shape(
        *model_assets.speech_tokenizer_weights,
        "model.encoder.conv1.weight",
        {speech.d_model, speech.feature_size, 3});

    const auto & flow = model_assets.config.flow;
    engine::assets::require_tensor_shape(
        *model_assets.flow_weights,
        "estimator.text_emb_layer.text_embed.weight",
        {flow.vocab_size + 1, flow.speech_token_dim});
    engine::assets::require_tensor_shape(
        *model_assets.flow_weights,
        "estimator.transformer_blocks.17.attn.to_q.weight",
        {flow.trans_dim, flow.trans_dim});
    engine::assets::require_tensor_shape(
        *model_assets.hift_weights,
        "conv_pre.parametrizations.weight.original1",
        {512, flow.mel_dim, 7});
    engine::assets::require_tensor_shape(
        *model_assets.campplus_weights,
        "speaker_encoder.xvector.dense.linear.weight",
        {192, 1024, 1});
}

}  // namespace

std::shared_ptr<const GlmTTSAssets> load_glm_tts_assets(
    const std::filesystem::path & model_path) {
    auto resources = engine::model_spec::load_resource_bundle_for_family(
        model_path, "glm_tts");

    GlmTTSAssets assets;
    assets.config.llama = parse_llama(resources);
    assets.config.speech_tokenizer = parse_speech_tokenizer(resources);
    assets.config.flow = parse_flow(resources);

    const auto runtime_config = resources.parse_json("audio_cpp_config");
    assets.config.sample_rate =
        json::optional_i64(runtime_config, "sample_rate", 24000);

    GlmTTSTextTokenizer tokenizer(
        resources.require_file("tokenizer_vocab"),
        resources.require_file("tokenizer_merges"),
        resources.require_file("tokenizer_config"));
    assets.config.audio_token_start =
        tokenizer.require_token_id("<|audio_0|>");
    assets.config.audio_token_end =
        tokenizer.require_token_id("<|audio_32767|>");
    assets.config.begin_audio_token =
        tokenizer.require_token_id("<|begin_of_audio|>");
    assets.config.end_audio_token =
        tokenizer.require_token_id("<|user|>");
    assets.config.pad_token =
        tokenizer.require_token_id("<|endoftext|>");

    assets.llama_weights = resources.open_tensor_source("llama_weights");
    assets.speech_tokenizer_weights =
        resources.open_tensor_source("speech_tokenizer_weights");
    assets.flow_weights = resources.open_tensor_source("flow_weights");
    assets.hift_weights = resources.open_tensor_source("hift_weights");
    assets.campplus_weights =
        resources.open_tensor_source("campplus_weights");
    assets.resources = std::move(resources);
    validate_anchors(assets);
    return std::make_shared<GlmTTSAssets>(std::move(assets));
}

}  // namespace engine::models::glm_tts
