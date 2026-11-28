#include "engine/models/mulacover/assets.h"

#include "engine/framework/io/config.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <utility>

namespace engine::models::mulacover {
namespace json = engine::io::json;

std::shared_ptr<const MuLaCoverAssets> load_mulacover_assets(const std::filesystem::path & model_path) {
    auto out = std::make_shared<MuLaCoverAssets>();
    out->resources = model_spec::load_resource_bundle_for_family(model_path, "mulacover");
    const auto root = out->resources.parse_json("mula_config");
    auto & config = out->mula_config;
    const auto model_type = json::require_string(root, "model_type");
    const auto backbone_flavor = json::require_string(root, "backbone_flavor");
    const auto decoder_flavor = json::require_string(root, "decoder_flavor");
    if (model_type != "mulacover" || backbone_flavor != "llama-3B" ||
        decoder_flavor != "llama-300M") {
        throw std::runtime_error("MuLaCover requires mulacover with llama-3B/llama-300M transformers");
    }
    json::require_string(root, "dtype");
    config.text_vocab_size = json::require_i64(root, "text_vocab_size");
    config.audio_vocab_size = json::require_i64(root, "audio_vocab_size");
    config.audio_num_codebooks = json::require_i64(root, "audio_num_codebooks");
    config.muq_dim = json::require_i64(root, "muq_dim");
    config.qwen_dim = json::require_i64(root, "qwen_dim");
    config.peft_layer_num = json::require_i64(root, "peft_layer_num");
    config.train_tag = root.require("train_tag").as_bool();
    // Released upstream flavors; scaled RoPE factors are bound with the weights.
    config.backbone.hidden_size = config.decoder.hidden_size = 3072;
    config.backbone.intermediate_size = config.decoder.intermediate_size = 8192;
    config.backbone.layers = 28;
    config.backbone.num_attention_heads = 24;
    config.backbone.num_key_value_heads = 8;
    config.backbone.head_dim = 128;
    config.decoder.layers = 3;
    config.decoder.num_attention_heads = 8;
    config.decoder.num_key_value_heads = 4;
    config.decoder.head_dim = 384;
    for (auto * stack : {&config.backbone, &config.decoder}) {
        stack->rope_theta = 500000.0F;
        stack->rope_type = GGML_ROPE_TYPE_NORMAL;
        stack->use_qk_norm = false;
        stack->runtime.static_cache.update_mode = modules::DecoderStaticCacheUpdateMode::DirectSetRows;
        stack->runtime.attention.prefill_mode = modules::DecoderAttentionMode::FlashGrouped;
        stack->runtime.attention.static_mode = modules::DecoderAttentionMode::FlashGrouped;
    }
    io::require_positive(config.text_vocab_size, "text_vocab_size");
    io::require_positive(config.audio_vocab_size, "audio_vocab_size");
    io::require_positive(config.muq_dim, "muq_dim");
    io::require_positive(config.qwen_dim, "qwen_dim");
    if (config.audio_num_codebooks < 2 || config.peft_layer_num != 8) {
        throw std::runtime_error("MuLaCover invalid codebook count or adapted layer count");
    }

    // The depth head has 384 channels per head. Grouped explicit attention
    // avoids materializing repeated F16 K/V (unsupported by Vulkan REPEAT).
    config.decoder.runtime.attention.prefill_mode = modules::DecoderAttentionMode::ManualRepeat;
    config.decoder.runtime.attention.static_mode = modules::DecoderAttentionMode::ManualRepeatThenGroupedQuery;
    config.decoder.runtime.attention.grouped_query_min_steps = 1;

    const auto generation = out->resources.parse_json("generation_config");
    auto & ids = out->generation_config;
    ids.text_bos_id = json::require_i64(generation, "text_bos_id");
    ids.text_eos_id = json::require_i64(generation, "text_eos_id");
    ids.audio_eos_id = json::require_i64(generation, "audio_eos_id");
    ids.empty_id = json::require_i64(generation, "empty_id");
    ids.tag_start_id = json::require_i64(generation, "tag_start_id");
    ids.tag_end_id = json::require_i64(generation, "tag_end_id");
    ids.muq_start_id = json::require_i64(generation, "muq_start_id");
    ids.muq_end_id = json::require_i64(generation, "muq_end_id");
    ids.qwen_start_id = json::require_i64(generation, "qwen_start_id");
    ids.qwen_end_id = json::require_i64(generation, "qwen_end_id");
    for (const auto id : {ids.text_bos_id, ids.text_eos_id, ids.tag_start_id, ids.tag_end_id,
                          ids.muq_start_id, ids.muq_end_id, ids.qwen_start_id, ids.qwen_end_id}) {
        if (id < 0 || id >= config.text_vocab_size) {
            throw std::runtime_error("MuLaCover text token ID outside vocabulary");
        }
    }
    if (ids.audio_eos_id < 0 || ids.audio_eos_id >= config.audio_vocab_size ||
        ids.empty_id < 0 || ids.empty_id >= config.audio_vocab_size) {
        throw std::runtime_error("MuLaCover audio token ID outside vocabulary");
    }

    out->mula_weights = out->resources.open_tensor_source("mula_weights");
    out->qwen_weights = out->resources.open_tensor_source("qwen_weights");
    const auto & weights = *out->mula_weights;
    assets::require_tensor_shape(weights, "text_embeddings.weight", {config.text_vocab_size, 3072});
    assets::require_tensor_shape(weights, "audio_embeddings.weight", {config.audio_vocab_size * config.audio_num_codebooks, 3072});
    assets::require_tensor_shape(weights, "audio_head", {config.audio_num_codebooks - 1, 3072, config.audio_vocab_size});
    assets::require_tensor_shape(weights, "qwen_linear.weight", {3072, config.qwen_dim});
    assets::require_tensor_shape(weights, "muq_linear.weight", {3072, config.muq_dim});
    const auto qwen = out->resources.parse_json("qwen_config");
    if (json::require_string(qwen, "model_type") != "qwen3" ||
        json::require_i64(qwen, "hidden_size") != config.qwen_dim) {
        throw std::runtime_error("MuLaCover requires Qwen3 embeddings matching qwen_dim");
    }
    assets::require_tensor_shape(*out->qwen_weights, "embed_tokens.weight",
        {json::require_i64(qwen, "vocab_size"), config.qwen_dim});
    out->qwen_weights->require_metadata("layers.0.self_attn.q_proj.weight");
    out->qwen_weights->require_metadata("norm.weight");

    out->transcriber_weights = out->resources.open_tensor_source("transcriber_weights");
    for (size_t fold = 0; fold < out->chord_weights.size(); ++fold) {
        out->chord_weights[fold] = out->resources.open_tensor_source("chord_weights_" + std::to_string(fold));
    }

    {
        const auto codec_root = out->resources.parse_json("codec_config");
        auto & codec_config = out->codec_config;
        const auto model_type = json::optional_string(codec_root, "model_type", "");
        if (model_type != "heartcodec") {
            throw std::runtime_error("HeartCodec config model_type must be heartcodec");
        }
        const auto torch_dtype = json::optional_string(codec_root, "torch_dtype", "");
        if (torch_dtype != "float32") {
            throw std::runtime_error("HeartCodec config torch_dtype must be float32");
        }
        codec_config.dim = json::require_i64(codec_root, "dim");
        codec_config.codebook_size = json::require_i64(codec_root, "codebook_size");
        const auto decay = json::optional_f32(codec_root, "decay", 0.0F);
        const auto commitment_weight = json::optional_f32(codec_root, "commitment_weight", 0.0F);
        const auto threshold_ema_dead_code = json::require_i64(codec_root, "threshold_ema_dead_code");
        (void) json::optional_bool(codec_root, "use_cosine_sim", false);
        codec_config.codebook_dim = json::require_i64(codec_root, "codebook_dim");
        codec_config.num_quantizers = json::require_i64(codec_root, "num_quantizers");
        codec_config.attention_head_dim = json::require_i64(codec_root, "attention_head_dim");
        codec_config.in_channels = json::require_i64(codec_root, "in_channels");
        const auto norm_type = json::require_string(codec_root, "norm_type");
        codec_config.num_attention_heads = json::require_i64(codec_root, "num_attention_heads");
        codec_config.num_layers = json::require_i64(codec_root, "num_layers");
        codec_config.num_layers_2 = json::require_i64(codec_root, "num_layers_2");
        codec_config.out_channels = json::require_i64(codec_root, "out_channels");
        codec_config.num_bands = json::require_i64(codec_root, "num_bands");
        codec_config.sample_rate = static_cast<int>(json::require_i64(codec_root, "sample_rate"));
        const auto causal = json::optional_bool(codec_root, "causal", true);
        codec_config.num_samples = json::require_i64(codec_root, "num_samples");
        const auto downsample_factors = json::require_i64_array(codec_root, "downsample_factors");
        const auto downsample_kernel_sizes = json::require_i64_array(codec_root, "downsample_kernel_sizes");
        codec_config.upsample_factors = json::require_i64_array(codec_root, "upsample_factors");
        codec_config.upsample_kernel_sizes = json::require_i64_array(codec_root, "upsample_kernel_sizes");
        codec_config.latent_hidden_dim = json::require_i64(codec_root, "latent_hidden_dim");
        codec_config.default_kernel_size = json::require_i64(codec_root, "default_kernel_size");
        codec_config.delay_kernel_size = json::require_i64(codec_root, "delay_kernel_size");
        codec_config.init_channel = json::require_i64(codec_root, "init_channel");
        codec_config.res_kernel_size = json::require_i64(codec_root, "res_kernel_size");

        engine::io::require_positive(codec_config.dim, "codec dim");
        engine::io::require_positive(codec_config.codebook_size, "codec codebook_size");
        engine::io::require_positive(decay, "codec decay");
        engine::io::require_positive(commitment_weight, "codec commitment_weight");
        engine::io::require_positive(threshold_ema_dead_code, "codec threshold_ema_dead_code");
        engine::io::require_positive(codec_config.codebook_dim, "codec codebook_dim");
        engine::io::require_positive(codec_config.num_quantizers, "codec num_quantizers");
        engine::io::require_positive(codec_config.attention_head_dim, "codec attention_head_dim");
        engine::io::require_positive(codec_config.in_channels, "codec in_channels");
        engine::io::require_positive(codec_config.num_attention_heads, "codec num_attention_heads");
        engine::io::require_positive(codec_config.num_layers, "codec num_layers");
        engine::io::require_positive(codec_config.num_layers_2, "codec num_layers_2");
        engine::io::require_positive(codec_config.out_channels, "codec out_channels");
        engine::io::require_positive(codec_config.num_bands, "codec num_bands");
        engine::io::require_positive(codec_config.sample_rate, "codec sample_rate");
        engine::io::require_positive(codec_config.num_samples, "codec num_samples");
        engine::io::require_positive(codec_config.latent_hidden_dim, "codec latent_hidden_dim");
        engine::io::require_positive(codec_config.default_kernel_size, "codec default_kernel_size");
        engine::io::require_positive(codec_config.delay_kernel_size, "codec delay_kernel_size");
        engine::io::require_positive(codec_config.init_channel, "codec init_channel");
        engine::io::require_positive(codec_config.res_kernel_size, "codec res_kernel_size");
        if (downsample_factors.empty() ||
            downsample_kernel_sizes.empty() ||
            codec_config.upsample_factors.empty() ||
            codec_config.upsample_kernel_sizes.empty()) {
            throw std::runtime_error("HeartCodec config contains empty sampling factors or kernels");
        }
        engine::io::require_all_positive(downsample_factors, "downsample_factors");
        engine::io::require_all_positive(downsample_kernel_sizes, "downsample_kernel_sizes");
        engine::io::require_all_positive(codec_config.upsample_factors, "upsample_factors");
        engine::io::require_all_positive(codec_config.upsample_kernel_sizes, "upsample_kernel_sizes");
        if (downsample_factors.size() != downsample_kernel_sizes.size() ||
            codec_config.upsample_factors.size() != codec_config.upsample_kernel_sizes.size()) {
            throw std::runtime_error("HeartCodec config sampling factor/kernel counts must match");
        }
        if (norm_type != "ada_norm_single") {
            throw std::runtime_error("HeartCodec config norm_type must be ada_norm_single");
        }
        if (!causal) {
            throw std::runtime_error("HeartCodec config must be causal");
        }
    }
    out->codec_weights = out->resources.open_tensor_source("codec_weights");
    if (out->codec_config.num_quantizers != config.audio_num_codebooks) {
        throw std::runtime_error("MuLaCover generator and HeartCodec codebook counts differ");
    }
    return out;
}

}  // namespace engine::models::mulacover
