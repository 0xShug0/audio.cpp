#include "engine/models/ming3_audio/assets.h"

#include "engine/framework/model_spec/package.h"

#include <stdexcept>

namespace engine::models::ming3_audio {
namespace {

modules::MingBackboneConfig parse_backbone(const io::json::Value & value) {
    modules::MingBackboneConfig config;
    config.hidden_size = value.require("hidden_size").as_i64();
    config.intermediate_size = value.require("intermediate_size").as_i64();
    config.layers = value.require("num_hidden_layers").as_i64();
    config.attention_heads = value.require("num_attention_heads").as_i64();
    config.key_value_heads = value.require("num_key_value_heads").as_i64();
    config.rms_norm_eps = value.require("rms_norm_eps").as_f32();
    config.rope_theta = value.require("rope_theta").as_f32();
    return config;
}

modules::MingPatchNetworkConfig parse_patch(const io::json::Value & value) {
    modules::MingPatchNetworkConfig config;
    config.hidden_size = value.require("hidden_size").as_i64();
    config.heads = value.require("num_heads").as_i64();
    config.depth = value.require("depth").as_i64();
    config.intermediate_size = config.hidden_size * value.require("mlp_ratio").as_i64();
    return config;
}

}  // namespace

MingAudioConfig parse_ming_audio_config(const io::json::Value & value) {
    MingAudioConfig config;
    const auto & text = value.require("text_config");
    auto & c = config.thinker;
    c.hidden_size = text.require("hidden_size").as_i64();
    c.vocabulary_size = text.require("vocab_size").as_i64();
    c.layers = text.require("num_hidden_layers").as_i64();
    c.heads = text.require("num_attention_heads").as_i64();
    c.head_dim = text.require("head_dim").as_i64();
    c.attention_interval = text.require("layer_group_size").as_i64();
    c.conv_kernel_size = text.require("short_conv_kernel_size").as_i64();
    c.intermediate_size = text.require("intermediate_size").as_i64();
    c.dense_layers = text.require("first_k_dense_replace").as_i64();
    c.experts = text.require("num_experts").as_i64();
    c.experts_per_token = text.require("num_experts_per_tok").as_i64();
    c.expert_intermediate_size = text.require("moe_intermediate_size").as_i64();
    c.shared_intermediate_size = text.require("moe_shared_expert_intermediate_size").as_i64()
        * text.require("num_shared_experts").as_i64();
    c.expert_groups = text.require("n_group").as_i64();
    c.selected_groups = text.require("topk_group").as_i64();
    c.routed_scaling_factor = text.require("routed_scaling_factor").as_f32();
    c.rms_norm_eps = text.require("rms_norm_eps").as_f32();
    c.kda_lower_bound = text.require("kda_lower_bound").as_f32();
    c.query_rank = text.require("q_lora_rank").as_i64();
    c.kv_rank = text.require("kv_lora_rank").as_i64();
    c.qk_nope_dim = text.require("qk_nope_head_dim").as_i64();
    c.qk_rope_dim = text.require("qk_rope_head_dim").as_i64();
    c.value_dim = text.require("v_head_dim").as_i64();
    c.rope_theta = text.require("rope_theta").as_f32();
    if (!text.require("no_kda_lora").as_bool() || !text.require("kda_safe_gate").as_bool() ||
        !text.require("rope_interleave").as_bool() ||
        text.require("gated_attention_proj_granularity_type").as_string() != "head_wise" ||
        text.require("scoring_func").as_string() != "sigmoid" ||
        text.require("hidden_act").as_string() != "silu") {
        throw std::runtime_error("Ming3 Audio requires direct KDA projections, bounded gates, interleaved RoPE, head-wise MLA gating and sigmoid/SwiGLU experts");
    }
    const auto & audio = value.require("audio_config");
    auto & a = config.audio;
    a.encoder.n_mels = audio.require("n_mels").as_i64();
    a.encoder.n_audio_ctx = audio.require("n_audio_ctx").as_i64();
    a.encoder.n_audio_state = audio.require("n_audio_state").as_i64();
    a.encoder.n_audio_head = audio.require("n_audio_head").as_i64();
    a.encoder.n_audio_layer = audio.require("n_audio_layer").as_i64();
    if (const auto * chunk = audio.find("whisper_chunk_size")) a.chunk_frames = chunk->as_i64();
    for (const auto & kernel : audio.require("downsampler_kernel_size_list").as_array()) {
        a.projector_kernels.push_back(kernel.as_i64());
    }
    for (const auto & stride : audio.require("downsampler_stride_list").as_array()) {
        a.projector_strides.push_back(stride.as_i64());
    }
    a.token_id = value.require("audio_token_id").as_i64();
    a.normalize_embeddings = value.require("norm_query_embeds").as_bool();
    if (a.projector_kernels.size() != a.projector_strides.size()) {
        throw std::runtime_error("Ming3 Audio projector kernel/stride counts differ");
    }
    return config;
}

std::shared_ptr<const MingAudioAssets> load_ming3_audio_assets(const std::filesystem::path & path) {
    auto assets = std::make_shared<MingAudioAssets>();
    assets->resources = model_spec::load_resource_bundle_for_family(path, "ming3_audio");
    assets->config = parse_ming_audio_config(assets->resources.parse_json("config"));
    const auto generation = assets->resources.parse_json("generation_config");
    for (const auto id : io::json::require_i64_array_or_scalar(generation, "eos_token_id")) {
        assets->eos_token_ids.push_back(static_cast<int32_t>(id));
    }
    assets->thinker_weights = assets->resources.open_tensor_source("thinker");
    assets->talker_weights = assets->resources.open_tensor_source("talker");
    assets->vae_weights = assets->resources.open_tensor_source("vae");
    assets->speaker_weights = assets->resources.open_tensor_source("speaker");
    const auto llm = assets->resources.parse_json("talker_llm_config");
    const auto talker = assets->resources.parse_json("talker_config");
    const auto vae = assets->resources.parse_json("vae_config");
    auto & t = assets->talker;
    t.backbone = parse_backbone(llm);
    t.vocabulary_size = llm.require("vocab_size").as_i64();
    t.context_length = llm.require("max_position_embeddings").as_i64();
    t.flow.network = parse_patch(talker.require("flowmodel"));
    t.flow.aggregator = parse_patch(talker.require("aggregator"));
    t.flow.network_prefix = "cfm.model";
    t.flow.aggregator_prefix = "aggregator";
    t.flow.latent_dim = talker.require("flowmodel").require("in_channels").as_i64();
    t.flow.conditioning_dim = t.backbone.hidden_size;
    if (talker.require("steps").as_i64() != 10 || talker.require("patch_size").as_i64() != 1 ||
        talker.require("history_patch_size").as_i64() != 1) {
        throw std::runtime_error("Ming3 Audio Talker requires ten flow steps and one-patch generation/history");
    }
    auto & codec = t.codec;
    codec.patch_size = vae.require("patch_size").as_i64();
    codec.sample_rate = vae.require("sample_rate").as_i64();
    const auto & encoder = vae.require("enc_kwargs");
    const auto & decoder = vae.require("dec_kwargs");
    codec.encoder.backbone = parse_backbone(encoder.require("backbone"));
    codec.encoder.input_dim = encoder.require("input_dim").as_i64();
    codec.encoder.hop_size = encoder.require("hop_size").as_i64();
    codec.encoder.latent_dim = encoder.require("latent_dim").as_i64();
    // The published encoder shares the aggregator's config and executes four layers.
    codec.encoder.backbone.layers = 4;
    // CUDA autocast evaluates posterior softplus and sampling arithmetic in F32.
    codec.encoder.bf16_posterior = false;
    codec.encoder.sliding_window = encoder.require("backbone").require("sliding_window").as_i64();
    codec.decoder.backbone = parse_backbone(decoder.require("backbone"));
    codec.decoder.latent_dim = decoder.require("latent_dim").as_i64();
    codec.decoder.output_dim = decoder.require("output_dim").as_i64();
    codec.decoder.sliding_window = decoder.require("backbone").require("sliding_window").as_i64();
    codec.encoder_prefix = "encoder";
    codec.decoder_prefix = "decoder";
    return assets;
}

}  // namespace engine::models::ming3_audio
