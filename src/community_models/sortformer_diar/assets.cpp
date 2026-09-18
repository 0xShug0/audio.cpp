#include "engine/community_models/sortformer_diar/assets.h"

#include "engine/framework/model_spec/package.h"

#include <stdexcept>

namespace engine::community_models::sortformer_diar {

namespace {

SortformerV2ModelConfig parse_model_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    const auto & frontend = root.require("frontend");
    const auto & encoder = root.require("encoder");
    const auto & transformer = root.require("transformer");
    const auto & streaming = root.require("streaming");
    const auto & loss_weights = root.require("loss_weights");

    SortformerV2ModelConfig config;
    config.model_type = root.require("model_type").as_string();
    config.variant = root.require("audiocpp_variant").as_string();
    config.num_speakers = root.require("num_speakers").as_i64();
    config.pil_weight = loss_weights.require("pil").as_f32();
    config.ats_weight = loss_weights.require("ats").as_f32();

    config.fc_encoder.hidden_size = encoder.require("d_model").as_i64();
    config.fc_encoder.intermediate_size = encoder.require("d_ff").as_i64();
    config.fc_encoder.num_attention_heads = encoder.require("n_heads").as_i64();
    config.fc_encoder.num_hidden_layers = encoder.require("n_layers").as_i64();
    config.fc_encoder.num_key_value_heads = config.fc_encoder.num_attention_heads;
    config.fc_encoder.num_mel_bins = encoder.require("feat_in").as_i64();
    config.fc_encoder.max_position_embeddings = encoder.require("pos_emb_max_len").as_i64();
    config.fc_encoder.conv_kernel_size = encoder.require("conv_kernel").as_i64();
    config.fc_encoder.subsampling_factor = encoder.require("subsampling_factor").as_i64();
    config.fc_encoder.subsampling_conv_channels = encoder.require("subsampling_channels").as_i64();
    config.fc_encoder.subsampling_conv_kernel_size = 3;
    config.fc_encoder.subsampling_conv_stride = 2;
    config.fc_encoder.attention_bias = encoder.require("attention_bias").as_bool();
    config.fc_encoder.scale_input = encoder.require("scale_input").as_bool();
    config.fc_encoder.hidden_act = "silu";

    config.tf_encoder.hidden_size = transformer.require("d_model").as_i64();
    config.tf_encoder.intermediate_size = transformer.require("d_ff").as_i64();
    config.tf_encoder.num_attention_heads = transformer.require("n_heads").as_i64();
    config.tf_encoder.num_hidden_layers = transformer.require("n_layers").as_i64();
    config.tf_encoder.max_source_positions = 1500;
    config.tf_encoder.layer_norm_eps = 1.0e-5f;
    config.tf_encoder.activation_function = transformer.require("activation").as_string();

    config.modules.num_speakers = config.num_speakers;
    config.modules.fc_d_model = config.fc_encoder.hidden_size;
    config.modules.tf_d_model = config.tf_encoder.hidden_size;
    config.modules.subsampling_factor = config.fc_encoder.subsampling_factor;
    config.modules.dropout_rate = 0.0f;

    if (frontend.require("sample_rate").as_i64() != 16000 ||
        frontend.require("num_mels").as_i64() != config.fc_encoder.num_mel_bins) {
        throw std::runtime_error("Sortformer v2 frontend/model mel configuration mismatch");
    }
    config.streaming.chunk_len = streaming.require("chunk_len").as_i64();
    config.streaming.chunk_left_context = streaming.require("chunk_left_context").as_i64();
    config.streaming.chunk_right_context = streaming.require("chunk_right_context").as_i64();
    config.streaming.fifo_len = streaming.require("fifo_len").as_i64();
    config.streaming.spkcache_len = streaming.require("spkcache_len").as_i64();
    config.streaming.spkcache_update_period = streaming.require("spkcache_update_period").as_i64();
    config.streaming.spkcache_sil_frames_per_spk = streaming.require("spkcache_sil_frames_per_spk").as_i64();
    config.streaming.pred_score_threshold = streaming.require("pred_score_threshold").as_f32();
    config.streaming.scores_boost_latest = streaming.require("scores_boost_latest").as_f32();
    config.streaming.sil_threshold = streaming.require("sil_threshold").as_f32();
    config.streaming.strong_boost_rate = streaming.require("strong_boost_rate").as_f32();
    config.streaming.weak_boost_rate = streaming.require("weak_boost_rate").as_f32();
    config.streaming.min_pos_scores_rate = streaming.require("min_pos_scores_rate").as_f32();
    config.streaming.max_index = streaming.require("max_index").as_i64();
    config.streaming.frame_hop_samples = streaming.require("frame_hop_samples").as_i64();
    if (config.streaming.frame_hop_samples <= 0) {
        throw std::runtime_error("Sortformer v2 streaming frame hop must be positive");
    }
    return config;
}

SortformerV2FeatureExtractorConfig parse_feature_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("processor");
    const auto & feature = root.require("feature_extractor");
    SortformerV2FeatureExtractorConfig config;
    config.sample_rate = feature.require("sampling_rate").as_i64();
    config.n_fft = feature.require("n_fft").as_i64();
    config.win_length = feature.require("win_length").as_i64();
    config.hop_length = feature.require("hop_length").as_i64();
    config.num_mel_bins = feature.require("feature_size").as_i64();
    config.preemphasis = feature.require("preemphasis").as_f32();
    config.return_attention_mask = feature.require("return_attention_mask").as_bool();
    return config;
}

}  // namespace

std::shared_ptr<const SortformerV2Assets> load_sortformer_v2_assets(
    const std::filesystem::path & model_root,
    const std::filesystem::path & spec_path) {
    auto assets = std::make_shared<SortformerV2Assets>();
    assets->resources = engine::model_spec::load_resource_bundle(model_root, spec_path);
    assets->model_config = parse_model_config(assets->resources);
    assets->feature_config = parse_feature_config(assets->resources);
    assets->model_weights = assets->resources.open_tensor_source("weights");
    if (assets->model_weights->has_tensor("preprocessor.fb")) {
        const int64_t expected_bins = assets->feature_config.n_fft / 2 + 1;
        assets->mel_filterbank = assets->model_weights->require_f32(
            "preprocessor.fb",
            {assets->feature_config.num_mel_bins, expected_bins});
    }
    return assets;
}

}  // namespace engine::community_models::sortformer_diar
