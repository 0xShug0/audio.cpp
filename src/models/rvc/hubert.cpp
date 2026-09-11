#include "engine/models/rvc/hubert.h"

#include "engine/framework/modules/speech_encoders/hubert_encoder.h"

#include <stdexcept>
#include <utility>

namespace engine::models::rvc {
namespace {

constexpr int64_t kHidden = 768;
constexpr int64_t kIntermediate = 3072;
constexpr int64_t kLayers = 12;
constexpr int64_t kHeads = 12;
constexpr int64_t kConvPos = 128;
constexpr int64_t kConvPosGroups = 16;
const std::vector<int64_t> kConvDim{512, 512, 512, 512, 512, 512, 512};
const std::vector<int64_t> kConvKernel{10, 3, 3, 3, 3, 2, 2};
const std::vector<int64_t> kConvStride{5, 2, 2, 2, 2, 2, 2};

engine::modules::HubertEncoderConfig rvc_hubert_config() {
    engine::modules::HubertEncoderConfig config;
    config.hidden_size = kHidden;
    config.intermediate_size = kIntermediate;
    config.num_hidden_layers = kLayers;
    config.output_hidden_layer = kLayers;
    config.num_attention_heads = kHeads;
    config.conv_in_channels = 1;
    config.num_conv_pos_embeddings = kConvPos;
    config.num_conv_pos_embedding_groups = kConvPosGroups;
    config.conv_dim = kConvDim;
    config.conv_kernel = kConvKernel;
    config.conv_stride = kConvStride;
    config.apply_encoder_input_layer_norm = true;
    config.apply_final_layer_norm = false;
    config.pad_odd_tokens_with_attention_mask = true;
    config.final_projection_size = 256;
    config.feature_extractor_norm = engine::modules::HubertFeatureExtractorNorm::FirstLayerGroupNorm;
    config.encoder_layer_norm_order = engine::modules::HubertEncoderLayerNormOrder::PostNorm;
    return config;
}

engine::modules::HubertEncoderWeightBinding rvc_hubert_binding(
    engine::assets::TensorStorageType storage_type) {
    engine::modules::HubertEncoderWeightBinding binding;
    binding.feature_extractor_conv = "0";
    binding.feature_extractor_layer_norm = "2";
    binding.feature_projection_layer_norm = "layer_norm";
    binding.feature_projection_projection = "post_extract_proj";
    binding.positional_conv = "encoder.pos_conv.0";
    binding.layer.attention = "self_attn";
    binding.layer.feed_forward_intermediate = "fc1";
    binding.layer.feed_forward_output = "fc2";
    binding.conv_storage_type = storage_type;
    binding.positional_conv_storage_type = storage_type;
    binding.projection_storage_type = storage_type;
    binding.attention_storage_type = storage_type;
    binding.feed_forward_storage_type = storage_type;
    binding.final_projection_storage_type = storage_type;
    return binding;
}

}  // namespace

struct RvcHubertEncoder::State {
    engine::modules::HubertEncoderComponent component;
};

RvcHubertEncoder::RvcHubertEncoder(
    std::shared_ptr<const engine::assets::TensorSource> source,
    engine::core::BackendConfig backend,
    engine::assets::TensorStorageType storage_type)
    : state_(std::make_shared<State>()) {
    state_->component = engine::modules::HubertEncoderComponent::load_from_tensor_source(
        std::move(source),
        std::move(backend),
        rvc_hubert_config(),
        rvc_hubert_binding(storage_type));
}

RvcHubertEncoder::~RvcHubertEncoder() = default;
RvcHubertEncoder::RvcHubertEncoder(RvcHubertEncoder &&) noexcept = default;
RvcHubertEncoder & RvcHubertEncoder::operator=(RvcHubertEncoder &&) noexcept = default;

RvcHubertFeatures RvcHubertEncoder::encode_16k_mono(
    const std::vector<float> & waveform_16k,
    bool v1_features) const {
    if (state_ == nullptr) {
        throw std::runtime_error("RVC HuBERT encoder is not initialized");
    }
    engine::modules::HubertEncoderRunConfig run_config;
    run_config.output_hidden_layer = v1_features ? 10 : kLayers;
    run_config.apply_final_projection = v1_features;
    const auto out = state_->component.encode(
        waveform_16k,
        1,
        static_cast<int64_t>(waveform_16k.size()),
        run_config);
    return {
        out.hidden_states,
        out.tokens,
        out.hidden_size,
    };
}

}  // namespace engine::models::rvc
