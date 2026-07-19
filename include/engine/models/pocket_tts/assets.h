#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/modules/attention/transformer_blocks.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/tokenizers/sentencepiece.h"
#include "engine/models/pocket_tts/types.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::runtime {
struct TransformerKVState;
}  // namespace engine::runtime

namespace engine::models::pocket_tts {

struct PocketTTSModelConfig {
    float default_temperature = 0.7F;
    int sample_rate = 24000;
    float frame_rate = 12.5F;
    float mimi_frame_rate = 12.5F;
    int64_t flow_layers = 6;
    int64_t flow_dim = 1024;
    int64_t flow_heads = 16;
    int64_t flow_hidden_size = 512;
    int64_t flow_intermediate_size = 4096;
    int64_t latent_dim = 32;
    int64_t mimi_layers = 2;
    int64_t mimi_dim = 512;
    int64_t mimi_heads = 8;
    int64_t mimi_intermediate_size = 2048;
    int64_t mimi_inner_dim = 32;
    int64_t mimi_outer_dim = 512;
    int64_t mimi_context = 250;
    int64_t mimi_seanet_dimension = 512;
    int64_t mimi_base_filters = 64;
    int64_t mimi_encoder_upsample_stride = 16;
    bool pad_with_spaces_for_short_inputs = false;
    bool remove_semicolons = false;
    bool insert_bos_before_voice = false;
    std::optional<int> model_recommended_frames_after_eos = std::nullopt;
};

struct PocketTTSBackendResidualBlockWeights {
    modules::StreamingConv1dWeights conv1;
    modules::StreamingConv1dWeights conv2;
};

struct PocketTTSBackendPackedAttentionWeights {
    core::TensorValue in_proj_weight;
    core::TensorValue out_proj_weight;
};

struct PocketTTSBackendTransformerLayerWeights {
    modules::NormWeights norm1;
    PocketTTSBackendPackedAttentionWeights self_attn;
    std::optional<modules::LayerScaleWeights> layer_scale_1;
    modules::NormWeights norm2;
    modules::FeedForwardWeights feed_forward;
    std::optional<modules::LayerScaleWeights> layer_scale_2;
};

struct PocketTTSBackendFlowWeights {
    modules::LinearWeights input_linear;
    std::vector<PocketTTSBackendTransformerLayerWeights> transformer_layers;
    modules::TimedConditionedFlowMLPWeights flow_net;
    modules::NormWeights out_norm;
    modules::LinearWeights out_eos;
    core::TensorValue speaker_proj_weight;
};

struct PocketTTSBackendMimiEncoderWeights {
    modules::StreamingConv1dWeights input_conv;
    PocketTTSBackendResidualBlockWeights block0;
    modules::StreamingConv1dWeights downsample0;
    PocketTTSBackendResidualBlockWeights block1;
    modules::StreamingConv1dWeights downsample1;
    PocketTTSBackendResidualBlockWeights block2;
    modules::StreamingConv1dWeights downsample2;
    modules::StreamingConv1dWeights output_conv;
    std::vector<PocketTTSBackendTransformerLayerWeights> transformer_layers;
    modules::StreamingConv1dWeights downsample_conv;
};

struct PocketTTSBackendMimiDecoderWeights {
    std::vector<PocketTTSBackendTransformerLayerWeights> transformer_layers;
    core::TensorValue quantizer_output_proj_weight;
    core::TensorValue encoder_upsample_weight;
    modules::StreamingConv1dWeights input_projection;
    modules::ConvTranspose1dWeights stage0_upsample;
    std::vector<float> stage0_upsample_bias_values;
    PocketTTSBackendResidualBlockWeights stage0_block;
    modules::ConvTranspose1dWeights stage1_upsample;
    std::vector<float> stage1_upsample_bias_values;
    PocketTTSBackendResidualBlockWeights stage1_block;
    modules::ConvTranspose1dWeights stage2_upsample;
    std::vector<float> stage2_upsample_bias_values;
    PocketTTSBackendResidualBlockWeights stage2_block;
    modules::StreamingConv1dWeights output_projection;
};

struct PocketTTSHostWeights {
    assets::TensorDataF32 conditioner_embedding_table;
    std::vector<float> bos_emb;
    std::optional<assets::TensorDataF32> bos_before_voice;
    std::vector<float> emb_mean;
    std::vector<float> emb_std;
};

struct PocketTTSBackendWeights {
    core::BackendType backend_type = core::BackendType::Cpu;
    std::shared_ptr<core::BackendWeightStore> flow_store;
    std::shared_ptr<core::BackendWeightStore> mimi_encoder_store;
    std::shared_ptr<core::BackendWeightStore> mimi_decoder_store;
    PocketTTSHostWeights host;
    PocketTTSBackendFlowWeights flow;
    PocketTTSBackendMimiEncoderWeights mimi_encoder;
    PocketTTSBackendMimiDecoderWeights mimi_decoder;
};

struct PocketTTSAssets {
    std::shared_ptr<const assets::TensorSource> model_weights;
    PocketTTSHostWeights host_weights;
    PocketTTSModelConfig model_config;
    std::vector<tokenizers::SentencePiecePiece> tokenizer_pieces;
    std::filesystem::path model_root;
    std::filesystem::path tokenizer_path;
    std::string language;
};

struct VoiceAttentionCache {
    int64_t heads = 0;
    int64_t head_dim = 0;
    int64_t cached_steps = 0;
    int64_t offset = 0;
    std::vector<float> key;
    std::vector<float> value;
};

struct VoiceStateAssets {
    std::vector<VoiceAttentionCache> transformer_layers;
};

PocketTTSAssets load_pocket_tts_assets(
    const std::filesystem::path & model_root,
    std::string language,
    const std::optional<std::filesystem::path> & config_path = std::nullopt);
std::shared_ptr<const PocketTTSBackendWeights> load_pocket_tts_backend_weights(
    const PocketTTSAssets & manifest,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type,
    size_t flow_context_bytes,
    size_t mimi_encoder_context_bytes,
    size_t mimi_decoder_context_bytes);
VoiceStateAssets load_voice_state_assets(const std::filesystem::path & source);
void save_voice_state_assets(
    const std::filesystem::path & destination,
    const runtime::TransformerKVState & state,
    int64_t heads,
    int64_t head_dim);
std::filesystem::path preset_embedding_path(const std::filesystem::path & model_root, const std::string & preset_name);
VoiceStateAssets load_voice_assets_for_plan(const VoiceConditioningPlan & plan, const PocketTTSAssets & manifest);

}  // namespace engine::models::pocket_tts
