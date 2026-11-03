#pragma once

#include "engine/models/pocket_tts_small/mimi_decoder.h"

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/transformers/transformer_blocks.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/framework/runtime/session.h"
#include "engine/framework/tokenizers/sentencepiece.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::pocket_tts_small {

enum class FlowSamplerKind { Drifting, Pointwise };

enum class VoiceSourceKind {
    NamedPreset,
    PreparedEmbedding,
    CloneAudio,
};

struct VoiceConfig {
    std::string preset_name;
    std::filesystem::path embedding_path;
    std::filesystem::path clone_audio_path;
    std::optional<runtime::AudioBuffer> clone_audio;
    std::string clone_prompt_text;
    bool truncate_clone_audio = false;
};

struct GenerationRequest {
    std::string text;
    VoiceConfig voice = {};
    int max_steps = 0;
    int max_tokens = 50;
    std::optional<int64_t> text_chunk_size = std::nullopt;
    int frames_after_eos = -1;
    float temperature = 0.3F;
    float noise_clamp = -1.0F;
    float eos_threshold = -1.0F;
    uint32_t seed = 1234;
    std::vector<float> noise_schedule;
    std::filesystem::path noise_schedule_path;
};

struct GenerationResult {
    int sample_rate = 24000;
    std::vector<float> audio;
};

struct VoiceConditioningPlan {
    VoiceSourceKind source = VoiceSourceKind::NamedPreset;
    std::filesystem::path asset_path;
    std::filesystem::path clone_audio_path;
    std::optional<runtime::AudioBuffer> clone_audio;
    std::string preset_name;
    std::string clone_prompt_text;
    bool truncate_clone_audio = false;
};

struct ModelConfig {
    float default_temperature = 0.3F;
    int sample_rate = 24000;
    float frame_rate = 12.5F;
    float mimi_frame_rate = 12.5F;
    int64_t flow_layers = 6;
    int64_t flow_depth = 6;
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
    // Added to the word-count guess (3 or 1) for the default frames_after_eos, which counts
    // the EOS frame itself.
    int frames_after_eos_offset = 2;
    float default_eos_threshold = -1.0F;
    FlowSamplerKind sampler = FlowSamplerKind::Drifting;
    int64_t flow_attention_dim = 0;
    int64_t flow_kv_heads = 0;
    int64_t flow_condition_dim = 0;
    int min_eos_steps = 0;
    int fade_in_samples = 0;
    std::vector<std::pair<std::string, std::string>> replace_characters;
    MimiDecoderConfig decoder;

};

struct HostWeights {
    assets::TensorDataF32 conditioner_embedding_table;
    std::optional<assets::TensorDataF32> conditioner_projection;
    std::vector<float> bos_emb;
    std::optional<assets::TensorDataF32> bos_before_voice;
    std::vector<float> emb_mean;
    std::vector<float> emb_std;
    // Constant added to input_linear on every generated frame, for checkpoints whose summed
    // conditioners are folded at load time.
    std::optional<std::vector<float>> input_linear_bias;
    // Projects `speaker_wavs` voice latents [frames, channels] to flow_dim on the host.
    std::optional<assets::TensorDataF32> speaker_proj;
};

struct Assets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> model_weights;
    std::filesystem::path voice_asset_root;
    HostWeights host_weights;
    ModelConfig model_config;
    std::vector<tokenizers::SentencePiecePiece> tokenizer_pieces;
    std::string language;
    std::string family = "pocket_tts_small";
    // Voice used when a request selects none.
    std::optional<std::filesystem::path> default_voice_path;
};

std::shared_ptr<const Assets> load_assets(const std::filesystem::path & model_path);

struct ResidualBlockWeights {
    modules::StreamingConv1dWeights conv1;
    modules::StreamingConv1dWeights conv2;
};

struct PackedAttentionWeights {
    core::TensorValue in_proj_weight;
    core::TensorValue out_proj_weight;
};

struct TransformerLayerWeights {
    modules::NormWeights norm1;
    PackedAttentionWeights self_attn;
    std::optional<modules::LayerScaleWeights> layer_scale_1;
    modules::NormWeights norm2;
    modules::FeedForwardWeights feed_forward;
    std::optional<modules::LayerScaleWeights> layer_scale_2;
};

struct FlowWeights {
    modules::LinearWeights input_linear;
    std::vector<TransformerLayerWeights> transformer_layers;
    modules::ConditionedFlowMLPWeights flow_net;
    std::optional<modules::LinearWeights> z_projection;
    modules::NormWeights out_norm;
    modules::LinearWeights out_eos;
    core::TensorValue speaker_proj_weight;
};

struct MimiEncoderWeights {
    modules::StreamingConv1dWeights input_conv;
    ResidualBlockWeights block0;
    modules::StreamingConv1dWeights downsample0;
    ResidualBlockWeights block1;
    modules::StreamingConv1dWeights downsample1;
    ResidualBlockWeights block2;
    modules::StreamingConv1dWeights downsample2;
    modules::StreamingConv1dWeights output_conv;
    std::vector<TransformerLayerWeights> transformer_layers;
    modules::StreamingConv1dWeights downsample_conv;
};

struct MimiDecoderWeights {
    std::vector<TransformerLayerWeights> transformer_layers;
    core::TensorValue quantizer_output_proj_weight;
    core::TensorValue encoder_upsample_weight;
    modules::StreamingConv1dWeights input_projection;
    modules::ConvTranspose1dWeights stage0_upsample;
    std::vector<float> stage0_upsample_bias_values;
    ResidualBlockWeights stage0_block;
    modules::ConvTranspose1dWeights stage1_upsample;
    std::vector<float> stage1_upsample_bias_values;
    ResidualBlockWeights stage1_block;
    modules::ConvTranspose1dWeights stage2_upsample;
    std::vector<float> stage2_upsample_bias_values;
    ResidualBlockWeights stage2_block;
    modules::StreamingConv1dWeights output_projection;
};

struct BackendWeights {
    core::BackendType backend_type = core::BackendType::Cpu;
    bool has_mimi_encoder = true;
    std::shared_ptr<core::BackendWeightStore> flow_store;
    std::shared_ptr<core::BackendWeightStore> mimi_encoder_store;
    std::shared_ptr<core::BackendWeightStore> mimi_decoder_store;
    HostWeights host;
    FlowWeights flow;
    MimiEncoderWeights mimi_encoder;
    MimiDecoderWeights mimi_decoder;
};

std::shared_ptr<const BackendWeights> load_pocket_tts_small_backend_weights(
    const Assets & manifest,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type,
    size_t flow_context_bytes,
    size_t mimi_encoder_context_bytes,
    size_t mimi_decoder_context_bytes);

struct Assets;

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

// Voice latents the checkpoint projects itself: a `speaker_wavs` tensor [1, channels, frames],
// returned frame-major.
struct SpeakerLatents {
    int64_t frames = 0;
    int64_t channels = 0;
    std::vector<float> values;
};

VoiceStateAssets load_voice_state_assets(const std::filesystem::path & source);
std::optional<SpeakerLatents> load_speaker_latents(const std::filesystem::path & source);
void save_voice_state_assets(
    const std::filesystem::path & destination,
    const runtime::TransformerKVState & state,
    int64_t heads,
    int64_t head_dim);
std::filesystem::path preset_embedding_path(const std::filesystem::path & model_root, const std::string & preset_name);
VoiceStateAssets load_voice_assets_for_plan(const VoiceConditioningPlan & plan, const Assets & manifest);

modules::TransformerEncoderBlockWeights make_transformer_block_weights(
    core::ModuleBuildContext & ctx,
    const TransformerLayerWeights & weights,
    int64_t hidden_size,
    int64_t attention_size = 0,
    int64_t kv_size = 0);

}  // namespace engine::models::pocket_tts_small
