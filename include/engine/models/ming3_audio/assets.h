#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/codecs/ming_audio_vae_runtime.h"
#include "engine/framework/modules/speech_decoders/ming_flow_runtime.h"
#include "engine/framework/modules/speech_encoders/whisper_embedding.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::ming3_audio {

struct ThinkerConfig {
    int64_t hidden_size = 0;
    int64_t vocabulary_size = 0;
    int64_t layers = 0;
    int64_t heads = 0;
    int64_t head_dim = 0;
    int64_t attention_interval = 0;
    int64_t conv_kernel_size = 0;
    int64_t intermediate_size = 0;
    int64_t dense_layers = 0;
    int64_t experts = 0;
    int64_t experts_per_token = 0;
    int64_t expert_intermediate_size = 0;
    int64_t shared_intermediate_size = 0;
    int64_t expert_groups = 0;
    int64_t selected_groups = 0;
    float routed_scaling_factor = 0;
    float rms_norm_eps = 0;
    float kda_lower_bound = 0;
    int64_t query_rank = 0;
    int64_t kv_rank = 0;
    int64_t qk_nope_dim = 0;
    int64_t qk_rope_dim = 0;
    int64_t value_dim = 0;
    float rope_theta = 0;
};

struct AudioConfig {
    modules::WhisperEmbeddingConfig encoder;
    int64_t chunk_frames = 3000;
    std::vector<int64_t> projector_kernels;
    std::vector<int64_t> projector_strides;
    int64_t token_id = 0;
    bool normalize_embeddings = false;
};

struct MingAudioConfig {
    ThinkerConfig thinker;
    AudioConfig audio;
};

struct TalkerConfig {
    modules::MingBackboneConfig backbone;
    modules::MingFlowConfig flow;
    modules::MingAudioCodecConfig codec;
    int64_t vocabulary_size = 0;
    int64_t context_length = 0;
};

MingAudioConfig parse_ming_audio_config(const io::json::Value & value);

struct MingAudioAssets {
    assets::ResourceBundle resources;
    MingAudioConfig config;
    std::vector<int32_t> eos_token_ids;
    TalkerConfig talker;
    std::shared_ptr<const assets::TensorSource> thinker_weights;
    std::shared_ptr<const assets::TensorSource> talker_weights;
    std::shared_ptr<const assets::TensorSource> vae_weights;
    std::shared_ptr<const assets::TensorSource> speaker_weights;
};

std::shared_ptr<const MingAudioAssets> load_ming3_audio_assets(const std::filesystem::path & path);

}  // namespace engine::models::ming3_audio
