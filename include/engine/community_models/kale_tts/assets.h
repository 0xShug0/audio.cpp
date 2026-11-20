#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::kale_tts {

struct KaleVoice {
    std::string id;                 // the release id, e.g. s057b
    std::string name;               // the voice id of requests, e.g. kenzie
    std::string accent;             // en-us or en-gb
    std::vector<float> style;       // style table [rows, 256]: row = phoneme count - 1
    int64_t style_rows = 0;
    std::vector<float> prosody;     // nano: the text side's voice row
    std::vector<float> sound;       // the decoder's voice row
    float pitch_floor = 0.0F;       // end clamp: pitch floor in Hz (0 = no clamp)
    float clamp_tail = 36.0F;       // end clamp: frames before the last voiced frame
    float clamp_threshold = 50.0F;  // end clamp: a frame above this is voiced
    float unvoice_hz = 80.0F;       // pitch below this is spoken as noise
};

// The "model" object of config.json (written by tools/community_models/convert_kale_tts.py).
struct KaleConfig {
    int64_t sample_rate = 24000;
    bool ettin = false;             // tiny: Ettin word reader + student text side; nano: ALBERT
    int64_t n_token = 178;
    int64_t hidden_dim = 0;
    int64_t style_dim = 0;
    int64_t n_layer = 3;
    int64_t max_dur = 50;
    int64_t decoder_width = 256;
    int64_t decoder_up_channels = 128;
    int64_t voice_sound_dim = 16;
    int64_t head_dim = 160;
    int64_t head_blocks = 4;
    int64_t head_points = 80;
    int64_t head_burst_points = 24;
    int64_t head_harmonics = 64;
    int64_t head_mod = 2;
    int64_t max_tokens = 510;
    // nano
    int64_t albert_hidden = 0;
    int64_t albert_embedding = 0;
    int64_t albert_heads = 0;
    int64_t albert_intermediate = 0;
    int64_t albert_layers = 0;
    int64_t albert_positions = 512;
    // tiny
    int64_t ettin_hidden = 0;
    int64_t ettin_intermediate = 0;
    int64_t ettin_layers = 0;
    int64_t ettin_heads = 0;
    int64_t ettin_global_every = 3;
    int64_t ettin_local_window = 128;
    float ettin_rope_theta = 160000.0F;
    int64_t student_channels = 0;
    int64_t student_heads = 4;
    int64_t student_conv_blocks = 3;
    int64_t student_transformer_blocks = 2;
    bool kokoro_native = true;      // phonemes in Kokoro's own symbols
    std::unordered_map<std::string, int32_t> vocab;
};

struct KaleAssets {
    assets::ResourceBundle resources;
    KaleConfig config;
    std::vector<KaleVoice> voices;
    std::shared_ptr<const assets::TensorSource> weights;
};

std::shared_ptr<const KaleAssets> load_kale_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::kale_tts
