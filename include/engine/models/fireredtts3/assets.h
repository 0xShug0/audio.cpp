#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>

namespace engine::models::fireredtts3 {

enum class FireRedTTS3Variant {
    Base,
    Instruct,
};

struct FireRedTTS3BaseConfig {
    int64_t vocab_size = 151936;
    int64_t hidden_size = 2048;
    int64_t intermediate_size = 6144;
    int64_t layers = 28;
    int64_t heads = 16;
    int64_t kv_heads = 8;
    int64_t head_dim = 128;
    int64_t redae_dim = 64;
    int64_t history_patches = 2;
    int64_t speaker_dim = 512;
    int64_t patch_size = 4;
    int64_t patch_hidden_size = 1024;
    int64_t patch_intermediate_size = 4096;
    int64_t patch_layers = 8;
    int64_t patch_heads = 16;
    int64_t dit_hidden_size = 1024;
    int64_t dit_intermediate_size = 3072;
    int64_t dit_layers = 11;
    int64_t dit_heads = 16;
};

struct FireRedTTS3RedAeConfig {
    int64_t sample_rate = 24000;
    int64_t audio_patch_size = 480;
    int64_t bottleneck_dim = 64;
    int64_t enc_hidden_size = 896;
    int64_t enc_intermediate_size = 3584;
    int64_t enc_layers = 18;
    int64_t enc_heads = 14;
    int64_t enc_kv_heads = 2;
    int64_t enc_head_dim = 64;
    int64_t enc_sliding_window = 0;
    int64_t enc_extra_downsample_rate = 2;
    int64_t enc_downsample_layers = 4;
    int64_t dec_hidden_size = 896;
    int64_t dec_intermediate_size = 3584;
    int64_t dec_layers = 18;
    int64_t dec_heads = 14;
    int64_t dec_kv_heads = 2;
    int64_t dec_head_dim = 64;
    int64_t dec_sliding_window = 0;
};

struct FireRedTTS3Assets {
    FireRedTTS3Variant variant = FireRedTTS3Variant::Base;
    std::filesystem::path model_root;
    std::filesystem::path gguf_path;
    engine::assets::ResourceBundle resources;
    FireRedTTS3BaseConfig base;
    FireRedTTS3RedAeConfig redae;
    std::shared_ptr<const engine::assets::TensorSource> base_weights;
    std::shared_ptr<const engine::assets::TensorSource> instruct_weights;
    std::shared_ptr<const engine::assets::TensorSource> redae_weights;
    std::shared_ptr<const engine::assets::TensorSource> campplus_weights;
};

std::shared_ptr<const FireRedTTS3Assets> load_fireredtts3_assets(
    const std::filesystem::path & model_path);

}  // namespace engine::models::fireredtts3
