#pragma once

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::zipvoice {

// Architecture hyper-parameters, mirroring model.json "model" + tokenizer info.
struct ZipVoiceConfig {
    std::vector<int> fm_downsampling_factor = {1, 2, 4, 2, 1};
    std::vector<int> fm_num_layers = {2, 2, 4, 4, 4};
    std::vector<int> fm_cnn_kernel = {31, 15, 7, 15, 31};
    int fm_feedforward_dim = 1536;
    int fm_num_heads = 4;
    int fm_dim = 512;
    int text_num_layers = 4;
    int text_feedforward_dim = 512;
    int text_cnn_kernel = 9;
    int text_num_heads = 4;
    int text_dim = 192;
    int time_embed_dim = 192;
    int text_embed_dim = 192;
    int query_head_dim = 32;
    int value_head_dim = 12;
    int pos_head_dim = 4;
    int pos_dim = 48;
    int feat_dim = 100;
    int vocab_size = 360;  // from tokens.txt
    int pad_id = 0;
    int sampling_rate = 24000;
    bool guidance_scale_embed = false;  // ZipVoice-Distill
};

// One Zipformer2 encoder layer's weights (torch module names).
struct ZipLayerWeights {
    core::TensorValue bypass_scale;       // [C]
    core::TensorValue bypass_mid_scale;   // [C]
    core::TensorValue attn_in_proj_w;     // [C, (2*qh+ph)*H]
    core::TensorValue attn_in_proj_b;
    core::TensorValue linear_pos_w;       // [H*ph, pos_dim]
    core::TensorValue sa1_in_w, sa1_in_b;      // [H*vh, C]
    core::TensorValue sa1_out_w, sa1_out_b;    // [C, H*vh]
    core::TensorValue sa2_in_w, sa2_in_b;
    core::TensorValue sa2_out_w, sa2_out_b;
    core::TensorValue ff1_in_w, ff1_in_b;
    core::TensorValue ff1_out_w, ff1_out_b;
    core::TensorValue ff2_in_w, ff2_in_b;
    core::TensorValue ff2_out_w, ff2_out_b;
    core::TensorValue ff3_in_w, ff3_in_b;
    core::TensorValue ff3_out_w, ff3_out_b;
    core::TensorValue na_in_w, na_in_b;    // [3*hidden, C]
    core::TensorValue na_out_w, na_out_b;  // [C, hidden]
    core::TensorValue cm1_in_w, cm1_in_b;
    core::TensorValue cm1_conv_w, cm1_conv_b;  // [C, 1, k], [C]
    core::TensorValue cm1_out_w, cm1_out_b;
    core::TensorValue cm2_in_w, cm2_in_b;
    core::TensorValue cm2_conv_w, cm2_conv_b;
    core::TensorValue cm2_out_w, cm2_out_b;
    core::TensorValue norm_bias;      // [C]
    float norm_log_scale = 0.0F;
};

struct ZipStackWeights {
    std::vector<ZipLayerWeights> layers;
    // present when the stack has a time embedding projection
    core::TensorValue time_proj_w, time_proj_b;  // [C, time_dim]
    // present when downsampling_factor > 1
    std::vector<float> downsample_bias;  // [ds] (softmax applied at load)
    core::TensorValue out_combiner_scale;  // [C]
};

struct TTSZipformerWeights {
    core::TensorValue in_proj_w, in_proj_b;
    core::TensorValue out_proj_w, out_proj_b;
    core::TensorValue embed_w;  // only for text encoder: [vocab, text_embed_dim]
    // fm decoder only:
    core::TensorValue time_mlp0_w, time_mlp0_b;      // [2*tdim, tdim]
    core::TensorValue time_mlp2_w, time_mlp2_b;      // [tdim, 2*tdim]
    core::TensorValue guidance_embed_w;              // [tdim, tdim] no bias (distill)
    std::vector<ZipStackWeights> stacks;
};

struct ZipVoiceWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    TTSZipformerWeights fm_decoder;
    TTSZipformerWeights text_encoder;
};

// Loads a converted checkpoint. `prefix` is "model" for GGUF packages and ""
// for the development safetensors export. Throws on shape mismatch.
// `config_path` (optional) overrides `model_dir / "model.json"` for
// spec-registered (materialized) sidecars.
ZipVoiceConfig load_zipvoice_config(
    const std::filesystem::path & model_dir,
    const class engine::assets::TensorSource * probe,
    const std::filesystem::path * config_path = nullptr);

ZipVoiceWeights load_zipvoice_weights(
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    const ZipVoiceConfig & config,
    ggml_backend_t backend,
    core::BackendType backend_type);

}  // namespace engine::models::zipvoice
