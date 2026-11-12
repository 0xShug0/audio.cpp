#pragma once

#include "engine/framework/assets/resource_bundle.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::zonos3 {

inline constexpr int kZonos3FrameRate = 15;
inline constexpr int kZonos3Codebooks = 10;
inline constexpr int kZonos3CodebookSize = 4096;
inline constexpr int kZonos3Eoa = 4096;
inline constexpr int kZonos3AudioPad = 4097;
inline constexpr int kZonos3AudioVocab = 4098;
inline constexpr int kZonos3TextPad = 8196;
inline constexpr int kZonos3ReferenceMaxFrames = 900;
inline constexpr int kZonos3SilencePrefillMax = 30;

// Backbone hyperparameters read from backbone-ptq.gguf (ZONOS3 zonos3_hparams).
struct Zonos3Hparams {
    int n_layer = 0;
    int n_embd = 0;
    int head_dim = 0;
    int n_head = 0;
    int n_head_kv = 0;
    int n_ff = 0;
    int n_expert = 0;
    int n_ctx_train = 0;
    float rope_freq_base = 10000.0f;
    float rms_eps = 1e-5f;
    float qk_norm_eps = 1e-6f;
    float logit_softcap = 15.0f;
    int n_codebooks = 0;
    int codebook_size = 0;
    int audio_vocab = 0;
    int eoa_id = 0;
    int audio_pad_id = 0;
    int text_vocab = 0;
    int gdn_heads = 0;
    int gdn_head_dim = 0;
    int gdn_conv_kernel = 0;
    std::vector<int32_t> expert_used;  // [n_layer]; 0 = dense layer
    int speaker_dim = 0;
    int n_cond = 0;
    std::vector<int32_t> cond_bucket_counts;  // table i has counts[i] + 1 rows; id counts[i] = unknown
    std::vector<std::string> cond_names;
    std::vector<std::string> cond_units;
    std::vector<std::vector<float>> cond_edges;  // [n_cond][counts[i] - 1], strictly increasing
    std::vector<int32_t> silence_frame;          // [n_codebooks]
    int text_extra_tokens = 0;
    int file_type = 0;  // general.file_type; > 1 runs the logits head as Q8_0
};

struct Zonos3Assets {
    assets::ResourceBundle resources;
    Zonos3Hparams hparams;
    std::string tokenizer_json;
    std::shared_ptr<const assets::TensorSource> backbone;
    std::shared_ptr<const assets::TensorSource> decoder;
    std::shared_ptr<const assets::TensorSource> speaker_encoder;
};

std::shared_ptr<const Zonos3Assets> load_zonos3_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::zonos3
