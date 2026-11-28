#pragma once

#include "engine/framework/assets/resource_bundle.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace engine::models::zonos2 {

// Backbone hyperparameters read from zonos2-*.gguf, plus the release's fixed conditioning layout.
struct Zonos2Hparams {
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
    int audio_vocab = 0;  // codebook_size + 2
    int eoa_id = 0;       // codebook_size
    int audio_pad_id = 0; // codebook_size + 1
    int text_vocab = 0;   // the text pad id; conditioning ids sit just below it
    int spk_dim = 0;
    int spk_lda_dim = 0;
    std::vector<int32_t> expert_used;  // [n_layer]; 0 = dense FFN, else MoE top-k

    // Conditioning layout of the Zyphra/ZONOS2 release (params.json); the GGUF does not carry it.
    int speaking_rate_buckets = 8;
    std::vector<int> quality_bucket_counts = {12, 12, 12, 8, 8, 8};
    int speaker_bg_buckets = 2;
    int accurate_buckets = 1;
    int default_quality_feature = 5;  // trailing_silence_s
    int default_quality_bucket = 3;
};

struct Zonos2Assets {
    assets::ResourceBundle resources;
    Zonos2Hparams hparams;
    std::shared_ptr<const assets::TensorSource> backbone;
    std::shared_ptr<const assets::TensorSource> decoder;
    std::shared_ptr<const assets::TensorSource> speaker_encoder;
};

std::shared_ptr<const Zonos2Assets> load_zonos2_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::zonos2
