#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::sopro_tts {

// samuel-vitorino/sopro-v2-turbo. Four stages run per request:
//   speaker encoder  (16 kHz reference -> id/style/style-ctrl embeddings)
//   semantic encoder (24 kHz reference -> FSQ semantic token ids)
//   semantic LM      (text + style prefix + prompt tokens -> semantic tokens)
//   acoustic head    (semantic tokens -> mel, rectified-flow Euler solve)
//   vocoder          (mel -> 24 kHz waveform, Vocos ConvNeXt + ISTFT head)
// Every field below mirrors one key of the checkpoint's config.json; nothing
// about the architecture is hardcoded so a retrained variant loads unchanged.

// config.json -> "model"
struct SoproModelConfig {
    int64_t latent_dim = 1280;
    int64_t semantic_vocab_size = 4375;
    int64_t text_vocab_size = 8192;
    int64_t max_text_len = 2048;
    int64_t cond_in_dim = 328;      // id_emb + style_emb + style_ctrl
    int64_t cond_hidden_dim = 512;

    int64_t ar_model_dim = 512;
    int64_t ar_blocks = 12;
    int64_t ar_heads = 8;
    int64_t ar_kv_heads = 8;        // defaults to ar_heads when absent
    float ar_ffn_mult = 4.0F;
    bool ar_qk_rms_norm = true;
    int64_t style_prefix_tokens = 8;

    int64_t acoustic_time_embed_dim = 256;
    float acoustic_sway_sampling_coef = -1.0F;
    int64_t acoustic_upsampler_kernel_size = 3;
    int64_t acoustic_dit_dim = 512;
    int64_t acoustic_dit_depth = 8;
    int64_t acoustic_dit_heads = 8;
    int64_t acoustic_dit_dim_head = 64;
    float acoustic_dit_ff_mult = 2.0F;
    int64_t acoustic_mu_dim = 100;  // defaults to acoustic_mel_n_mels
    int64_t acoustic_spk_dim = 80;
    int64_t acoustic_pre_lookahead_frames = 3;
    int64_t acoustic_pos_kernel_size = 31;
    float acoustic_sigma_min = 1.0e-6F;
    int64_t acoustic_num_left_chunks = -1;
    int64_t acoustic_mel_n_mels = 100;
    int64_t acoustic_mel_hop_length = 256;
    std::vector<float> acoustic_mel_mean;
    std::vector<float> acoustic_mel_std;

    // BOS/EOS live just past the FSQ codebook; the AR head is
    // Linear(dim -> semantic_vocab_size + 2).
    int64_t semantic_bos_id() const noexcept { return semantic_vocab_size; }
    int64_t semantic_eos_id() const noexcept { return semantic_vocab_size + 1; }
    int64_t ar_head_dim() const noexcept { return ar_model_dim / ar_heads; }
    int64_t ar_ffn_dim() const noexcept;
    int64_t acoustic_dit_ff_dim() const noexcept;
};

// config.json -> "semantic_encoder"
struct SoproSemanticEncoderConfig {
    int64_t n_mels = 80;
    int64_t d_model = 512;
    int64_t layers = 6;
    int64_t heads = 8;
    int64_t ffn_dim = 2048;
    int64_t max_positions = 1500;
    std::vector<int64_t> fsq_levels{7, 5, 5, 5, 5};
    int64_t sample_rate = 16000;
    int64_t n_fft = 400;
    int64_t hop_length = 160;
    int64_t token_samples_24k = 1024;

    int64_t head_dim() const noexcept { return d_model / heads; }
    int64_t digit_dim() const noexcept;   // sum(fsq_levels)
    int64_t codebook_size() const noexcept;  // prod(fsq_levels)
};

// config.json -> "speaker_encoder"
struct SoproSpeakerEncoderConfig {
    int64_t sample_rate = 16000;
    int64_t n_mels = 80;
    int64_t n_fft = 1024;
    int64_t win_length = 400;
    int64_t hop_length = 160;
    float f_min = 20.0F;
    float f_max = 7600.0F;
    float mel_log_floor = 1.0e-5F;
    int64_t stem_channels = 128;
    std::vector<int64_t> stage_channels{160, 192, 224};
    std::vector<int64_t> blocks_per_stage{4, 4, 4};
    std::vector<int64_t> dilation_cycle{1, 2, 4, 8};
    int64_t depthwise_kernel_size = 5;
    int64_t se_reduction = 8;
    int64_t id_emb_dim = 192;
    int64_t style_emb_dim = 128;
    int64_t style_ctrl_dim = 8;
    int64_t id_head_hidden = 256;
    int64_t style_head_hidden = 256;
    int64_t attn_hidden = 128;
};

// config.json -> "vocoder" / "vocoder_streaming"
struct SoproVocoderConfig {
    int64_t sample_rate = 24000;
    int64_t n_fft = 1024;
    int64_t hop_length = 256;
    int64_t n_mels = 100;
    int64_t dim = 512;
    int64_t intermediate_dim = 1536;
    int64_t num_layers = 14;
    float max_magnitude = 100.0F;
    // sopro/config.py VocoderConfig.band_limit_hz. Zero (or a negative value)
    // disables the cut; the published checkpoints do not carry the key, so the
    // default has to match the reference dataclass.
    float band_limit_hz = 10900.0F;
    bool causal = false;
    int64_t lookahead_frames = 0;
    std::vector<int64_t> block_lookaheads;
};

// config.json -> "generation"
struct SoproGenerationConfig {
    float temperature = 0.8F;
    float top_p = 0.9F;
    int64_t top_k = 25;
    int64_t steps = 2;
    float max_seconds = 30.0F;
    float min_seconds = 0.4F;
    int64_t max_segment_chars = 300;
    float ref_seconds = 10.0F;
    int64_t style_tokens = 160;
    int64_t prompt_tokens = 120;
    int64_t stream_chunk_frames = 64;
};

struct SoproTTSConfig {
    int64_t sample_rate = 24000;
    SoproModelConfig model;
    SoproSemanticEncoderConfig semantic_encoder;
    SoproSpeakerEncoderConfig speaker_encoder;
    SoproVocoderConfig vocoder;
    SoproVocoderConfig vocoder_streaming;
    SoproGenerationConfig generation;

    // Mel frames produced per semantic token (token_samples_24k / mel hop).
    int64_t hop_ratio() const noexcept;
};

struct SoproTTSAssets {
    assets::ResourceBundle resources;
    SoproTTSConfig config;
    std::shared_ptr<const assets::TensorSource> model_weights;
    std::shared_ptr<const assets::TensorSource> semantic_encoder_weights;
    std::shared_ptr<const assets::TensorSource> speaker_encoder_weights;
    std::shared_ptr<const assets::TensorSource> vocoder_weights;
    std::filesystem::path tokenizer_path;
};

// Per-request knobs; defaults come from config.json "generation".
struct SoproRequestOptions {
    std::string language;   // "", en, pt, fr, de
    float temperature = 0.8F;
    float top_p = 0.9F;
    int64_t top_k = 25;
    int64_t steps = 2;
    float max_seconds = 30.0F;
    float min_seconds = 0.4F;
    int64_t max_segment_chars = 300;
    float ref_seconds = 10.0F;
    uint64_t seed = 0;
    bool has_seed = false;
};

std::shared_ptr<const SoproTTSAssets> load_sopro_tts_assets(
    const std::filesystem::path & model_path);

// The front ends reuse the analysis window and mel filterbank that torchaudio
// stores as persistent buffers instead of rebuilding them, which is what keeps
// them comparable with the reference pipeline. Fail early and say why when a
// checkpoint was exported without them.
void require_frontend_buffers(
    const assets::TensorSource & source,
    const char * stage,
    std::initializer_list<const char *> tensor_names);

}  // namespace engine::community_models::sopro_tts
