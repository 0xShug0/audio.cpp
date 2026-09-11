#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/linear_module.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace engine::models::echo_tts {

// Architecture constants. These mirror the EchoDiT constructor arguments in
// upstream inference.py::load_model_from_hf, which are not stored in the
// checkpoint. The converter re-emits them as GGUF metadata and the loader
// cross-checks the values it reads back against these defaults, so a future
// upstream config change surfaces as a load error rather than silent garbage.
struct EchoTtsConfig {
    // Denoiser (EchoDiT).
    int64_t latent_size = 80;
    int64_t model_size = 2048;
    int64_t num_layers = 24;
    int64_t num_heads = 16;
    int64_t intermediate_size = 5888;
    float norm_eps = 1.0e-5F;

    // Text encoder.
    int64_t text_vocab_size = 256;
    int64_t text_model_size = 1280;
    int64_t text_num_layers = 14;
    int64_t text_num_heads = 10;
    int64_t text_intermediate_size = 3328;

    // Speaker encoder.
    int64_t speaker_patch_size = 4;
    int64_t speaker_model_size = 1280;
    int64_t speaker_num_layers = 14;
    int64_t speaker_num_heads = 10;
    int64_t speaker_intermediate_size = 3328;

    // Conditioning.
    int64_t timestep_embed_size = 512;
    int64_t adaln_rank = 256;

    // Sampling / windowing limits, all fixed by training.
    int64_t max_sequence_length = 640;      // 640 * 2048 / 44100 = 29.7215 s
    int64_t max_text_length = 768;          // hard truncation, UTF-8 bytes
    int64_t max_speaker_latent_length = 6400;
    int64_t speaker_chunk_latents = 640;    // 640 * 2048 samples per encode chunk

    // Autoencoder.
    int64_t ae_downsample_factor = 2048;
    int64_t ae_latent_dim = 1024;           // Fish S1-DAC z_q channel count
    int64_t sample_rate = 44100;

    int64_t head_dim() const { return model_size / num_heads; }
    int64_t text_head_dim() const { return text_model_size / text_num_heads; }
    int64_t speaker_head_dim() const { return speaker_model_size / speaker_num_heads; }

    // The DiT applies RoPE to the first half of the heads only
    // (model.py::JointAttention::_apply_rotary_half chunks along the head axis).
    int64_t rope_heads() const { return num_heads / 2; }

    void validate() const;
};

// RMSNorm in this model always uses a weight and never a bias. Head-wise norms
// (q_norm / k_norm) carry a (num_heads, head_dim) weight applied after the
// reduction over head_dim.
struct EchoRmsNormWeights {
    core::TensorValue weight;
};

struct EchoMlpWeights {
    modules::LinearWeights w1;
    modules::LinearWeights w2;
    modules::LinearWeights w3;
};

// SelfAttention, used by both encoders. `gate` is applied as
// output * sigmoid(gate) before the output projection.
struct EchoSelfAttentionWeights {
    modules::LinearWeights wq;
    modules::LinearWeights wk;
    modules::LinearWeights wv;
    modules::LinearWeights wo;
    modules::LinearWeights gate;
    EchoRmsNormWeights q_norm;
    EchoRmsNormWeights k_norm;
};

struct EchoEncoderBlockWeights {
    EchoSelfAttentionWeights attention;
    EchoMlpWeights mlp;
    EchoRmsNormWeights attention_norm;
    EchoRmsNormWeights mlp_norm;
};

struct EchoTextEncoderWeights {
    core::TensorValue text_embedding;
    std::vector<EchoEncoderBlockWeights> blocks;
};

struct EchoSpeakerEncoderWeights {
    modules::LinearWeights in_proj;  // (latent_size * patch_size) -> speaker_model_size
    std::vector<EchoEncoderBlockWeights> blocks;
};

// LowRankAdaLN: each of shift/scale/gate is refined by a rank-256 residual
// MLP, `up(down(silu(v))) + v`. down has no bias, up does.
struct EchoAdaLnWeights {
    modules::LinearWeights shift_down;
    modules::LinearWeights scale_down;
    modules::LinearWeights gate_down;
    modules::LinearWeights shift_up;
    modules::LinearWeights scale_up;
    modules::LinearWeights gate_up;
};

struct EchoJointAttentionWeights {
    modules::LinearWeights wq;
    modules::LinearWeights wk;
    modules::LinearWeights wv;
    modules::LinearWeights wk_text;
    modules::LinearWeights wv_text;
    modules::LinearWeights wk_speaker;
    modules::LinearWeights wv_speaker;
    modules::LinearWeights gate;
    modules::LinearWeights wo;
    EchoRmsNormWeights q_norm;
    EchoRmsNormWeights k_norm;
};

struct EchoDitBlockWeights {
    EchoJointAttentionWeights attention;
    EchoMlpWeights mlp;
    EchoAdaLnWeights attention_adaln;
    EchoAdaLnWeights mlp_adaln;
};

struct EchoDitWeights {
    EchoTextEncoderWeights text_encoder;
    EchoSpeakerEncoderWeights speaker_encoder;
    EchoRmsNormWeights text_norm;
    EchoRmsNormWeights speaker_norm;

    modules::LinearWeights cond_0;  // timestep_embed_size -> model_size
    modules::LinearWeights cond_2;  // model_size -> model_size
    modules::LinearWeights cond_4;  // model_size -> model_size * 3

    modules::LinearWeights in_proj;   // latent_size -> model_size, bias
    std::vector<EchoDitBlockWeights> blocks;
    EchoRmsNormWeights out_norm;
    modules::LinearWeights out_proj;  // model_size -> latent_size, bias
};

// PCA basis mapping the DiT's 80-D working space to the 1024-D Fish z_q space.
// Stored row-major as (latent_size, ae_latent_dim).
struct EchoPcaState {
    std::vector<float> components;  // latent_size * ae_latent_dim
    std::vector<float> mean;        // ae_latent_dim
    float latent_scale = 1.0F;
};

// Request-level sampler configuration, parsed from spec options.
struct EchoSamplerOptions {
    int num_steps = 40;
    float cfg_scale_text = 3.0F;
    float cfg_scale_speaker = 8.0F;
    float cfg_min_t = 0.5F;
    float cfg_max_t = 1.0F;
    // Evaluate the two unconditional lanes only every Nth step inside the CFG
    // window, reusing the previous guidance correction in between. 1 reproduces
    // upstream exactly.
    //
    // The correction -- w_text*(v_cond - v_text) + w_speaker*(v_cond - v_speaker)
    // -- varies slowly in t even though v_cond does not, so it tolerates being
    // resampled. What matters is how many times it is actually measured across
    // the guided phase, which is num_steps/2 rounded up, divided by this value:
    //
    //   num_steps=40, interval=2 -> 10 refreshes, 80 -> 60 lane-evals (1.33x)
    //   num_steps=30, interval=2 ->  8 refreshes, 60 -> 46 lane-evals (1.30x)
    //   num_steps=14, interval=2 ->  4 refreshes, 28 -> 22 lane-evals (1.27x)
    //
    // Ten refreshes is dense enough that each reused correction is one small
    // t-step stale; four is not. So this is worth raising at 30+ steps and is
    // not worth it at 14, where the correction is already coarsely sampled and
    // the speaker term's weight of 8.0 multiplies any staleness straight into
    // timbre and pronunciation -- a failure mode you hear rather than see.
    //
    // Note that num_steps=40 with interval=2 and num_steps=30 with interval=1
    // both cost 60 lane-evals. They spend the same compute differently: fewer
    // steps coarsens the whole ODE trajectory, while a longer interval leaves
    // the trajectory intact and only lets the guidance go stale. Neither
    // dominates on paper; compare them by listening before committing.
    int cfg_interval = 1;
    std::optional<float> truncation_factor = 0.8F;
    std::optional<float> speaker_kv_scale;
    std::optional<int> speaker_kv_max_layers;
    std::optional<float> speaker_kv_min_t;
    int64_t sequence_length = 640;
    // True when the caller pinned sequence_length explicitly, which suppresses
    // the per-chunk window estimate.
    bool window_pinned = false;
    uint64_t seed = 0;
};

}  // namespace engine::models::echo_tts
