#pragma once

// AuK's Flux2Edit backbone: an MMDiT over a text stream and an audio stream.
//
// Tencent-Hunyuan/AuK, flux2_edit.py. Its module vocabulary is F5-TTS's -- the same
// ConvPositionEmbedding (grouped k31 g16, Mish twice), the same sinusoidal timestep
// embedding, the same adaLN modulation -- so src/community_models/f5_tts/dit_modules.cpp
// is the precedent for everything except the double-stream blocks, which F5 has no
// equivalent of.
//
// Built in slices, each diffed against a PyTorch tap:
//   (a) time embedding, text projection, audio embedding      <- this stage
//   (b) the 10 double-stream MMDiT blocks
//   (c) the 20 single-stream DiT blocks, output projection

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::auk {

struct AukDitConfig {
    int64_t dim = 1536;
    int64_t heads = 24;
    int64_t head_dim = 64;
    int64_t latent_dim = 64;
    int64_t text_hidden_dim = 2048;
    int64_t double_layers = 10;
    int64_t single_layers = 20;
    double ff_mult = 2.0;
    int64_t freq_embed_dim = 256;
    int64_t conv_pos_kernel = 31;
    int64_t conv_pos_groups = 16;
    float rms_norm_eps = 1e-6F;
    // SinusPositionEmbedding's `scale`, which is 1000 and not 1: the timestep is
    // multiplied by it before the sinusoids, so dropping it silently shifts every
    // frequency.
    float time_scale = 1000.0F;
    // x_transformers' RotaryEmbedding default. inv_freq is also stored in the
    // checkpoint, so this is a cross-check rather than the source of truth.
    float rope_theta = 10000.0F;

    void validate() const;
};

struct AukDitInputs {
    std::vector<float> text;        // [text_tokens, text_hidden_dim] from the conditioner
    std::vector<float> reference;   // [ref_frames, latent_dim], may be empty
    std::vector<float> noised;      // [gen_frames, latent_dim]
    float timestep = 0.0F;
    int64_t text_tokens = 0;
    int64_t ref_frames = 0;
    int64_t gen_frames = 0;
    // CFG's unconditional branch: the context is zeroed after projection and the
    // reference latent is zeroed before embedding. The second is the caller's job --
    // pass zeros in `reference` -- because the embedding has biases, so zeroing the
    // INPUT is not the same as zeroing the embedding.
    bool drop_text = false;
};

// Everything the reference taps, so a wrong number can be localized instead of only
// detected: the input stages, the first block of each phase, and the final velocity.
struct AukDitStageOutputs {
    std::vector<float> time_embedding;   // [dim]
    std::vector<float> projected_text;   // [text_tokens, dim]
    std::vector<float> embedded_audio;   // [gen_frames, dim] -- target stream only,
                                         // matching the reference's audio_embed tap
    std::vector<float> first_double;     // [text_tokens, dim] -- block 0's context output
    std::vector<float> first_single;     // [text_tokens + ref + gen, dim]
    std::vector<float> velocity;         // [gen_frames, latent_dim]
};

class AukDit {
public:
    AukDit(
        AukDitConfig config,
        const assets::TensorSource & source,
        core::ExecutionContext & execution,
        std::string prefix = "transformer");
    ~AukDit();

    AukDit(const AukDit &) = delete;
    AukDit & operator=(const AukDit &) = delete;

    AukDitStageOutputs forward(const AukDitInputs & inputs);

    int64_t loaded_tensor_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::auk
