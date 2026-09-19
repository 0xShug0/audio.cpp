#pragma once

// Qwen2.5-Omni's audio tower, as AuK's conditioner uses it.
//
// Most of AuK's claimed capabilities go through here: content editing, acoustic
// editing, paralinguistic editing, denoising, separation, speaker extraction and voice
// cloning all pass the SOURCE AUDIO to the Thinker as part of the message, not only to
// the DiT as a latent. Text-only generation is the exception, not the rule.
//
// Shape: log-mel (128 bins, 16 kHz) -> conv1 -> conv2 stride 2 -> sinusoidal positions
// -> 32 pre-norm layers with CHUNKED attention -> ln_post -> AvgPool1d(2) -> proj.
// Four mel frames become one token: stride 2 in conv2 and the average pool.
//
// ⚠ 16 kHz here, 24 kHz for the VAE. The same reference audio feeds both at different
// rates, and feeding either one the other's audio is a silent quality failure.

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::auk {

struct AukAudioTowerConfig {
    int64_t mel_bins = 128;
    int64_t d_model = 1280;
    int64_t heads = 20;
    int64_t layers = 32;
    int64_t ffn_dim = 5120;
    int64_t output_dim = 2048;
    int64_t max_source_positions = 1500;
    // Attention runs within windows of this many OUTPUT frames (200 mel frames), not
    // over the whole utterance: an hour of audio does not become an hour-square
    // attention matrix.
    int64_t window = 100;
    int64_t sample_rate = 16000;
    float layer_norm_eps = 1e-5F;

    assets::TensorStorageType weight_storage = assets::TensorStorageType::Native;

    void validate() const;
};

struct AukAudioEmbeddings {
    std::vector<float> values;   // [tokens, output_dim]
    int64_t tokens = 0;
    int64_t dim = 0;
};

class AukAudioTower {
public:
    AukAudioTower(
        AukAudioTowerConfig config,
        const assets::TensorSource & source,
        core::ExecutionContext & execution,
        std::string prefix = "thinker.audio_tower");
    ~AukAudioTower();

    AukAudioTower(const AukAudioTower &) = delete;
    AukAudioTower & operator=(const AukAudioTower &) = delete;

    // `mel` is [mel_bins, frames], channel-major, as the log-mel extractor produces.
    AukAudioEmbeddings encode(const std::vector<float> & mel, int64_t frames);

    // 16 kHz mono samples in, log-mel out -- the framework's Whisper extractor with
    // Omni's parameters (n_fft 400, hop 160, 128 bins).
    static std::vector<float> log_mel(const std::vector<float> & samples, int64_t & frames);

    int64_t loaded_tensor_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::auk
