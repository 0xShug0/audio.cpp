#pragma once

// HiFi-GAN vocoder + ResNet speaker encoder for xTTS v2.
//
// Converts GPT hidden states (d=1024) to 24 kHz waveform using the HiFi-GAN
// neural vocoder architecture with global speaker conditioning.

#include "engine/framework/core/backend.h"
#include "engine/framework/audio/resampling.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

struct ggml_context;

namespace engine::models::xtts_v2 {

// Opaque vocoder context (loads HiFi-GAN + speaker encoder from a GGUF).
class XttSv2Vocoder {
public:
    XttSv2Vocoder() = default;
    ~XttSv2Vocoder();

    XttSv2Vocoder(const XttSv2Vocoder &) = delete;
    XttSv2Vocoder & operator=(const XttSv2Vocoder &) = delete;

    // Load from GGUF. Returns true on success.
    bool load(const std::filesystem::path & path, const core::BackendConfig & backend_config);

    // Generate waveform from GPT hidden states.
    // latent: [d_model, T] float array (channels-first, d_model=1024).
    // spk_emb: [512] speaker embedding.
    // Returns output samples (24kHz mono).
    std::vector<float> generate(const float * latent, int T, const float * spk_emb) const;

    // Compute speaker embedding from reference audio (mono, 22050 Hz).
    std::vector<float> speaker_embed(const float * ref_pcm, int n_samples) const;

    // Mel normalization stats (80 values).
    const std::vector<float> & mel_stats() const { return mel_stats_; }

private:
    struct Impl;
    Impl* impl_ = nullptr;

    std::vector<float> mel_stats_;
    int n_threads_ = 4;
};

}  // namespace engine::models::xtts_v2

