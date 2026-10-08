#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/speech_encoders/wavlm_encoder.h"

#include <memory>
#include <vector>

namespace engine::models::plapre {

struct KanadeTransformerConfig {
    int64_t hidden = 768;
    int64_t intermediate = 2048;
    int64_t layers = 6;
    int64_t heads = 12;
    int64_t window_size = 65;
    int64_t speaker_dim = 128;
    float norm_eps = 1.0e-5f;
    float rope_theta = 10000.0f;
};

struct KanadeMelConfig : KanadeTransformerConfig {
    // Upstream CUDA autocast; disable only for a controlled F32 comparison.
    bool bf16_autocast = true;
    int64_t mel_bins = 80;
    int64_t upsample_factor = 2;
    int64_t postnet_channels = 256;
    int64_t postnet_layers = 4;
    int64_t postnet_kernel = 7;
    std::vector<int64_t> levels = {8, 8, 8, 5, 5};
};

struct KanadeEncoderConfig {
    KanadeTransformerConfig local;
    modules::WavlmEncoderConfig wavlm;
    int64_t downsample_factor = 2;
    int64_t global_channels = 384;
    int64_t global_intermediate = 1152;
    int64_t global_layers = 4;
    int64_t attention_channels = 128;
    std::vector<int64_t> local_ssl_layers = {6, 9};
    std::vector<int64_t> global_ssl_layers = {1, 2, 3, 4};
    std::vector<int64_t> levels = {8, 8, 8, 5, 5};
    bool normalize_ssl_features = true;
};

struct KanadeAudioFeatures {
    std::vector<int32_t> codes;
    std::vector<float> speaker;
};

class KanadeAudioEncoder {
public:
    KanadeAudioEncoder(std::shared_ptr<const assets::TensorSource> source,
        std::shared_ptr<const assets::TensorSource> wavlm_source,
        KanadeEncoderConfig config, core::ExecutionContext & execution);
    ~KanadeAudioEncoder();

    KanadeAudioFeatures encode(const std::vector<float> & audio_24khz);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class KanadeMelDecoder {
public:
    KanadeMelDecoder(std::shared_ptr<const assets::TensorSource> source,
        KanadeMelConfig config, core::ExecutionContext & execution);
    ~KanadeMelDecoder();

    std::vector<float> decode(const std::vector<int32_t> & codes,
        const std::vector<float> & speaker);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::plapre
