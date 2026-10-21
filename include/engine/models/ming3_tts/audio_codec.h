#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/attention_fallback.h"
#include "engine/models/ming3_tts/assets.h"

#include <memory>
#include <vector>

namespace engine::models::ming3_tts {

class MingAudioEncoder {
public:
    MingAudioEncoder(std::shared_ptr<const assets::TensorSource> source,
        const MingAudioCodecConfig & config, core::ExecutionContext & execution,
        core::AttentionPreference attention = core::AttentionPreference::Auto);
    ~MingAudioEncoder();

    int64_t latent_frames(size_t samples) const;
    std::vector<float> encode(const std::vector<float> & waveform, const std::vector<float> & noise);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class MingAudioDecoder {
public:
    MingAudioDecoder(std::shared_ptr<const assets::TensorSource> source,
        const MingAudioCodecConfig & config, core::ExecutionContext & execution,
        core::AttentionPreference attention = core::AttentionPreference::Auto);
    ~MingAudioDecoder();

    std::vector<float> decode(const std::vector<float> & latent);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::ming3_tts
