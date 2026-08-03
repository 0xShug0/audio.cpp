#pragma once

#include "engine/framework/audio/istft_graph.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/speech_encoders/wavlm_encoder.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/miocodec/assets.h"
#include "engine/models/miocodec/components.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::miocodec {

struct MioCodecSslFeatures {
    std::vector<float> values;
    int64_t frames = 0;
};

std::vector<float> prepare_miocodec_mono_audio(
    const runtime::AudioBuffer & audio,
    int target_sample_rate);

class MioCodecSslFeatureExtractor {
public:
    MioCodecSslFeatureExtractor(
        std::shared_ptr<const MioCodecAssets> assets,
        engine::modules::WavlmEncoderComponent wavlm,
        std::vector<int> layer_indices,
        bool normalize_features,
        std::string timing_prefix);

    MioCodecSslFeatures extract(const std::vector<float> & mono_44k) const;
    const engine::modules::WavlmEncoderComponent & wavlm() const noexcept;

private:
    std::shared_ptr<const MioCodecAssets> assets_;
    engine::modules::WavlmEncoderComponent wavlm_;
    std::vector<int> layer_indices_;
    bool normalize_features_ = false;
    std::string timing_prefix_;
};

class MioCodecGlobalReferenceEncoder {
public:
    MioCodecGlobalReferenceEncoder(
        MioCodecSslFeatureExtractor ssl_extractor,
        MioCodecGlobalEncoderRuntime & global_encoder);

    MioCodecGlobalEmbedding embedding_for_reference(const std::vector<float> & reference_audio) const;

private:
    struct CachedReference {
        std::vector<float> audio;
        MioCodecGlobalEmbedding embedding;
    };

    MioCodecSslFeatureExtractor ssl_extractor_;
    MioCodecGlobalEncoderRuntime * global_encoder_ = nullptr;
    mutable std::vector<CachedReference> cached_references_;
};

class MioCodecWaveformReconstructor {
public:
    MioCodecWaveformReconstructor(
        std::shared_ptr<const MioCodecAssets> assets,
        core::ExecutionContext & execution_context);

    std::vector<float> reconstruct(
        const MioCodecWaveHead & head,
        const std::vector<float> & window);

private:
    std::shared_ptr<const MioCodecAssets> assets_;
    core::ExecutionContext * execution_context_ = nullptr;
    std::unique_ptr<engine::audio::HostLogMagnitudePhaseISTFT> host_istft_;
    int64_t host_istft_frames_ = 0;
    std::unique_ptr<engine::audio::CudaLogMagnitudePhaseISTFT> cuda_istft_;
    int64_t cuda_istft_frames_ = 0;
};

}  // namespace engine::models::miocodec
