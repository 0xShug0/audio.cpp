#pragma once

#include "engine/community_models/sopro_tts/assets.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::core {
class ExecutionContext;
}
namespace engine::assets {
enum class TensorStorageType;
}

namespace engine::community_models::sopro_tts {

struct SoproSpeakerWeights;
struct SoproSpeakerGraph;

struct SoproSpeakerEmbeddings {
    std::vector<float> id_emb;      // id_emb_dim, L2-normalised
    std::vector<float> style_emb;   // style_emb_dim
    std::vector<float> style_ctrl;  // style_ctrl_dim
};

// sopro/encoders/speaker.py. A log-mel front end feeding a three-stage
// gated depthwise ResNet with squeeze-excite, then two pooling heads:
// attentive statistics for speaker identity and multi-scale mean/std for
// style. The convolution trunk runs on the backend; the pooling heads and
// their small MLPs run on the host, where they cost a few hundred kFLOP.
class SoproSpeakerEncoderRuntime final {
public:
    SoproSpeakerEncoderRuntime(
        const SoproTTSAssets & assets,
        engine::core::ExecutionContext & execution_context,
        size_t weight_context_bytes,
        size_t graph_context_bytes,
        engine::assets::TensorStorageType matmul_storage_type,
        engine::assets::TensorStorageType conv_storage_type);
    ~SoproSpeakerEncoderRuntime();

    SoproSpeakerEncoderRuntime(const SoproSpeakerEncoderRuntime &) = delete;
    SoproSpeakerEncoderRuntime & operator=(const SoproSpeakerEncoderRuntime &) = delete;

    // audio16: mono 16 kHz reference waveform.
    SoproSpeakerEmbeddings encode(const std::vector<float> & audio16) const;

    int64_t sample_rate() const noexcept;

private:
    // Returns the fused trunk features [channels, frames], channel-major.
    std::vector<float> trunk(const std::vector<float> & log_mel, int64_t frames, int64_t & out_frames) const;

    const SoproSpeakerEncoderConfig & config_;
    engine::core::ExecutionContext & execution_context_;
    size_t graph_context_bytes_ = 0;
    std::shared_ptr<const SoproSpeakerWeights> weights_;
    mutable std::unique_ptr<SoproSpeakerGraph> graph_;
};

}  // namespace engine::community_models::sopro_tts
