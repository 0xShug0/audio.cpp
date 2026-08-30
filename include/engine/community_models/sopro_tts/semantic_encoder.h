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

struct SoproSemanticEncoderWeights;
struct SoproSemanticEncoderGraph;

// sopro/encoders/semantic.py. A Whisper-style log-mel front end and two
// striding convolutions feed six non-causal transformer layers; the result is
// resampled to one frame per 1024 output samples and quantised by a finite
// scalar quantiser whose per-level arg-maxes are packed into a single id.
class SoproSemanticEncoderRuntime final {
public:
    SoproSemanticEncoderRuntime(
        const SoproTTSAssets & assets,
        engine::core::ExecutionContext & execution_context,
        size_t weight_context_bytes,
        size_t graph_context_bytes,
        engine::assets::TensorStorageType matmul_storage_type,
        engine::assets::TensorStorageType conv_storage_type);
    ~SoproSemanticEncoderRuntime();

    SoproSemanticEncoderRuntime(const SoproSemanticEncoderRuntime &) = delete;
    SoproSemanticEncoderRuntime & operator=(const SoproSemanticEncoderRuntime &) = delete;

    // audio24: mono 24 kHz reference waveform. Returns ceil(n / 1024) token ids.
    std::vector<int32_t> encode(const std::vector<float> & audio24) const;

private:
    const SoproSemanticEncoderConfig & config_;
    engine::core::ExecutionContext & execution_context_;
    size_t graph_context_bytes_ = 0;
    std::shared_ptr<const SoproSemanticEncoderWeights> weights_;
    mutable std::unique_ptr<SoproSemanticEncoderGraph> graph_;
};

}  // namespace engine::community_models::sopro_tts
