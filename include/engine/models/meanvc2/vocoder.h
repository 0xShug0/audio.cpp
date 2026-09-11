#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::meanvc2 {

struct MeanVC2VocoderWeights;
struct MeanVC2VocoderGraph;

class MeanVC2VocoderRuntime final {
public:
    MeanVC2VocoderRuntime(
        std::shared_ptr<const engine::assets::TensorSource> source,
        engine::core::ExecutionContext & execution_context,
        size_t weight_context_bytes,
        size_t graph_context_bytes,
        engine::assets::TensorStorageType matmul_weight_storage_type,
        engine::assets::TensorStorageType conv_weight_storage_type);
    ~MeanVC2VocoderRuntime();

    runtime::AudioBuffer decode(const std::vector<float> & mel_frames, int64_t frames) const;
    runtime::AudioBuffer decode_streaming(const std::vector<float> & mel_frames, int64_t frames) const;
    void reset_streaming_state() const;
    runtime::AudioBuffer decode_streaming_chunk(const std::vector<float> & mel_frames, int64_t frames) const;
    runtime::AudioBuffer finish_streaming() const;

private:
    MeanVC2VocoderGraph & graph_for_frames(int64_t frames) const;

    engine::core::ExecutionContext & execution_context_;
    size_t graph_context_bytes_ = 0;
    std::shared_ptr<const engine::assets::TensorSource> source_;
    std::shared_ptr<const MeanVC2VocoderWeights> weights_;
    mutable std::vector<std::unique_ptr<MeanVC2VocoderGraph>> graphs_;
    mutable std::vector<float> streaming_mel_cache_;
    mutable std::vector<float> streaming_last_wav_;
};

}  // namespace engine::models::meanvc2
