#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/qwen3_tts/assets.h"
#include "engine/models/qwen3_tts/types.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace engine::models::qwen3_tts {

class Qwen3SpeakerEncoderGraph;
struct Qwen3SpeakerEncoderWeights;

struct Qwen3SpeakerFeatures {
    std::vector<float> values;
    int64_t mel_bins = 0;
    int64_t frames = 0;
};

class Qwen3SpeakerEncoderRuntime {
public:
    Qwen3SpeakerEncoderRuntime(
        std::shared_ptr<const Qwen3TTSAssets> assets,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        engine::assets::TensorStorageType conv_weight_storage_type);
    ~Qwen3SpeakerEncoderRuntime();

    Qwen3SpeakerEmbedding encode(const runtime::AudioBuffer & audio) const;
    Qwen3SpeakerFeatures extract_features(const runtime::AudioBuffer & audio) const;
    Qwen3SpeakerEmbedding encode_features(const Qwen3SpeakerFeatures & features) const;

private:
    std::shared_ptr<const Qwen3TTSAssets> assets_;
    std::shared_ptr<const Qwen3SpeakerEncoderWeights> weights_;
    core::ExecutionContext * execution_context_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    mutable std::unique_ptr<Qwen3SpeakerEncoderGraph> graph_;
};

}  // namespace engine::models::qwen3_tts
