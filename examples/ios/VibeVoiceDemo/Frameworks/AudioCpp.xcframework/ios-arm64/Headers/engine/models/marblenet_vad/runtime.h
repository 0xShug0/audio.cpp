#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/models/marblenet_vad/assets.h"
#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::marblenet_vad {

struct MarbleNetBackendWeights;

struct MarbleNetInferenceResult {
    int64_t frames = 0;
    int64_t num_classes = 0;
    std::vector<float> logits;
};

class MarbleNetRuntime {
public:
    MarbleNetRuntime(
        std::shared_ptr<const MarbleNetWeights> weights,
        core::ExecutionContext & execution_context,
        assets::TensorStorageType weight_storage_type);
    ~MarbleNetRuntime();

    std::vector<float> extract_audio_features(const runtime::AudioBuffer & audio) const;
    MarbleNetInferenceResult infer_features(const std::vector<float> & features, int64_t frames);
    MarbleNetInferenceResult infer_audio(const runtime::AudioBuffer & audio);
    std::vector<runtime::SpeechSegment> detect_speech(
        const runtime::AudioBuffer & audio,
        float threshold);

private:
    class Graph;
    Graph & ensure_graph(int64_t frames);

    std::shared_ptr<const MarbleNetWeights> weights_;
    std::shared_ptr<const MarbleNetBackendWeights> backend_weights_;
    core::ExecutionContext * execution_context_ = nullptr;
    std::unique_ptr<Graph> graph_;
};
}  // namespace engine::models::marblenet_vad
