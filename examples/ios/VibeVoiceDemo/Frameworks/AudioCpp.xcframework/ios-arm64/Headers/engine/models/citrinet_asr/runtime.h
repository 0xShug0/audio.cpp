#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/models/citrinet_asr/assets.h"
#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::citrinet_asr {

struct CitrinetBackendWeights;

struct CitrinetInferenceResult {
    int64_t input_frames = 0;
    int64_t output_frames = 0;
    int64_t num_classes = 0;
    std::vector<float> logits;
};

struct CitrinetTranscriptionResult {
    std::string text;
    std::vector<int32_t> token_ids;
    CitrinetInferenceResult inference;
};

class CitrinetRuntime {
public:
    CitrinetRuntime(
        std::shared_ptr<const CitrinetWeights> weights,
        core::ExecutionContext & execution_context,
        assets::TensorStorageType weight_storage_type);
    ~CitrinetRuntime();

    CitrinetInferenceResult infer_features(const std::vector<float> & features, int64_t frames);
    CitrinetInferenceResult infer_audio(const runtime::AudioBuffer & audio);
    CitrinetTranscriptionResult transcribe_features(const std::vector<float> & features, int64_t frames);
    CitrinetTranscriptionResult transcribe_audio(const runtime::AudioBuffer & audio);

private:
    class Graph;
    Graph & ensure_graph(int64_t frames);

    std::shared_ptr<const CitrinetWeights> weights_;
    std::shared_ptr<const CitrinetBackendWeights> backend_weights_;
    core::ExecutionContext * execution_context_ = nullptr;
    std::unique_ptr<Graph> graph_;
};
}  // namespace engine::models::citrinet_asr
