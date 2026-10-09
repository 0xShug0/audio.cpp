#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"

#include <memory>

namespace engine::models::firered_vad {

struct DetectionOptions {
    double threshold = 0.4;
    int smooth_window_frames = 5;
    int min_speech_frames = 20;
    int max_speech_frames = 2000;
    int min_silence_frames = 20;
    int merge_silence_frames = 0;
    int extend_speech_frames = 0;
    int pad_start_frames = 5;
    int chunk_frames = 30000;
};

class FireRedDfsmnRuntime {
public:
    FireRedDfsmnRuntime(std::shared_ptr<const assets::TensorSource> source,
                     core::ExecutionContext & execution);
    ~FireRedDfsmnRuntime();
    bool causal() const;
    std::vector<float> extract_features(const runtime::AudioBuffer & audio) const;
    std::vector<float> infer_features(const std::vector<float> & features, bool retain_state = false);
    runtime::TaskResult detect(const runtime::AudioBuffer & audio, const DetectionOptions & options);
    void reset(const DetectionOptions & options);
    runtime::StreamEvent process(const runtime::AudioChunk & chunk);
    runtime::TaskResult finalize();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Frame probabilities and segment rules are tested independently of inference.
std::vector<runtime::SpeechSegment> segment_probabilities(
    const std::vector<float> & probabilities, int64_t samples, const DetectionOptions & options);

}  // namespace engine::models::firered_vad
