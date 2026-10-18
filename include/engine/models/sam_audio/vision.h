#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"

#include <map>

namespace engine::models::sam_audio {

struct SourceVideo;

class PECoreVisionEncoderModule {
public:
    PECoreVisionEncoderModule(const assets::TensorSource & source, core::ExecutionContext & execution);
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & pixels,
                           std::map<std::string, core::TensorValue> * boundaries = nullptr) const;

private:
    core::BackendWeightStore store_;
    std::map<std::string, core::TensorValue> weights_;
    core::TensorValue x_positions_, y_positions_;
};

class PECoreVisionEncoder {
public:
    PECoreVisionEncoder(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution);
    ~PECoreVisionEncoder();
    // Normalized RGB frames in NCHW order, fixed to the upstream 336px input.
    std::vector<float> encode(const std::vector<float> & pixels, int64_t batch);
    // Returns [1024, audio_frames] features aligned to the audio timeline.
    std::vector<float> encode_video(const SourceVideo & video, int64_t audio_frames,
                                    int64_t hop, int sample_rate);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::sam_audio
