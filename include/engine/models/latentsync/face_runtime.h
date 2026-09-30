#pragma once

#include "engine/models/latentsync/scrfd.h"
#include "engine/models/latentsync/face_alignment.h"
#include "engine/framework/core/execution_context.h"

#include <memory>

namespace engine::models::latentsync {

struct LatentSyncAlignedFace {
    std::vector<float> pixels;
    std::vector<float> landmarks;
    LatentSyncAffine transform;
};

class LatentSyncFaceRuntime {
public:
    LatentSyncFaceRuntime(core::ExecutionContext & execution,
        const assets::TensorSource & detector_source, const std::filesystem::path & detector_config,
        const assets::TensorSource & landmark_source, const std::filesystem::path & landmark_config,
        assets::TensorStorageType weight_type);
    ~LatentSyncFaceRuntime();

    // RGB NCHW: normalized 512x512 detector input; raw 0..255 192x192 landmark crop.
    std::vector<LatentSyncFaceDetection> detect(const std::vector<float> & image, float resize_scale);
    std::vector<float> landmarks(const std::vector<float> & image);
    LatentSyncAlignedFace align_frame(const std::vector<float> & rgb, int height, int width);
    void reset_alignment();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::latentsync
