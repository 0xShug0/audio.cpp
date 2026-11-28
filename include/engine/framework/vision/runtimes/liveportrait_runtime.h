#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <array>
#include <memory>
#include <vector>

namespace engine::modules {

struct LivePortraitRuntimeConfig {
    assets::TensorStorageType weight_type = assets::TensorStorageType::Native;
    bool cudnn = false;
    bool f16_activations = false;
};

// LivePortrait's learned appearance/warping/decoding pipeline. Inputs use planar
// RGB in [0,1]. All weights, prepared graphs and scratch are session-owned;
// ExecutionContext must outlive the runtime. Calls are serialized by the session.
class LivePortraitRuntime {
    public:
    LivePortraitRuntime(const assets::TensorSource & source, core::ExecutionContext & execution,
                        LivePortraitRuntimeConfig config = {});
    ~LivePortraitRuntime();
    LivePortraitRuntime(const LivePortraitRuntime &) = delete;
    LivePortraitRuntime & operator=(const LivePortraitRuntime &) = delete;

    // Component boundaries used for reference validation and renderer composition.
    // Appearance is C,D,H,W; warped features are C,H,W; decoded frames are C,H,W.
    // Pitch, yaw, roll, translation, expression, scale and canonical keypoints.
    std::array<std::vector<float>, 7> extract_motion(const std::vector<float> & rgb);
    // Landmark crop: RGB 224x224 in [0,1]; returns 203 normalized XY pairs.
    std::vector<float> extract_landmarks203(const std::vector<float> & rgb);
    std::vector<float> extract_appearance(const std::vector<float> & rgb);
    std::vector<float> warp_features(const std::vector<float> & appearance, const std::vector<float> & source_keypoints,
                                     const std::vector<float> & driving_keypoints);
    std::vector<float> decode_features(const std::vector<float> & features);
    // Source preparation and per-frame rendering share one PreparedGraph owner.
    // Source-only results persist on the backend; only completed RGB frames are read back.
    void set_source(const std::vector<float> & appearance, const std::vector<float> & source_keypoints);
    std::vector<float> render_frame(const std::vector<float> & driving_keypoints);
    std::vector<float> stitch_keypoints(const std::vector<float> & source, const std::vector<float> & driving);
    void release_graphs();

    private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::modules
