#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"

#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace engine::models::latentsync {

struct LatentSyncScrfdNode {
    std::string op, output;
    std::vector<std::string> inputs;
    core::TensorShape shape;
    std::array<int, 4> axes{0, 1, 2, 3};
    modules::Conv2dConfig convolution_config;
    modules::Conv2dWeights convolution;
    core::TensorValue scale;
};

struct LatentSyncScrfdWeights {
    std::string input;
    core::TensorShape input_shape;
    std::vector<LatentSyncScrfdNode> nodes;
    std::vector<std::string> outputs;
    core::TensorValue negative_one;
};

LatentSyncScrfdWeights load_scrfd_weights(core::BackendWeightStore & store,
    const assets::TensorSource & source, const std::filesystem::path & config_path,
    assets::TensorStorageType type);

class LatentSyncScrfdDetector {
public:
    // Nine heads: scores at strides 8/16/32, then boxes, then five-point landmarks.
    std::vector<core::TensorValue> build(core::ModuleBuildContext & ctx,
        const core::TensorValue & input, const LatentSyncScrfdWeights & weights) const;
};

struct LatentSyncFaceDetection {
    std::array<float, 4> box;
    float score;
    std::array<float, 10> landmarks;
};

std::vector<LatentSyncFaceDetection> decode_scrfd_faces(
    const std::array<std::vector<float>, 9> & heads, float resize_scale,
    float score_threshold = 0.5F, float nms_threshold = 0.4F);

struct LatentSyncLandmarkLayer {
    modules::Conv2dConfig config;
    bool depthwise = false;
    modules::Conv2dWeights convolution;
    core::TensorValue slope, positive_scale;
};

struct LatentSyncLandmarkWeights {
    core::TensorValue input_bias, input_scale;
    std::vector<LatentSyncLandmarkLayer> layers;
    modules::LinearWeights output;
};

LatentSyncLandmarkWeights load_landmark_weights(core::BackendWeightStore & store,
    const assets::TensorSource & source, const std::filesystem::path & config_path,
    assets::TensorStorageType type);

class LatentSyncDepthwiseCnnLandmarks {
public:
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & image,
        const LatentSyncLandmarkWeights & weights) const;
};

using LatentSyncFacePoints = std::array<std::array<float, 2>, 3>;
using LatentSyncAffine = std::array<float, 6>;

LatentSyncFacePoints latentsync_alignment_points(const std::vector<float> & landmarks);

// Planar image layout; the matrix maps source pixels to destination pixels.
std::vector<float> latentsync_warp_affine(const std::vector<float> & image,
    int channels, int height, int width, const LatentSyncAffine & matrix,
    int output_height, int output_width, float fill, int threads);

std::vector<float> latentsync_resize_lanczos4(const std::vector<float> & image,
    int channels, int height, int width, int output_height, int output_width, int threads);

class LatentSyncFaceAlignment {
public:
    explicit LatentSyncFaceAlignment(int resolution = 512);
    LatentSyncAffine transform(const LatentSyncFacePoints & points, bool smooth = true);
    void reset();
    int crop_width() const { return width_; }
    int crop_height() const { return height_; }

private:
    LatentSyncFacePoints target_;
    int width_, height_;
    std::optional<std::array<float, 2>> previous_bias_;
};

std::vector<float> latentsync_resize_bicubic(const std::vector<float> & image,
    int channels, int height, int width, int output_height, int output_width);

// Original and returned RGB pixels are planar 0..255; generated face is planar -1..1, 512x512.
std::vector<float> latentsync_restore_face(const std::vector<float> & original,
    int height, int width, const std::vector<float> & face, const LatentSyncAffine & transform, int threads = 1);

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
