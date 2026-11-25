#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/vision/modules/face_modules.h"

#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace engine::models::latentsync {

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
    std::vector<modules::ScrfdFaceDetection> detect(const std::vector<float> & image, float resize_scale);
    std::vector<float> landmarks(const std::vector<float> & image);
    LatentSyncAlignedFace align_frame(const std::vector<float> & rgb, int height, int width);
    void reset_alignment();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::latentsync
