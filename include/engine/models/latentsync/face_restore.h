#pragma once

#include "engine/models/latentsync/face_alignment.h"

namespace engine::models::latentsync {

std::vector<float> latentsync_resize_bicubic(const std::vector<float> & image,
    int channels, int height, int width, int output_height, int output_width);

// Original and returned RGB pixels are planar 0..255; generated face is planar -1..1, 512x512.
std::vector<float> latentsync_restore_face(const std::vector<float> & original,
    int height, int width, const std::vector<float> & face, const LatentSyncAffine & transform, int threads = 1);

}  // namespace engine::models::latentsync
