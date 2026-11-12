#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace engine::models::zonos3 {

// Speaker embedding .npy: little-endian f4/f8, C order, shape [192] or [1,192]. Raw encoder output.
std::vector<float> read_speaker_embedding_npy(const std::filesystem::path & path);
// Reference codes .npy: little-endian i4/i8/u4/u8/f4/f8, C order, shape [T,10], integer values 0..4095.
// Every frame is validated; the first max_frames frames are returned row-major.
std::vector<int32_t> read_reference_codes_npy(const std::filesystem::path & path, int max_frames);

// '<f4' (192,) and '<i4' (T, 10) files, as ZONOS3's encoders write them.
std::vector<std::byte> speaker_embedding_npy(const std::vector<float> & embedding);
std::vector<std::byte> reference_codes_npy(const std::vector<int32_t> & codes);

// Any little-endian float/int NPY as F32 (parity oracle files).
struct Zonos3NpyF32 {
    std::vector<int64_t> shape;
    std::vector<float> data;
};
Zonos3NpyF32 read_f32_npy(const std::filesystem::path & path);

}  // namespace engine::models::zonos3
