#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace engine::models::zonos2 {

inline constexpr int kZonos2SpeakerDim = 2048;

// Speaker embedding .npy: little-endian f4/f8, C order, shape [2048] or [1,2048]. Raw ECAPA output.
std::vector<float> read_speaker_embedding_npy(const std::filesystem::path & path);

// '<f4' (2048,), as Zyphra's spk-encoder-cli writes it.
std::vector<std::byte> speaker_embedding_npy(const std::vector<float> & embedding);

// Any little-endian float/int NPY as F32 (parity oracle files).
struct Zonos2NpyF32 {
    std::vector<int64_t> shape;
    std::vector<float> data;
};
Zonos2NpyF32 read_f32_npy(const std::filesystem::path & path);

}  // namespace engine::models::zonos2
