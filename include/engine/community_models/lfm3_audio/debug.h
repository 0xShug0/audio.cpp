#pragma once

// LFM3-Audio stage dumps and MaskGIT order injection, for parity checks
// against the python reference.

#include "engine/community_models/lfm3_audio/depth.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace engine::community_models::lfm3_audio {

// LFM3_AUDIO_DUMP_DIR=<dir>: each call writes <dir>/<name>.npy (NumPy v1.0 header, '<f4' or '<i4', C order).
bool lfm3_dump_enabled();
void lfm3_dump(const std::string & name, const float * data, const std::vector<int64_t> & shape);
void lfm3_dump(const std::string & name, const int32_t * data, const std::vector<int64_t> & shape);
int64_t lfm3_dump_frames();  // LFM3_AUDIO_DUMP_FRAMES (default 4): frames whose per-pass depth logits are dumped
// LFM3_AUDIO_MASKGIT_ORDERS=<file.npy int32 [F,12]>: orders injected for the first F frames of every generation (test/debug).
std::vector<Lfm3Codes> lfm3_injected_orders();
std::vector<int32_t> lfm3_read_npy_i32(const std::filesystem::path & path, std::vector<int64_t> * shape);
std::vector<float> lfm3_read_npy_f32(const std::filesystem::path & path, std::vector<int64_t> * shape);

}  // namespace engine::community_models::lfm3_audio
