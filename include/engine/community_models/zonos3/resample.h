#pragma once

#include <vector>

namespace engine::models::zonos3 {

// 44.1 kHz -> 16 kHz with the torchaudio Resample(44100, 16000) polyphase kernel a ZONOS3 GGUF stores as
// frontend.resample_44100 (row-major [160][475], phase-major). Output length ceil(160 * n / 441).
std::vector<float> zonos3_resample_44k_to_16k(const std::vector<float> & kernel, const std::vector<float> & x);

}  // namespace engine::models::zonos3
