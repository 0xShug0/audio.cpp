#pragma once

#include <cstdint>
#include <vector>

namespace engine::models::kale_tts {

// The decoder head's signal part, on the host: a 64-harmonic sine source from the pitch, shaped per STFT frame by the
// network's filter (gain + phase at mel-spaced points, a pulse shift), plus plain and pitch-pulsing noise and a
// short-window noise burst path. Port of the release graphs' head (paradee SourceFilterHead2, Apache-2.0).
struct SourceFilterConfig {
    int64_t sample_rate = 24000;
    int64_t f0_hop = 300;
    int64_t n_fft = 600;
    int64_t hop = 150;
    int64_t harmonics = 64;
    int64_t mod = 2;
    int64_t points = 80;
    int64_t burst_points = 24;
};

// head [n, up * (channels * points + 1)] and burst [n, 4 * burst_points] per trunk frame, f0 [n] in Hz, noise [n * 300]
std::vector<float> source_filter(const SourceFilterConfig & config, const std::vector<float> & head, const std::vector<float> & burst,
                                 const std::vector<float> & f0, const std::vector<float> & noise);

}  // namespace engine::models::kale_tts
