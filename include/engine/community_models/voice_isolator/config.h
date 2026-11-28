#pragma once

#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

// Voice Isolator architecture: the checkpoint's own description of the network
// and the band split of its spectrum. Pure data, no backend.
namespace engine::models::voice_isolator {

enum class BlockKind {
    Attention,
    Mamba,
};

struct Config {
    int64_t width = 320;
    int64_t num_heads = 8;
    int64_t n_fft = 2048;
    int64_t hop_length = 480;
    int64_t sample_rate = 48000;
    // (band width in Hz, upper edge in Hz), low to high.
    std::vector<std::pair<int64_t, int64_t>> band_plan = {
        {100, 1000}, {250, 4000}, {500, 8000}, {1000, 16000}, {2000, 24000}};
    std::vector<BlockKind> blocks = {BlockKind::Attention, BlockKind::Mamba, BlockKind::Attention,
        BlockKind::Mamba, BlockKind::Attention, BlockKind::Mamba, BlockKind::Attention};
    float correction_bound = 0.5F;
    double chunk_seconds = 12.0;
    double overlap_seconds = 2.0;
    bool exclusive_time = true;
    bool exclusive_band = false;
    int64_t mamba_state = 16;
    int64_t mamba_conv = 4;
    int64_t mamba_expand = 2;
    // Causal checkpoints stream; two-sided ones render whole chunks offline.
    bool causal = false;
    int64_t lookahead_frames = 0;
    std::vector<int64_t> lookahead_choices;
    // Causal only: each frame's time attention sees itself and window - 1 frames before it.
    int64_t attention_window_frames = 0;

    int64_t head_dim() const noexcept { return width / num_heads; }
    int64_t bins() const noexcept { return n_fft / 2 + 1; }
    int64_t mamba_inner() const noexcept { return mamba_expand * width; }
    int64_t dt_rank() const noexcept { return (width + 15) / 16; }
    // Streaming look-ahead positions the checkpoint was trained at.
    std::vector<int64_t> streaming_lookaheads() const;
};

// Parses the JSON architecture a released checkpoint stores in its safetensors
// "config" metadata. Settings this port does not implement are refused.
Config parse_config(std::string_view json);

// The bands' bin ranges [start, end), covering every bin once; the encoder and
// decoder work on bands zero-padded to the widest one.
struct BandLayout {
    std::vector<std::pair<int64_t, int64_t>> edges;
    int64_t bins = 0;
    int64_t max_band_bins = 0;

    int64_t bands() const noexcept { return static_cast<int64_t>(edges.size()); }
    // (re, im) of every bin of the band.
    int64_t encoder_inputs() const noexcept { return 2 * max_band_bins; }
    // (mask re, mask im, residual re, residual im) of every bin of the band.
    int64_t decoder_outputs() const noexcept { return 4 * max_band_bins; }
};

BandLayout make_band_layout(const Config & config);

}  // namespace engine::models::voice_isolator
