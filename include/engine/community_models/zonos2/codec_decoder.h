#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::zonos2 {

// The Descript Audio Codec 44.1 kHz decoder (dac.gguf): 9 x 1024-entry folded codebooks -> conv_in -> four
// snake / transposed-conv / residual-unit blocks (strides 8, 8, 4, 2) -> snake -> conv_out -> tanh. 512 samples per
// frame. Owns the weights and a small per-length graph cache; one per session.
class Zonos2CodecDecoder {
public:
    Zonos2CodecDecoder(std::shared_ptr<const assets::TensorSource> dac, core::ExecutionContext & execution);
    ~Zonos2CodecDecoder();
    Zonos2CodecDecoder(const Zonos2CodecDecoder &) = delete;
    Zonos2CodecDecoder & operator=(const Zonos2CodecDecoder &) = delete;

    int sample_rate() const;        // 44100
    int samples_per_frame() const;  // 512 = product of dac.decoder_rates

    // frames: [n][9] undelayed codes 0..1023, including `left` / `right` context frames at either end. Returns the
    // 44.1 kHz samples of the middle n - left - right frames only.
    std::vector<float> decode(const int32_t * frames, int n, int left, int right);
    // All n frames in windows of `block` frames with up to `context` frames on each side (clamped at the ends).
    // With context >= 16 the result equals one decode(frames, n, 0, 0) pass while the graph stays bounded.
    std::vector<float> decode_tiled(const int32_t * frames, int n, int block, int context);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Raw delayed rows [row_count][9] -> undelayed frames [frame_begin, frame_end): frame t, codebook j is
// rows[t + j][j] when t + j < row_count, else the audio pad 1025; every code is then clamped to 0..1023. Frame t is
// final once row t + 8 exists.
std::vector<int32_t> zonos2_undelay(const int32_t * rows, int row_count, int frame_begin, int frame_end);

}  // namespace engine::models::zonos2
