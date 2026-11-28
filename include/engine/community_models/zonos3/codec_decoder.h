#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::zonos3 {

namespace detail {
struct Zonos3DecoderState;
}  // namespace detail

// The ZONOS3 MR-iSTFT codec decoder (zac-keep2.gguf, mossdual_istft): 15 Hz frames of 10 x 4096 codes to 44.1 kHz.
// Semantic RVQ -> causal ConvNeXt head, + residual RVQ -> 5 windowed-causal MOSS stages with patch upsampling to
// 240 Hz -> 8 causal ConvNeXt blocks + LayerNorm + two spectral projections run as ggml graphs; the iSTFT
// (n_fft 736 and 2208, hop 184), the 2->1 FIR mixer and the exact 44160->44100 resampler run on the host.
// Owns the weights and a small graph cache; one per session. Streams decode one at a time.
class Zonos3CodecDecoder {
public:
    Zonos3CodecDecoder(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution);
    ~Zonos3CodecDecoder();
    Zonos3CodecDecoder(const Zonos3CodecDecoder &) = delete;
    Zonos3CodecDecoder & operator=(const Zonos3CodecDecoder &) = delete;

    int sample_rate() const;        // 44100
    int samples_per_frame() const;  // 2940
    float headroom_db() const;      // mossdual_istft.headroom_db

private:
    friend class Zonos3DecoderStream;
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Per-request incremental decode state (MOSS K/V rings, ConvNeXt histories, iSTFT / mixer / resampler holdback).
// The decoder must outlive its streams.
class Zonos3DecoderStream {
public:
    explicit Zonos3DecoderStream(Zonos3CodecDecoder & decoder);
    ~Zonos3DecoderStream();
    Zonos3DecoderStream(const Zonos3DecoderStream &) = delete;
    Zonos3DecoderStream & operator=(const Zonos3DecoderStream &) = delete;

    // frames: [n][10] undelayed codes 0..4095. Returns only the 44.1 kHz samples newly final after this call, at raw
    // decoder level (the caller applies the headroom gain).
    std::vector<float> decode(const int32_t * frames, int n);
    // End of stream: the remaining samples; all decode() returns + flush() total frames_decoded * 2940.
    std::vector<float> flush();

private:
    Zonos3CodecDecoder & decoder_;
    std::unique_ptr<detail::Zonos3DecoderState> state_;
};

}  // namespace engine::models::zonos3
