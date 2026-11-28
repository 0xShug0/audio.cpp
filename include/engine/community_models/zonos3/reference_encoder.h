#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::zonos3 {

// Raw-reference encoder: ZAC dual encoder (zac-encoder.gguf, mossdual_encoder) with its w2v-BERT SSL teacher
// (ssl-teacher.gguf, w2vbert_ssl). Turns 44.1 kHz mono audio into ordinary (undelayed) 10-codebook codes.
// Weights load in the constructor; destroy the encoder to release them.
class Zonos3ReferenceEncoder {
public:
    Zonos3ReferenceEncoder(std::shared_ptr<const assets::TensorSource> zac_encoder,
                           std::shared_ptr<const assets::TensorSource> ssl_teacher, core::ExecutionContext & execution);
    ~Zonos3ReferenceEncoder();
    Zonos3ReferenceEncoder(const Zonos3ReferenceEncoder &) = delete;
    Zonos3ReferenceEncoder & operator=(const Zonos3ReferenceEncoder &) = delete;

    // mono_44k: finite samples in [-1, 1] at 44100 Hz, 1..600 s. Returns row-major [frames][10] codes 0..4095,
    // frames = ceil(n / 2940).
    std::vector<int32_t> encode(const std::vector<float> & mono_44k) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::zonos3
