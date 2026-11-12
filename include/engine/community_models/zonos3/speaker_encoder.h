#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <memory>
#include <vector>

namespace engine::models::zonos3 {

// MECT-B2 speaker encoder (GGUF architecture `mect_spk`): Kaldi fbank + CMN, MECT-B2 blocks, ASTP pooling,
// BN affine and a 192-d linear projection.
class Zonos3SpeakerEncoder {
public:
    Zonos3SpeakerEncoder(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution);
    ~Zonos3SpeakerEncoder();
    Zonos3SpeakerEncoder(const Zonos3SpeakerEncoder &) = delete;
    Zonos3SpeakerEncoder & operator=(const Zonos3SpeakerEncoder &) = delete;

    // mono_44k: float [-1, 1] at 44100 Hz; the first 60 s are used. Returns 192 floats, not normalized.
    std::vector<float> embed(const std::vector<float> & mono_44k) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::zonos3
