#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <memory>
#include <vector>

namespace engine::models::zonos2 {

inline constexpr int kZonos2SpeakerSampleRate = 24000;

// The ECAPA-TDNN speaker encoder (spk-encoder.gguf): 24 kHz mono -> 128-bin log-mel (n_fft 1024, hop 256, reflect
// padding, Slaney 0-12 kHz) -> conv k5 -> three SE-Res2Net blocks -> MFA -> attentive statistics pooling -> fc ->
// [2048], unnormalized. The graph is built for the exact frame count of each call.
class Zonos2SpeakerEncoder {
public:
    Zonos2SpeakerEncoder(std::shared_ptr<const assets::TensorSource> spk, core::ExecutionContext & execution);
    ~Zonos2SpeakerEncoder();
    Zonos2SpeakerEncoder(const Zonos2SpeakerEncoder &) = delete;
    Zonos2SpeakerEncoder & operator=(const Zonos2SpeakerEncoder &) = delete;

    std::vector<float> embed(const std::vector<float> & mono_24k) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::zonos2
