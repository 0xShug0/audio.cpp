#pragma once

#include "engine/community_models/sopro_tts/assets.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::core {
class ExecutionContext;
}
namespace engine::assets {
enum class TensorStorageType;
}

namespace engine::community_models::sopro_tts {

struct SoproVocoderWeights;
struct SoproVocoderGraph;

// ISTFTHead band limit (sopro/vocoder.py band_limit_bin): the first FFT bin the
// head zeroes, i.e. how many of the n_fft/2 + 1 bins it actually synthesises.
// A band_limit_hz of zero or less keeps every bin.
int64_t band_limit_bin(const SoproVocoderConfig & config);

// sopro/vocoder.py, offline path. A Vocos backbone (Conv1d embed, 14 ConvNeXt
// blocks with per-channel gamma, final LayerNorm) feeding one ISTFT head:
// Linear(dim -> n_fft + 2) split into log-magnitude and phase, then a single
// centred inverse STFT with the checkpoint's Hann window.
//
// The same object also owns the analysis mel filterbank, because the acoustic
// stage conditions on the reference mel produced by exactly this extractor.
class SoproVocoderRuntime final {
public:
    SoproVocoderRuntime(
        const SoproTTSAssets & assets,
        engine::core::ExecutionContext & execution_context,
        size_t weight_context_bytes,
        size_t graph_context_bytes,
        engine::assets::TensorStorageType matmul_storage_type,
        engine::assets::TensorStorageType conv_storage_type);
    ~SoproVocoderRuntime();

    SoproVocoderRuntime(const SoproVocoderRuntime &) = delete;
    SoproVocoderRuntime & operator=(const SoproVocoderRuntime &) = delete;

    // mel: [n_mels, frames], channel-major (mel[c * frames + t]).
    // Returns (frames - 1) * hop_length mono samples at config.sample_rate.
    std::vector<float> decode(const std::vector<float> & mel, int64_t frames) const;

    // MelFeatures.forward: log(clamp(|STFT|, min=1e-7)) with the torchaudio
    // MelSpectrogram buffers stored in the checkpoint (power=1, centred).
    // Returns [n_mels, frames] channel-major.
    std::vector<float> log_mel(const std::vector<float> & audio) const;

    int64_t mel_frames(int64_t samples) const noexcept;
    int64_t hop_length() const noexcept;
    int64_t n_mels() const noexcept;
    int sample_rate() const noexcept;

private:
    const SoproVocoderConfig & config_;
    engine::core::ExecutionContext & execution_context_;
    size_t graph_context_bytes_ = 0;
    std::shared_ptr<const SoproVocoderWeights> weights_;
    mutable std::unique_ptr<SoproVocoderGraph> graph_;
};

}  // namespace engine::community_models::sopro_tts
