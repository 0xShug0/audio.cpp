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

struct SoproVocosWeights;
struct SoproVocosGraph;

// sopro/vocoder.py. A Vocos backbone (Conv1d embed, num_layers ConvNeXt blocks
// with per-channel gamma, final LayerNorm, convs padded from the config's
// lookaheads) feeding one ISTFT head: Linear(dim -> n_fft + 2) split into
// log-magnitude and phase, then a centred inverse STFT with the checkpoint's
// Hann window.
//
// The same object also owns the analysis mel filterbank, because the acoustic
// stage conditions on the reference mel produced by exactly this extractor.
class SoproVocosRuntime final {
public:
    SoproVocosRuntime(
        const SoproTTSAssets & assets,
        engine::core::ExecutionContext & execution_context,
        size_t weight_context_bytes,
        size_t graph_context_bytes,
        engine::assets::TensorStorageType matmul_storage_type,
        engine::assets::TensorStorageType conv_storage_type);
    ~SoproVocosRuntime();

    SoproVocosRuntime(const SoproVocosRuntime &) = delete;
    SoproVocosRuntime & operator=(const SoproVocosRuntime &) = delete;

    // mel: [n_mels, frames], channel-major (mel[c * frames + t]).
    // Returns (frames - 1) * hop_length mono samples at config.sample_rate.
    std::vector<float> decode(const std::vector<float> & mel, int64_t frames) const;

    // Backbone + ISTFT head linear: one row of n_fft + 2 values per frame.
    std::vector<float> head(const std::vector<float> & mel, int64_t frames) const;
    // Windowed inverse FFT of `frames` head rows: [frames, n_fft].
    std::vector<float> synthesis_frames(const float * head, int64_t frames) const;
    const std::vector<float> & synthesis_window() const noexcept;
    // Input frames the backbone reads on the left (right) of an output frame.
    int64_t context_frames(bool right) const;

    // MelFeatures.forward: log(clamp(|STFT|, min=1e-7)) with the torchaudio
    // MelSpectrogram buffers stored in the checkpoint (power=1, centred).
    // Returns [n_mels, frames] channel-major.
    std::vector<float> log_mel(const std::vector<float> & audio) const;

    int64_t mel_frames(int64_t samples) const noexcept;
    int64_t hop_length() const noexcept;
    int64_t n_fft() const noexcept;
    int64_t n_mels() const noexcept;

private:
    const SoproVocoderConfig & config_;
    engine::core::ExecutionContext & execution_context_;
    size_t graph_context_bytes_ = 0;
    std::shared_ptr<const SoproVocosWeights> weights_;
    mutable std::unique_ptr<SoproVocosGraph> graph_;
};

// Vocoder.decode_stream: the vocoder fed chunk by chunk. Each push decodes the
// frames whose receptive field is now complete by re-running the backbone over
// them plus that field on both sides, which equals the causal convs' streaming
// state; an overlap-add ISTFT then emits every sample whose window sum is
// complete. A flush decodes the rest and resets the stream.
class SoproVocoderStream final {
public:
    explicit SoproVocoderStream(const SoproVocosRuntime & vocoder);

    // mel: [n_mels, frames], channel-major, in the vocoder's (denormalised) space.
    std::vector<float> push(const std::vector<float> & mel, int64_t frames, bool flush);

private:
    std::vector<float> overlap_add(const std::vector<float> & framed, int64_t frames, bool flush);

    const SoproVocosRuntime & vocoder_;
    int64_t left_ = 0;
    int64_t right_ = 0;
    std::vector<float> history_;  // [frames][n_mels], starting at history_start_
    int64_t history_start_ = 0;
    int64_t received_ = 0;  // mel frames pushed
    int64_t decoded_ = 0;   // frames handed to the ISTFT
    int64_t processed_ = 0;
    int64_t emitted_ = 0;
    int64_t tail_start_ = 0;
    std::vector<float> ola_;
    std::vector<float> envelope_;
};

}  // namespace engine::community_models::sopro_tts
