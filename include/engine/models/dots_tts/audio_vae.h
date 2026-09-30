#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/models/dots_tts/types.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::dots_tts {

struct DotsEncoderLatents {
    std::vector<float> values;
    int64_t channels = 0;
    int64_t frames = 0;
};

struct DotsDecodedAudio {
    std::vector<float> samples;
    int64_t sample_rate = 0;
};

class DotsAudioVAEStreamState {
public:
    DotsAudioVAEStreamState();
    ~DotsAudioVAEStreamState();
    DotsAudioVAEStreamState(DotsAudioVAEStreamState &&) noexcept;
    DotsAudioVAEStreamState & operator=(DotsAudioVAEStreamState &&) noexcept;
    DotsAudioVAEStreamState(const DotsAudioVAEStreamState &) = delete;
    DotsAudioVAEStreamState & operator=(const DotsAudioVAEStreamState &) = delete;

private:
    friend class DotsAudioVAEComponent;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class DotsAudioVAEComponent {
public:
    static DotsAudioVAEComponent load_from_tensor_source(
        std::shared_ptr<const assets::TensorSource> source,
        core::BackendConfig backend,
        DotsVocoderConfig config,
        assets::TensorStorageType weight_storage_type,
        assets::TensorStorageType conv_weight_storage_type);

    DotsAudioVAEComponent();
    DotsAudioVAEComponent(DotsAudioVAEComponent &&) noexcept;
    DotsAudioVAEComponent & operator=(DotsAudioVAEComponent &&) noexcept;
    DotsAudioVAEComponent(const DotsAudioVAEComponent &) = delete;
    DotsAudioVAEComponent & operator=(const DotsAudioVAEComponent &) = delete;
    ~DotsAudioVAEComponent();

    int64_t sample_rate() const noexcept;
    int64_t hop_size() const noexcept;
    bool is_loaded() const noexcept;

    DotsEncoderLatents extract_latents(const std::vector<float> & waveform) const;
    DotsDecodedAudio decode_latents(const std::vector<float> & latents, int64_t frames) const;

    DotsAudioVAEStreamState create_stream_state(int64_t chunk_frames) const;
    DotsDecodedAudio stream_step(
        const std::vector<float> & latents,
        int64_t frames,
        DotsAudioVAEStreamState & state) const;
    DotsDecodedAudio flush_stream(DotsAudioVAEStreamState & state) const;

    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::dots_tts
