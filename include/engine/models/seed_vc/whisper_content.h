#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/modules/speech_encoders/whisper_frontend.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::seed_vc {

class SeedVcWhisperContentEncoder {
public:
    SeedVcWhisperContentEncoder() = default;
    SeedVcWhisperContentEncoder(
        std::shared_ptr<const engine::assets::TensorSource> source,
        engine::core::BackendConfig backend,
        engine::assets::TensorStorageType storage_type);
    ~SeedVcWhisperContentEncoder();

    SeedVcWhisperContentEncoder(SeedVcWhisperContentEncoder &&) noexcept;
    SeedVcWhisperContentEncoder & operator=(SeedVcWhisperContentEncoder &&) noexcept;
    SeedVcWhisperContentEncoder(const SeedVcWhisperContentEncoder &) = delete;
    SeedVcWhisperContentEncoder & operator=(const SeedVcWhisperContentEncoder &) = delete;

    int64_t channels() const noexcept;
    std::vector<float> extract_16k_mono(
        const std::vector<float> & waveform_16k,
        size_t threads) const;

private:
    engine::modules::WhisperFrontendComponent frontend_;
};

}  // namespace engine::models::seed_vc
