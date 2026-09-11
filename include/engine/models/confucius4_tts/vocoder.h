#pragma once

#include "engine/framework/modules/vocoders/bigvgan_vocoder.h"
#include "engine/models/confucius4_tts/assets.h"

#include <memory>
#include <vector>

namespace engine::models::confucius4_tts {

struct ConfuciusVocoderOutput {
    std::vector<float> waveform;
    int64_t samples = 0;
    int sample_rate = 0;
};

class ConfuciusBigVganVocoder {
public:
    ConfuciusBigVganVocoder(
        std::shared_ptr<const ConfuciusAssets> assets,
        core::BackendConfig backend,
        engine::assets::TensorStorageType weight_storage_type);

    ConfuciusVocoderOutput synthesize(const std::vector<float> & mel, int64_t frames) const;
    void release_runtime_graph();

private:
    std::shared_ptr<const ConfuciusAssets> assets_;
    engine::modules::BigVganVocoderComponent component_;
};

}  // namespace engine::models::confucius4_tts
