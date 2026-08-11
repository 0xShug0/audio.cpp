#pragma once

#include "engine/framework/modules/vocoders/bigvgan_vocoder.h"
#include "engine/models/index_tts2_5/assets.h"

#include <memory>
#include <vector>

namespace engine::models::index_tts2_5 {

struct IndexTTS25VocoderOutput {
    std::vector<float> waveform;
    int64_t samples = 0;
    int sample_rate = 0;
};

class IndexTTS25BigVganVocoder {
public:
    IndexTTS25BigVganVocoder(
        std::shared_ptr<const IndexTTS25Assets> assets,
        core::BackendConfig backend,
        engine::assets::TensorStorageType weight_storage_type);

    IndexTTS25VocoderOutput synthesize(
        const std::vector<float> & mel,
        int64_t frames) const;
    void release_runtime_graph();

private:
    std::shared_ptr<const IndexTTS25Assets> assets_;
    engine::modules::BigVganVocoderComponent component_;
};

}  // namespace engine::models::index_tts2_5
