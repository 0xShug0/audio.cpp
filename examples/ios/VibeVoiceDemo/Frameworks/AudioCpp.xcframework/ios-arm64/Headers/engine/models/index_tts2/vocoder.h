#pragma once

#include "engine/framework/modules/vocoders/bigvgan_vocoder.h"
#include "engine/models/index_tts2/assets.h"

#include <memory>
#include <vector>

namespace engine::models::index_tts2 {

struct IndexTTS2VocoderOutput {
    std::vector<float> waveform;
    int64_t samples = 0;
    int sample_rate = 0;
};

class IndexTTS2BigVganVocoder {
public:
    IndexTTS2BigVganVocoder(
        std::shared_ptr<const IndexTTS2Assets> assets,
        core::BackendConfig backend,
        engine::assets::TensorStorageType weight_storage_type);

    IndexTTS2VocoderOutput synthesize(
        const std::vector<float> & mel,
        int64_t frames) const;
    void release_runtime_graph();

private:
    std::shared_ptr<const IndexTTS2Assets> assets_;
    engine::modules::BigVganVocoderComponent component_;
};

}  // namespace engine::models::index_tts2
