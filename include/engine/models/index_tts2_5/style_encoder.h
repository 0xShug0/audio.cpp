#pragma once

#include "engine/framework/modules/speech_encoders/campplus_encoder.h"
#include "engine/models/index_tts2_5/assets.h"

#include <memory>
#include <vector>

namespace engine::models::index_tts2_5 {

struct IndexTTS25StyleEmbedding {
    std::vector<float> values;
    int64_t dims = 0;
};

class IndexTTS25StyleEncoder {
public:
    IndexTTS25StyleEncoder(
        std::shared_ptr<const IndexTTS25Assets> assets,
        core::BackendConfig backend,
        engine::assets::TensorStorageType weight_storage_type);

    IndexTTS25StyleEmbedding embed_fbank(
        const std::vector<float> & features,
        int64_t frames,
        int64_t dims) const;
    void release_graph();

private:
    std::shared_ptr<const IndexTTS25Assets> assets_;
    engine::modules::CampplusEncoderComponent component_;
};

}  // namespace engine::models::index_tts2_5
