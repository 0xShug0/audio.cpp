#pragma once

#include "engine/framework/modules/speech_encoders/campplus_encoder.h"
#include "engine/models/index_tts2/assets.h"

#include <memory>
#include <vector>

namespace engine::models::index_tts2 {

struct IndexTTS2StyleEmbedding {
    std::vector<float> values;
    int64_t dims = 0;
};

class IndexTTS2StyleEncoder {
public:
    IndexTTS2StyleEncoder(
        std::shared_ptr<const IndexTTS2Assets> assets,
        core::BackendConfig backend,
        engine::assets::TensorStorageType weight_storage_type);

    IndexTTS2StyleEmbedding embed_fbank(
        const std::vector<float> & features,
        int64_t frames,
        int64_t dims) const;
    void release_graph();

private:
    std::shared_ptr<const IndexTTS2Assets> assets_;
    engine::modules::CampplusEncoderComponent component_;
};

}  // namespace engine::models::index_tts2
