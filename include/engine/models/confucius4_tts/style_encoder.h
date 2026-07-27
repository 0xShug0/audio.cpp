#pragma once

#include "engine/framework/modules/speech_encoders/campplus_encoder.h"
#include "engine/models/confucius4_tts/assets.h"

#include <memory>
#include <vector>

namespace engine::models::confucius4_tts {

struct ConfuciusStyleEmbedding {
    std::vector<float> values;
    int64_t dims = 0;
};

class ConfuciusStyleEncoder {
public:
    ConfuciusStyleEncoder(
        std::shared_ptr<const ConfuciusAssets> assets,
        core::BackendConfig backend,
        engine::assets::TensorStorageType weight_storage_type);

    ConfuciusStyleEmbedding embed_fbank(
        const std::vector<float> & features,
        int64_t frames,
        int64_t dims) const;
    void release_runtime_graph();

private:
    std::shared_ptr<const ConfuciusAssets> assets_;
    engine::modules::CampplusEncoderComponent component_;
};

}  // namespace engine::models::confucius4_tts
