#pragma once
#include "engine/community_models/soulx_flashhead/assets.h"
#include "engine/framework/modules/speech_encoders/wav2vec2_encoder.h"
namespace engine::community_models::soulx_flashhead {
class AudioConditioner {
  public:
    AudioConditioner(core::BackendConfig backend, const Assets & assets);
    std::vector<float> encode(const std::vector<float> & window);

  private:
    modules::Wav2Vec2EncoderRuntime runtime_;
};
} // namespace engine::community_models::soulx_flashhead
