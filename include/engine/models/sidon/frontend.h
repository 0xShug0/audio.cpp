#pragma once

#include "engine/framework/modules/speech_encoders/wav2vec2_bert_encoder.h"
#include "engine/framework/runtime/session.h"

namespace engine::models::sidon {

std::vector<float> prepare_audio(const runtime::AudioBuffer & audio);
modules::Wav2Vec2BertEncoderInput extract_features(const std::vector<float> & padded_audio, int threads);

}  // namespace engine::models::sidon
