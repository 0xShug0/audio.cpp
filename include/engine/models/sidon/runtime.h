#pragma once

#include "engine/framework/modules/speech_encoders/wav2vec2_bert_encoder.h"

namespace engine::models::sidon {

class SidonRuntime {
public:
    SidonRuntime(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution);
    ~SidonRuntime();

    modules::Wav2Vec2BertEncoderOutput encode(const modules::Wav2Vec2BertEncoderInput & input);
    std::vector<float> decode(const std::vector<float> & hidden, int64_t frames);

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace engine::models::sidon
