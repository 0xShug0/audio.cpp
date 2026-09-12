#pragma once

#include "engine/models/bark_tts/codec.h"
#include "engine/models/bark_tts/tokenizer.h"
#include "engine/models/bark_tts/transformer.h"
#include "engine/models/bark_tts/types.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::core { class ExecutionContext; }
namespace engine::models::bark_tts {

class BarkGenerator {
public:
    BarkGenerator(std::shared_ptr<const BarkAssets> assets,
                  engine::core::ExecutionContext & execution,
                  engine::assets::TensorStorageType transformer_storage,
                  engine::assets::TensorStorageType codec_storage);
    std::vector<float> synthesize(const std::string & text, const BarkGenerationOptions & options) const;
private:
    std::shared_ptr<const BarkAssets> assets_;
    BarkTokenizer tokenizer_;
    BarkTransformer semantic_;
    BarkTransformer coarse_;
    BarkTransformer fine_;
    BarkCodecDecoder codec_;
};

}  // namespace engine::models::bark_tts
