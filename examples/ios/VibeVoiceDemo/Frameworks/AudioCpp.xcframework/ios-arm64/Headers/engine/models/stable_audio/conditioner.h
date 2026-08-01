#pragma once

#include "engine/framework/tokenizers/sentencepiece.h"
#include "engine/models/stable_audio/assets.h"
#include "engine/models/stable_audio/types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::stable_audio {

struct StableAudioTokenBatch {
    int64_t batch = 0;
    int64_t tokens = 0;
    std::vector<int32_t> input_ids;
    std::vector<uint8_t> attention_mask;
};

struct StableAudioConditioningBatch {
    StableAudioTokenBatch prompt_tokens;
    StableAudioTokenBatch negative_prompt_tokens;
    std::vector<float> normalized_seconds;
    std::vector<float> seconds_fourier_features;
};

class StableAudioConditionerInputs {
public:
    explicit StableAudioConditionerInputs(std::shared_ptr<const StableAudioAssets> assets);

    StableAudioConditioningBatch build(const StableAudioRequest & request) const;

private:
    StableAudioTokenBatch tokenize_prompts(const std::vector<std::string> & prompts) const;
    std::vector<float> normalized_seconds(const std::vector<float> & durations) const;
    std::vector<float> seconds_fourier_features(const std::vector<float> & normalized) const;

    std::shared_ptr<const StableAudioAssets> assets_;
    std::vector<engine::tokenizers::SentencePiecePiece> t5_pieces_;
};

}  // namespace engine::models::stable_audio
