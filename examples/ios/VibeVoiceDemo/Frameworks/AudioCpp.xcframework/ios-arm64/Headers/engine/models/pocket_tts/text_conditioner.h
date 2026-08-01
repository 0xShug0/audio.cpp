#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::pocket_tts {

struct PocketTTSAssets;
struct PocketTTSHostWeights;

struct TextConditionerConfig {
    int64_t hidden_size = 1024;
};

struct TextConditioningResult {
    std::string prepared_text;
    std::vector<int32_t> tokens;
    std::vector<float> text_embeddings;
};

class TextConditioner {
public:
    explicit TextConditioner(TextConditionerConfig config = {});

    TextConditioningResult prepare(
        const PocketTTSAssets & manifest,
        const PocketTTSHostWeights & weights,
        const std::string & text) const;

private:
    TextConditionerConfig config_;
};

}  // namespace engine::models::pocket_tts
