#pragma once

#include "engine/community_models/kale_tts/assets.h"
#include "engine/framework/core/backend.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace engine::models::kale_tts {

struct KaleRequest {
    std::vector<int32_t> ids;               // one sentence, no pads
    const KaleVoice * voice = nullptr;
    float speed = 1.0F;
    uint32_t seed = 0;
    // tiny only: the sentence's Ettin inputs and the word of every id
    std::vector<int32_t> ettin_ids;
    std::vector<float> ettin_wavg;
    int64_t ettin_words = 0;
    std::vector<int32_t> word_index;
    // tests only: use this noise instead of the seeded one (samples)
    const std::vector<float> * noise = nullptr;
};

class KaleRuntime {
public:
    KaleRuntime(std::shared_ptr<const KaleAssets> assets, core::BackendConfig backend);
    ~KaleRuntime();
    std::vector<float> synthesize(const KaleRequest & request);

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace engine::models::kale_tts
