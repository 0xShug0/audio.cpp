#pragma once

#include "engine/models/dramabox/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::dramabox {

struct DramaBoxGemmaTokenBatch {
    int64_t batch = 0;
    int64_t tokens = 0;
    std::vector<int32_t> input_ids;
    std::vector<int32_t> attention_mask;
};

class DramaBoxGemmaTokenizer {
public:
    explicit DramaBoxGemmaTokenizer(std::shared_ptr<const DramaBoxAssets> assets);

    DramaBoxGemmaTokenBatch encode(const std::vector<std::string> & prompts) const;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::models::dramabox
