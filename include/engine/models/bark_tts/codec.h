#pragma once

#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::core { class ExecutionContext; }
namespace engine::models::bark_tts {

struct BarkAssets;

class BarkCodecDecoder {
public:
    BarkCodecDecoder(std::shared_ptr<const BarkAssets> assets,
                     engine::core::ExecutionContext & execution,
                     engine::assets::TensorStorageType storage_type);
    ~BarkCodecDecoder();
    std::vector<float> decode(const std::vector<std::vector<int32_t>> & codes) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::bark_tts
