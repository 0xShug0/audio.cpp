#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/models/bark_tts/assets.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::core { class ExecutionContext; }

namespace engine::models::bark_tts {

class BarkTransformer {
public:
    BarkTransformer(std::shared_ptr<const BarkAssets> assets,
                    engine::core::ExecutionContext & execution,
                    std::string prefix,
                    BarkTransformerConfig config,
                    bool causal,
                    engine::assets::TensorStorageType storage_type);
    ~BarkTransformer();

    // For causal models, embedding_channels may contain one or more token rows
    // whose embeddings are summed. Returns logits for the final position.
    std::vector<float> causal_logits(const std::vector<std::vector<int32_t>> & embedding_channels) const;

    // Fine Bark sums codebook embeddings 0..codebook_idx and returns one logit
    // row per sequence position.
    std::vector<float> fine_logits(const std::vector<std::vector<int32_t>> & codes,
                                   int codebook_idx) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::bark_tts
