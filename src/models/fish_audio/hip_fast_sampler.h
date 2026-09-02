#pragma once

#include <cstdint>

namespace engine::models::fish_audio::detail {

constexpr int kHipFastSamplerMaxTopK = 256;

struct HipFastTopKResult {
    int32_t count = 0;
    float max_logit = 0.0F;
    double denom = 0.0;
    float logits[kHipFastSamplerMaxTopK]{};
    int32_t indices[kHipFastSamplerMaxTopK]{};
};

void * hip_fast_sampler_create(int32_t vocab_size);
void hip_fast_sampler_destroy(void * workspace);
void hip_fast_sampler_topk(
    void * workspace,
    const float * device_logits,
    int32_t top_k,
    HipFastTopKResult * host_result);

}  // namespace engine::models::fish_audio::detail
