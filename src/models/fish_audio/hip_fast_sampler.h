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

void hip_fast_sampler_upload_embeddings(
    void * workspace,
    const float * host_embeddings,
    int32_t rows,
    int32_t dim);

void hip_fast_sampler_gather_embedding(
    void * workspace,
    int32_t row,
    float * device_output,
    void * stream);

void hip_fast_sampler_prepare_step(
    void * workspace,
    int32_t position,
    int32_t mask_length,
    int32_t * device_position,
    void * device_mask,
    void * stream);

}  // namespace engine::models::fish_audio::detail
