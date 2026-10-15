#pragma once

#include <cstddef>

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::fish_drama3 {

using Drama3Frame = std::array<int32_t, 11>;

struct Drama3TowerConfig {
    int64_t hidden_size = 2048;
    int64_t intermediate_size = 6144;
    int64_t heads = 16;
    int64_t kv_heads = 2;
    int64_t head_dim = 256;
    int64_t codebook_size = 4096;
    int64_t shared_layers = 4;
    int64_t residual_layers = 9;
    int64_t backbone_codebooks = 1;
    int32_t semantic_begin = 248099;
    float norm_eps = 1e-6F;
    float rope_theta = 1e7F;
    float rotary_fraction = 0.25F;
};

struct Drama3TowerOutput {
    std::vector<float> hidden;
    std::vector<float> logits;
};

// Each tower layer attends over the speech timeline, including prompt rows.
// Its cache is independent of the backbone and persists across codebooks.
class Drama3Tower {
public:
    Drama3Tower(std::shared_ptr<const assets::TensorSource> weights,
                Drama3TowerConfig config, core::ExecutionContext & execution,
                int64_t cache_steps, size_t prefill_cache_bytes = 256u * 1024u * 1024u);
    ~Drama3Tower();
    Drama3Tower(const Drama3Tower &) = delete;
    Drama3Tower & operator=(const Drama3Tower &) = delete;

    void reset();
    void mix_backbone_embeddings(std::vector<float> & embeddings,
                                 const std::vector<Drama3Frame> & frames) const;
    std::vector<float> shared_input(const std::vector<Drama3Frame> & frames) const;
    void prefill(const std::vector<float> & hidden, const std::vector<Drama3Frame> & frames);
    Drama3TowerOutput run_layer(int64_t layer, const std::vector<float> & hidden,
                              const std::vector<int32_t> & codes,
                              const std::vector<float> & semantic_mask,
                              const std::vector<float> & extra_input,
                              int64_t position, int64_t steps = 1);
    const Drama3TowerConfig & config() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::fish_drama3
