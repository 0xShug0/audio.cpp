#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/io/json.h"

#include <memory>
#include <vector>

namespace engine::models::plapre {

struct PlapreLlamaConfig {
    int64_t hidden = 0;
    int64_t intermediate = 0;
    int64_t layers = 0;
    int64_t heads = 0;
    int64_t kv_heads = 0;
    int64_t head_dim = 0;
    int64_t vocabulary = 0;
    int64_t context_length = 0;
    int64_t speaker_dim = 128;
    float norm_eps = 1.0e-5f;
    float rope_theta = 10000.0f;

    static PlapreLlamaConfig from_json(const io::json::Value & value);
};

struct PlapreSamplingOptions {
    int64_t max_tokens = 500;
    int64_t min_tokens = 0;
    float temperature = 0.8f;
    float top_p = 0.95f;
    int64_t top_k = 50;
    uint64_t seed = 0;
    std::vector<int32_t> stop_ids;
};

class PlapreLlamaGenerator {
public:
    PlapreLlamaGenerator(std::shared_ptr<const assets::TensorSource> source,
        std::shared_ptr<const assets::TensorSource> speaker_projection,
        PlapreLlamaConfig config, core::ExecutionContext & execution,
        assets::TensorStorageType storage = assets::TensorStorageType::Native);
    ~PlapreLlamaGenerator();

    std::vector<float> prepare_inputs(const std::vector<int32_t> & prompt, const std::vector<float> & speaker);
    std::vector<int32_t> generate(const std::vector<float> & embeddings, const PlapreSamplingOptions & options);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::plapre
