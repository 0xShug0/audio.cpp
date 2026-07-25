#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/community_models/parakeet_tdt/assets.h"
#include "engine/community_models/parakeet_tdt/frontend.h"
#include "engine/community_models/parakeet_tdt/weights.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace engine::community_models::parakeet_tdt {

struct ParakeetEncodedAudio {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t valid_frames = 0;
    int64_t hidden_size = 0;
};

struct ParakeetEncoderStreamState {
    int64_t attention_seen_frames = 0;
    int64_t attention_cached_frames = 0;
    bool first_chunk = true;
    bool backend_cache_valid = false;
    const void * backend_cache_owner = nullptr;
};

class ParakeetEncoderRuntime {
public:
    ParakeetEncoderRuntime(
        std::shared_ptr<const ParakeetTDTAssets> assets,
        std::shared_ptr<const ParakeetWeights> weights,
        engine::core::ExecutionContext & execution_context,
        size_t graph_arena_bytes);
    ~ParakeetEncoderRuntime();

    void prepare_capacity(int64_t input_frames, int64_t feature_dim);
    void release_offline_graph();

    ParakeetEncodedAudio encode(
        const ParakeetFrontendFeatures & features);

    ParakeetEncoderStreamState make_stream_state() const;

private:
    struct Graph;
    std::shared_ptr<const ParakeetTDTAssets> assets_;
    std::shared_ptr<const ParakeetWeights> weights_;
    engine::core::ExecutionContext * execution_context_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::unique_ptr<Graph> graph_;
    std::vector<float> output_scratch_;
    std::vector<int32_t> mask_scratch_;
    std::vector<float> attention_mask_scratch_;

    void ensure_graph(int64_t input_frames, int64_t feature_dim);
    const std::vector<float> & relative_positional_encoding(int64_t frames);
    std::unordered_map<int64_t, std::vector<float>> relative_positional_encoding_cache_;
};

}  // namespace engine::community_models::parakeet_tdt
