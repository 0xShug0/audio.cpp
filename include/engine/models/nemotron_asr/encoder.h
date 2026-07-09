#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/models/nemotron_asr/assets.h"
#include "engine/models/nemotron_asr/frontend.h"
#include "engine/models/nemotron_asr/weights.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace engine::models::nemotron_asr {

struct NemotronEncodedAudio {
    std::vector<float> values;
    std::vector<int32_t> durations;
    int64_t frames = 0;
    int64_t valid_frames = 0;
    int64_t hidden_size = 0;
};

struct NemotronEncoderStreamState {
    int64_t attention_seen_frames = 0;
    int64_t attention_cached_frames = 0;
    bool first_chunk = true;
    bool backend_cache_valid = false;
    const void * backend_cache_owner = nullptr;
};

class NemotronEncoderRuntime {
public:
    NemotronEncoderRuntime(
        std::shared_ptr<const NemotronAssets> assets,
        std::shared_ptr<const NemotronWeights> weights,
        engine::core::ExecutionContext & execution_context,
        size_t graph_arena_bytes);
    ~NemotronEncoderRuntime();

    void prepare_capacity(int64_t input_frames, int64_t feature_dim, int64_t lookahead_tokens);
    void prepare_streaming_capacity(int64_t feature_dim, int64_t lookahead_tokens);
    void release_offline_graph();
    NemotronEncodedAudio encode(const NemotronFrontendFeatures & features, int64_t prompt_id, int64_t lookahead_tokens);
    NemotronEncoderStreamState make_stream_state() const;
    NemotronEncodedAudio encode_stream_chunk(
        const NemotronFrontendFeatures & features,
        int64_t prompt_id,
        int64_t lookahead_tokens,
        NemotronEncoderStreamState & state);

private:
    struct Graph;

    void ensure_graph(int64_t input_frames, int64_t feature_dim, int64_t lookahead_tokens);
    Graph & ensure_stream_graph(
        int64_t input_frames,
        int64_t feature_dim,
        int64_t lookahead_tokens,
        int64_t prefix_frames,
        bool first_chunk);
    const std::vector<float> & relative_positional_encoding(int64_t frames);

    std::shared_ptr<const NemotronAssets> assets_;
    std::shared_ptr<const NemotronWeights> weights_;
    engine::core::ExecutionContext * execution_context_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::unique_ptr<Graph> graph_;
    std::vector<std::unique_ptr<Graph>> stream_graphs_;
    std::vector<float> input_scratch_;
    std::vector<float> output_scratch_;
    std::vector<float> prompt_scratch_;
    std::vector<int32_t> mask_scratch_;
    std::vector<float> attention_mask_scratch_;
    std::unordered_map<int64_t, std::vector<float>> relative_positional_encoding_cache_;
};

}  // namespace engine::models::nemotron_asr
