#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
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

// Builds a single FastConformer encoder layer's graph ops (feed-forward 1,
// relative-position self-attention, conv module, feed-forward 2, final
// layer norm — matching NeMo's ConformerLayer.forward exactly). Exported so
// test/parity harnesses can exercise the real production layer-building
// code directly on a hand-fed input and compare against a NeMo reference
// capture, instead of maintaining a separate duplicate implementation that
// could silently drift out of sync with the real encoder. See
// tests/parakeet_tdt/parity/ for the harness that uses this.
engine::core::TensorValue build_encoder_layer(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & attention_mask,
    const engine::core::TensorValue & keep_mask,
    const engine::core::TensorValue & projected_pos_emb,
    const ParakeetEncoderLayerWeights & weights,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t heads,
    int64_t conv_kernel,
    bool use_flash_attention = false);

class ParakeetEncoderRuntime {
public:
    ParakeetEncoderRuntime(
        std::shared_ptr<const ParakeetTDTAssets> assets,
        std::shared_ptr<const ParakeetWeights> weights,
        engine::core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        bool use_flash_attention = false);
    ~ParakeetEncoderRuntime();

    void prepare_capacity(int64_t input_frames, int64_t feature_dim);
    void release_offline_graph();

    ParakeetEncodedAudio encode(
        const ParakeetFrontendFeatures & features);

private:
    struct Graph;
    std::shared_ptr<const ParakeetTDTAssets> assets_;
    std::shared_ptr<const ParakeetWeights> weights_;
    engine::core::ExecutionContext * execution_context_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    bool use_flash_attention_ = false;
    std::unique_ptr<Graph> graph_;
    std::vector<float> output_scratch_;
    std::vector<int32_t> mask_scratch_;
    std::vector<float> attention_mask_scratch_;

    void ensure_graph(int64_t input_frames, int64_t feature_dim);
    const std::vector<float> & relative_positional_encoding(int64_t frames);
    std::unordered_map<int64_t, std::vector<float>> relative_positional_encoding_cache_;
};

}  // namespace engine::community_models::parakeet_tdt
