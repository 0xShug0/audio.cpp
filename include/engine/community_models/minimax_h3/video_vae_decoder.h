#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/community_models/minimax_h3/assets.h"
#include "engine/community_models/minimax_h3/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace engine::models::minimax_h3 {

class VideoVaeWeightStore {
public:
    VideoVaeWeightStore(
        core::ExecutionContext & execution_context,
        std::shared_ptr<const assets::TensorSource> tensor_source,
        const MiniMaxH3Config & cfg,
        size_t weight_context_bytes);

    const core::TensorValue & require(std::string_view name) const;
    const engine::modules::LinearWeights & linear(std::string_view name) const;

    core::ExecutionContext & execution;

private:
    std::shared_ptr<const assets::TensorSource> source_;
    core::BackendWeightStore store_;
    std::unordered_map<std::string, core::TensorValue> weights_;
    std::unordered_map<std::string, engine::modules::LinearWeights> linear_weights_;
};

class VideoVaeTileGraph;

class VideoVaeDecodeCache {
public:
    VideoVaeDecodeCache();
    ~VideoVaeDecodeCache();

    VideoVaeDecodeCache(const VideoVaeDecodeCache &) = delete;
    VideoVaeDecodeCache & operator=(const VideoVaeDecodeCache &) = delete;

    VideoVaeTileGraph & graph(VideoVaeWeightStore & weights, const MiniMaxH3Config & cfg, int64_t latent_t, int64_t latent_h, int64_t latent_w);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

MiniMaxH3VideoFrames run_video_vae_decode_graph(
    VideoVaeWeightStore & weights,
    const MiniMaxH3Config & cfg,
    const std::vector<float> & video_rows,
    VideoVaeDecodeCache & cache);

}  // namespace engine::models::minimax_h3
