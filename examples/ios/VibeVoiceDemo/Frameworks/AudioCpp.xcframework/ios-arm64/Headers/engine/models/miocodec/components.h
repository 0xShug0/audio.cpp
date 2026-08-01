#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/models/miocodec/assets.h"
#include "engine/models/miocodec/weights.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models {

namespace miocodec {

namespace graphs {
class MioCodecContentEncoderGraph;
class MioCodecGlobalEncoderGraph;
class MioCodecWaveDecoderGraph;
}  // namespace graphs

struct MioCodecContentEmbedding {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t dim = 768;
};

struct MioCodecGlobalEmbedding {
    std::vector<float> values;
    int64_t dim = 128;
};

struct MioCodecWaveHead {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t bins = 0;
};

MioCodecContentEmbedding content_embedding_from_tokens(
    const MioCodecWeights & weights,
    const std::vector<int32_t> & content_tokens);

class MioCodecContentEncoderRuntime {
public:
    MioCodecContentEncoderRuntime(
        std::shared_ptr<const MioCodecAssets> assets,
        std::shared_ptr<const MioCodecWeights> weights,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        size_t constant_context_bytes);
    ~MioCodecContentEncoderRuntime();

    void ensure_graph(int64_t ssl_frames) const;
    MioCodecContentEmbedding encode(const std::vector<float> & ssl_features, int64_t ssl_frames) const;

private:
    std::shared_ptr<const MioCodecAssets> assets_;
    std::shared_ptr<const MioCodecWeights> weights_;
    core::ExecutionContext * execution_context_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    size_t constant_context_bytes_ = 0;
    mutable std::unique_ptr<graphs::MioCodecContentEncoderGraph> graph_;
};

class MioCodecGlobalEncoderRuntime {
public:
    MioCodecGlobalEncoderRuntime(
        std::shared_ptr<const MioCodecAssets> assets,
        std::shared_ptr<const MioCodecWeights> weights,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes);
    ~MioCodecGlobalEncoderRuntime();

    void ensure_graph(int64_t ssl_frames) const;
    MioCodecGlobalEmbedding encode(const std::vector<float> & ssl_features, int64_t ssl_frames) const;

private:
    std::shared_ptr<const MioCodecAssets> assets_;
    std::shared_ptr<const MioCodecWeights> weights_;
    core::ExecutionContext * execution_context_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    mutable std::unique_ptr<graphs::MioCodecGlobalEncoderGraph> graph_;
};

class MioCodecWaveDecoderRuntime {
public:
    MioCodecWaveDecoderRuntime(
        std::shared_ptr<const MioCodecAssets> assets,
        std::shared_ptr<const MioCodecWeights> weights,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        size_t constant_context_bytes);
    ~MioCodecWaveDecoderRuntime();

    void ensure_graph(int64_t content_frames, int64_t stft_frames) const;
    MioCodecWaveHead decode(
        const MioCodecContentEmbedding & content,
        const MioCodecGlobalEmbedding & condition,
        int64_t stft_frames) const;
    MioCodecWaveHead decode_tokens(
        const std::vector<int32_t> & content_tokens,
        const MioCodecGlobalEmbedding & condition,
        int64_t stft_frames) const;

private:
    std::shared_ptr<const MioCodecAssets> assets_;
    std::shared_ptr<const MioCodecWeights> weights_;
    core::ExecutionContext * execution_context_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    size_t constant_context_bytes_ = 0;
    mutable std::unique_ptr<graphs::MioCodecWaveDecoderGraph> graph_;
};

}  // namespace miocodec
}  // namespace engine::models
