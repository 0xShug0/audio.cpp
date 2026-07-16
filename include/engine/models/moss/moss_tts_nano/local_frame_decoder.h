#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/models/moss/moss_tts_nano/assets.h"
#include "engine/models/moss/moss_tts_nano/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace engine::models::moss_tts_nano {

class MossTTSNanoLocalFrameDecoderRuntime {
public:
    MossTTSNanoLocalFrameDecoderRuntime(
        std::shared_ptr<const MossTTSNanoAssets> assets,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType weight_storage_type);
    ~MossTTSNanoLocalFrameDecoderRuntime();

    void prepare(int64_t active_codebooks);
    int32_t assistant_slot_token_id() const noexcept;
    int32_t audio_pad_token_id() const noexcept;
    std::vector<int32_t> generate_frame(
        const std::vector<float> & global_hidden,
        int64_t active_codebooks,
        const std::vector<int32_t> & history,
        const MossTTSNanoSamplingOptions & sampling,
        std::mt19937 & rng,
        uint64_t seed,
        uint64_t & sample_call_index);

private:
    struct Weights;
    struct TextGraph;
    struct AudioGraph;
    std::shared_ptr<const MossTTSNanoAssets> assets_;
    core::ExecutionContext & execution_context_;
    engine::sampling::TorchCudaSamplingPolicy sampling_policy_;
    std::shared_ptr<Weights> weights_;
    size_t graph_arena_bytes_ = 0;
    std::unique_ptr<TextGraph> text_graph_;
    std::vector<std::unique_ptr<AudioGraph>> audio_graphs_;
};

}  // namespace engine::models::moss_tts_nano
