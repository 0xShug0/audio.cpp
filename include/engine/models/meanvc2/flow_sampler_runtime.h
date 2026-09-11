#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/sampling/torch_random.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace engine::modules {
class FlowSamplerRuntime;
}  // namespace engine::modules

namespace engine::models::meanvc2 {

struct MeanVC2GtmMemory {
    std::vector<float> keys;
    std::vector<float> values;
};

struct MeanVC2FlowWeights;
struct MeanVC2GtmGraph;
class MeanVC2DenoiserRuntime;

class MeanVC2FlowSamplerRuntime final {
public:
    MeanVC2FlowSamplerRuntime(
        std::shared_ptr<const engine::assets::TensorSource> source,
        engine::core::ExecutionContext & execution_context,
        size_t weight_context_bytes,
        size_t graph_context_bytes,
        engine::assets::TensorStorageType weight_storage_type);
    ~MeanVC2FlowSamplerRuntime();

    MeanVC2GtmMemory encode_speaker_memory(const std::vector<float> & speaker_embedding) const;
    void start_streaming(
        const std::vector<float> & speaker_embedding,
        const MeanVC2GtmMemory & speaker_memory,
        uint64_t seed);
    void reset_streaming();
    std::vector<float> synthesize_streaming_chunk(const std::vector<float> & condition_frames);
    std::vector<float> synthesize_mel(
        const std::vector<float> & condition_frames,
        int64_t condition_frame_count,
        const std::vector<float> & speaker_embedding,
        const MeanVC2GtmMemory & speaker_memory,
        uint64_t seed);

private:
    engine::core::ExecutionContext & execution_context_;
    size_t graph_context_bytes_ = 0;
    std::shared_ptr<const engine::assets::TensorSource> source_;
    std::shared_ptr<const MeanVC2FlowWeights> weights_;
    engine::sampling::TorchCudaSamplingPolicy rng_policy_;
    mutable std::unique_ptr<MeanVC2GtmGraph> gtm_graph_;
    mutable MeanVC2DenoiserRuntime * denoiser_ = nullptr;
    mutable std::unique_ptr<engine::modules::FlowSamplerRuntime> sampler_runtime_;
    std::vector<float> stream_noise_cache_;
    std::vector<float> stream_speaker_embedding_;
    MeanVC2GtmMemory stream_speaker_memory_;
    uint64_t stream_seed_ = 42;
    uint64_t stream_rng_offset_blocks_ = 0;
    bool stream_has_noise_cache_ = false;
    bool stream_started_ = false;
};

}  // namespace engine::models::meanvc2
