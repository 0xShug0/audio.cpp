#pragma once

#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/models/chatterbox/s3gen_types.h"

#include <filesystem>
#include <memory>

namespace engine::models::chatterbox {

struct S3FlowDecoderRunTiming;
struct S3FlowCFMTimingBreakdown;

class S3FlowSessionCache {
public:
    explicit S3FlowSessionCache(engine::core::BackendConfig backend = {});
    ~S3FlowSessionCache();
    S3FlowSessionCache(S3FlowSessionCache &&) noexcept;
    S3FlowSessionCache & operator=(S3FlowSessionCache &&) noexcept;

    S3FlowSessionCache(const S3FlowSessionCache &) = delete;
    S3FlowSessionCache & operator=(const S3FlowSessionCache &) = delete;

    void release_encoder_graphs();
    void release_decoder_graphs();

private:
    struct State;
    std::unique_ptr<State> state_;

    friend S3FlowEncoderOutputs compute_s3_flow_encoder_forward(
        S3FlowSessionCache & cache,
        const S3FlowEncoderWeights & weights,
        const std::vector<float> & input,
        int64_t frames,
        int64_t capacity_frames,
        int64_t hidden_size,
        engine::core::BackendConfig backend);
    friend S3FlowDecoderOutputs compute_s3_flow_decoder_forward(
        S3FlowSessionCache & cache,
        const S3FlowDecoderWeights & weights,
        const std::vector<float> & x,
        const std::vector<float> & mask,
        const std::vector<float> & mu,
        const std::vector<float> & t,
        const std::vector<float> & spks,
        const std::vector<float> & cond,
        int64_t batch,
        int64_t frames,
        int64_t capacity_frames,
        engine::core::BackendConfig backend);
    friend S3FlowCFMOutputs compute_s3_flow_cfm_euler(
        S3FlowSessionCache & cache,
        const S3FlowDecoderWeights & weights,
        const std::vector<float> & noise,
        const std::vector<float> & mask,
        const std::vector<float> & mu,
        const std::vector<float> & spks,
        const std::vector<float> & cond,
        int64_t batch,
        int64_t frames,
        int64_t capacity_frames,
        int64_t num_steps,
        float cfg_rate,
        bool cosine_schedule,
        engine::core::BackendConfig backend,
        S3FlowCFMTimingBreakdown * timing);
};

struct S3FlowDecoderRunTiming {
    int64_t calls = 0;
    double conditioning_write_ms = 0.0;
    double time_embedding_ms = 0.0;
    double input_write_ms = 0.0;
    double time_write_ms = 0.0;
    double graph_compute_ms = 0.0;
    double output_read_ms = 0.0;
};

struct S3FlowCFMTimingBreakdown {
    int64_t steps = 0;
    int64_t decoder_calls = 0;
    double initial_state_ms = 0.0;
    double schedule_ms = 0.0;
    double zero_conditioning_ms = 0.0;
    double runner_setup_ms = 0.0;
    double host_update_ms = 0.0;
    S3FlowDecoderRunTiming conditioned;
    S3FlowDecoderRunTiming unconditioned;
};

std::shared_ptr<const S3FlowEncoderWeights> load_s3_flow_encoder_weights(
    const engine::assets::TensorSource & source,
    const engine::core::ExecutionContext & execution_context,
    engine::assets::TensorStorageType weight_storage_type = engine::assets::TensorStorageType::Native);
S3FlowEncoderOutputs compute_s3_flow_encoder_forward(
    S3FlowSessionCache & cache,
    const S3FlowEncoderWeights & weights,
    const std::vector<float> & input,
    int64_t frames,
    int64_t capacity_frames,
    int64_t hidden_size,
    engine::core::BackendConfig backend = {});
std::shared_ptr<const S3FlowDecoderWeights> load_s3_flow_decoder_weights(
    const engine::assets::TensorSource & source,
    const engine::core::ExecutionContext & execution_context,
    engine::assets::TensorStorageType weight_storage_type = engine::assets::TensorStorageType::Native);
S3FlowDecoderOutputs compute_s3_flow_decoder_forward(
    S3FlowSessionCache & cache,
    const S3FlowDecoderWeights & weights,
    const std::vector<float> & x,
    const std::vector<float> & mask,
    const std::vector<float> & mu,
    const std::vector<float> & t,
    const std::vector<float> & spks,
    const std::vector<float> & cond,
    int64_t batch,
    int64_t frames,
    int64_t capacity_frames,
    engine::core::BackendConfig backend = {});
S3FlowCFMOutputs compute_s3_flow_cfm_euler(
    S3FlowSessionCache & cache,
    const S3FlowDecoderWeights & weights,
    const std::vector<float> & noise,
    const std::vector<float> & mask,
    const std::vector<float> & mu,
    const std::vector<float> & spks,
    const std::vector<float> & cond,
    int64_t batch,
    int64_t frames,
    int64_t capacity_frames,
    int64_t num_steps,
    float cfg_rate,
    bool cosine_schedule,
    engine::core::BackendConfig backend = {},
    S3FlowCFMTimingBreakdown * timing = nullptr);

}  // namespace engine::models::chatterbox
