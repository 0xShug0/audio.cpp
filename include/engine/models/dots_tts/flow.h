#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/models/dots_tts/latent.h"
#include "engine/models/dots_tts/types.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::dots_tts {

struct DotsProjectedSequence {
    std::vector<float> values;
    int64_t steps = 0;
    int64_t hidden_size = 0;
};

struct DotsVelocityOutput {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t latent_dim = 0;
};

struct DotsFlowRuntimeStats {
    int64_t velocity_calls = 0;
    double modulation_graph_ms = 0.0;
    double modulation_input_upload_ms = 0.0;
    double modulation_compute_ms = 0.0;
    double velocity_graph_ms = 0.0;
    double velocity_input_upload_ms = 0.0;
    double velocity_compute_ms = 0.0;
    double velocity_output_read_ms = 0.0;
};

class DotsFlowDecodeState {
public:
    DotsFlowDecodeState();
    DotsFlowDecodeState(DotsFlowDecodeState &&) noexcept;
    DotsFlowDecodeState & operator=(DotsFlowDecodeState &&) noexcept;
    DotsFlowDecodeState(const DotsFlowDecodeState &) = delete;
    DotsFlowDecodeState & operator=(const DotsFlowDecodeState &) = delete;
    ~DotsFlowDecodeState();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class DotsFlowComponent;
};

struct DotsFlowDecodeRequest {
    const std::vector<float> * sequence = nullptr;
    const std::vector<float> * cfg_sequence = nullptr;
    const std::vector<float> * speaker_condition = nullptr;
    int64_t fm_seq_len = 0;
    int64_t num_inference_steps = 0;
    DotsOdeMethod ode_method = DotsOdeMethod::Euler;
    float guidance_scale = 1.0F;
    uint64_t seed = 0;
    uint64_t rng_offset_blocks = 0;
    int64_t cache_capacity = 0;
    bool use_cached_dit = false;
    DotsFlowRuntimeStats * runtime_stats = nullptr;
    DotsFlowDecodeState * decode_state = nullptr;
};

class DotsFlowComponent {
public:
    static DotsFlowComponent load_from_tensor_source(
        std::shared_ptr<const assets::TensorSource> source,
        core::BackendConfig backend,
        DotsConfig config,
        assets::TensorStorageType weight_storage_type);

    DotsFlowComponent();
    DotsFlowComponent(DotsFlowComponent &&) noexcept;
    DotsFlowComponent & operator=(DotsFlowComponent &&) noexcept;
    DotsFlowComponent(const DotsFlowComponent &) = delete;
    DotsFlowComponent & operator=(const DotsFlowComponent &) = delete;
    ~DotsFlowComponent();

    bool is_loaded() const noexcept;
    DotsProjectedSequence project_llm_hidden(const std::vector<float> & hidden, int64_t steps) const;
    DotsProjectedSequence project_latents(const std::vector<float> & latents, int64_t frames) const;
    DotsProjectedSequence project_speaker(const std::vector<float> & speaker) const;
    DotsVelocityOutput predict_velocity(
        const std::vector<float> & sequence,
        int64_t steps,
        const std::vector<float> & timesteps,
        const std::vector<float> & durations,
        const std::vector<float> & speaker_condition,
        int64_t batch_size,
        const std::vector<int32_t> & positions = {},
        const std::vector<uint8_t> & attention_mask = {}) const;
    DotsLatentMatrix decode_next_soar(const DotsFlowDecodeRequest & request) const;
    DotsLatentMatrix decode_next_meanflow(const DotsFlowDecodeRequest & request) const;

    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::dots_tts
