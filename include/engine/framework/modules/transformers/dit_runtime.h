#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/transformers/dit.h"
#include "engine/framework/sampling/device_denoiser_runtime.h"
#include "engine/framework/sampling/torch_random.h"

#include <memory>
#include <optional>
#include <vector>

namespace engine::modules {

enum class DitTimeFeatures { SinCos, CosSin, Fourier };
enum class DitSampler { Euler, PingPong };
enum class DitConditioningCache { Unprojected, ProjectedKV };
enum class DitGuidance { Linear, AdaptiveProjected };
enum class DitSchedule { Linear, Cosine, Shifted };

struct DitRuntimeConfig {
    std::string label = "dit";
    DitConfig model;
    assets::TensorStorageType weight_type = assets::TensorStorageType::Native;
    size_t weight_metadata_bytes = 4 * 1024 * 1024;
    size_t graph_metadata_bytes = 64 * 1024 * 1024;
    size_t graph_nodes = 65536;
    DitTimeFeatures time_features = DitTimeFeatures::CosSin;
    // Sinusoidal frequencies are period^(-i / denominator). Zero uses half-width.
    double time_frequency_denominator = 0;
    double time_period = 10000;
    float time_frequency_min = 0.5f, time_frequency_max = 10000.f;
    float time_scale = 1;
    // Cached modulation dtype is a numerical boundary, independent of weight storage.
    ggml_type modulation_storage_type = GGML_TYPE_F32;
    DitGuidance guidance = DitGuidance::Linear;
    DitConditioningCache conditioning_cache = DitConditioningCache::ProjectedKV;
    DeviceDenoiserOptimizations optimizations{};
};

struct DitConditioning {
    int64_t batch = 1;
    int64_t frames = 0;
    int64_t branches = 1;
    int64_t context_tokens = 0;
    // Conditions are branch-major: all positive batches, then all negative batches.
    // Context/frame features use B,T,C. Vectors use B,C. Latents use B,C,T.
    std::vector<float> context, global;
    std::vector<int64_t> frame_channels, vector_channels;
    std::vector<std::vector<float>> frame_features, vector_features;
    std::vector<float> value_mask, latent_mask, local_condition;
};

struct DitSamplingRequest {
    DitSampler sampler = DitSampler::Euler;
    std::vector<float> noise;
    // Flow schedule, batch-major. Includes the terminal point.
    std::vector<float> schedule;
    int64_t steps = 0;
    DitSchedule schedule_type = DitSchedule::Linear;
    float sigma_shift = 1;
    float timestep_scale = 1;
    float guidance = 1;
    float apg_scale = 0;
    uint64_t seed = 0;
    uint64_t * rng_offset_blocks = nullptr;
    sampling::TorchCudaSamplingPolicy rng_policy;
};

struct DitOperationStats {
    double prepare_ms = 0, upload_ms = 0, compute_ms = 0, read_ms = 0;
};

struct DitModulationBank {
    int64_t rows = 0, width = 0;
};

struct DitSequenceRequest {
    const std::vector<float> & tokens;
    int64_t batch = 1, frames = 0;
    const std::vector<float> * timesteps = nullptr;
    const std::vector<float> * durations = nullptr;
    const std::vector<float> * global = nullptr;
    const std::vector<int32_t> * positions = nullptr;
    const std::vector<uint8_t> * attention_mask = nullptr;
    // A contiguous row range in this runtime's prepared modulation bank.
    // Negative selects direct timestep/global conditioning instead.
    int64_t modulation_start = -1;
    int64_t output_start = 0, output_tokens = 0;
    // Omitted returns raw branch predictions; supplied combines paired branches.
    std::optional<float> guidance = std::nullopt;
};

struct DitPrefixCacheConfig {
    int64_t capacity = 0, banks = 0, batch = 1, branches = 1;
    int64_t unit_tokens = 0, output_tokens = 0;
    ggml_type storage_type = GGML_TYPE_F32;
};

// Complete session-owned estimator: loads/binds weights, prepares conditioning,
// builds/caches the configured model, and executes predictions or solver sequences.
// All mutable graphs and caches belong to this instance. Serial use only.
class DitRuntime {
public:
    DitRuntime(core::ExecutionContext & execution, const assets::TensorSource & source,
        DitRuntimeConfig config, const DitWeightNames & names);
    ~DitRuntime();
    DitRuntime(const DitRuntime &) = delete;
    DitRuntime & operator=(const DitRuntime &) = delete;

    void prepare_conditioning(DitConditioning conditioning);
    std::vector<float> predict(const std::vector<float> & latent, float timestep, int64_t branch = 0);
    std::vector<float> predict_cfg(const std::vector<float> & latent, float timestep);
    std::vector<float> sample(const DitSamplingRequest & request);
    // Reuses the bank for identical conditioning. Global rows may describe one
    // repeating branch group whose row count divides the timestep row count.
    // Changed conditioning invalidates prefix history that used the old bank.
    DitModulationBank prepare_modulations(const std::vector<float> & timesteps,
        const std::vector<float> & durations, const std::vector<float> & global,
        DitOperationStats * stats = nullptr);
    std::vector<float> predict_sequence(const DitSequenceRequest & request,
        DitOperationStats * stats = nullptr);
    void reserve_prefix_cache(DitPrefixCacheConfig config);
    void reset_prefix_cache();
    // End a decode phase without retaining its device KV/input/scratch buffers.
    // Graph metadata is kept and reused when the next phase has the same shape.
    void release_prefix_buffers();
    int64_t prefix_tokens(int64_t bank) const;
    void prefill_prefix(const std::vector<float> & tokens, int64_t frames, int64_t bank,
        DitOperationStats * stats = nullptr);
    std::vector<float> predict_prefix(const std::vector<float> & tokens, int64_t past_tokens,
        int64_t bank, float guidance, DitOperationStats * stats = nullptr);
    void release_graphs();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::modules
