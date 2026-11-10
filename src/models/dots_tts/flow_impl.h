#pragma once

#include "engine/models/dots_tts/flow.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/transformers/dit_runtime.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/framework/sampling/flow_sampler_runtime.h"
#include "engine/framework/core/backend.h"

#include <ggml-alloc.h>
#include <ggml.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace engine::models::dots_tts::detail {

namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

constexpr int64_t kTimestepFrequencyDim = 256;
constexpr float kLayerNormEps = 1.0e-5F;
constexpr float kTorchBFloat16Eps = 0.0078125F;
constexpr size_t kWeightContextBytes = 4ull * 1024ull * 1024ull;
constexpr size_t kSmallGraphContextBytes = 32ull * 1024ull * 1024ull;
constexpr size_t kLargeGraphContextBytes = 128ull * 1024ull * 1024ull;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept;
};

struct GgmlGallocrDeleter {
    void operator()(ggml_gallocr_t alloc) const noexcept;
};

struct DotFlowWeights {
    DotsConfig config;
    std::shared_ptr<core::ExecutionContext> execution_context;
    std::shared_ptr<core::BackendWeightStore> store;
    modules::LinearWeights hidden_proj;
    modules::LinearWeights latent_proj;
    modules::LinearWeights coordinate_proj;
    modules::LinearWeights xvec_proj;
    modules::NormWeights xvec_norm;
};

using DotModulationOutput = modules::DitModulationBank;

class LinearProjectionRunner;
class SpeakerProjectionRunner;

struct DotFlowSharedRuntime {
    DotFlowSharedRuntime(std::shared_ptr<const DotFlowWeights> weights,
        const assets::TensorSource & source, assets::TensorStorageType storage);
    ~DotFlowSharedRuntime();

    std::shared_ptr<const DotFlowWeights> weights;
    std::unique_ptr<LinearProjectionRunner> hidden_projection;
    std::unique_ptr<LinearProjectionRunner> latent_projection;
    std::unique_ptr<LinearProjectionRunner> coordinate_projection;
    std::unique_ptr<SpeakerProjectionRunner> speaker_projection;
    std::unique_ptr<modules::DitRuntime> dit;
};

struct FlowDecodeCacheState {
    bool initialized = false;
};

modules::DitRuntimeConfig dit_runtime_config(const DotsConfig & config,
    core::BackendType backend, assets::TensorStorageType storage);
modules::DitWeightNames dit_weight_names(const DotsConfig & config);

std::unique_ptr<DotFlowSharedRuntime> load_flow_shared_runtime(
    std::shared_ptr<const assets::TensorSource> source,
    core::BackendConfig backend,
    DotsConfig config,
    assets::TensorStorageType storage_type);
DotsProjectedSequence run_hidden_projection(DotFlowSharedRuntime & runtime, const std::vector<float> & hidden, int64_t steps);
DotsProjectedSequence run_latent_projection(DotFlowSharedRuntime & runtime, const std::vector<float> & latents, int64_t frames);
DotsProjectedSequence run_coordinate_projection(DotFlowSharedRuntime & runtime, const std::vector<float> & latents, int64_t frames);
DotsProjectedSequence run_speaker_projection(DotFlowSharedRuntime & runtime, const std::vector<float> & speaker);
DotModulationOutput run_modulation(
    DotFlowSharedRuntime & runtime,
    const std::vector<float> & timesteps,
    const std::vector<float> & durations,
    const std::vector<float> & speaker_condition,
    DotsFlowRuntimeStats * runtime_stats);
DotsVelocityOutput run_velocity(
    DotFlowSharedRuntime & runtime,
    const std::vector<float> & sequence,
    int64_t steps,
    const std::vector<float> & timesteps,
    const std::vector<float> & durations,
    const std::vector<float> & speaker_condition,
    int64_t batch_size,
    const std::vector<int32_t> & positions,
    const std::vector<uint8_t> & attention_mask,
    const DotModulationOutput * modulations = nullptr,
    int64_t modulation_row_start = 0,
    int64_t output_start = 0,
    int64_t output_length = 0,
    DotsFlowRuntimeStats * runtime_stats = nullptr,
    std::optional<float> guidance = std::nullopt);
void release_flow_graphs(DotFlowSharedRuntime & runtime);

int64_t head_dim(const DotsTransformerConfig & config);
ggml_type cached_dit_kv_type(core::BackendType backend_type);
std::vector<uint8_t> build_decode_mask(int64_t batch_size, int64_t total_len, int64_t fm_seq_len, int64_t latent_patch_size, int64_t hidden_patch_size);
std::vector<int32_t> build_decode_positions(int64_t total_len, int64_t fm_seq_len, int64_t latent_patch_size);
int64_t resolve_dit_cache_capacity_tokens(int64_t fm_seq_len, int64_t unit_len);
void append_branch_sequence(
    std::vector<float> & out,
    const std::vector<float> & prefix,
    const std::vector<float> & latent_projection,
    int64_t prefix_steps,
    int64_t hidden_size,
    int64_t latent_patch_size);

DotsLatentMatrix decode_next_soar(DotFlowSharedRuntime & runtime, const DotsFlowDecodeRequest & request, FlowDecodeCacheState * decode_state);
DotsLatentMatrix decode_next_meanflow(DotFlowSharedRuntime & runtime, const DotsFlowDecodeRequest & request, FlowDecodeCacheState * decode_state);

}  // namespace engine::models::dots_tts::detail
