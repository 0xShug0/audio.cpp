#pragma once

#include "engine/models/dots_tts/flow.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/framework/sampling/torch_random.h"
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
constexpr size_t kWeightContextBytes = 128ull * 1024ull * 1024ull;
constexpr size_t kSmallGraphContextBytes = 32ull * 1024ull * 1024ull;
constexpr size_t kLargeGraphContextBytes = 128ull * 1024ull * 1024ull;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept;
};

struct GgmlGallocrDeleter {
    void operator()(ggml_gallocr_t alloc) const noexcept;
};

struct DotDitBlockWeights {
    modules::LinearWeights adaln;
    core::TensorValue qkv_proj;
    core::TensorValue q_norm;
    core::TensorValue k_norm;
    core::TensorValue o_proj;
    core::TensorValue o_bias;
    modules::LinearWeights fc1;
    modules::LinearWeights fc2;
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
    modules::LinearWeights input_layer;
    core::TensorValue timestep_freqs;
    modules::LinearWeights timestep_fc1;
    modules::LinearWeights timestep_fc2;
    std::optional<modules::LinearWeights> duration_fc1;
    std::optional<modules::LinearWeights> duration_fc2;
    std::vector<DotDitBlockWeights> blocks;
    modules::LinearWeights final_adaln;
    modules::LinearWeights final_linear;
};

struct DotModulationOutput {
    core::TensorValue backend_value;
    int64_t rows = 0;
    int64_t width = 0;
};

struct DotAttentionOutput {
    core::TensorValue output;
    core::TensorValue key;
    core::TensorValue value;
};

struct DotBlockOutput {
    core::TensorValue output;
    core::TensorValue key;
    core::TensorValue value;
};

class LinearProjectionRunner;
class SpeakerProjectionRunner;
class ModulationRunner;
class VelocityRunner;

struct DotFlowSharedRuntime {
    explicit DotFlowSharedRuntime(std::shared_ptr<const DotFlowWeights> weights);
    ~DotFlowSharedRuntime();

    std::shared_ptr<const DotFlowWeights> weights;
    std::unique_ptr<LinearProjectionRunner> hidden_projection;
    std::unique_ptr<LinearProjectionRunner> latent_projection;
    std::unique_ptr<LinearProjectionRunner> coordinate_projection;
    std::unique_ptr<SpeakerProjectionRunner> speaker_projection;
    std::unique_ptr<ModulationRunner> modulation;
    std::unique_ptr<VelocityRunner> velocity;
};

class SoarDiTRunner {
public:
    virtual ~SoarDiTRunner() = default;
    virtual void ensure_cache(int64_t capacity) = 0;
    virtual int64_t valid_steps(int64_t ode_index) const = 0;
    virtual void prefill(
        int64_t ode_index,
        const std::vector<float> & branch_sequence,
        int64_t steps,
        const DotModulationOutput & modulations,
        DotsFlowRuntimeStats * runtime_stats) = 0;
    virtual DotsVelocityOutput run_step(
        int64_t ode_index,
        const std::vector<float> & tail_sequence,
        int64_t persistent_len,
        int64_t unit_len,
        const DotModulationOutput & modulations,
        float guidance_scale,
        DotsFlowRuntimeStats * runtime_stats) = 0;
};

class MeanFlowDiTRunner {
public:
    virtual ~MeanFlowDiTRunner() = default;
    virtual void ensure_cache(int64_t capacity) = 0;
    virtual int64_t valid_steps() const noexcept = 0;
    virtual void prefill(
        const std::vector<float> & sequence,
        int64_t steps,
        const DotModulationOutput & modulations,
        DotsFlowRuntimeStats * runtime_stats) = 0;
    virtual DotsVelocityOutput run_step(
        const std::vector<float> & tail_sequence,
        int64_t persistent_len,
        int64_t unit_len,
        const DotModulationOutput & modulations,
        DotsFlowRuntimeStats * runtime_stats) = 0;
};

struct FlowDecodeCacheState {
    struct CachedModulations {
        DotModulationOutput output;
        std::vector<float> speaker_condition;
        int64_t num_inference_steps = 0;
        DotsOdeMethod ode_method = DotsOdeMethod::Euler;
        bool use_duration_embedding = false;
    };

    std::optional<CachedModulations> soar;
    std::optional<CachedModulations> meanflow;
    std::unique_ptr<SoarDiTRunner> soar_dit;
    int64_t soar_dit_calls = 0;
    int64_t soar_dit_capacity = 0;
    std::vector<std::unique_ptr<MeanFlowDiTRunner>> meanflow_dit;
    int64_t meanflow_dit_steps = 0;
    int64_t meanflow_dit_capacity = 0;
};

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
    DotsFlowRuntimeStats * runtime_stats = nullptr);
void release_flow_graphs(DotFlowSharedRuntime & runtime);

int64_t head_dim(const DotsTransformerConfig & config);
ggml_type cached_dit_kv_type(core::BackendType backend_type);
DotBlockOutput dit_block_with_mods_and_cache(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & mods,
    const core::TensorValue & positions,
    const DotDitBlockWeights & weights,
    const DotsTransformerConfig & config,
    const std::optional<core::TensorValue> & attention_mask = std::nullopt,
    const std::optional<core::TensorValue> & prefix_key = std::nullopt,
    const std::optional<core::TensorValue> & prefix_value = std::nullopt,
    bool use_16bit_kv = false);
core::TensorValue final_projection_with_mods(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & mods,
    const DotFlowWeights & weights);
std::vector<uint8_t> build_decode_mask(int64_t batch_size, int64_t total_len, int64_t fm_seq_len, int64_t latent_patch_size, int64_t hidden_patch_size);
std::vector<int32_t> build_decode_positions(int64_t total_len, int64_t fm_seq_len, int64_t latent_patch_size);
std::vector<int32_t> build_position_range(int64_t start, int64_t steps);
int64_t resolve_dit_cache_capacity_tokens(int64_t fm_seq_len, int64_t unit_len);
std::vector<ggml_fp16_t> build_prefill_mask_values(int64_t steps);
std::vector<ggml_fp16_t> build_cached_update_mask_values(int64_t capacity, int64_t persistent_len, int64_t unit_len);
core::TensorValue view_dit_cache_banks(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache,
    int64_t bank_start,
    int64_t capacity,
    int64_t heads,
    int64_t dim);
core::TensorValue set_dit_cache_flat_rows(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache,
    const core::TensorValue & rows,
    const core::TensorValue & slots);
void add_scaled(std::vector<float> & lhs, const std::vector<float> & rhs, float scale);
std::vector<float> sum_scaled(const std::vector<float> & base, const std::vector<float> & delta, float scale);
void append_branch_sequence(
    std::vector<float> & out,
    const std::vector<float> & prefix,
    const std::vector<float> & latent_projection,
    int64_t prefix_steps,
    int64_t hidden_size,
    int64_t latent_patch_size);
std::vector<float> cfg_combine_velocity(const DotsVelocityOutput & velocity, int64_t latent_patch_size, float guidance_scale);

std::unique_ptr<SoarDiTRunner> make_soar_dit_runner(std::shared_ptr<const DotFlowWeights> weights, int64_t call_count);
std::unique_ptr<MeanFlowDiTRunner> make_meanflow_dit_runner(std::shared_ptr<const DotFlowWeights> weights, int64_t ode_index);
DotsLatentMatrix decode_next_soar(DotFlowSharedRuntime & runtime, const DotsFlowDecodeRequest & request, FlowDecodeCacheState * decode_state);
DotsLatentMatrix decode_next_meanflow(DotFlowSharedRuntime & runtime, const DotsFlowDecodeRequest & request, FlowDecodeCacheState * decode_state);

}  // namespace engine::models::dots_tts::detail
