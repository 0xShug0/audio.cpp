#pragma once

#include "engine/models/pocket_tts_small/assets.h"

#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/transformers/transformer_blocks.h"
#include "engine/framework/runtime/kv_cache.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>

namespace engine::models::pocket_tts_small {

struct Assets;
struct BackendWeights;
struct VoiceStateAssets;

struct FlowLMConfig {
    int64_t hidden_size = 512;
    int64_t num_heads = 8;
    int64_t intermediate_size = 512;
    int64_t latent_size = 32;
    int64_t flow_hidden_size = 192;
    int64_t layers = 8;
    float norm_eps = 1.0e-5F;
    float flow_eps = 1.0e-6F;
    int64_t flow_depth = 4;  // flow-net residual blocks, see ModelConfig::flow_depth
    int64_t attention_size = 0;
    int64_t num_kv_heads = 0;
    int64_t condition_size = 64;
    FlowSamplerKind sampler = FlowSamplerKind::Drifting;
    bool native_cpu = false;
    int native_threads = 0;
    ggml_type kv_cache_type = GGML_TYPE_F32;
    bool flash_attention = false;
};

struct FlowLMWeights {
    modules::LinearWeights input_linear;
    std::vector<modules::TransformerEncoderBlockWeights> transformer_layers;
    modules::NormWeights out_norm;
    modules::LinearWeights out_eos;
    modules::ConditionedFlowMLPWeights flow_net;
    std::optional<modules::LinearWeights> z_projection;
};

using FlowLMCacheState = runtime::KVLayerState;
using FlowLMState = runtime::TransformerKVState;

struct FlowLMStepResult {
    std::vector<float> next_latent;
    float eos_logit = 0.0F;
};

class FlowLMStepRuntime;
class FlowLMWeightsRuntime;
class FlowLMDecodeRuntime;

class FlowLM {
public:
    explicit FlowLM(FlowLMConfig config = {});

    const FlowLMConfig & config() const noexcept;

    FlowLMState make_empty_state() const;
    FlowLMState make_state(const VoiceStateAssets & voice_assets) const;

    void apply_prompt(
        FlowLMStepRuntime & runtime,
        const std::vector<float> & input_embeddings,
        int64_t steps,
        const FlowLMState & state) const;

    FlowLMState prompt_state(
        ggml_backend_t backend,
        int threads,
        const Assets & manifest,
        const BackendWeights & weights,
        const std::vector<float> & input_embeddings,
        int64_t steps,
        const FlowLMState & state,
        size_t weights_view_context_bytes,
        size_t step_graph_context_bytes) const;

    std::shared_ptr<const FlowLMWeightsRuntime> create_weights_runtime(
        ggml_backend_t backend,
        int threads,
        const BackendWeights & weights,
        size_t weights_view_context_bytes) const;

    std::shared_ptr<FlowLMStepRuntime> create_step_runtime(
        ggml_backend_t backend,
        int threads,
        std::shared_ptr<const FlowLMWeightsRuntime> weights,
        int64_t cache_steps,
        int64_t prompt_steps,
        int64_t prompt_prefix_steps,
        size_t step_graph_context_bytes) const;

    FlowLMStepResult run_native_step(
        FlowLMStepRuntime & runtime,
        const std::vector<float> & input_latent,
        const std::vector<float> & noise) const;

private:
    struct PromptCache {
        ggml_backend_t backend = nullptr;
        int threads = 0;
        const BackendWeights * weights = nullptr;
        size_t weights_context_bytes = 0;
        size_t graph_context_bytes = 0;
        int64_t steps = 0;
        int64_t prefix_steps = 0;
        std::shared_ptr<const FlowLMWeightsRuntime> weights_runtime;
        std::shared_ptr<FlowLMStepRuntime> runtime;
    };

    FlowLMConfig config_;
    mutable PromptCache prompt_cache_;
};

struct Assets;
struct BackendWeights;

struct AcousticGenerationConfig {
    int max_steps = 0;
    int frames_after_eos = -1;
    float temperature = 0.7F;
    float noise_clamp = -1.0F;
    float eos_threshold = -4.0F;
    int min_eos_steps = 0;
    uint32_t seed = 1234;
    std::vector<float> noise_schedule;
};

struct AcousticModelResult {
    int generated_steps = 0;
    std::vector<float> latents;
    std::vector<float> eos_logits;
};

struct AcousticPreparedRuntime {
    std::shared_ptr<FlowLMStepRuntime> step_runtime;
    std::shared_ptr<FlowLMDecodeRuntime> decode_runtime;
    ggml_backend_t backend = nullptr;
    int threads = 1;
    size_t weights_context_bytes = 0;
    size_t graph_context_bytes = 0;
};

struct AcousticRowState {
    AcousticGenerationConfig config;
    std::vector<float> current_input;
    std::mt19937 rng;
    int step = 0;
    int eos_step = -1;
    bool done = false;
};

struct AcousticStreamState {
    AcousticPreparedRuntime runtime;
    std::vector<AcousticRowState> rows;
    std::vector<float> inputs;
    std::vector<float> noise;
    std::vector<int64_t> active;
};

class AcousticModel {
public:
    explicit AcousticModel(FlowLMConfig config = {});

    const FlowLMConfig & config() const noexcept;

    AcousticPreparedRuntime prepare_runtime(
        ggml_backend_t backend,
        int threads,
        const Assets & manifest,
        const BackendWeights & weights,
        const std::vector<float> & text_embeddings,
        const FlowLMState & initial_state,
        const AcousticGenerationConfig & config,
        int64_t prompt_capacity,
        int64_t initial_cache_capacity,
        int max_steps_capacity,
        size_t flow_weights_view_context_bytes,
        size_t flow_step_graph_context_bytes) const;

    AcousticModelResult generate(
        const AcousticPreparedRuntime & runtime,
        const Assets & manifest,
        const BackendWeights & weights,
        const std::vector<float> & text_embeddings,
        const FlowLMState & initial_state,
        const AcousticGenerationConfig & config) const;

    AcousticStreamState start_stream(
        const AcousticPreparedRuntime & runtime,
        const Assets & manifest,
        const BackendWeights & weights,
        const std::vector<float> & text_embeddings,
        const FlowLMState & initial_state,
        const AcousticGenerationConfig & config) const;

    std::optional<FlowLMStepResult> next_stream_step(AcousticStreamState & state) const;

    std::vector<AcousticModelResult> generate_batch(
        ggml_backend_t backend, int threads, const Assets & manifest, const BackendWeights & weights,
        const std::vector<std::vector<float>> & text_embeddings,
        const std::vector<FlowLMState> & initial_states,
        const std::vector<AcousticGenerationConfig> & configs, int64_t cache_capacity,
        size_t weights_context_bytes, size_t graph_context_bytes) const;

    void clear_runtime_cache() const noexcept;
    int64_t prepared_prompt_capacity() const noexcept;
    int prepared_max_steps_capacity() const noexcept;

private:
    struct RuntimeCache {
        ggml_backend_t backend = nullptr;
        int threads = 1;
        const Assets * manifest = nullptr;
        int64_t prompt_capacity = 0;
        int64_t initial_cache_capacity = 0;
        int max_steps_capacity = 0;
        size_t flow_weights_view_context_bytes = 0;
        size_t flow_step_graph_context_bytes = 0;
        std::shared_ptr<const FlowLMWeightsRuntime> weights_runtime;
        std::shared_ptr<FlowLMStepRuntime> runtime;
    };

    AcousticStreamState start_rows(
        const AcousticPreparedRuntime & runtime, const Assets & manifest, const BackendWeights & weights,
        const std::vector<std::vector<float>> & text_embeddings, const std::vector<FlowLMState> & initial_states,
        const std::vector<AcousticGenerationConfig> & configs) const;
    std::vector<std::optional<FlowLMStepResult>> next_rows(AcousticStreamState & state) const;
    std::vector<AcousticModelResult> generate_rows(
        const AcousticPreparedRuntime & runtime, const Assets & manifest, const BackendWeights & weights,
        const std::vector<std::vector<float>> & text_embeddings, const std::vector<FlowLMState> & initial_states,
        const std::vector<AcousticGenerationConfig> & configs) const;

    std::shared_ptr<FlowLMDecodeRuntime> prepare_decode_runtime(
        ggml_backend_t backend, int threads, const BackendWeights & weights,
        int64_t batch, int64_t capacity, size_t weights_context_bytes, size_t graph_context_bytes) const;

    FlowLM flow_lm_;
    mutable RuntimeCache runtime_cache_;
    mutable std::shared_ptr<FlowLMDecodeRuntime> decode_runtime_;
};

struct FlowLMWeights;

class NativeFlowLM {
public:
    NativeFlowLM(FlowLMConfig config, const FlowLMWeights & weights, int64_t capacity, int threads);
    ~NativeFlowLM();
    void apply_prompt(const std::vector<float> & embeddings, int64_t steps, const FlowLMState & state);
    FlowLMStepResult step(const std::vector<float> & latent, const std::vector<float> & noise);
    FlowLMState export_state() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::pocket_tts_small
