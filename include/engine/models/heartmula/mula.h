#pragma once

#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/models/heartmula/assets.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace engine::core {
class BackendWeightStore;
}

namespace engine::core {
class ConstantTensorCache;
}

namespace engine::models::heartmula {

class HeartMuLaBackbonePrefillGraph;
class HeartMuLaBackboneCachedStepGraph;
class HeartMuLaDecoderPrefillGraph;
class HeartMuLaDecoderCachedStepGraph;
class HeartMuLaFrameEmbeddingGraph;

struct HeartMuLaTransformerLayerWeights {
    assets::TensorDataF32 sa_norm;
    core::TensorValue q_proj;
    core::TensorValue k_proj;
    core::TensorValue v_proj;
    core::TensorValue output_proj;
    assets::TensorDataF32 mlp_norm;
    core::TensorValue w1;
    core::TensorValue w2;
    core::TensorValue w3;
};

struct HeartMuLaTransformerWeights {
    std::vector<HeartMuLaTransformerLayerWeights> layers;
    assets::TensorDataF32 norm;
};

struct HeartMuLaWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue text_embeddings;
    core::TensorValue audio_embeddings;
    core::TensorValue unconditional_text_embedding;
    core::TensorValue projection;
    core::TensorValue codebook0_head;
    core::TensorValue audio_head;
    core::TensorValue muq_linear_weight;
    core::TensorValue muq_linear_bias;
    core::TensorValue backbone_rope_factors;
    core::TensorValue decoder_rope_factors;
    HeartMuLaTransformerWeights backbone;
    HeartMuLaTransformerWeights decoder;
};

struct HeartMuLaTransformerLayerOutputs {
    core::TensorValue output;
    core::TensorValue key;
    core::TensorValue value;
};

struct HeartMuLaFrameEmbeddingInputs {
    int64_t batch_size = 0;
    int64_t steps = 0;
    std::vector<int32_t> audio_token_ids;
    std::vector<int32_t> text_token_ids;
    std::vector<float> audio_mask;
    std::vector<float> text_cond_mask;
    std::vector<float> text_uncond_mask;
    std::vector<float> muq_embed;
    std::vector<float> muq_cond_mask;
    std::vector<float> muq_uncond_mask;
    int64_t muq_row = 0;
    bool apply_muq = false;
};

struct HeartMuLaMergedEmbeddings {
    std::vector<float> values;
    int64_t batch_size = 0;
    int64_t steps = 0;
    int64_t dims = 0;
};

struct HeartMuLaBackboneLogits {
    std::vector<float> values;
    int64_t vocab_size = 0;
};

struct HeartMuLaBackboneHidden {
    std::vector<float> values;
    int64_t dims = 0;
};

struct HeartMuLaBackboneResult {
    HeartMuLaBackboneLogits logits;
    HeartMuLaBackboneHidden last_hidden;
};

struct HeartMuLaBackbonePrefillOutput {
    HeartMuLaBackboneResult result;
    runtime::TransformerKVState state;
};

struct HeartMuLaDecoderLogits {
    std::vector<float> values;
    int64_t vocab_size = 0;
};

struct HeartMuLaDecoderHidden {
    std::vector<float> values;
    int64_t dims = 0;
};

struct HeartMuLaDecoderResult {
    HeartMuLaDecoderLogits logits;
    HeartMuLaDecoderHidden last_hidden;
};

struct HeartMuLaDecoderPrefillOutput {
    HeartMuLaDecoderResult result;
    runtime::TransformerKVState state;
};

class HeartMuLaBackboneCachedState final {
public:
    HeartMuLaBackboneCachedState();
    ~HeartMuLaBackboneCachedState();

    HeartMuLaBackboneCachedState(const HeartMuLaBackboneCachedState &) = delete;
    HeartMuLaBackboneCachedState & operator=(const HeartMuLaBackboneCachedState &) = delete;
    HeartMuLaBackboneCachedState(HeartMuLaBackboneCachedState &&) noexcept;
    HeartMuLaBackboneCachedState & operator=(HeartMuLaBackboneCachedState &&) noexcept;

private:
    friend class HeartMuLaWeightsRuntime;

    runtime::TransformerKVState pending_state_;
    int64_t prefill_steps_ = 0;
    bool graph_has_state_ = false;
};

class HeartMuLaDecoderCachedState final {
public:
    HeartMuLaDecoderCachedState();
    ~HeartMuLaDecoderCachedState();

    HeartMuLaDecoderCachedState(const HeartMuLaDecoderCachedState &) = delete;
    HeartMuLaDecoderCachedState & operator=(const HeartMuLaDecoderCachedState &) = delete;
    HeartMuLaDecoderCachedState(HeartMuLaDecoderCachedState &&) noexcept;
    HeartMuLaDecoderCachedState & operator=(HeartMuLaDecoderCachedState &&) noexcept;

private:
    friend class HeartMuLaWeightsRuntime;

    runtime::TransformerKVState pending_state_;
    bool graph_has_state_ = false;
};

class HeartMuLaWeightsRuntime final {
public:
    HeartMuLaWeightsRuntime(
        std::shared_ptr<const HeartMuLaAssets> assets,
        core::BackendType backend_type,
        int device,
        int threads,
        size_t weight_context_bytes = 512ull * 1024ull * 1024ull,
        size_t constant_context_bytes = 256ull * 1024ull * 1024ull,
        size_t backbone_prefill_graph_arena_bytes = 1536ull * 1024ull * 1024ull,
        size_t backbone_step_graph_arena_bytes = 1536ull * 1024ull * 1024ull,
        size_t decoder_prefill_graph_arena_bytes = 512ull * 1024ull * 1024ull,
        size_t decoder_step_graph_arena_bytes = 512ull * 1024ull * 1024ull,
        size_t frame_embedding_graph_arena_bytes = 512ull * 1024ull * 1024ull,
        assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native);
    ~HeartMuLaWeightsRuntime();

    HeartMuLaWeightsRuntime(const HeartMuLaWeightsRuntime &) = delete;
    HeartMuLaWeightsRuntime & operator=(const HeartMuLaWeightsRuntime &) = delete;

    const HeartMuLaAssets & assets() const noexcept;
    const HeartMuLaWeights & weights() const noexcept;
    ggml_backend_t backend() const noexcept;
    core::BackendType backend_type() const noexcept;
    int device() const noexcept;
    core::ConstantTensorCache & backbone_constants() const noexcept;
    core::ConstantTensorCache & decoder_constants() const noexcept;
    core::ConstantTensorCache & embedding_constants() const noexcept;
    int threads() const noexcept;
    size_t backbone_prefill_graph_arena_bytes() const noexcept;
    size_t backbone_step_graph_arena_bytes() const noexcept;
    size_t decoder_prefill_graph_arena_bytes() const noexcept;
    size_t decoder_step_graph_arena_bytes() const noexcept;
    size_t frame_embedding_graph_arena_bytes() const noexcept;

    HeartMuLaBackbonePrefillOutput backbone_prefill_embeddings(
        const std::vector<float> & embeddings,
        int64_t batch_size,
        int64_t steps) const;
    void reset_backbone_cached_state(
        HeartMuLaBackboneCachedState & state,
        runtime::TransformerKVState prefill_state) const;
    HeartMuLaBackboneResult backbone_cached_step(
        const std::vector<float> & embedding,
        int64_t batch_size,
        HeartMuLaBackboneCachedState & state,
        int64_t cache_capacity) const;
    HeartMuLaDecoderPrefillOutput decoder_prefill_embeddings(
        const std::vector<float> & embeddings,
        int64_t batch_size,
        int64_t steps,
        int64_t codebook_index) const;
    void reset_decoder_cached_state(
        HeartMuLaDecoderCachedState & state,
        runtime::TransformerKVState prefill_state) const;
    HeartMuLaDecoderResult decoder_cached_step(
        const std::vector<float> & embedding,
        int64_t batch_size,
        int64_t codebook_index,
        HeartMuLaDecoderCachedState & state,
        int64_t cache_capacity) const;
    HeartMuLaMergedEmbeddings merge_frame_embeddings(const HeartMuLaFrameEmbeddingInputs & inputs) const;
    void release_graph_workspaces() const;
    void clear_graph_cache() const;

private:
    std::shared_ptr<const HeartMuLaAssets> assets_;
    std::shared_ptr<const HeartMuLaWeights> weights_;
    std::unique_ptr<core::ConstantTensorCache> backbone_constants_;
    std::unique_ptr<core::ConstantTensorCache> decoder_constants_;
    std::unique_ptr<core::ConstantTensorCache> embedding_constants_;
    mutable std::unique_ptr<HeartMuLaBackbonePrefillGraph> backbone_prefill_graph_;
    mutable std::unique_ptr<HeartMuLaBackboneCachedStepGraph> backbone_cached_step_graph_;
    mutable std::unique_ptr<HeartMuLaDecoderPrefillGraph> decoder_prefill_graph_;
    mutable std::unique_ptr<HeartMuLaDecoderCachedStepGraph> decoder_cached_step_graph_;
    mutable std::unique_ptr<HeartMuLaFrameEmbeddingGraph> prompt_frame_embedding_graph_;
    mutable std::unique_ptr<HeartMuLaFrameEmbeddingGraph> step_frame_embedding_graph_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int device_ = 0;
    int threads_ = 1;
    size_t backbone_prefill_graph_arena_bytes_ = 1536ull * 1024ull * 1024ull;
    size_t backbone_step_graph_arena_bytes_ = 1536ull * 1024ull * 1024ull;
    size_t decoder_prefill_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
    size_t decoder_step_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
    size_t frame_embedding_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
};

HeartMuLaWeights load_heartmula_weights(
    const HeartMuLaAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type);

HeartMuLaTransformerLayerOutputs build_heartmula_transformer_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const HeartMuLaTransformerLayerWeights & weights,
    const HeartMuLaTransformerConfig & config,
    const core::TensorValue & rope_factors,
    core::ConstantTensorCache & constants,
    const std::optional<core::TensorValue> & prefix_key = std::nullopt,
    const std::optional<core::TensorValue> & prefix_value = std::nullopt,
    const std::optional<core::TensorValue> & attention_mask = std::nullopt);

}  // namespace engine::models::heartmula
