#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/models/vibevoice/assets.h"

#include <ggml-backend.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace engine::core {
class BackendWeightStore;
}

namespace engine::models::common {
class ConstantTensorCache;
}

namespace engine::models::vibevoice {

class VibeVoiceDecoderPrefillGraph;
class VibeVoiceDecoderCachedStepGraph;
class VibeVoiceDecoderCachedBatchStepGraph;
class VibeVoiceDecoderEmbeddingGraph;

class VibeVoiceDecoderCachedState final {
public:
    VibeVoiceDecoderCachedState();
    ~VibeVoiceDecoderCachedState();

    VibeVoiceDecoderCachedState(const VibeVoiceDecoderCachedState &) = delete;
    VibeVoiceDecoderCachedState & operator=(const VibeVoiceDecoderCachedState &) = delete;
    VibeVoiceDecoderCachedState(VibeVoiceDecoderCachedState &&) noexcept;
    VibeVoiceDecoderCachedState & operator=(VibeVoiceDecoderCachedState &&) noexcept;

private:
    friend class VibeVoiceDecoderWeightsRuntime;
    friend class VibeVoiceDecoderCachedBatchStepGraph;

    std::unique_ptr<VibeVoiceDecoderCachedStepGraph> graph_;
    runtime::TransformerKVState pending_state_;
    const VibeVoiceDecoderCachedBatchStepGraph * batch_owner_ = nullptr;
    bool graph_has_state_ = false;
};

struct VibeVoiceDecoderLogits {
    std::vector<float> values;
    int64_t vocab_size = 0;
};

struct VibeVoiceTokenEmbeddings {
    std::vector<float> values;
    int64_t steps = 0;
    int64_t hidden_size = 0;
};

struct VibeVoiceDecoderHidden {
    std::vector<float> values;
    int64_t dims = 0;
};

struct VibeVoiceDecoderResult {
    VibeVoiceDecoderLogits logits;
    VibeVoiceDecoderHidden last_hidden;
};

struct VibeVoiceDecoderPrefillOutput {
    VibeVoiceDecoderResult result;
    runtime::TransformerKVState state;
};

struct VibeVoiceDecoderMLPWeights {
    modules::LinearWeights gate_proj;
    modules::LinearWeights up_proj;
    modules::LinearWeights down_proj;
};

struct VibeVoiceDecoderLayerWeights {
    assets::TensorDataF32 input_norm;
    modules::AttentionWeights self_attention;
    assets::TensorDataF32 post_norm;
    VibeVoiceDecoderMLPWeights mlp;
};

struct VibeVoiceDecoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue token_embedding;
    core::TensorValue lm_head;
    std::vector<VibeVoiceDecoderLayerWeights> layers;
    assets::TensorDataF32 norm;
};

struct VibeVoiceDecoderLayerOutputs {
    core::TensorValue output;
    core::TensorValue key;
    core::TensorValue value;
};

class VibeVoiceDecoderWeightsRuntime final {
public:
    VibeVoiceDecoderWeightsRuntime(
        std::shared_ptr<const VibeVoiceAssets> assets,
        core::BackendType backend_type,
        int device,
        int threads,
        size_t weight_context_bytes = 256ull * 1024ull * 1024ull,
        size_t constant_context_bytes = 128ull * 1024ull * 1024ull,
        assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native);

    ~VibeVoiceDecoderWeightsRuntime();

    VibeVoiceDecoderWeightsRuntime(const VibeVoiceDecoderWeightsRuntime &) = delete;
    VibeVoiceDecoderWeightsRuntime & operator=(const VibeVoiceDecoderWeightsRuntime &) = delete;

    const VibeVoiceAssets & assets() const noexcept;
    const VibeVoiceDecoderWeights & weights() const noexcept;
    ggml_backend_t backend() const noexcept;
    common::ConstantTensorCache & constants() const noexcept;
    int threads() const noexcept;

    VibeVoiceTokenEmbeddings embed_tokens(const std::vector<int32_t> & input_ids) const;
    VibeVoiceDecoderPrefillOutput prefill_embeddings(const std::vector<float> & embeddings, int64_t steps) const;
    std::vector<VibeVoiceDecoderPrefillOutput> prefill_embeddings_batch(
        const std::vector<std::vector<float>> & embeddings,
        int64_t steps) const;
    void reset_cached_state(VibeVoiceDecoderCachedState & state, runtime::TransformerKVState prefill_state) const;
    VibeVoiceDecoderResult cached_step(
        const std::vector<float> & embedding,
        VibeVoiceDecoderCachedState & state,
        int64_t cache_capacity) const;
    std::vector<VibeVoiceDecoderResult> cached_step_batch(
        const std::vector<std::vector<float>> & embeddings,
        const std::vector<VibeVoiceDecoderCachedState *> & states,
        int64_t cache_capacity) const;

private:
    VibeVoiceDecoderCachedBatchStepGraph * find_cached_batch_graph(
        const VibeVoiceDecoderCachedState & state) const;
    void export_and_drop_cached_batch_graph(const VibeVoiceDecoderCachedBatchStepGraph * owner) const;

    std::shared_ptr<const VibeVoiceAssets> assets_;
    std::shared_ptr<const VibeVoiceDecoderWeights> weights_;
    std::unique_ptr<common::ConstantTensorCache> constants_;
    mutable std::unique_ptr<VibeVoiceDecoderEmbeddingGraph> embedding_graph_;
    mutable std::unique_ptr<VibeVoiceDecoderPrefillGraph> prefill_graph_;
    mutable std::vector<std::unique_ptr<VibeVoiceDecoderCachedBatchStepGraph>> cached_batch_graphs_;
    ggml_backend_t backend_ = nullptr;
    int threads_ = 1;
};

VibeVoiceDecoderWeights load_vibevoice_decoder_weights(
    const VibeVoiceAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type);

VibeVoiceDecoderLayerOutputs build_vibevoice_decoder_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const VibeVoiceDecoderLayerWeights & weights,
    const VibeVoiceDecoderConfig & config,
    common::ConstantTensorCache & constants,
    const std::optional<core::TensorValue> & prefix_key = std::nullopt,
    const std::optional<core::TensorValue> & prefix_value = std::nullopt,
    const std::optional<core::TensorValue> & attention_mask = std::nullopt);

VibeVoiceDecoderLayerOutputs build_vibevoice_decoder_layer_static_tail(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const VibeVoiceDecoderLayerWeights & weights,
    const VibeVoiceDecoderConfig & config,
    common::ConstantTensorCache & constants,
    const core::TensorValue & cache_key,
    const core::TensorValue & cache_value,
    const core::TensorValue & cache_slot,
    const core::TensorValue & attention_mask);

}  // namespace engine::models::vibevoice
