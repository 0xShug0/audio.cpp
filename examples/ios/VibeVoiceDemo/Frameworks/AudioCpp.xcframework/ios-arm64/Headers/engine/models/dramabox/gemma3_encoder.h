#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/transformers/gemma_decoder.h"
#include "engine/models/dramabox/assets.h"
#include "engine/models/dramabox/gemma_tokenizer.h"

#include <ggml-backend.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace engine::core {
class BackendWeightStore;
class ExecutionContext;
}

namespace engine::models::dramabox {

struct DramaBoxGemma3PromptWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::GemmaDecoderComponent encoder;
    std::vector<modules::LinearWeights> aggregate_layers;
    core::TensorValue aggregate_bias;
};

struct DramaBoxPromptEncoding {
    int64_t batch = 0;
    int64_t tokens = 0;
    int64_t hidden_size = 0;
    std::vector<float> audio_features;
    std::vector<int32_t> attention_mask;
};

DramaBoxGemma3PromptWeights load_dramabox_gemma3_prompt_weights(
    const DramaBoxAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType gemma_weight_storage_type,
    assets::TensorStorageType projection_weight_storage_type);

core::TensorValue build_dramabox_gemma3_prompt_encoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_ids,
    const core::TensorValue & positions,
    const core::TensorValue & additive_attention_mask,
    const core::TensorValue & hidden_attention_mask,
    const DramaBoxGemma3PromptWeights & weights,
    const DramaBoxConfig & config);

class DramaBoxGemma3PromptRuntime {
public:
    DramaBoxGemma3PromptRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const DramaBoxAssets> assets,
        assets::TensorStorageType gemma_weight_storage_type,
        assets::TensorStorageType projection_weight_storage_type,
        int64_t max_batch);
    ~DramaBoxGemma3PromptRuntime();

    DramaBoxGemma3PromptRuntime(const DramaBoxGemma3PromptRuntime &) = delete;
    DramaBoxGemma3PromptRuntime & operator=(const DramaBoxGemma3PromptRuntime &) = delete;

    void prepare(int64_t batch) const;
    DramaBoxPromptEncoding encode(const DramaBoxGemmaTokenBatch & tokens) const;
    void release_runtime_state() const;

private:
    class Graph;

    core::ExecutionContext * execution_ = nullptr;
    std::shared_ptr<const DramaBoxAssets> assets_;
    assets::TensorStorageType gemma_weight_storage_type_ = assets::TensorStorageType::Native;
    assets::TensorStorageType projection_weight_storage_type_ = assets::TensorStorageType::Native;
    int64_t max_batch_ = 1;
    mutable std::unique_ptr<DramaBoxGemma3PromptWeights> weights_;
    mutable std::unique_ptr<Graph> graph_;
};

}  // namespace engine::models::dramabox
