#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/transformers/qwen_causal_decoder.h"
#include "engine/models/neutts/assets.h"

#include <cstddef>
#include <memory>

#include <ggml-backend.h>

namespace engine::core {
class BackendWeightStore;
}

namespace engine::models::neutts {

struct NeuTTSBackboneWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue token_embedding;
    modules::QwenCausalDecoderWeights decoder;
};

modules::QwenCausalDecoderConfig make_neutts_qwen_config(
    const NeuTTSBackboneConfig & config,
    core::BackendType backend_type);

NeuTTSBackboneWeights load_neutts_backbone_weights(
    const NeuTTSAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type);

}  // namespace engine::models::neutts
