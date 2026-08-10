#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/models/neutts/assets.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <ggml-backend.h>

namespace engine::core {
class BackendWeightStore;
class ExecutionContext;
}

namespace engine::models::neutts {

struct NeuTTSCodecResnetBlockWeights {
    modules::NormWeights norm1;
    modules::Conv1dWeights conv1;
    modules::NormWeights norm2;
    modules::Conv1dWeights conv2;
};

struct NeuTTSCodecTransformerLayerWeights {
    modules::NormWeights attention_norm;
    std::optional<modules::LinearWeights> qkv_proj;
    modules::LinearWeights q_proj;
    modules::LinearWeights k_proj;
    modules::LinearWeights v_proj;
    modules::LinearWeights out_proj;
    modules::NormWeights ffn_norm;
    modules::LinearWeights fc1;
    modules::LinearWeights fc2;
};

struct NeuTTSCodecDecoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::LinearWeights quantizer_project_out;
    modules::LinearWeights acoustic_fc;
    modules::Conv1dWeights embed;
    std::vector<NeuTTSCodecResnetBlockWeights> prior_blocks;
    std::vector<NeuTTSCodecTransformerLayerWeights> transformer_layers;
    std::vector<NeuTTSCodecResnetBlockWeights> post_blocks;
    modules::NormWeights final_norm;
    modules::LinearWeights istft_head;
};

struct NeuTTSCodecHead {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t out_dim = 0;
};

class NeuTTSCodecDecoderRuntime {
public:
    NeuTTSCodecDecoderRuntime(
        std::shared_ptr<const NeuTTSAssets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType matmul_storage_type,
        assets::TensorStorageType conv_storage_type);
    ~NeuTTSCodecDecoderRuntime();

    NeuTTSCodecDecoderRuntime(const NeuTTSCodecDecoderRuntime &) = delete;
    NeuTTSCodecDecoderRuntime & operator=(const NeuTTSCodecDecoderRuntime &) = delete;
    NeuTTSCodecDecoderRuntime(NeuTTSCodecDecoderRuntime &&) noexcept;
    NeuTTSCodecDecoderRuntime & operator=(NeuTTSCodecDecoderRuntime &&) noexcept;

    NeuTTSCodecHead decode_head(const std::vector<int32_t> & codes);
    std::vector<float> decode_audio(const std::vector<int32_t> & codes);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

NeuTTSCodecDecoderWeights load_neutts_codec_decoder_weights(
    const NeuTTSAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type);

std::vector<float> decode_neutts_fsq_levels(
    const std::vector<int32_t> & codes,
    const std::vector<int64_t> & levels);

}  // namespace engine::models::neutts
