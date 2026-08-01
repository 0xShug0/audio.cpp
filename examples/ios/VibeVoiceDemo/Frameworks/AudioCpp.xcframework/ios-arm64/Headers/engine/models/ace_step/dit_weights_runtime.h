#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/transformers/qwen_decoder.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/models/ace_step/assets.h"

#include <memory>
#include <optional>
#include <vector>

namespace engine::models::ace_step {

struct AceStepDetokenizerWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::LinearWeights embed_tokens;
    modules::QwenDecoderStackWeights layers;
    core::TensorValue norm;
    modules::LinearWeights proj_out;
    std::vector<float> quantizer_project_out_weight;
    std::vector<float> quantizer_project_out_bias;
    std::vector<float> special_tokens_host;
};

struct AceStepCoverTokenizerWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::LinearWeights audio_acoustic_proj;
    modules::LinearWeights attention_pooler_embed_tokens;
    modules::QwenDecoderStackWeights attention_pooler_layers;
    core::TensorValue attention_pooler_norm;
    modules::LinearWeights quantizer_project_in;
    std::vector<float> attention_pooler_special_token_host;
};

struct AceStepConditionEncoderLayerWeights {
    core::TensorValue input_norm;
    core::TensorValue q_proj;
    core::TensorValue k_proj;
    core::TensorValue v_proj;
    core::TensorValue o_proj;
    core::TensorValue q_norm;
    core::TensorValue k_norm;
    core::TensorValue post_norm;
    core::TensorValue gate_proj;
    core::TensorValue up_proj;
    core::TensorValue down_proj;
};

struct AceStepConditionEncoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue text_projector;
    core::TensorValue lyric_embed_weight;
    core::TensorValue lyric_embed_bias;
    std::vector<AceStepConditionEncoderLayerWeights> lyric_layers;
    core::TensorValue lyric_norm;
    core::TensorValue timbre_embed_weight;
    core::TensorValue timbre_embed_bias;
    std::vector<AceStepConditionEncoderLayerWeights> timbre_layers;
    core::TensorValue timbre_norm;
};

struct AceStepTimeEmbeddingWeights {
    modules::LinearWeights fc1;
    modules::LinearWeights fc2;
    modules::LinearWeights time_proj;
};

struct AceStepDiTAttentionWeights {
    core::TensorValue q_weight;
    core::TensorValue k_weight;
    core::TensorValue v_weight;
    core::TensorValue out_weight;
    core::TensorValue q_norm;
    core::TensorValue k_norm;
};

struct AceStepDiTLayerWeights {
    core::TensorValue self_attn_norm;
    AceStepDiTAttentionWeights self_attn;
    core::TensorValue cross_attn_norm;
    AceStepDiTAttentionWeights cross_attn;
    core::TensorValue mlp_norm;
    modules::LinearWeights mlp_gate;
    modules::LinearWeights mlp_up;
    modules::LinearWeights mlp_down;
    core::TensorValue scale_shift_table;
};

struct AceStepDiffusionWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue one;
    modules::Conv1dWeights proj_in;
    AceStepTimeEmbeddingWeights time_embed;
    AceStepTimeEmbeddingWeights time_embed_r;
    modules::LinearWeights condition_embedder;
    std::vector<float> null_condition_emb_host;
    std::vector<AceStepDiTLayerWeights> layers;
    core::TensorValue norm_out;
    modules::ConvTranspose1dWeights proj_out;
    core::TensorValue final_scale_shift_table;
};

class AceStepDitWeightsRuntime {
public:
    AceStepDitWeightsRuntime(
        std::shared_ptr<const AceStepAssets> assets,
        core::ExecutionContext & execution,
        assets::TensorStorageType storage_type = assets::TensorStorageType::Native,
        size_t weight_context_bytes = 256ull * 1024ull * 1024ull);

    const std::shared_ptr<const AceStepAssets> & assets() const noexcept;
    std::shared_ptr<const AceStepDetokenizerWeights> detokenizer_weights() const noexcept;
    std::shared_ptr<const AceStepConditionEncoderWeights> condition_encoder_weights() const noexcept;
    std::shared_ptr<const AceStepDiffusionWeights> diffusion_weights() const noexcept;

private:
    std::shared_ptr<const AceStepAssets> assets_;
    std::shared_ptr<core::BackendWeightStore> store_;
    std::shared_ptr<const AceStepDetokenizerWeights> detokenizer_weights_;
    std::shared_ptr<const AceStepConditionEncoderWeights> condition_encoder_weights_;
    std::shared_ptr<const AceStepDiffusionWeights> diffusion_weights_;
};

}  // namespace engine::models::ace_step
