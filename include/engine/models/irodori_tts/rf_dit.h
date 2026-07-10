#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/models/irodori_tts/assets.h"

#include <ggml-backend.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace engine::core {
class BackendWeightStore;
}

namespace engine::models::irodori_tts {

struct IrodoriLowRankAdaLNWeights {
  modules::LinearWeights shift_down;
  modules::LinearWeights shift_up;
  modules::LinearWeights scale_down;
  modules::LinearWeights scale_up;
  modules::LinearWeights gate_down;
  modules::LinearWeights gate_up;
};

struct IrodoriJointAttentionWeights {
  modules::LinearWeights wq;
  modules::LinearWeights wk;
  modules::LinearWeights wv;
  modules::LinearWeights qkvg;
  modules::LinearWeights gate;
  modules::LinearWeights wk_text;
  modules::LinearWeights wv_text;
  modules::LinearWeights wk_speaker;
  modules::LinearWeights wv_speaker;
  modules::LinearWeights wk_caption;
  modules::LinearWeights wv_caption;
  modules::LinearWeights wo;
  core::TensorValue q_norm;
  core::TensorValue k_norm;
};

struct IrodoriDiffusionBlockWeights {
  IrodoriJointAttentionWeights attention;
  IrodoriLowRankAdaLNWeights attention_adaln;
  IrodoriLowRankAdaLNWeights mlp_adaln;
  modules::LinearWeights mlp_w1;
  modules::LinearWeights mlp_w2;
  modules::LinearWeights mlp_w3;
};

struct IrodoriRfDitWeights {
  std::shared_ptr<core::BackendWeightStore> store;
  core::TensorValue timestep_freqs;
  modules::LinearWeights cond_fc0;
  modules::LinearWeights cond_fc1;
  modules::LinearWeights cond_fc2;
  modules::LinearWeights in_proj;
  std::vector<IrodoriDiffusionBlockWeights> blocks;
  core::TensorValue out_norm;
  modules::LinearWeights out_proj;
};

struct IrodoriLayerContextKV {
  core::TensorValue k_context;
  core::TensorValue v_context;
};

struct IrodoriAdaLNModulation {
  core::TensorValue shift;
  core::TensorValue scale;
  core::TensorValue gate;
};

struct IrodoriLayerAdaLNModulation {
  IrodoriAdaLNModulation attention;
  IrodoriAdaLNModulation mlp;
};

IrodoriRfDitWeights
load_irodori_rf_dit_weights(const IrodoriAssets &assets, ggml_backend_t backend,
                            core::BackendType backend_type,
                            size_t weight_context_bytes,
                            assets::TensorStorageType weight_storage_type);

std::vector<IrodoriLayerContextKV> build_irodori_context_kv_cache(
    core::ModuleBuildContext &ctx, const core::TensorValue &text_state,
    const core::TensorValue &speaker_state,
    const core::TensorValue &caption_state, const IrodoriRfDitWeights &weights,
    const IrodoriModelConfig &config);

std::vector<IrodoriLayerAdaLNModulation> build_irodori_adaln_modulation_cache(
    core::ModuleBuildContext &ctx, const core::TensorValue &t,
    const IrodoriRfDitWeights &weights, const IrodoriModelConfig &config);

core::TensorValue build_irodori_rf_dit(
    core::ModuleBuildContext &ctx, const core::TensorValue &x_t,
    const core::TensorValue &t, const core::TensorValue &text_state,
    const core::TensorValue &speaker_state,
    const core::TensorValue &caption_state,
    const core::TensorValue &attention_mask, const core::TensorValue &positions,
    const IrodoriRfDitWeights &weights, const IrodoriModelConfig &config,
    const std::vector<IrodoriLayerContextKV> *context_kv_cache = nullptr,
    const std::vector<IrodoriLayerAdaLNModulation> *modulation_cache = nullptr);

} // namespace engine::models::irodori_tts
