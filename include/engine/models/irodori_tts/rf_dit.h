#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/models/irodori_tts/assets.h"
#include "engine/models/irodori_tts/condition_encoder.h"

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

class IrodoriRfSampler {
public:
  class ContextCache {
  public:
    ContextCache();
    ~ContextCache();

  private:
    friend class IrodoriRfSampler;
    class State;
    std::shared_ptr<State> state_;
  };

  class ModulationCache {
  public:
    ModulationCache();
    ~ModulationCache();

  private:
    friend class IrodoriRfSampler;
    class State;
    std::shared_ptr<State> state_;
  };

  IrodoriRfSampler(std::shared_ptr<const IrodoriAssets> assets,
                   core::ExecutionContext &execution_context,
                   size_t graph_arena_bytes, size_t weight_context_bytes,
                   assets::TensorStorageType weight_storage_type,
                   bool mem_saver);
  ~IrodoriRfSampler();

  ContextCache build_context_cache(const std::vector<float> &text_state_cond,
                                   const std::vector<uint8_t> &text_mask_cond,
                                   const std::vector<float> &caption_state_cond,
                                   const IrodoriCaptionCondition &caption,
                                   const IrodoriSpeakerCondition &speaker,
                                   bool text_cfg_enabled,
                                   bool speaker_cfg_enabled,
                                   bool caption_cfg_enabled);
  ModulationCache build_modulation_cache(const std::vector<float> &timesteps);
  void run_step(const std::vector<float> &x_t, int64_t step,
                const ModulationCache &modulation_cache,
                const ContextCache &context_cache, bool text_cfg_enabled,
                bool speaker_cfg_enabled, bool caption_cfg_enabled,
                float text_guidance_scale, float speaker_guidance_scale,
                float caption_guidance_scale, int64_t latent_steps,
                std::vector<float> &velocity);
  void release_graphs();
  int64_t context_graph_rebuilds() const noexcept;
  int64_t step_graph_rebuilds() const noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace engine::models::irodori_tts
