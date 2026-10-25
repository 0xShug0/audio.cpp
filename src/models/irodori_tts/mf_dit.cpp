#include "mf_dit.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/models/irodori_tts/schedule.h"

#include <utility>

namespace engine::models::irodori_tts {

core::TensorValue build_irodori_meanflow_conditioning(
    core::ModuleBuildContext &ctx, const core::TensorValue &interval_embedding,
    const core::TensorValue &time_condition,
    const std::array<modules::LinearWeights, 3> &weights,
    const IrodoriModelConfig &config) {
  namespace binding = modules::binding;
  auto delta = modules::LinearModule(binding::linear_config(
      config.timestep_embed_dim, config.model_dim, false))
                   .build(ctx, interval_embedding, weights[0]);
  delta = modules::SiluModule{}.build(ctx, delta);
  delta = modules::LinearModule(binding::linear_config(
      config.model_dim, config.model_dim, false)).build(ctx, delta, weights[1]);
  delta = modules::SiluModule{}.build(ctx, delta);
  delta = modules::LinearModule(binding::linear_config(
      config.model_dim, 3 * config.model_dim, false)).build(ctx, delta, weights[2]);
  delta = core::reshape_tensor(ctx, delta, time_condition.shape);
  return modules::AddModule{}.build(ctx, time_condition, delta);
}

std::vector<float> sample_irodori_meanflow(
    const IrodoriGenerationOptions &options, int64_t patched_steps,
    int64_t patched_dim, bool mem_saver,
    modules::FlowSamplerSingleBranchDenoiserConfig denoiser) {
  auto latent = sampling::generate_torch_cuda_randn(
      static_cast<size_t>(patched_steps * patched_dim), options.seed,
      sampling::TorchRandnPrecision::Float32);

  modules::FlowSamplerRuntimeConfig config;
  config.label = "Irodori-TTS MeanFlow";
  config.latent_shape = {patched_steps, patched_dim};
  config.schedule = make_irodori_meanflow_schedule(options.num_inference_steps);
  config.branches = {{"velocity", 1.0F}};
  config.release_graph_after_sequence = mem_saver;
  debug::timing_log_scalar("irodori_tts.meanflow.nfe", options.num_inference_steps);
  denoiser.label = "Irodori-TTS MeanFlow denoiser";
  denoiser.sampler_mode = [](const modules::FlowSamplerStepState &) {
    return "irodori_meanflow";
  };
  modules::FlowSamplerRuntime flow(
      std::move(config),
      modules::make_flow_sampler_single_branch_denoiser(std::move(denoiser)));
  flow.run_sequence(latent);
  return flow.latent();
}

} // namespace engine::models::irodori_tts
