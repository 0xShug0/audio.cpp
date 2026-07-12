#include "engine/models/irodori_tts/rf_dit.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_projection_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::irodori_tts {
namespace {

namespace binding = modules::binding;

constexpr size_t kRfDitWeightContextBytes = 768ull * 1024ull * 1024ull;
constexpr size_t kRfDitIoContextBytes = 64ull * 1024ull * 1024ull;

modules::LinearWeights
load_linear(core::BackendWeightStore &store, const assets::TensorSource &source,
            const std::string &prefix, assets::TensorStorageType storage_type,
            int64_t out_features, int64_t in_features, bool use_bias) {
  modules::LinearWeights weights;
  weights.weight = store.load_tensor(source, prefix + ".weight", storage_type,
                                     {out_features, in_features});
  if (use_bias) {
    weights.bias =
        store.load_f32_tensor(source, prefix + ".bias", {out_features});
  }
  return weights;
}

assets::TensorStorageType
resolve_derived_storage_type(const assets::TensorSource &source,
                             const std::string &tensor_name,
                             assets::TensorStorageType requested_type) {
  if (requested_type != assets::TensorStorageType::Native) {
    return requested_type;
  }
  return assets::tensor_storage_type_for_dtype(
      source.require_metadata(tensor_name).dtype);
}

modules::LinearWeights load_packed_qkvg(core::BackendWeightStore &store,
                                        const assets::TensorSource &source,
                                        const std::string &prefix,
                                        assets::TensorStorageType storage_type,
                                        const IrodoriModelConfig &config) {
  const int64_t out_features = config.model_dim;
  const int64_t in_features = config.model_dim;
  const auto expected_shape = std::vector<int64_t>{out_features, in_features};
  const auto q = source.require_f32(prefix + ".wq.weight", expected_shape);
  const auto k = source.require_f32(prefix + ".wk.weight", expected_shape);
  const auto v = source.require_f32(prefix + ".wv.weight", expected_shape);
  const auto gate = source.require_f32(prefix + ".gate.weight", expected_shape);
  std::vector<float> packed(
      static_cast<size_t>(4 * out_features * in_features));
  auto copy_rows = [&](const std::vector<float> &source_values,
                       int64_t block_index) {
    const size_t block_offset =
        static_cast<size_t>(block_index * out_features * in_features);
    std::copy(source_values.begin(), source_values.end(),
              packed.begin() + static_cast<std::ptrdiff_t>(block_offset));
  };
  copy_rows(q, 0);
  copy_rows(k, 1);
  copy_rows(v, 2);
  copy_rows(gate, 3);

  modules::LinearWeights weights;
  weights.weight = store.make_from_f32(
      core::TensorShape::from_dims({4 * out_features, in_features}),
      resolve_derived_storage_type(source, prefix + ".wq.weight", storage_type),
      std::move(packed));
  return weights;
}

int64_t head_dim(const IrodoriModelConfig &config) {
  return config.model_dim / config.num_heads;
}

int64_t ffn_dim(const IrodoriModelConfig &config) {
  return static_cast<int64_t>(static_cast<double>(config.model_dim) *
                              static_cast<double>(config.mlp_ratio));
}

std::vector<float> make_timestep_freqs(int64_t dim) {
  const int64_t half = dim / 2;
  std::vector<float> freqs(static_cast<size_t>(half));
  for (int64_t i = 0; i < half; ++i) {
    freqs[static_cast<size_t>(i)] =
        1000.0F * std::exp(-std::log(10000.0F) * static_cast<float>(i) /
                           static_cast<float>(half));
  }
  return freqs;
}

IrodoriLowRankAdaLNWeights load_adaln(core::BackendWeightStore &store,
                                      const assets::TensorSource &source,
                                      const std::string &prefix,
                                      assets::TensorStorageType storage_type,
                                      const IrodoriModelConfig &config) {
  IrodoriLowRankAdaLNWeights weights;
  weights.shift_down =
      load_linear(store, source, prefix + ".shift_down", storage_type,
                  config.adaln_rank, config.model_dim, false);
  weights.shift_up =
      load_linear(store, source, prefix + ".shift_up", storage_type,
                  config.model_dim, config.adaln_rank, true);
  weights.scale_down =
      load_linear(store, source, prefix + ".scale_down", storage_type,
                  config.adaln_rank, config.model_dim, false);
  weights.scale_up =
      load_linear(store, source, prefix + ".scale_up", storage_type,
                  config.model_dim, config.adaln_rank, true);
  weights.gate_down =
      load_linear(store, source, prefix + ".gate_down", storage_type,
                  config.adaln_rank, config.model_dim, false);
  weights.gate_up =
      load_linear(store, source, prefix + ".gate_up", storage_type,
                  config.model_dim, config.adaln_rank, true);
  return weights;
}

IrodoriJointAttentionWeights load_joint_attention(
    core::BackendWeightStore &store, const assets::TensorSource &source,
    const std::string &prefix, assets::TensorStorageType storage_type,
    const IrodoriModelConfig &config, core::BackendType backend_type) {
  IrodoriJointAttentionWeights weights;
  if (backend_type == core::BackendType::Cuda) {
    weights.qkvg =
        load_packed_qkvg(store, source, prefix, storage_type, config);
  } else {
    weights.wq = load_linear(store, source, prefix + ".wq", storage_type,
                             config.model_dim, config.model_dim, false);
    weights.wk = load_linear(store, source, prefix + ".wk", storage_type,
                             config.model_dim, config.model_dim, false);
    weights.wv = load_linear(store, source, prefix + ".wv", storage_type,
                             config.model_dim, config.model_dim, false);
    weights.gate = load_linear(store, source, prefix + ".gate", storage_type,
                               config.model_dim, config.model_dim, false);
  }
  weights.wk_text =
      load_linear(store, source, prefix + ".wk_text", storage_type,
                  config.model_dim, config.text_dim, false);
  weights.wv_text =
      load_linear(store, source, prefix + ".wv_text", storage_type,
                  config.model_dim, config.text_dim, false);
  weights.wk_speaker =
      load_linear(store, source, prefix + ".wk_speaker", storage_type,
                  config.model_dim, config.speaker_dim, false);
  weights.wv_speaker =
      load_linear(store, source, prefix + ".wv_speaker", storage_type,
                  config.model_dim, config.speaker_dim, false);
  if (config.use_caption_condition) {
    weights.wk_caption =
        load_linear(store, source, prefix + ".wk_caption", storage_type,
                    config.model_dim, config.caption_dim_resolved(), false);
    weights.wv_caption =
        load_linear(store, source, prefix + ".wv_caption", storage_type,
                    config.model_dim, config.caption_dim_resolved(), false);
  }
  weights.wo = load_linear(store, source, prefix + ".wo", storage_type,
                           config.model_dim, config.model_dim, false);
  weights.q_norm = store.load_f32_tensor(source, prefix + ".q_norm.weight",
                                         {config.num_heads, head_dim(config)});
  weights.k_norm = store.load_f32_tensor(source, prefix + ".k_norm.weight",
                                         {config.num_heads, head_dim(config)});
  return weights;
}

IrodoriDiffusionBlockWeights
load_block(core::BackendWeightStore &store, const assets::TensorSource &source,
           const std::string &prefix, assets::TensorStorageType storage_type,
           const IrodoriModelConfig &config, core::BackendType backend_type) {
  IrodoriDiffusionBlockWeights weights;
  weights.attention = load_joint_attention(store, source, prefix + ".attention",
                                           storage_type, config, backend_type);
  weights.attention_adaln = load_adaln(
      store, source, prefix + ".attention_adaln", storage_type, config);
  weights.mlp_adaln =
      load_adaln(store, source, prefix + ".mlp_adaln", storage_type, config);
  weights.mlp_w1 = load_linear(store, source, prefix + ".mlp.w1", storage_type,
                               ffn_dim(config), config.model_dim, false);
  weights.mlp_w2 = load_linear(store, source, prefix + ".mlp.w2", storage_type,
                               config.model_dim, ffn_dim(config), false);
  weights.mlp_w3 = load_linear(store, source, prefix + ".mlp.w3", storage_type,
                               ffn_dim(config), config.model_dim, false);
  return weights;
}

core::TensorValue reshape_heads(core::ModuleBuildContext &ctx,
                                const core::TensorValue &input, int64_t heads,
                                int64_t dim) {
  return core::reshape_tensor(
      ctx, core::ensure_backend_addressable_layout(ctx, input),
      core::TensorShape::from_dims(
          {input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

core::TensorValue head_rms_norm(core::ModuleBuildContext &ctx,
                                const core::TensorValue &input,
                                const core::TensorValue &weight, float eps) {
  core::validate_shape(
      weight,
      core::TensorShape::from_dims({input.shape.dims[2], input.shape.dims[3]}),
      "head_rms_norm.weight");
  auto normalized = core::wrap_tensor(
      ggml_rms_norm(ctx.ggml,
                    core::ensure_backend_addressable_layout(ctx, input).tensor,
                    eps),
      input.shape, GGML_TYPE_F32);
  auto weight_view = core::reshape_tensor(
      ctx, weight,
      core::TensorShape::from_dims(
          {1, 1, input.shape.dims[2], input.shape.dims[3]}));
  return core::wrap_tensor(ggml_mul(ctx.ggml, normalized.tensor,
                                    modules::RepeatModule({normalized.shape})
                                        .build(ctx, weight_view)
                                        .tensor),
                           input.shape, GGML_TYPE_F32);
}

core::TensorValue build_timestep_embedding(core::ModuleBuildContext &ctx,
                                           const core::TensorValue &t,
                                           const IrodoriRfDitWeights &weights,
                                           const IrodoriModelConfig &config) {
  const int64_t batch = t.shape.dims[0];
  const int64_t half = config.timestep_embed_dim / 2;
  auto freqs = core::reshape_tensor(ctx, weights.timestep_freqs,
                                    core::TensorShape::from_dims({1, half}));
  freqs = modules::RepeatModule({core::TensorShape::from_dims({batch, half})})
              .build(ctx, freqs);
  auto t_expanded =
      core::reshape_tensor(ctx, t, core::TensorShape::from_dims({batch, 1}));
  t_expanded =
      modules::RepeatModule({core::TensorShape::from_dims({batch, half})})
          .build(ctx, t_expanded);
  auto args = modules::MulModule{}.build(ctx, t_expanded, freqs);
  auto cos_part = core::wrap_tensor(ggml_cos(ctx.ggml, args.tensor), args.shape,
                                    GGML_TYPE_F32);
  auto sin_part = core::wrap_tensor(ggml_sin(ctx.ggml, args.tensor), args.shape,
                                    GGML_TYPE_F32);
  return modules::ConcatModule({1}).build(ctx, cos_part, sin_part);
}

core::TensorValue build_condition_embedding(core::ModuleBuildContext &ctx,
                                            const core::TensorValue &t,
                                            const IrodoriRfDitWeights &weights,
                                            const IrodoriModelConfig &config,
                                            bool collapse_batch) {
  const int64_t batch = t.shape.dims[0];
  auto t_for_embedding = collapse_batch && batch > 1
                             ? modules::SliceModule({0, 0, 1}).build(ctx, t)
                             : t;
  auto hidden = build_timestep_embedding(ctx, t_for_embedding, weights, config);
  hidden =
      modules::LinearModule(binding::linear_config(config.timestep_embed_dim,
                                                   config.model_dim, false))
          .build(ctx, hidden, weights.cond_fc0);
  hidden = modules::SiluModule{}.build(ctx, hidden);
  hidden = modules::LinearModule(binding::linear_config(
                                     config.model_dim, config.model_dim, false))
               .build(ctx, hidden, weights.cond_fc1);
  hidden = modules::SiluModule{}.build(ctx, hidden);
  hidden =
      modules::LinearModule(
          binding::linear_config(config.model_dim, 3 * config.model_dim, false))
          .build(ctx, hidden, weights.cond_fc2);
  return core::reshape_tensor(
      ctx, hidden,
      core::TensorShape::from_dims(
          {hidden.shape.dims[0], 1, hidden.shape.dims[1]}));
}

core::TensorValue adaln_part(core::ModuleBuildContext &ctx,
                             const core::TensorValue &input,
                             const modules::LinearWeights &down,
                             const modules::LinearWeights &up,
                             const core::TensorValue &residual,
                             const IrodoriModelConfig &config) {
  auto hidden = modules::SiluModule{}.build(ctx, input);
  hidden =
      modules::LinearModule(
          binding::linear_config(config.model_dim, config.adaln_rank, false))
          .build(ctx, hidden, down);
  hidden = modules::LinearModule(binding::linear_config(config.adaln_rank,
                                                        config.model_dim, true))
               .build(ctx, hidden, up);
  return modules::AddModule{}.build(ctx, hidden, residual);
}

IrodoriAdaLNModulation
build_adaln_modulation(core::ModuleBuildContext &ctx,
                       const core::TensorValue &cond_embed,
                       const IrodoriLowRankAdaLNWeights &weights,
                       const IrodoriModelConfig &config) {
  auto shift_base =
      modules::SliceModule({2, 0, config.model_dim}).build(ctx, cond_embed);
  auto scale_base =
      modules::SliceModule({2, config.model_dim, config.model_dim})
          .build(ctx, cond_embed);
  auto gate_base =
      modules::SliceModule({2, 2 * config.model_dim, config.model_dim})
          .build(ctx, cond_embed);
  auto shift = adaln_part(ctx, shift_base, weights.shift_down, weights.shift_up,
                          shift_base, config);
  auto scale = adaln_part(ctx, scale_base, weights.scale_down, weights.scale_up,
                          scale_base, config);
  auto gate = modules::TanhModule{}.build(
      ctx, adaln_part(ctx, gate_base, weights.gate_down, weights.gate_up,
                      gate_base, config));
  return {shift, scale, gate};
}

struct AdaLNOutput {
  core::TensorValue hidden;
  core::TensorValue gate;
};

AdaLNOutput apply_adaln_modulation(core::ModuleBuildContext &ctx,
                                   const core::TensorValue &x,
                                   const IrodoriAdaLNModulation &modulation,
                                   const IrodoriModelConfig &config) {
  auto hidden =
      modules::RMSNormModule({config.model_dim, config.norm_eps, false, false})
          .build(ctx, x, {std::nullopt, std::nullopt});
  auto one_plus_scale = core::wrap_tensor(
      ggml_scale_bias(ctx.ggml, modulation.scale.tensor, 1.0F, 1.0F),
      modulation.scale.shape, GGML_TYPE_F32);
  auto scaled = core::wrap_tensor(
      ggml_mul(ctx.ggml, hidden.tensor, one_plus_scale.tensor), hidden.shape,
      GGML_TYPE_F32);
  hidden = core::wrap_tensor(
      ggml_add(ctx.ggml, scaled.tensor, modulation.shift.tensor), hidden.shape,
      GGML_TYPE_F32);
  return {hidden, modulation.gate};
}

AdaLNOutput low_rank_adaln(core::ModuleBuildContext &ctx,
                           const core::TensorValue &x,
                           const core::TensorValue &cond_embed,
                           const IrodoriLowRankAdaLNWeights &weights,
                           const IrodoriModelConfig &config) {
  return apply_adaln_modulation(
      ctx, x, build_adaln_modulation(ctx, cond_embed, weights, config), config);
}

core::TensorValue apply_rotary_half_heads(core::ModuleBuildContext &ctx,
                                          const core::TensorValue &input,
                                          const core::TensorValue &positions,
                                          int64_t dim) {
  const int64_t half_heads = input.shape.dims[2] / 2;
  auto rot = modules::SliceModule({2, 0, half_heads}).build(ctx, input);
  auto passthrough =
      modules::SliceModule({2, half_heads, input.shape.dims[2] - half_heads})
          .build(ctx, input);
  rot = modules::RoPEModule({dim, GGML_ROPE_TYPE_NORMAL, 10000.0F})
            .build(ctx, rot, positions);
  return modules::ConcatModule({2}).build(ctx, rot, passthrough);
}

core::TensorValue sub_tensor(core::ModuleBuildContext &ctx,
                             const core::TensorValue &lhs,
                             const core::TensorValue &rhs) {
  return core::wrap_tensor(
      ggml_sub(ctx.ggml,
               core::ensure_backend_addressable_layout(ctx, lhs).tensor,
               core::ensure_backend_addressable_layout(ctx, rhs).tensor),
      lhs.shape, GGML_TYPE_F32);
}

core::TensorValue scalar_like(core::ModuleBuildContext &ctx,
                              const core::TensorValue &scalar,
                              const core::TensorShape &shape) {
  auto reshaped = core::reshape_tensor(ctx, scalar,
                                       core::TensorShape::from_dims({1, 1, 1}));
  return modules::RepeatModule({shape}).build(ctx, reshaped);
}

core::TensorValue select_modulation_step(core::ModuleBuildContext &ctx,
                                         const core::TensorValue &cache,
                                         const core::TensorValue &step_index,
                                         const IrodoriModelConfig &config) {
  auto flat_cache = core::reshape_tensor(
      ctx, cache,
      core::TensorShape::from_dims({cache.shape.dims[0], config.model_dim}));
  auto selected = core::wrap_tensor(
      ggml_get_rows(
          ctx.ggml,
          core::ensure_backend_addressable_layout(ctx, flat_cache).tensor,
          step_index.tensor),
      core::TensorShape::from_dims({1, config.model_dim}), GGML_TYPE_F32);
  return core::reshape_tensor(
      ctx, selected, core::TensorShape::from_dims({1, 1, config.model_dim}));
}

IrodoriLayerAdaLNModulation select_modulation_step(
    core::ModuleBuildContext &ctx, const IrodoriLayerAdaLNModulation &cache,
    const core::TensorValue &step_index, const IrodoriModelConfig &config) {
  IrodoriLayerAdaLNModulation out;
  out.attention.shift =
      select_modulation_step(ctx, cache.attention.shift, step_index, config);
  out.attention.scale =
      select_modulation_step(ctx, cache.attention.scale, step_index, config);
  out.attention.gate =
      select_modulation_step(ctx, cache.attention.gate, step_index, config);
  out.mlp.shift =
      select_modulation_step(ctx, cache.mlp.shift, step_index, config);
  out.mlp.scale =
      select_modulation_step(ctx, cache.mlp.scale, step_index, config);
  out.mlp.gate =
      select_modulation_step(ctx, cache.mlp.gate, step_index, config);
  return out;
}

core::TensorValue flash_attention_from_heads(
    core::ModuleBuildContext &ctx, const core::TensorValue &q_heads,
    const core::TensorValue &k_heads, const core::TensorValue &v_heads,
    const core::TensorValue &attention_mask, int64_t dim) {
  const auto q_contiguous =
      core::ensure_backend_addressable_layout(ctx, q_heads);
  const auto k_contiguous =
      core::ensure_backend_addressable_layout(ctx, k_heads);
  const auto v_contiguous =
      core::ensure_backend_addressable_layout(ctx, v_heads);
  auto *flash = ggml_flash_attn_ext(
      ctx.ggml, q_contiguous.tensor, k_contiguous.tensor, v_contiguous.tensor,
      attention_mask.tensor, 1.0F / std::sqrt(static_cast<float>(dim)), 0.0F,
      0.0F);
  ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
  return core::wrap_tensor(
      flash,
      core::TensorShape::from_dims({q_contiguous.shape.dims[0],
                                    q_contiguous.shape.dims[2],
                                    q_contiguous.shape.dims[1], dim}),
      GGML_TYPE_F32);
}

IrodoriLayerContextKV
build_layer_context_kv(core::ModuleBuildContext &ctx,
                       const core::TensorValue &text_state,
                       const core::TensorValue &speaker_state,
                       const core::TensorValue &caption_state,
                       const IrodoriJointAttentionWeights &weights,
                       const IrodoriModelConfig &config) {
  const int64_t dim = head_dim(config);
  IrodoriLayerContextKV out;
  auto k_text =
      modules::LinearModule(
          binding::linear_config(config.text_dim, config.model_dim, false))
          .build(ctx, text_state, weights.wk_text);
  auto v_text =
      modules::LinearModule(
          binding::linear_config(config.text_dim, config.model_dim, false))
          .build(ctx, text_state, weights.wv_text);
  k_text = core::ensure_backend_addressable_layout(
      ctx, head_rms_norm(ctx, reshape_heads(ctx, k_text, config.num_heads, dim),
                         weights.k_norm, config.norm_eps));
  v_text = core::ensure_backend_addressable_layout(
      ctx, reshape_heads(ctx, v_text, config.num_heads, dim));

  auto k_speaker =
      modules::LinearModule(
          binding::linear_config(config.speaker_dim, config.model_dim, false))
          .build(ctx, speaker_state, weights.wk_speaker);
  auto v_speaker =
      modules::LinearModule(
          binding::linear_config(config.speaker_dim, config.model_dim, false))
          .build(ctx, speaker_state, weights.wv_speaker);
  k_speaker = core::ensure_backend_addressable_layout(
      ctx,
      head_rms_norm(ctx, reshape_heads(ctx, k_speaker, config.num_heads, dim),
                    weights.k_norm, config.norm_eps));
  v_speaker = core::ensure_backend_addressable_layout(
      ctx, reshape_heads(ctx, v_speaker, config.num_heads, dim));

  core::TensorValue k_caption;
  core::TensorValue v_caption;
  if (config.use_caption_condition && caption_state.tensor != nullptr &&
      caption_state.shape.dims[1] > 0) {
    k_caption = modules::LinearModule(
                    binding::linear_config(config.caption_dim_resolved(),
                                           config.model_dim, false))
                    .build(ctx, caption_state, weights.wk_caption);
    v_caption = modules::LinearModule(
                    binding::linear_config(config.caption_dim_resolved(),
                                           config.model_dim, false))
                    .build(ctx, caption_state, weights.wv_caption);
    k_caption = core::ensure_backend_addressable_layout(
        ctx,
        head_rms_norm(ctx, reshape_heads(ctx, k_caption, config.num_heads, dim),
                      weights.k_norm, config.norm_eps));
    v_caption = core::ensure_backend_addressable_layout(
        ctx, reshape_heads(ctx, v_caption, config.num_heads, dim));
  }
  out.k_context = modules::ConcatModule({1}).build(ctx, k_text, k_speaker);
  out.v_context = modules::ConcatModule({1}).build(ctx, v_text, v_speaker);
  if (config.use_caption_condition && k_caption.tensor != nullptr &&
      k_caption.shape.dims[1] > 0) {
    out.k_context =
        modules::ConcatModule({1}).build(ctx, out.k_context, k_caption);
    out.v_context =
        modules::ConcatModule({1}).build(ctx, out.v_context, v_caption);
  }
  out.k_context = core::ensure_backend_addressable_layout(ctx, out.k_context);
  out.v_context = core::ensure_backend_addressable_layout(ctx, out.v_context);
  return out;
}

core::TensorValue build_joint_attention(
    core::ModuleBuildContext &ctx, const core::TensorValue &x,
    const core::TensorValue &text_state, const core::TensorValue &speaker_state,
    const core::TensorValue &caption_state,
    const core::TensorValue &attention_mask, const core::TensorValue &positions,
    const IrodoriJointAttentionWeights &weights,
    const IrodoriModelConfig &config, int64_t layer_index,
    const IrodoriLayerContextKV *context_kv) {
  const int64_t dim = head_dim(config);
  core::TensorValue q;
  core::TensorValue k_self;
  core::TensorValue v_self;
  core::TensorValue gate;
  if (ctx.backend_type == core::BackendType::Cuda) {
    auto packed_projection = modules::FastPackedProjection4Module(
                                 {config.model_dim, 4 * config.model_dim})
                                 .build(ctx, x, weights.qkvg);
    q = modules::SliceModule({2, 0, config.model_dim})
            .build(ctx, packed_projection);
    k_self = modules::SliceModule({2, config.model_dim, config.model_dim})
                 .build(ctx, packed_projection);
    v_self = modules::SliceModule({2, 2 * config.model_dim, config.model_dim})
                 .build(ctx, packed_projection);
    auto gate_slice =
        modules::SliceModule({2, 3 * config.model_dim, config.model_dim})
            .build(ctx, packed_projection);
    q = core::ensure_backend_addressable_layout(ctx, q);
    k_self = core::ensure_backend_addressable_layout(ctx, k_self);
    v_self = core::ensure_backend_addressable_layout(ctx, v_self);
    gate_slice = core::ensure_backend_addressable_layout(ctx, gate_slice);
    gate = modules::SigmoidModule{}.build(ctx, gate_slice);
  } else {
    q = modules::LinearModule(
            binding::linear_config(config.model_dim, config.model_dim, false))
            .build(ctx, x, weights.wq);
    k_self =
        modules::LinearModule(
            binding::linear_config(config.model_dim, config.model_dim, false))
            .build(ctx, x, weights.wk);
    v_self =
        modules::LinearModule(
            binding::linear_config(config.model_dim, config.model_dim, false))
            .build(ctx, x, weights.wv);
    gate = modules::SigmoidModule{}.build(
        ctx,
        modules::LinearModule(
            binding::linear_config(config.model_dim, config.model_dim, false))
            .build(ctx, x, weights.gate));
  }

  q = head_rms_norm(ctx, reshape_heads(ctx, q, config.num_heads, dim),
                    weights.q_norm, config.norm_eps);
  k_self = head_rms_norm(ctx, reshape_heads(ctx, k_self, config.num_heads, dim),
                         weights.k_norm, config.norm_eps);
  v_self = reshape_heads(ctx, v_self, config.num_heads, dim);
  q = apply_rotary_half_heads(ctx, q, positions, dim);
  k_self = apply_rotary_half_heads(ctx, k_self, positions, dim);

  IrodoriLayerContextKV projected;
  if (context_kv == nullptr) {
    projected = build_layer_context_kv(ctx, text_state, speaker_state,
                                       caption_state, weights, config);
    context_kv = &projected;
  }
  if (context_kv->k_context.tensor == nullptr ||
      context_kv->v_context.tensor == nullptr) {
    throw std::runtime_error("Irodori-TTS RF context KV cache is missing");
  }
  auto k = modules::ConcatModule({1}).build(ctx, k_self, context_kv->k_context);
  auto v = modules::ConcatModule({1}).build(ctx, v_self, context_kv->v_context);
  k = core::ensure_backend_addressable_layout(ctx, k);
  v = core::ensure_backend_addressable_layout(ctx, v);
  auto q_heads =
      modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
  auto k_heads =
      modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
  auto v_heads =
      modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
  (void)layer_index;
  auto context = flash_attention_from_heads(ctx, q_heads, k_heads, v_heads,
                                            attention_mask, dim);
  context = core::reshape_tensor(
      ctx, core::ensure_backend_addressable_layout(ctx, context),
      core::TensorShape::from_dims(
          {x.shape.dims[0], x.shape.dims[1], config.model_dim}));
  context = modules::MulModule{}.build(ctx, context, gate);
  return modules::LinearModule(
             binding::linear_config(config.model_dim, config.model_dim, false))
      .build(ctx, context, weights.wo);
}

core::TensorValue build_swiglu(core::ModuleBuildContext &ctx,
                               const core::TensorValue &input,
                               const modules::LinearWeights &w1,
                               const modules::LinearWeights &w2,
                               const modules::LinearWeights &w3,
                               const IrodoriModelConfig &config) {
  auto gate =
      modules::LinearModule(
          binding::linear_config(config.model_dim, ffn_dim(config), false))
          .build(ctx, input, w1);
  gate = modules::SiluModule{}.build(ctx, gate);
  auto up = modules::LinearModule(binding::linear_config(
                                      config.model_dim, ffn_dim(config), false))
                .build(ctx, input, w3);
  return modules::LinearModule(
             binding::linear_config(ffn_dim(config), config.model_dim, false))
      .build(ctx, modules::MulModule{}.build(ctx, gate, up), w2);
}

core::TensorValue build_block(
    core::ModuleBuildContext &ctx, const core::TensorValue &x,
    const core::TensorValue &cond_embed, const core::TensorValue &text_state,
    const core::TensorValue &speaker_state,
    const core::TensorValue &caption_state,
    const core::TensorValue &attention_mask, const core::TensorValue &positions,
    const IrodoriDiffusionBlockWeights &weights,
    const IrodoriModelConfig &config, int64_t layer_index,
    const IrodoriLayerContextKV *context_kv,
    const IrodoriLayerAdaLNModulation *modulation) {
  auto attn_adaln =
      modulation == nullptr
          ? low_rank_adaln(ctx, x, cond_embed, weights.attention_adaln, config)
          : apply_adaln_modulation(ctx, x, modulation->attention, config);
  auto attn =
      build_joint_attention(ctx, attn_adaln.hidden, text_state, speaker_state,
                            caption_state, attention_mask, positions,
                            weights.attention, config, layer_index, context_kv);
  auto gated_attn =
      core::wrap_tensor(ggml_mul(ctx.ggml, attn.tensor, attn_adaln.gate.tensor),
                        attn.shape, GGML_TYPE_F32);
  auto hidden = core::wrap_tensor(
      ggml_add(ctx.ggml, x.tensor, gated_attn.tensor), x.shape, GGML_TYPE_F32);
  auto mlp_adaln =
      modulation == nullptr
          ? low_rank_adaln(ctx, hidden, cond_embed, weights.mlp_adaln, config)
          : apply_adaln_modulation(ctx, hidden, modulation->mlp, config);
  auto mlp = build_swiglu(ctx, mlp_adaln.hidden, weights.mlp_w1, weights.mlp_w2,
                          weights.mlp_w3, config);
  auto gated_mlp =
      core::wrap_tensor(ggml_mul(ctx.ggml, mlp.tensor, mlp_adaln.gate.tensor),
                        mlp.shape, GGML_TYPE_F32);
  return core::wrap_tensor(ggml_add(ctx.ggml, hidden.tensor, gated_mlp.tensor),
                           hidden.shape, GGML_TYPE_F32);
}

} // namespace

IrodoriRfDitWeights
load_irodori_rf_dit_weights(const IrodoriAssets &assets, ggml_backend_t backend,
                            core::BackendType backend_type,
                            size_t weight_context_bytes,
                            assets::TensorStorageType weight_storage_type) {
  const auto &config = assets.config;
  const auto &source = *assets.model_weights;
  IrodoriRfDitWeights weights;
  weights.store = std::make_shared<core::BackendWeightStore>(
      backend, backend_type, "irodori_tts.rf_dit.weights",
      weight_context_bytes == 0 ? kRfDitWeightContextBytes
                                : weight_context_bytes);
  weights.timestep_freqs = weights.store->make_f32(
      core::TensorShape::from_dims({config.timestep_embed_dim / 2}),
      make_timestep_freqs(config.timestep_embed_dim));
  weights.cond_fc0 =
      load_linear(*weights.store, source, "cond_module.0", weight_storage_type,
                  config.model_dim, config.timestep_embed_dim, false);
  weights.cond_fc1 =
      load_linear(*weights.store, source, "cond_module.2", weight_storage_type,
                  config.model_dim, config.model_dim, false);
  weights.cond_fc2 =
      load_linear(*weights.store, source, "cond_module.4", weight_storage_type,
                  3 * config.model_dim, config.model_dim, false);
  weights.in_proj =
      load_linear(*weights.store, source, "in_proj", weight_storage_type,
                  config.model_dim, config.patched_latent_dim(), true);
  weights.blocks.reserve(static_cast<size_t>(config.num_layers));
  for (int64_t layer = 0; layer < config.num_layers; ++layer) {
    weights.blocks.push_back(
        load_block(*weights.store, source, "blocks." + std::to_string(layer),
                   weight_storage_type, config, backend_type));
  }
  weights.out_norm = weights.store->load_f32_tensor(source, "out_norm.weight",
                                                    {config.model_dim});
  weights.out_proj =
      load_linear(*weights.store, source, "out_proj", weight_storage_type,
                  config.patched_latent_dim(), config.model_dim, true);
  weights.store->upload();
  return weights;
}

std::vector<IrodoriLayerContextKV> build_irodori_context_kv_cache(
    core::ModuleBuildContext &ctx, const core::TensorValue &text_state,
    const core::TensorValue &speaker_state,
    const core::TensorValue &caption_state, const IrodoriRfDitWeights &weights,
    const IrodoriModelConfig &config) {
  std::vector<IrodoriLayerContextKV> out;
  out.reserve(weights.blocks.size());
  for (const auto &block : weights.blocks) {
    out.push_back(build_layer_context_kv(ctx, text_state, speaker_state,
                                         caption_state, block.attention,
                                         config));
  }
  return out;
}

std::vector<IrodoriLayerAdaLNModulation> build_irodori_adaln_modulation_cache(
    core::ModuleBuildContext &ctx, const core::TensorValue &t,
    const IrodoriRfDitWeights &weights, const IrodoriModelConfig &config) {
  auto cond_embed = build_condition_embedding(ctx, t, weights, config, false);
  std::vector<IrodoriLayerAdaLNModulation> out;
  out.reserve(weights.blocks.size());
  for (const auto &block : weights.blocks) {
    IrodoriLayerAdaLNModulation layer;
    layer.attention =
        build_adaln_modulation(ctx, cond_embed, block.attention_adaln, config);
    layer.mlp =
        build_adaln_modulation(ctx, cond_embed, block.mlp_adaln, config);
    out.push_back(layer);
  }
  return out;
}

core::TensorValue build_irodori_rf_dit(
    core::ModuleBuildContext &ctx, const core::TensorValue &x_t,
    const core::TensorValue &t, const core::TensorValue &text_state,
    const core::TensorValue &speaker_state,
    const core::TensorValue &caption_state,
    const core::TensorValue &attention_mask, const core::TensorValue &positions,
    const IrodoriRfDitWeights &weights, const IrodoriModelConfig &config,
    const std::vector<IrodoriLayerContextKV> *context_kv_cache,
    const std::vector<IrodoriLayerAdaLNModulation> *modulation_cache) {
  if (context_kv_cache != nullptr &&
      context_kv_cache->size() != weights.blocks.size()) {
    throw std::runtime_error(
        "Irodori-TTS RF context KV cache layer count mismatch");
  }
  if (modulation_cache != nullptr &&
      modulation_cache->size() != weights.blocks.size()) {
    throw std::runtime_error(
        "Irodori-TTS RF AdaLN modulation layer count mismatch");
  }
  core::TensorValue cond_embed;
  if (modulation_cache == nullptr) {
    cond_embed = build_condition_embedding(ctx, t, weights, config, true);
  }
  auto hidden =
      modules::LinearModule(binding::linear_config(config.patched_latent_dim(),
                                                   config.model_dim, true))
          .build(ctx, x_t, weights.in_proj);
  for (size_t layer = 0; layer < weights.blocks.size(); ++layer) {
    const IrodoriLayerContextKV *context_kv =
        context_kv_cache == nullptr ? nullptr : &(*context_kv_cache)[layer];
    const IrodoriLayerAdaLNModulation *modulation =
        modulation_cache == nullptr ? nullptr : &(*modulation_cache)[layer];
    hidden = build_block(ctx, hidden, cond_embed, text_state, speaker_state,
                         caption_state, attention_mask, positions,
                         weights.blocks[layer], config,
                         static_cast<int64_t>(layer), context_kv, modulation);
  }
  hidden =
      modules::RMSNormModule({config.model_dim, config.norm_eps, true, false})
          .build(ctx, hidden, {weights.out_norm, std::nullopt});
  return modules::LinearModule(
             binding::linear_config(config.model_dim,
                                    config.patched_latent_dim(), true))
      .build(ctx, hidden, weights.out_proj);
}

core::TensorValue build_guided_velocity(
    core::ModuleBuildContext &ctx, const core::TensorValue &raw_output,
    const core::TensorValue &text_guidance,
    const core::TensorValue &speaker_guidance,
    const core::TensorValue &caption_guidance, bool text_cfg_enabled,
    bool speaker_cfg_enabled, bool caption_cfg_enabled) {
  const auto cond = modules::SliceModule({0, 0, 1}).build(ctx, raw_output);
  auto velocity = cond;
  int64_t branch = 1;
  auto add_branch = [&](const core::TensorValue &guidance) {
    auto uncond = modules::SliceModule({0, branch, 1}).build(ctx, raw_output);
    auto diff = sub_tensor(ctx, cond, uncond);
    auto scaled = modules::MulModule{}.build(
        ctx, diff, scalar_like(ctx, guidance, diff.shape));
    velocity = modules::AddModule{}.build(ctx, velocity, scaled);
    ++branch;
  };
  if (text_cfg_enabled) {
    add_branch(text_guidance);
  }
  if (speaker_cfg_enabled) {
    add_branch(speaker_guidance);
  }
  if (caption_cfg_enabled) {
    add_branch(caption_guidance);
  }
  return velocity;
}

namespace {

struct GgmlContextDeleter {
  void operator()(ggml_context *ctx) const noexcept {
    if (ctx != nullptr) {
      ggml_free(ctx);
    }
  }
};

std::vector<float>
make_rf_attention_mask(const std::vector<uint8_t> &text_mask,
                       const std::vector<uint8_t> &speaker_mask,
                       const std::vector<uint8_t> &caption_mask, int64_t batch,
                       int64_t latent_steps, int64_t text_tokens,
                       int64_t speaker_tokens, int64_t caption_tokens) {
  const int64_t keys =
      latent_steps + text_tokens + speaker_tokens + caption_tokens;
  std::vector<float> out(static_cast<size_t>(batch * latent_steps * keys),
                         -INFINITY);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if(batch * latent_steps * keys >= 4096)
#endif
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t q = 0; q < latent_steps; ++q) {
      for (int64_t k = 0; k < keys; ++k) {
        bool keep = k < latent_steps;
        if (k >= latent_steps && k < latent_steps + text_tokens) {
          keep = text_mask[static_cast<size_t>(b * text_tokens +
                                               (k - latent_steps))] != 0;
        } else if (k >= latent_steps + text_tokens &&
                   k < latent_steps + text_tokens + speaker_tokens) {
          keep =
              speaker_mask[static_cast<size_t>(
                  b * speaker_tokens + (k - latent_steps - text_tokens))] != 0;
        } else if (k >= latent_steps + text_tokens + speaker_tokens &&
                   caption_tokens > 0) {
          keep = caption_mask[static_cast<size_t>(
                     b * caption_tokens +
                     (k - latent_steps - text_tokens - speaker_tokens))] != 0;
        }
        out[static_cast<size_t>((b * latent_steps + q) * keys + k)] =
            keep ? 0.0F : -INFINITY;
      }
    }
  }
  return out;
}

std::vector<int32_t> positions(int64_t count) {
  std::vector<int32_t> out(static_cast<size_t>(count));
#ifdef _OPENMP
#pragma omp parallel for if(count >= 4096)
#endif
  for (int64_t i = 0; i < count; ++i) {
    out[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  }
  return out;
}

} // namespace

class IrodoriRfSampler::Impl {
  class ContextGraph;
  class ModulationGraph;
  class Graph;
  class MemSaverContextGraph;
  class MemSaverModulationGraph;
  class MemSaverGraph;

public:
  struct ContextCache {
    uint64_t id = 0;
    int64_t batch = 0;
    int64_t speaker_tokens = 0;
    int64_t caption_tokens = 0;
    std::vector<uint8_t> text_mask;
    std::vector<uint8_t> speaker_mask;
    std::vector<uint8_t> caption_mask;
    const std::vector<IrodoriLayerContextKV> *backend_kv = nullptr;
    const std::vector<IrodoriLayerContextKV> *first_branch_backend_kv = nullptr;
  };

  struct ModulationCache {
    uint64_t id = 0;
    int64_t steps = 0;
    const std::vector<IrodoriLayerAdaLNModulation> *backend_layers = nullptr;
  };

  Impl(std::shared_ptr<const IrodoriAssets> assets,
       core::ExecutionContext &execution_context, size_t graph_arena_bytes,
       size_t weight_context_bytes,
       assets::TensorStorageType weight_storage_type, bool mem_saver)
      : assets_(std::move(assets)),
        weights_(load_irodori_rf_dit_weights(
            *assets_, execution_context.backend(),
            execution_context.backend_type(), weight_context_bytes,
            weight_storage_type)),
        backend_(execution_context.backend()),
        backend_type_(execution_context.backend_type()),
        threads_(std::max(1, execution_context.config().threads)),
        graph_arena_bytes_(graph_arena_bytes), mem_saver_(mem_saver) {
    if (assets_ == nullptr) {
      throw std::runtime_error("Irodori-TTS RF graph runner requires assets");
    }
  }

  ContextCache build_context_cache(const std::vector<float> &text_state_cond,
                                   const std::vector<uint8_t> &text_mask_cond,
                                   const std::vector<float> &caption_state_cond,
                                   const IrodoriCaptionCondition &caption,
                                   const IrodoriSpeakerCondition &speaker,
                                   bool text_cfg_enabled,
                                   bool speaker_cfg_enabled,
                                   bool caption_cfg_enabled) {
    const auto &config = assets_->config;
    const int64_t batch = 1 + (text_cfg_enabled ? 1 : 0) +
                          (speaker_cfg_enabled ? 1 : 0) +
                          (caption_cfg_enabled ? 1 : 0);
    const int64_t caption_tokens =
        config.use_caption_condition ? static_cast<int64_t>(caption.mask.size())
                                     : 0;
    if (static_cast<int64_t>(text_state_cond.size()) !=
            config.max_text_len * config.text_dim ||
        static_cast<int64_t>(text_mask_cond.size()) != config.max_text_len ||
        static_cast<int64_t>(speaker.state.size()) !=
            speaker.tokens * config.speaker_dim ||
        static_cast<int64_t>(speaker.mask.size()) != speaker.tokens) {
      throw std::runtime_error("Irodori-TTS RF context input shape mismatch");
    }
    if (config.use_caption_condition && caption_tokens > 0 &&
        (static_cast<int64_t>(caption_state_cond.size()) !=
             caption_tokens * config.caption_dim_resolved() ||
         static_cast<int64_t>(caption.mask.size()) != caption_tokens)) {
      throw std::runtime_error(
          "Irodori-TTS RF caption context input shape mismatch");
    }

    ContextCache cache;
    cache.id = ++next_context_id_;
    cache.batch = batch;
    cache.speaker_tokens = speaker.tokens;
    cache.caption_tokens = caption_tokens;

    const int64_t text_elems = config.max_text_len * config.text_dim;
    std::vector<float> text_state(static_cast<size_t>(batch * text_elems),
                                  0.0F);
    cache.text_mask.assign(static_cast<size_t>(batch * config.max_text_len), 0);
    auto copy_text_branch = [&](int64_t branch, bool enabled) {
      if (enabled) {
        std::copy(text_state_cond.begin(), text_state_cond.end(),
                  text_state.begin() +
                      static_cast<std::ptrdiff_t>(branch * text_elems));
        std::copy(text_mask_cond.begin(), text_mask_cond.end(),
                  cache.text_mask.begin() + static_cast<std::ptrdiff_t>(
                                                branch * config.max_text_len));
      }
    };
    copy_text_branch(0, true);
    int64_t branch = 1;
    if (text_cfg_enabled) {
      copy_text_branch(branch, false);
      ++branch;
    }
    if (speaker_cfg_enabled) {
      copy_text_branch(branch, true);
      ++branch;
    }
    if (caption_cfg_enabled) {
      copy_text_branch(branch, true);
    }

    const int64_t speaker_elems = speaker.tokens * config.speaker_dim;
    std::vector<float> speaker_state(static_cast<size_t>(batch * speaker_elems),
                                     0.0F);
    cache.speaker_mask.assign(static_cast<size_t>(batch * speaker.tokens), 0);
    auto copy_speaker_branch = [&](int64_t target_branch, bool enabled) {
      if (enabled) {
        std::copy(speaker.state.begin(), speaker.state.end(),
                  speaker_state.begin() + static_cast<std::ptrdiff_t>(
                                              target_branch * speaker_elems));
        std::copy(
            speaker.mask.begin(), speaker.mask.end(),
            cache.speaker_mask.begin() +
                static_cast<std::ptrdiff_t>(target_branch * speaker.tokens));
      }
    };
    copy_speaker_branch(0, true);
    branch = 1;
    if (text_cfg_enabled) {
      copy_speaker_branch(branch, true);
      ++branch;
    }
    if (speaker_cfg_enabled) {
      copy_speaker_branch(branch, false);
      ++branch;
    }
    if (caption_cfg_enabled) {
      copy_speaker_branch(branch, true);
    }

    std::vector<float> caption_state;
    if (config.use_caption_condition && caption_tokens > 0) {
      const int64_t caption_elems =
          caption_tokens * config.caption_dim_resolved();
      caption_state.assign(static_cast<size_t>(batch * caption_elems), 0.0F);
      cache.caption_mask.assign(static_cast<size_t>(batch * caption_tokens), 0);
      auto copy_caption_branch = [&](int64_t target_branch, bool enabled) {
        if (enabled) {
          std::copy(caption_state_cond.begin(), caption_state_cond.end(),
                    caption_state.begin() + static_cast<std::ptrdiff_t>(
                                                target_branch * caption_elems));
          std::copy(
              caption.mask.begin(), caption.mask.end(),
              cache.caption_mask.begin() +
                  static_cast<std::ptrdiff_t>(target_branch * caption_tokens));
        }
      };
      copy_caption_branch(0, true);
      branch = 1;
      if (text_cfg_enabled) {
        copy_caption_branch(branch, true);
        ++branch;
      }
      if (speaker_cfg_enabled) {
        copy_caption_branch(branch, true);
        ++branch;
      }
      if (caption_cfg_enabled) {
        copy_caption_branch(branch, false);
      }
    }

    if (mem_saver_) {
      if (mem_saver_context_graph_ == nullptr ||
          mem_saver_context_graph_->speaker_tokens() != speaker.tokens ||
          mem_saver_context_graph_->caption_tokens() != caption_tokens ||
          mem_saver_context_graph_->batch() != batch) {
        ++context_graph_rebuilds_;
        mem_saver_context_graph_.reset();
        mem_saver_context_graph_ = std::make_unique<MemSaverContextGraph>(
            *this, speaker.tokens, caption_tokens, batch, graph_arena_bytes_);
      }
      mem_saver_context_graph_->run(text_state, speaker_state, caption_state,
                                    cache);
    } else {
      if (context_graph_ == nullptr ||
          context_graph_->speaker_tokens() != speaker.tokens ||
          context_graph_->caption_tokens() != caption_tokens ||
          context_graph_->batch() != batch) {
        ++context_graph_rebuilds_;
        context_graph_.reset();
        context_graph_ = std::make_unique<ContextGraph>(
            *this, speaker.tokens, caption_tokens, batch, graph_arena_bytes_);
      }
      context_graph_->run(text_state, speaker_state, caption_state, cache);
    }
    return cache;
  }

  ModulationCache build_modulation_cache(const std::vector<float> &timesteps) {
    const int64_t steps = static_cast<int64_t>(timesteps.size());
    if (steps <= 0) {
      throw std::runtime_error("Irodori-TTS RF modulation cache needs steps");
    }
    if (mem_saver_) {
      if (mem_saver_modulation_graph_ == nullptr ||
          mem_saver_modulation_graph_->steps() != steps) {
        mem_saver_modulation_graph_.reset();
        mem_saver_modulation_graph_ = std::make_unique<MemSaverModulationGraph>(
            *this, steps, graph_arena_bytes_);
      }
      auto cache = mem_saver_modulation_graph_->run(timesteps);
      cache.id = ++next_modulation_id_;
      return cache;
    }
    if (modulation_graph_ == nullptr || modulation_graph_->steps() != steps) {
      modulation_graph_.reset();
      modulation_graph_ =
          std::make_unique<ModulationGraph>(*this, steps, graph_arena_bytes_);
    }
    auto cache = modulation_graph_->run(timesteps);
    cache.id = ++next_modulation_id_;
    return cache;
  }

  void run_step(const std::vector<float> &x_t, int64_t step,
                const ModulationCache &modulation_cache,
                const ContextCache &context_cache, bool text_cfg_enabled,
                bool speaker_cfg_enabled, bool caption_cfg_enabled,
                float text_guidance_scale, float speaker_guidance_scale,
                float caption_guidance_scale, int64_t latent_steps,
                std::vector<float> &velocity) {
    const int64_t batch = 1 + (text_cfg_enabled ? 1 : 0) +
                          (speaker_cfg_enabled ? 1 : 0) +
                          (caption_cfg_enabled ? 1 : 0);
    const int64_t caption_tokens = context_cache.caption_tokens;
    if (mem_saver_) {
      std::unique_ptr<MemSaverGraph> &graph =
          batch == 1 ? mem_saver_cond_graph_ : mem_saver_cfg_graph_;
      std::unique_ptr<MemSaverGraph> &inactive_graph =
          batch == 1 ? mem_saver_cfg_graph_ : mem_saver_cond_graph_;
      inactive_graph.reset();
      if (graph == nullptr || graph->latent_steps() != latent_steps ||
          graph->speaker_tokens() != context_cache.speaker_tokens ||
          graph->caption_tokens() != caption_tokens ||
          graph->batch() != batch ||
          graph->modulation_steps() != modulation_cache.steps ||
          graph->text_cfg_enabled() != text_cfg_enabled ||
          graph->speaker_cfg_enabled() != speaker_cfg_enabled ||
          graph->caption_cfg_enabled() != caption_cfg_enabled) {
        ++step_graph_rebuilds_;
        graph.reset();
        graph = std::make_unique<MemSaverGraph>(
            *this, latent_steps, context_cache.speaker_tokens, caption_tokens,
            batch, modulation_cache.steps, text_cfg_enabled,
            speaker_cfg_enabled, caption_cfg_enabled, graph_arena_bytes_);
      }
      graph->set_context(context_cache);
      graph->run(x_t, step, modulation_cache, context_cache,
                 text_guidance_scale, speaker_guidance_scale,
                 caption_guidance_scale, velocity);
      return;
    }
    std::unique_ptr<Graph> &graph = batch == 1 ? cond_graph_ : cfg_graph_;
    std::unique_ptr<Graph> &inactive_graph = batch == 1 ? cfg_graph_ : cond_graph_;
    inactive_graph.reset();
    if (graph == nullptr || graph->latent_steps() != latent_steps ||
        graph->speaker_tokens() != context_cache.speaker_tokens ||
        graph->caption_tokens() != caption_tokens || graph->batch() != batch ||
        graph->modulation_steps() != modulation_cache.steps ||
        graph->text_cfg_enabled() != text_cfg_enabled ||
        graph->speaker_cfg_enabled() != speaker_cfg_enabled ||
        graph->caption_cfg_enabled() != caption_cfg_enabled) {
      ++step_graph_rebuilds_;
      graph.reset();
      graph = std::make_unique<Graph>(
          *this, latent_steps, context_cache.speaker_tokens, caption_tokens,
          batch, modulation_cache.steps, text_cfg_enabled, speaker_cfg_enabled,
          caption_cfg_enabled, graph_arena_bytes_);
    }
    graph->set_context(context_cache);
    graph->run(x_t, step, modulation_cache, context_cache, text_guidance_scale,
               speaker_guidance_scale, caption_guidance_scale, velocity);
  }

  void release_graphs() {
    context_graph_.reset();
    modulation_graph_.reset();
    cond_graph_.reset();
    cfg_graph_.reset();
    mem_saver_context_graph_.reset();
    mem_saver_modulation_graph_.reset();
    mem_saver_cond_graph_.reset();
    mem_saver_cfg_graph_.reset();
  }

  int64_t context_graph_rebuilds() const noexcept {
    return context_graph_rebuilds_;
  }

  int64_t step_graph_rebuilds() const noexcept { return step_graph_rebuilds_; }

private:
  class ContextGraph {
  public:
    ContextGraph(IrodoriRfSampler::Impl &owner, int64_t speaker_tokens,
                 int64_t caption_tokens, int64_t batch,
                 size_t graph_arena_bytes)
        : owner_(&owner), speaker_tokens_(speaker_tokens),
          caption_tokens_(caption_tokens), batch_(batch) {
      const auto &config = owner.assets_->config;
      ggml_init_params params{graph_arena_bytes, nullptr, true};
      ctx_.reset(ggml_init(params));
      if (ctx_ == nullptr) {
        throw std::runtime_error(
            "failed to initialize Irodori-TTS RF context graph context");
      }
      core::ModuleBuildContext build_ctx{
          ctx_.get(), "irodori_tts.rf_dit.context_cache", owner.backend_type_};
      text_state_ = core::make_tensor(
          build_ctx, GGML_TYPE_F32,
          core::TensorShape::from_dims(
              {batch_, config.max_text_len, config.text_dim}));
      speaker_state_ =
          core::make_tensor(build_ctx, GGML_TYPE_F32,
                            core::TensorShape::from_dims(
                                {batch_, speaker_tokens_, config.speaker_dim}));
      if (config.use_caption_condition && caption_tokens_ > 0) {
        caption_state_ = core::make_tensor(
            build_ctx, GGML_TYPE_F32,
            core::TensorShape::from_dims(
                {batch_, caption_tokens_, config.caption_dim_resolved()}));
      }
      ggml_set_input(text_state_.tensor);
      ggml_set_input(speaker_state_.tensor);
      if (config.use_caption_condition && caption_tokens_ > 0) {
        ggml_set_input(caption_state_.tensor);
      }
      outputs_ = build_irodori_context_kv_cache(build_ctx, text_state_,
                                                speaker_state_, caption_state_,
                                                owner.weights_, config);
      if (batch_ > 1) {
        first_branch_outputs_.reserve(outputs_.size());
        for (auto &layer : outputs_) {
          IrodoriLayerContextKV first;
          first.k_context =
              modules::SliceModule({0, 0, 1}).build(build_ctx, layer.k_context);
          first.v_context =
              modules::SliceModule({0, 0, 1}).build(build_ctx, layer.v_context);
          first_branch_outputs_.push_back(first);
        }
      }
      for (auto &layer : outputs_) {
        ggml_set_output(layer.k_context.tensor);
        ggml_set_output(layer.v_context.tensor);
      }
      graph_ = ggml_new_graph_custom(ctx_.get(), 131072, false);
      for (auto &layer : outputs_) {
        ggml_build_forward_expand(graph_, layer.k_context.tensor);
        ggml_build_forward_expand(graph_, layer.v_context.tensor);
      }
      buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), owner.backend_);
      if (buffer_ == nullptr) {
        throw std::runtime_error(
            "failed to allocate Irodori-TTS RF context graph");
      }
    }

    ~ContextGraph() {
      engine::core::release_backend_graph_resources(owner_->backend_, graph_);
      if (buffer_ != nullptr) {
        ggml_backend_buffer_free(buffer_);
      }
    }

    int64_t speaker_tokens() const noexcept { return speaker_tokens_; }

    int64_t caption_tokens() const noexcept { return caption_tokens_; }

    int64_t batch() const noexcept { return batch_; }

    void run(const std::vector<float> &text_state,
             const std::vector<float> &speaker_state,
             const std::vector<float> &caption_state,
             ContextCache &cache) const {
      const auto &config = owner_->assets_->config;
      if (static_cast<int64_t>(text_state.size()) !=
              batch_ * config.max_text_len * config.text_dim ||
          static_cast<int64_t>(speaker_state.size()) !=
              batch_ * speaker_tokens_ * config.speaker_dim) {
        throw std::runtime_error(
            "Irodori-TTS RF context graph input shape mismatch");
      }
      if (config.use_caption_condition && caption_tokens_ > 0 &&
          static_cast<int64_t>(caption_state.size()) !=
              batch_ * caption_tokens_ * config.caption_dim_resolved()) {
        throw std::runtime_error(
            "Irodori-TTS RF caption context graph input shape mismatch");
      }
      core::write_tensor_f32(text_state_, text_state);
      core::write_tensor_f32(speaker_state_, speaker_state);
      if (config.use_caption_condition && caption_tokens_ > 0) {
        core::write_tensor_f32(caption_state_, caption_state);
      }
      core::set_backend_threads(owner_->backend_, owner_->threads_);
      const ggml_status status =
          core::compute_backend_graph(owner_->backend_, graph_, nullptr,
                                      "irodori_tts.rf_dit.context_cache");
      ggml_backend_synchronize(owner_->backend_);
      if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Irodori-TTS RF context graph compute failed");
      }
      cache.backend_kv = &outputs_;
      cache.first_branch_backend_kv =
          batch_ > 1 ? &first_branch_outputs_ : &outputs_;
    }

  private:
    IrodoriRfSampler::Impl *owner_ = nullptr;
    int64_t speaker_tokens_ = 0;
    int64_t caption_tokens_ = 0;
    int64_t batch_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue text_state_;
    core::TensorValue speaker_state_;
    core::TensorValue caption_state_;
    std::vector<IrodoriLayerContextKV> outputs_;
    std::vector<IrodoriLayerContextKV> first_branch_outputs_;
    ggml_cgraph *graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
  };

  class ModulationGraph {
  public:
    ModulationGraph(IrodoriRfSampler::Impl &owner, int64_t steps,
                    size_t graph_arena_bytes)
        : owner_(&owner), steps_(steps) {
      const auto &config = owner.assets_->config;
      ggml_init_params params{graph_arena_bytes, nullptr, true};
      ctx_.reset(ggml_init(params));
      if (ctx_ == nullptr) {
        throw std::runtime_error(
            "failed to initialize Irodori-TTS RF modulation graph context");
      }
      core::ModuleBuildContext build_ctx{ctx_.get(),
                                         "irodori_tts.rf_dit.adaln_modulation",
                                         owner.backend_type_};
      timesteps_ = core::make_tensor(build_ctx, GGML_TYPE_F32,
                                     core::TensorShape::from_dims({steps_}));
      ggml_set_input(timesteps_.tensor);
      outputs_ = build_irodori_adaln_modulation_cache(build_ctx, timesteps_,
                                                      owner.weights_, config);
      for (auto &layer : outputs_) {
        ggml_set_output(layer.attention.shift.tensor);
        ggml_set_output(layer.attention.scale.tensor);
        ggml_set_output(layer.attention.gate.tensor);
        ggml_set_output(layer.mlp.shift.tensor);
        ggml_set_output(layer.mlp.scale.tensor);
        ggml_set_output(layer.mlp.gate.tensor);
      }
      graph_ = ggml_new_graph_custom(ctx_.get(), 131072, false);
      for (auto &layer : outputs_) {
        ggml_build_forward_expand(graph_, layer.attention.shift.tensor);
        ggml_build_forward_expand(graph_, layer.attention.scale.tensor);
        ggml_build_forward_expand(graph_, layer.attention.gate.tensor);
        ggml_build_forward_expand(graph_, layer.mlp.shift.tensor);
        ggml_build_forward_expand(graph_, layer.mlp.scale.tensor);
        ggml_build_forward_expand(graph_, layer.mlp.gate.tensor);
      }
      buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), owner.backend_);
      if (buffer_ == nullptr) {
        throw std::runtime_error(
            "failed to allocate Irodori-TTS RF modulation graph");
      }
    }

    ~ModulationGraph() {
      engine::core::release_backend_graph_resources(owner_->backend_, graph_);
      if (buffer_ != nullptr) {
        ggml_backend_buffer_free(buffer_);
      }
    }

    int64_t steps() const noexcept { return steps_; }

    ModulationCache run(const std::vector<float> &timesteps) const {
      if (static_cast<int64_t>(timesteps.size()) != steps_) {
        throw std::runtime_error(
            "Irodori-TTS RF modulation timestep count mismatch");
      }
      core::write_tensor_f32(timesteps_, timesteps);
      core::set_backend_threads(owner_->backend_, owner_->threads_);
      const ggml_status status =
          core::compute_backend_graph(owner_->backend_, graph_, nullptr,
                                      "irodori_tts.rf_dit.adaln_modulation");
      ggml_backend_synchronize(owner_->backend_);
      if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(
            "Irodori-TTS RF modulation graph compute failed");
      }
      ModulationCache cache;
      cache.steps = steps_;
      cache.backend_layers = &outputs_;
      return cache;
    }

  private:
    IrodoriRfSampler::Impl *owner_ = nullptr;
    int64_t steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue timesteps_;
    std::vector<IrodoriLayerAdaLNModulation> outputs_;
    ggml_cgraph *graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
  };

  class Graph {
  public:
    Graph(IrodoriRfSampler::Impl &owner, int64_t latent_steps,
          int64_t speaker_tokens, int64_t caption_tokens, int64_t batch,
          int64_t modulation_steps, bool text_cfg_enabled,
          bool speaker_cfg_enabled, bool caption_cfg_enabled,
          size_t graph_arena_bytes)
        : owner_(&owner), latent_steps_(latent_steps),
          speaker_tokens_(speaker_tokens), caption_tokens_(caption_tokens),
          batch_(batch), modulation_steps_(modulation_steps),
          text_cfg_enabled_(text_cfg_enabled),
          speaker_cfg_enabled_(speaker_cfg_enabled),
          caption_cfg_enabled_(caption_cfg_enabled) {
      const auto &config = owner.assets_->config;
      ggml_init_params params{graph_arena_bytes, nullptr, true};
      ctx_.reset(ggml_init(params));
      if (ctx_ == nullptr) {
        throw std::runtime_error(
            "failed to initialize Irodori-TTS RF graph context");
      }
      core::ModuleBuildContext build_ctx{ctx_.get(), "irodori_tts.rf_dit",
                                         owner.backend_type_};
      x_t_ = core::make_tensor(
          build_ctx, GGML_TYPE_F32,
          core::TensorShape::from_dims(
              {batch_, latent_steps_, config.patched_latent_dim()}));
      step_index_ = core::make_tensor(build_ctx, GGML_TYPE_I32,
                                      core::TensorShape::from_dims({1}));
      const int64_t dim = config.model_dim / config.num_heads;
      const int64_t context_tokens =
          config.max_text_len + speaker_tokens_ + caption_tokens_;
      context_kv_inputs_.reserve(owner.weights_.blocks.size());
      for (size_t layer = 0; layer < owner.weights_.blocks.size(); ++layer) {
        IrodoriLayerContextKV kv;
        kv.k_context = core::make_tensor(
            build_ctx, GGML_TYPE_F32,
            core::TensorShape::from_dims(
                {batch_, context_tokens, config.num_heads, dim}));
        kv.v_context = core::make_tensor(
            build_ctx, GGML_TYPE_F32,
            core::TensorShape::from_dims(
                {batch_, context_tokens, config.num_heads, dim}));
        ggml_set_input(kv.k_context.tensor);
        ggml_set_input(kv.v_context.tensor);
        context_kv_inputs_.push_back(kv);
      }
      modulation_cache_inputs_.reserve(owner.weights_.blocks.size());
      for (size_t layer = 0; layer < owner.weights_.blocks.size(); ++layer) {
        IrodoriLayerAdaLNModulation modulation;
        auto make_mod_tensor = [&]() {
          auto tensor =
              core::make_tensor(build_ctx, GGML_TYPE_F32,
                                core::TensorShape::from_dims(
                                    {modulation_steps_, 1, config.model_dim}));
          ggml_set_input(tensor.tensor);
          return tensor;
        };
        modulation.attention.shift = make_mod_tensor();
        modulation.attention.scale = make_mod_tensor();
        modulation.attention.gate = make_mod_tensor();
        modulation.mlp.shift = make_mod_tensor();
        modulation.mlp.scale = make_mod_tensor();
        modulation.mlp.gate = make_mod_tensor();
        modulation_cache_inputs_.push_back(modulation);
      }
      attention_mask_ =
          core::make_tensor(build_ctx, GGML_TYPE_F16,
                            core::TensorShape::from_dims(
                                {batch_, 1, latent_steps_,
                                 latent_steps_ + config.max_text_len +
                                     speaker_tokens_ + caption_tokens_}));
      positions_ =
          core::make_tensor(build_ctx, GGML_TYPE_I32,
                            core::TensorShape::from_dims({latent_steps_}));
      ggml_set_input(x_t_.tensor);
      ggml_set_input(step_index_.tensor);
      ggml_set_input(attention_mask_.tensor);
      ggml_set_input(positions_.tensor);
      if (batch_ > 1) {
        text_guidance_ = core::make_tensor(build_ctx, GGML_TYPE_F32,
                                           core::TensorShape::from_dims({1}));
        speaker_guidance_ = core::make_tensor(
            build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
        caption_guidance_ = core::make_tensor(
            build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
        ggml_set_input(text_guidance_.tensor);
        ggml_set_input(speaker_guidance_.tensor);
        ggml_set_input(caption_guidance_.tensor);
      }
      modulation_inputs_.reserve(modulation_cache_inputs_.size());
      for (const auto &layer : modulation_cache_inputs_) {
        modulation_inputs_.push_back(
            select_modulation_step(build_ctx, layer, step_index_, config));
      }
      auto output = build_irodori_rf_dit(
          build_ctx, x_t_, {}, {}, {}, {}, attention_mask_, positions_,
          owner.weights_, config, &context_kv_inputs_, &modulation_inputs_);
      if (batch_ > 1) {
        output = build_guided_velocity(build_ctx, output, text_guidance_,
                                       speaker_guidance_, caption_guidance_,
                                       text_cfg_enabled_, speaker_cfg_enabled_,
                                       caption_cfg_enabled_);
      }
      output_ = core::ensure_backend_addressable_layout(build_ctx, output);
      ggml_set_output(output_.tensor);
      graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
      ggml_build_forward_expand(graph_, output_.tensor);
      buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), owner.backend_);
      if (buffer_ == nullptr) {
        throw std::runtime_error("failed to allocate Irodori-TTS RF graph");
      }
      core::write_tensor_i32(positions_, positions(latent_steps_));
    }

    ~Graph() {
      engine::core::release_backend_graph_resources(owner_->backend_, graph_);
      if (buffer_ != nullptr) {
        ggml_backend_buffer_free(buffer_);
      }
    }

    int64_t latent_steps() const noexcept { return latent_steps_; }

    int64_t speaker_tokens() const noexcept { return speaker_tokens_; }

    int64_t caption_tokens() const noexcept { return caption_tokens_; }

    int64_t batch() const noexcept { return batch_; }

    int64_t modulation_steps() const noexcept { return modulation_steps_; }

    bool text_cfg_enabled() const noexcept { return text_cfg_enabled_; }

    bool speaker_cfg_enabled() const noexcept { return speaker_cfg_enabled_; }

    bool caption_cfg_enabled() const noexcept { return caption_cfg_enabled_; }

    void set_context(const ContextCache &cache) {
      if (loaded_context_id_ == cache.id) {
        return;
      }
      const auto &config = owner_->assets_->config;
      const bool use_first_branch =
          cache.batch != batch_ && batch_ == 1 && cache.batch > 1;
      if ((!use_first_branch && cache.batch != batch_) ||
          cache.speaker_tokens != speaker_tokens_ ||
          cache.caption_tokens != caption_tokens_) {
        throw std::runtime_error("Irodori-TTS RF context cache shape mismatch");
      }
      if (cache.backend_kv == nullptr ||
          cache.backend_kv->size() != context_kv_inputs_.size()) {
        throw std::runtime_error(
            "Irodori-TTS RF context cache layer count mismatch");
      }
      const auto *source_kv = cache.backend_kv;
      if (use_first_branch) {
        source_kv = cache.first_branch_backend_kv;
        if (source_kv == nullptr ||
            source_kv->size() != context_kv_inputs_.size()) {
          throw std::runtime_error(
              "Irodori-TTS RF first-branch context cache layer count mismatch");
        }
      }
      const int64_t context_tokens =
          config.max_text_len + speaker_tokens_ + caption_tokens_;
      const size_t context_values =
          static_cast<size_t>(cache.batch * context_tokens * config.num_heads *
                              (config.model_dim / config.num_heads));
      const size_t write_context_values =
          static_cast<size_t>(batch_ * context_tokens * config.num_heads *
                              (config.model_dim / config.num_heads));
      auto upload_context_tensor =
          [&](const core::TensorValue &source, const core::TensorValue &target,
              size_t source_values, size_t target_values) {
            if (source.shape.num_elements() !=
                    static_cast<int64_t>(source_values) ||
                target.shape.num_elements() !=
                    static_cast<int64_t>(target_values)) {
              throw std::runtime_error(
                  "Irodori-TTS RF context cache tensor shape mismatch");
            }
            ggml_backend_tensor_copy(source.tensor, target.tensor);
          };
      const size_t source_context_values =
          use_first_branch ? write_context_values : context_values;
      for (size_t layer = 0; layer < context_kv_inputs_.size(); ++layer) {
        const auto &source = (*source_kv)[layer];
        upload_context_tensor(source.k_context,
                              context_kv_inputs_[layer].k_context,
                              source_context_values, write_context_values);
        upload_context_tensor(source.v_context,
                              context_kv_inputs_[layer].v_context,
                              source_context_values, write_context_values);
      }
      const auto attention_mask_values = make_rf_attention_mask(
          cache.text_mask, cache.speaker_mask, cache.caption_mask, batch_,
          latent_steps_, config.max_text_len, speaker_tokens_, caption_tokens_);
      core::write_tensor_f16(attention_mask_, attention_mask_values);
      loaded_context_id_ = cache.id;
    }

    void set_modulation(const ModulationCache &cache) {
      if (loaded_modulation_id_ == cache.id) {
        return;
      }
      if (cache.steps != modulation_steps_ || cache.backend_layers == nullptr ||
          cache.backend_layers->size() != modulation_cache_inputs_.size()) {
        throw std::runtime_error(
            "Irodori-TTS RF modulation cache shape mismatch");
      }
      auto copy_modulation_tensor = [](const core::TensorValue &source,
                                       const core::TensorValue &target) {
        if (source.shape.num_elements() != target.shape.num_elements()) {
          throw std::runtime_error(
              "Irodori-TTS RF modulation cache tensor shape mismatch");
        }
        ggml_backend_tensor_copy(source.tensor, target.tensor);
      };
      for (size_t layer = 0; layer < modulation_cache_inputs_.size(); ++layer) {
        const auto &source = (*cache.backend_layers)[layer];
        auto &target = modulation_cache_inputs_[layer];
        copy_modulation_tensor(source.attention.shift, target.attention.shift);
        copy_modulation_tensor(source.attention.scale, target.attention.scale);
        copy_modulation_tensor(source.attention.gate, target.attention.gate);
        copy_modulation_tensor(source.mlp.shift, target.mlp.shift);
        copy_modulation_tensor(source.mlp.scale, target.mlp.scale);
        copy_modulation_tensor(source.mlp.gate, target.mlp.gate);
      }
      loaded_modulation_id_ = cache.id;
    }

    void run(const std::vector<float> &x_t, int64_t step,
             const ModulationCache &modulation_cache,
             const ContextCache &context_cache, float text_guidance_scale,
             float speaker_guidance_scale, float caption_guidance_scale,
             std::vector<float> &velocity) {
      const auto &config = owner_->assets_->config;
      const int64_t x_elems = latent_steps_ * config.patched_latent_dim();
      const bool use_first_branch = context_cache.batch != batch_ &&
                                    batch_ == 1 && context_cache.batch > 1;
      if (static_cast<int64_t>(x_t.size()) != x_elems ||
          (!use_first_branch && context_cache.batch != batch_) ||
          context_cache.speaker_tokens != speaker_tokens_ ||
          context_cache.caption_tokens != caption_tokens_ ||
          static_cast<int64_t>(context_cache.text_mask.size()) !=
              context_cache.batch * config.max_text_len ||
          static_cast<int64_t>(context_cache.speaker_mask.size()) !=
              context_cache.batch * speaker_tokens_) {
        throw std::runtime_error("Irodori-TTS RF step input shape mismatch");
      }
      if (config.use_caption_condition && caption_tokens_ > 0 &&
          static_cast<int64_t>(context_cache.caption_mask.size()) !=
              context_cache.batch * caption_tokens_) {
        throw std::runtime_error("Irodori-TTS RF caption input shape mismatch");
      }
      if (step < 0 || step >= modulation_cache.steps ||
          modulation_cache.steps != modulation_steps_) {
        throw std::runtime_error("Irodori-TTS RF modulation cache mismatch");
      }
      set_modulation(modulation_cache);
      const int32_t step_value = static_cast<int32_t>(step);
      core::write_tensor_i32(step_index_, &step_value, 1);
      if (batch_ > 1) {
        core::write_tensor_f32(text_guidance_, &text_guidance_scale, 1);
        core::write_tensor_f32(speaker_guidance_, &speaker_guidance_scale, 1);
        core::write_tensor_f32(caption_guidance_, &caption_guidance_scale, 1);
      }
      if (batch_ == 1) {
        core::write_tensor_f32(x_t_, x_t.data(), x_t.size());
      } else {
        x_batch_scratch_.resize(static_cast<size_t>(batch_ * x_elems));
        for (int64_t b = 0; b < batch_; ++b) {
          std::copy(x_t.begin(), x_t.end(),
                    x_batch_scratch_.begin() +
                        static_cast<std::ptrdiff_t>(b * x_elems));
        }
        core::write_tensor_f32(x_t_, x_batch_scratch_.data(),
                               x_batch_scratch_.size());
      }

      core::set_backend_threads(owner_->backend_, owner_->threads_);
      const ggml_status status = core::compute_backend_graph(
          owner_->backend_, graph_, nullptr, "irodori_tts.rf_dit");
      ggml_backend_synchronize(owner_->backend_);
      if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Irodori-TTS RF graph compute failed");
      }
      core::read_tensor_f32_into(output_.tensor, output_scratch_);
      velocity.resize(static_cast<size_t>(x_elems));
      std::copy(output_scratch_.begin(),
                output_scratch_.begin() + static_cast<std::ptrdiff_t>(x_elems),
                velocity.begin());
    }

  private:
    IrodoriRfSampler::Impl *owner_ = nullptr;
    int64_t latent_steps_ = 0;
    int64_t speaker_tokens_ = 0;
    int64_t caption_tokens_ = 0;
    int64_t batch_ = 0;
    int64_t modulation_steps_ = 0;
    bool text_cfg_enabled_ = false;
    bool speaker_cfg_enabled_ = false;
    bool caption_cfg_enabled_ = false;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue x_t_;
    std::vector<IrodoriLayerContextKV> context_kv_inputs_;
    std::vector<IrodoriLayerAdaLNModulation> modulation_cache_inputs_;
    std::vector<IrodoriLayerAdaLNModulation> modulation_inputs_;
    core::TensorValue attention_mask_;
    core::TensorValue positions_;
    core::TensorValue step_index_;
    core::TensorValue text_guidance_;
    core::TensorValue speaker_guidance_;
    core::TensorValue caption_guidance_;
    core::TensorValue output_;
    std::vector<float> x_batch_scratch_;
    std::vector<float> output_scratch_;
    uint64_t loaded_context_id_ = 0;
    uint64_t loaded_modulation_id_ = 0;
    ggml_cgraph *graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
  };

  class MemSaverContextGraph {
  public:
    MemSaverContextGraph(IrodoriRfSampler::Impl &owner,
                         int64_t speaker_tokens, int64_t caption_tokens,
                         int64_t batch, size_t graph_arena_bytes)
        : owner_(&owner), speaker_tokens_(speaker_tokens),
          caption_tokens_(caption_tokens), batch_(batch) {
      const auto &config = owner.assets_->config;
      ggml_init_params params{graph_arena_bytes, nullptr, true};
      ctx_.reset(ggml_init(params));
      if (ctx_ == nullptr) {
        throw std::runtime_error(
            "failed to initialize Irodori-TTS RF context graph context");
      }
      core::ModuleBuildContext build_ctx{
          ctx_.get(), "irodori_tts.rf_dit.context_cache", owner.backend_type_};
      text_state_ = core::make_tensor(
          build_ctx, GGML_TYPE_F32,
          core::TensorShape::from_dims(
              {batch_, config.max_text_len, config.text_dim}));
      speaker_state_ =
          core::make_tensor(build_ctx, GGML_TYPE_F32,
                            core::TensorShape::from_dims(
                                {batch_, speaker_tokens_, config.speaker_dim}));
      if (config.use_caption_condition && caption_tokens_ > 0) {
        caption_state_ = core::make_tensor(
            build_ctx, GGML_TYPE_F32,
            core::TensorShape::from_dims(
                {batch_, caption_tokens_, config.caption_dim_resolved()}));
      }
      ggml_set_input(text_state_.tensor);
      ggml_set_input(speaker_state_.tensor);
      if (config.use_caption_condition && caption_tokens_ > 0) {
        ggml_set_input(caption_state_.tensor);
      }
      outputs_ = build_irodori_context_kv_cache(build_ctx, text_state_,
                                                speaker_state_, caption_state_,
                                                owner.weights_, config);
      for (auto &layer : outputs_) {
        ggml_set_output(layer.k_context.tensor);
        ggml_set_output(layer.v_context.tensor);
      }
      graph_ = ggml_new_graph_custom(ctx_.get(), 131072, false);
      for (auto &layer : outputs_) {
        ggml_build_forward_expand(graph_, layer.k_context.tensor);
        ggml_build_forward_expand(graph_, layer.v_context.tensor);
      }
      gallocr_ =
          ggml_gallocr_new(ggml_backend_get_default_buffer_type(owner.backend_));
      if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) ||
          !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
        throw std::runtime_error(
            "failed to allocate Irodori-TTS RF context graph");
      }
    }

    ~MemSaverContextGraph() {
      engine::core::release_backend_graph_resources(owner_->backend_, graph_);
      if (gallocr_ != nullptr) {
        ggml_gallocr_free(gallocr_);
      }
    }

    int64_t speaker_tokens() const noexcept { return speaker_tokens_; }

    int64_t caption_tokens() const noexcept { return caption_tokens_; }

    int64_t batch() const noexcept { return batch_; }

    void run(const std::vector<float> &text_state,
             const std::vector<float> &speaker_state,
             const std::vector<float> &caption_state,
             ContextCache &cache) const {
      const auto &config = owner_->assets_->config;
      if (static_cast<int64_t>(text_state.size()) !=
              batch_ * config.max_text_len * config.text_dim ||
          static_cast<int64_t>(speaker_state.size()) !=
              batch_ * speaker_tokens_ * config.speaker_dim) {
        throw std::runtime_error(
            "Irodori-TTS RF context graph input shape mismatch");
      }
      if (config.use_caption_condition && caption_tokens_ > 0 &&
          static_cast<int64_t>(caption_state.size()) !=
              batch_ * caption_tokens_ * config.caption_dim_resolved()) {
        throw std::runtime_error(
            "Irodori-TTS RF caption context graph input shape mismatch");
      }
      core::write_tensor_f32(text_state_, text_state);
      core::write_tensor_f32(speaker_state_, speaker_state);
      if (config.use_caption_condition && caption_tokens_ > 0) {
        core::write_tensor_f32(caption_state_, caption_state);
      }
      core::set_backend_threads(owner_->backend_, owner_->threads_);
      const ggml_status status =
          core::compute_backend_graph(owner_->backend_, graph_, nullptr,
                                      "irodori_tts.rf_dit.context_cache");
      ggml_backend_synchronize(owner_->backend_);
      if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Irodori-TTS RF context graph compute failed");
      }
      cache.backend_kv = &outputs_;
      cache.first_branch_backend_kv = nullptr;
    }

  private:
    IrodoriRfSampler::Impl *owner_ = nullptr;
    int64_t speaker_tokens_ = 0;
    int64_t caption_tokens_ = 0;
    int64_t batch_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue text_state_;
    core::TensorValue speaker_state_;
    core::TensorValue caption_state_;
    std::vector<IrodoriLayerContextKV> outputs_;
    ggml_cgraph *graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
  };

  class MemSaverModulationGraph {
  public:
    MemSaverModulationGraph(IrodoriRfSampler::Impl &owner, int64_t steps,
                            size_t graph_arena_bytes)
        : owner_(&owner), steps_(steps) {
      const auto &config = owner.assets_->config;
      ggml_init_params params{graph_arena_bytes, nullptr, true};
      ctx_.reset(ggml_init(params));
      if (ctx_ == nullptr) {
        throw std::runtime_error(
            "failed to initialize Irodori-TTS RF modulation graph context");
      }
      core::ModuleBuildContext build_ctx{ctx_.get(),
                                         "irodori_tts.rf_dit.adaln_modulation",
                                         owner.backend_type_};
      timesteps_ = core::make_tensor(build_ctx, GGML_TYPE_F32,
                                     core::TensorShape::from_dims({steps_}));
      ggml_set_input(timesteps_.tensor);
      outputs_ = build_irodori_adaln_modulation_cache(build_ctx, timesteps_,
                                                      owner.weights_, config);
      for (auto &layer : outputs_) {
        ggml_set_output(layer.attention.shift.tensor);
        ggml_set_output(layer.attention.scale.tensor);
        ggml_set_output(layer.attention.gate.tensor);
        ggml_set_output(layer.mlp.shift.tensor);
        ggml_set_output(layer.mlp.scale.tensor);
        ggml_set_output(layer.mlp.gate.tensor);
      }
      graph_ = ggml_new_graph_custom(ctx_.get(), 131072, false);
      for (auto &layer : outputs_) {
        ggml_build_forward_expand(graph_, layer.attention.shift.tensor);
        ggml_build_forward_expand(graph_, layer.attention.scale.tensor);
        ggml_build_forward_expand(graph_, layer.attention.gate.tensor);
        ggml_build_forward_expand(graph_, layer.mlp.shift.tensor);
        ggml_build_forward_expand(graph_, layer.mlp.scale.tensor);
        ggml_build_forward_expand(graph_, layer.mlp.gate.tensor);
      }
      gallocr_ =
          ggml_gallocr_new(ggml_backend_get_default_buffer_type(owner.backend_));
      if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) ||
          !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
        throw std::runtime_error(
            "failed to allocate Irodori-TTS RF modulation graph");
      }
    }

    ~MemSaverModulationGraph() {
      engine::core::release_backend_graph_resources(owner_->backend_, graph_);
      if (gallocr_ != nullptr) {
        ggml_gallocr_free(gallocr_);
      }
    }

    int64_t steps() const noexcept { return steps_; }

    ModulationCache run(const std::vector<float> &timesteps) const {
      if (static_cast<int64_t>(timesteps.size()) != steps_) {
        throw std::runtime_error(
            "Irodori-TTS RF modulation timestep count mismatch");
      }
      core::write_tensor_f32(timesteps_, timesteps);
      core::set_backend_threads(owner_->backend_, owner_->threads_);
      const ggml_status status =
          core::compute_backend_graph(owner_->backend_, graph_, nullptr,
                                      "irodori_tts.rf_dit.adaln_modulation");
      ggml_backend_synchronize(owner_->backend_);
      if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(
            "Irodori-TTS RF modulation graph compute failed");
      }
      ModulationCache cache;
      cache.steps = steps_;
      cache.backend_layers = &outputs_;
      return cache;
    }

  private:
    IrodoriRfSampler::Impl *owner_ = nullptr;
    int64_t steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue timesteps_;
    std::vector<IrodoriLayerAdaLNModulation> outputs_;
    ggml_cgraph *graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
  };

  class MemSaverGraph {
  public:
    MemSaverGraph(IrodoriRfSampler::Impl &owner, int64_t latent_steps,
                  int64_t speaker_tokens, int64_t caption_tokens,
                  int64_t batch, int64_t modulation_steps,
                  bool text_cfg_enabled, bool speaker_cfg_enabled,
                  bool caption_cfg_enabled, size_t graph_arena_bytes)
        : owner_(&owner), latent_steps_(latent_steps),
          speaker_tokens_(speaker_tokens), caption_tokens_(caption_tokens),
          batch_(batch), modulation_steps_(modulation_steps),
          text_cfg_enabled_(text_cfg_enabled),
          speaker_cfg_enabled_(speaker_cfg_enabled),
          caption_cfg_enabled_(caption_cfg_enabled) {
      const auto &config = owner.assets_->config;
      ggml_init_params params{graph_arena_bytes, nullptr, true};
      ctx_.reset(ggml_init(params));
      if (ctx_ == nullptr) {
        throw std::runtime_error(
            "failed to initialize Irodori-TTS RF graph context");
      }
      ggml_init_params io_params{kRfDitIoContextBytes, nullptr, true};
      io_ctx_.reset(ggml_init(io_params));
      if (io_ctx_ == nullptr) {
        throw std::runtime_error(
            "failed to initialize Irodori-TTS RF graph IO context");
      }
      core::ModuleBuildContext build_ctx{ctx_.get(), "irodori_tts.rf_dit",
                                         owner.backend_type_};
      core::ModuleBuildContext io_build_ctx{io_ctx_.get(),
                                            "irodori_tts.rf_dit.io",
                                            owner.backend_type_};
      x_t_ = core::make_tensor(
          io_build_ctx, GGML_TYPE_F32,
          core::TensorShape::from_dims(
              {batch_, latent_steps_, config.patched_latent_dim()}));
      step_index_ = core::make_tensor(io_build_ctx, GGML_TYPE_I32,
                                      core::TensorShape::from_dims({1}));
      const int64_t dim = config.model_dim / config.num_heads;
      const int64_t context_tokens =
          config.max_text_len + speaker_tokens_ + caption_tokens_;
      context_kv_inputs_.reserve(owner.weights_.blocks.size());
      for (size_t layer = 0; layer < owner.weights_.blocks.size(); ++layer) {
        IrodoriLayerContextKV kv;
        kv.k_context = core::make_tensor(
            io_build_ctx, GGML_TYPE_F32,
            core::TensorShape::from_dims(
                {batch_, context_tokens, config.num_heads, dim}));
        kv.v_context = core::make_tensor(
            io_build_ctx, GGML_TYPE_F32,
            core::TensorShape::from_dims(
                {batch_, context_tokens, config.num_heads, dim}));
        ggml_set_input(kv.k_context.tensor);
        ggml_set_input(kv.v_context.tensor);
        context_kv_inputs_.push_back(kv);
      }
      modulation_cache_inputs_.reserve(owner.weights_.blocks.size());
      for (size_t layer = 0; layer < owner.weights_.blocks.size(); ++layer) {
        IrodoriLayerAdaLNModulation modulation;
        auto make_mod_tensor = [&]() {
          auto tensor =
              core::make_tensor(io_build_ctx, GGML_TYPE_F32,
                                core::TensorShape::from_dims(
                                    {modulation_steps_, 1, config.model_dim}));
          ggml_set_input(tensor.tensor);
          return tensor;
        };
        modulation.attention.shift = make_mod_tensor();
        modulation.attention.scale = make_mod_tensor();
        modulation.attention.gate = make_mod_tensor();
        modulation.mlp.shift = make_mod_tensor();
        modulation.mlp.scale = make_mod_tensor();
        modulation.mlp.gate = make_mod_tensor();
        modulation_cache_inputs_.push_back(modulation);
      }
      attention_mask_ =
          core::make_tensor(io_build_ctx, GGML_TYPE_F16,
                            core::TensorShape::from_dims(
                                {batch_, 1, latent_steps_,
                                 latent_steps_ + config.max_text_len +
                                     speaker_tokens_ + caption_tokens_}));
      positions_ =
          core::make_tensor(io_build_ctx, GGML_TYPE_I32,
                            core::TensorShape::from_dims({latent_steps_}));
      ggml_set_input(x_t_.tensor);
      ggml_set_input(step_index_.tensor);
      ggml_set_input(attention_mask_.tensor);
      ggml_set_input(positions_.tensor);
      if (batch_ > 1) {
        text_guidance_ = core::make_tensor(io_build_ctx, GGML_TYPE_F32,
                                           core::TensorShape::from_dims({1}));
        speaker_guidance_ = core::make_tensor(
            io_build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
        caption_guidance_ = core::make_tensor(
            io_build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
        ggml_set_input(text_guidance_.tensor);
        ggml_set_input(speaker_guidance_.tensor);
        ggml_set_input(caption_guidance_.tensor);
      }
      modulation_inputs_.reserve(modulation_cache_inputs_.size());
      for (const auto &layer : modulation_cache_inputs_) {
        modulation_inputs_.push_back(
            select_modulation_step(build_ctx, layer, step_index_, config));
      }
      auto output = build_irodori_rf_dit(
          build_ctx, x_t_, {}, {}, {}, {}, attention_mask_, positions_,
          owner.weights_, config, &context_kv_inputs_, &modulation_inputs_);
      if (batch_ > 1) {
        output = build_guided_velocity(build_ctx, output, text_guidance_,
                                       speaker_guidance_, caption_guidance_,
                                       text_cfg_enabled_, speaker_cfg_enabled_,
                                       caption_cfg_enabled_);
      }
      output_ = core::ensure_backend_addressable_layout(build_ctx, output);
      ggml_set_output(output_.tensor);
      graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
      ggml_build_forward_expand(graph_, output_.tensor);
      io_buffer_ = ggml_backend_alloc_ctx_tensors(io_ctx_.get(),
                                                  owner.backend_);
      if (io_buffer_ == nullptr) {
        throw std::runtime_error(
            "failed to allocate Irodori-TTS RF graph IO buffer");
      }
      gallocr_ =
          ggml_gallocr_new(ggml_backend_get_default_buffer_type(owner.backend_));
      if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) ||
          !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
        throw std::runtime_error("failed to allocate Irodori-TTS RF graph");
      }
      core::write_tensor_i32(positions_, positions(latent_steps_));
    }

    ~MemSaverGraph() {
      engine::core::release_backend_graph_resources(owner_->backend_, graph_);
      if (gallocr_ != nullptr) {
        ggml_gallocr_free(gallocr_);
      }
      if (io_buffer_ != nullptr) {
        ggml_backend_buffer_free(io_buffer_);
      }
    }

    int64_t latent_steps() const noexcept { return latent_steps_; }

    int64_t speaker_tokens() const noexcept { return speaker_tokens_; }

    int64_t caption_tokens() const noexcept { return caption_tokens_; }

    int64_t batch() const noexcept { return batch_; }

    int64_t modulation_steps() const noexcept { return modulation_steps_; }

    bool text_cfg_enabled() const noexcept { return text_cfg_enabled_; }

    bool speaker_cfg_enabled() const noexcept { return speaker_cfg_enabled_; }

    bool caption_cfg_enabled() const noexcept { return caption_cfg_enabled_; }

    void set_context(const ContextCache &cache) {
      if (loaded_context_id_ == cache.id) {
        return;
      }
      const auto &config = owner_->assets_->config;
      const bool use_first_branch =
          cache.batch != batch_ && batch_ == 1 && cache.batch > 1;
      if ((!use_first_branch && cache.batch != batch_) ||
          cache.speaker_tokens != speaker_tokens_ ||
          cache.caption_tokens != caption_tokens_) {
        throw std::runtime_error("Irodori-TTS RF context cache shape mismatch");
      }
      if (cache.backend_kv == nullptr ||
          cache.backend_kv->size() != context_kv_inputs_.size()) {
        throw std::runtime_error(
            "Irodori-TTS RF context cache layer count mismatch");
      }
      const auto *source_kv = cache.backend_kv;
      const int64_t context_tokens =
          config.max_text_len + speaker_tokens_ + caption_tokens_;
      const size_t context_values =
          static_cast<size_t>(cache.batch * context_tokens * config.num_heads *
                              (config.model_dim / config.num_heads));
      const size_t write_context_values =
          static_cast<size_t>(batch_ * context_tokens * config.num_heads *
                              (config.model_dim / config.num_heads));
      auto upload_context_tensor =
          [&](const core::TensorValue &source, const core::TensorValue &target,
              size_t source_values, size_t target_values) {
            if (source.shape.num_elements() !=
                    static_cast<int64_t>(source_values) ||
                target.shape.num_elements() !=
                    static_cast<int64_t>(target_values)) {
              throw std::runtime_error(
                  "Irodori-TTS RF context cache tensor shape mismatch");
            }
            if (use_first_branch) {
              context_first_branch_scratch_ =
                  core::read_tensor_f32(source.tensor);
              if (context_first_branch_scratch_.size() < target_values) {
                throw std::runtime_error(
                    "Irodori-TTS RF first-branch context cache is too small");
              }
              core::write_tensor_f32(target,
                                     context_first_branch_scratch_.data(),
                                     target_values);
              return;
            }
            ggml_backend_tensor_copy(source.tensor, target.tensor);
          };
      for (size_t layer = 0; layer < context_kv_inputs_.size(); ++layer) {
        const auto &source = (*source_kv)[layer];
        upload_context_tensor(source.k_context,
                              context_kv_inputs_[layer].k_context,
                              context_values, write_context_values);
        upload_context_tensor(source.v_context,
                              context_kv_inputs_[layer].v_context,
                              context_values, write_context_values);
      }
      const auto attention_mask_values = make_rf_attention_mask(
          cache.text_mask, cache.speaker_mask, cache.caption_mask, batch_,
          latent_steps_, config.max_text_len, speaker_tokens_, caption_tokens_);
      core::write_tensor_f16(attention_mask_, attention_mask_values);
      loaded_context_id_ = cache.id;
    }

    void set_modulation(const ModulationCache &cache) {
      if (loaded_modulation_id_ == cache.id) {
        return;
      }
      if (cache.steps != modulation_steps_ || cache.backend_layers == nullptr ||
          cache.backend_layers->size() != modulation_cache_inputs_.size()) {
        throw std::runtime_error(
            "Irodori-TTS RF modulation cache shape mismatch");
      }
      auto copy_modulation_tensor = [](const core::TensorValue &source,
                                       const core::TensorValue &target) {
        if (source.shape.num_elements() != target.shape.num_elements()) {
          throw std::runtime_error(
              "Irodori-TTS RF modulation cache tensor shape mismatch");
        }
        ggml_backend_tensor_copy(source.tensor, target.tensor);
      };
      for (size_t layer = 0; layer < modulation_cache_inputs_.size(); ++layer) {
        const auto &source = (*cache.backend_layers)[layer];
        auto &target = modulation_cache_inputs_[layer];
        copy_modulation_tensor(source.attention.shift, target.attention.shift);
        copy_modulation_tensor(source.attention.scale, target.attention.scale);
        copy_modulation_tensor(source.attention.gate, target.attention.gate);
        copy_modulation_tensor(source.mlp.shift, target.mlp.shift);
        copy_modulation_tensor(source.mlp.scale, target.mlp.scale);
        copy_modulation_tensor(source.mlp.gate, target.mlp.gate);
      }
      loaded_modulation_id_ = cache.id;
    }

    void run(const std::vector<float> &x_t, int64_t step,
             const ModulationCache &modulation_cache,
             const ContextCache &context_cache, float text_guidance_scale,
             float speaker_guidance_scale, float caption_guidance_scale,
             std::vector<float> &velocity) {
      const auto &config = owner_->assets_->config;
      const int64_t x_elems = latent_steps_ * config.patched_latent_dim();
      const bool use_first_branch = context_cache.batch != batch_ &&
                                    batch_ == 1 && context_cache.batch > 1;
      if (static_cast<int64_t>(x_t.size()) != x_elems ||
          (!use_first_branch && context_cache.batch != batch_) ||
          context_cache.speaker_tokens != speaker_tokens_ ||
          context_cache.caption_tokens != caption_tokens_ ||
          static_cast<int64_t>(context_cache.text_mask.size()) !=
              context_cache.batch * config.max_text_len ||
          static_cast<int64_t>(context_cache.speaker_mask.size()) !=
              context_cache.batch * speaker_tokens_) {
        throw std::runtime_error("Irodori-TTS RF step input shape mismatch");
      }
      if (config.use_caption_condition && caption_tokens_ > 0 &&
          static_cast<int64_t>(context_cache.caption_mask.size()) !=
              context_cache.batch * caption_tokens_) {
        throw std::runtime_error("Irodori-TTS RF caption input shape mismatch");
      }
      if (step < 0 || step >= modulation_cache.steps ||
          modulation_cache.steps != modulation_steps_) {
        throw std::runtime_error("Irodori-TTS RF modulation cache mismatch");
      }
      set_modulation(modulation_cache);
      const int32_t step_value = static_cast<int32_t>(step);
      core::write_tensor_i32(step_index_, &step_value, 1);
      if (batch_ > 1) {
        core::write_tensor_f32(text_guidance_, &text_guidance_scale, 1);
        core::write_tensor_f32(speaker_guidance_, &speaker_guidance_scale, 1);
        core::write_tensor_f32(caption_guidance_, &caption_guidance_scale, 1);
      }
      if (batch_ == 1) {
        core::write_tensor_f32(x_t_, x_t.data(), x_t.size());
      } else {
        x_batch_scratch_.resize(static_cast<size_t>(batch_ * x_elems));
        for (int64_t b = 0; b < batch_; ++b) {
          std::copy(x_t.begin(), x_t.end(),
                    x_batch_scratch_.begin() +
                        static_cast<std::ptrdiff_t>(b * x_elems));
        }
        core::write_tensor_f32(x_t_, x_batch_scratch_.data(),
                               x_batch_scratch_.size());
      }

      core::set_backend_threads(owner_->backend_, owner_->threads_);
      const ggml_status status = core::compute_backend_graph(
          owner_->backend_, graph_, nullptr, "irodori_tts.rf_dit");
      ggml_backend_synchronize(owner_->backend_);
      if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Irodori-TTS RF graph compute failed");
      }
      core::read_tensor_f32_into(output_.tensor, output_scratch_);
      velocity.resize(static_cast<size_t>(x_elems));
      std::copy(output_scratch_.begin(),
                output_scratch_.begin() + static_cast<std::ptrdiff_t>(x_elems),
                velocity.begin());
    }

  private:
    IrodoriRfSampler::Impl *owner_ = nullptr;
    int64_t latent_steps_ = 0;
    int64_t speaker_tokens_ = 0;
    int64_t caption_tokens_ = 0;
    int64_t batch_ = 0;
    int64_t modulation_steps_ = 0;
    bool text_cfg_enabled_ = false;
    bool speaker_cfg_enabled_ = false;
    bool caption_cfg_enabled_ = false;
    std::unique_ptr<ggml_context, GgmlContextDeleter> io_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue x_t_;
    std::vector<IrodoriLayerContextKV> context_kv_inputs_;
    std::vector<IrodoriLayerAdaLNModulation> modulation_cache_inputs_;
    std::vector<IrodoriLayerAdaLNModulation> modulation_inputs_;
    core::TensorValue attention_mask_;
    core::TensorValue positions_;
    core::TensorValue step_index_;
    core::TensorValue text_guidance_;
    core::TensorValue speaker_guidance_;
    core::TensorValue caption_guidance_;
    core::TensorValue output_;
    std::vector<float> x_batch_scratch_;
    std::vector<float> output_scratch_;
    std::vector<float> context_first_branch_scratch_;
    uint64_t loaded_context_id_ = 0;
    uint64_t loaded_modulation_id_ = 0;
    ggml_cgraph *graph_ = nullptr;
    ggml_backend_buffer_t io_buffer_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
  };

  std::shared_ptr<const IrodoriAssets> assets_;
  IrodoriRfDitWeights weights_;
  ggml_backend_t backend_ = nullptr;
  core::BackendType backend_type_ = core::BackendType::Cpu;
  int threads_ = 1;
  size_t graph_arena_bytes_ = 0;
  bool mem_saver_ = false;
  uint64_t next_context_id_ = 0;
  uint64_t next_modulation_id_ = 0;
  int64_t context_graph_rebuilds_ = 0;
  int64_t step_graph_rebuilds_ = 0;
  std::unique_ptr<ContextGraph> context_graph_;
  std::unique_ptr<ModulationGraph> modulation_graph_;
  std::unique_ptr<Graph> cond_graph_;
  std::unique_ptr<Graph> cfg_graph_;
  std::unique_ptr<MemSaverContextGraph> mem_saver_context_graph_;
  std::unique_ptr<MemSaverModulationGraph> mem_saver_modulation_graph_;
  std::unique_ptr<MemSaverGraph> mem_saver_cond_graph_;
  std::unique_ptr<MemSaverGraph> mem_saver_cfg_graph_;
};

class IrodoriRfSampler::ContextCache::State {
public:
  explicit State(IrodoriRfSampler::Impl::ContextCache cache)
      : cache(std::move(cache)) {}

  IrodoriRfSampler::Impl::ContextCache cache;
};

class IrodoriRfSampler::ModulationCache::State {
public:
  explicit State(IrodoriRfSampler::Impl::ModulationCache cache)
      : cache(std::move(cache)) {}

  IrodoriRfSampler::Impl::ModulationCache cache;
};

IrodoriRfSampler::ContextCache::ContextCache() = default;
IrodoriRfSampler::ContextCache::~ContextCache() = default;

IrodoriRfSampler::ModulationCache::ModulationCache() = default;
IrodoriRfSampler::ModulationCache::~ModulationCache() = default;

IrodoriRfSampler::IrodoriRfSampler(
    std::shared_ptr<const IrodoriAssets> assets,
    core::ExecutionContext &execution_context, size_t graph_arena_bytes,
    size_t weight_context_bytes, assets::TensorStorageType weight_storage_type,
    bool mem_saver)
    : impl_(std::make_unique<Impl>(std::move(assets), execution_context,
                                   graph_arena_bytes, weight_context_bytes,
                                   weight_storage_type, mem_saver)) {}

IrodoriRfSampler::~IrodoriRfSampler() = default;

IrodoriRfSampler::ContextCache IrodoriRfSampler::build_context_cache(
    const std::vector<float> &text_state_cond,
    const std::vector<uint8_t> &text_mask_cond,
    const std::vector<float> &caption_state_cond,
    const IrodoriCaptionCondition &caption,
    const IrodoriSpeakerCondition &speaker, bool text_cfg_enabled,
    bool speaker_cfg_enabled, bool caption_cfg_enabled) {
  IrodoriRfSampler::ContextCache out;
  out.state_ = std::make_shared<IrodoriRfSampler::ContextCache::State>(
      impl_->build_context_cache(
          text_state_cond, text_mask_cond, caption_state_cond, caption, speaker,
          text_cfg_enabled, speaker_cfg_enabled, caption_cfg_enabled));
  return out;
}

IrodoriRfSampler::ModulationCache
IrodoriRfSampler::build_modulation_cache(const std::vector<float> &timesteps) {
  IrodoriRfSampler::ModulationCache out;
  out.state_ = std::make_shared<IrodoriRfSampler::ModulationCache::State>(
      impl_->build_modulation_cache(timesteps));
  return out;
}

void IrodoriRfSampler::run_step(
    const std::vector<float> &x_t, int64_t step,
    const IrodoriRfSampler::ModulationCache &modulation_cache,
    const IrodoriRfSampler::ContextCache &context_cache, bool text_cfg_enabled,
    bool speaker_cfg_enabled, bool caption_cfg_enabled,
    float text_guidance_scale, float speaker_guidance_scale,
    float caption_guidance_scale, int64_t latent_steps,
    std::vector<float> &velocity) {
  if (modulation_cache.state_ == nullptr || context_cache.state_ == nullptr) {
    throw std::runtime_error("Irodori-TTS RF cache is not initialized");
  }
  impl_->run_step(x_t, step, modulation_cache.state_->cache,
                  context_cache.state_->cache, text_cfg_enabled,
                  speaker_cfg_enabled, caption_cfg_enabled, text_guidance_scale,
                  speaker_guidance_scale, caption_guidance_scale, latent_steps,
                  velocity);
}

void IrodoriRfSampler::release_graphs() { impl_->release_graphs(); }

int64_t IrodoriRfSampler::context_graph_rebuilds() const noexcept {
  return impl_->context_graph_rebuilds();
}

int64_t IrodoriRfSampler::step_graph_rebuilds() const noexcept {
  return impl_->step_graph_rebuilds();
}

} // namespace engine::models::irodori_tts
