#include "engine/models/irodori_tts/rf_dit.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_projection_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

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

assets::TensorStorageType resolve_derived_storage_type(
    const assets::TensorSource &source, const std::string &tensor_name,
    assets::TensorStorageType requested_type) {
  if (requested_type != assets::TensorStorageType::Native) {
    return requested_type;
  }
  return assets::tensor_storage_type_for_dtype(
      source.require_metadata(tensor_name).dtype);
}

modules::LinearWeights load_packed_qkvg(
    core::BackendWeightStore &store, const assets::TensorSource &source,
    const std::string &prefix, assets::TensorStorageType storage_type,
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

IrodoriDiffusionBlockWeights load_block(core::BackendWeightStore &store,
                                        const assets::TensorSource &source,
                                        const std::string &prefix,
                                        assets::TensorStorageType storage_type,
                                        const IrodoriModelConfig &config,
                                        core::BackendType backend_type) {
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
      ggml_rms_norm(ctx.ggml, core::ensure_backend_addressable_layout(ctx, input).tensor, eps),
      input.shape, GGML_TYPE_F32);
  auto weight_view = core::reshape_tensor(
      ctx, weight,
      core::TensorShape::from_dims(
          {1, 1, input.shape.dims[2], input.shape.dims[3]}));
  return core::wrap_tensor(
      ggml_mul(ctx.ggml, normalized.tensor,
               modules::RepeatModule({normalized.shape}).build(ctx, weight_view).tensor),
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
                                            const IrodoriModelConfig &config) {
  const int64_t batch = t.shape.dims[0];
  auto t_for_embedding =
      batch > 1 ? modules::SliceModule({0, 0, 1}).build(ctx, t) : t;
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

struct AdaLNOutput {
  core::TensorValue hidden;
  core::TensorValue gate;
};

AdaLNOutput low_rank_adaln(core::ModuleBuildContext &ctx,
                           const core::TensorValue &x,
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
  auto hidden =
      modules::RMSNormModule({config.model_dim, config.norm_eps, false, false})
          .build(ctx, x, {std::nullopt, std::nullopt});
  auto one_plus_scale = core::wrap_tensor(
      ggml_scale_bias(ctx.ggml, scale.tensor, 1.0F, 1.0F), scale.shape,
      GGML_TYPE_F32);
  auto scaled = core::wrap_tensor(
      ggml_mul(ctx.ggml, hidden.tensor, one_plus_scale.tensor), hidden.shape,
      GGML_TYPE_F32);
  hidden = core::wrap_tensor(ggml_add(ctx.ggml, scaled.tensor, shift.tensor),
                             hidden.shape, GGML_TYPE_F32);
  return {hidden, gate};
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
  out.k_text =
      modules::LinearModule(
          binding::linear_config(config.text_dim, config.model_dim, false))
          .build(ctx, text_state, weights.wk_text);
  out.v_text =
      modules::LinearModule(
          binding::linear_config(config.text_dim, config.model_dim, false))
          .build(ctx, text_state, weights.wv_text);
  out.k_text = core::ensure_backend_addressable_layout(
      ctx,
      head_rms_norm(ctx, reshape_heads(ctx, out.k_text, config.num_heads, dim),
                    weights.k_norm, config.norm_eps));
  out.v_text = core::ensure_backend_addressable_layout(
      ctx, reshape_heads(ctx, out.v_text, config.num_heads, dim));

  out.k_speaker =
      modules::LinearModule(
          binding::linear_config(config.speaker_dim, config.model_dim, false))
          .build(ctx, speaker_state, weights.wk_speaker);
  out.v_speaker =
      modules::LinearModule(
          binding::linear_config(config.speaker_dim, config.model_dim, false))
          .build(ctx, speaker_state, weights.wv_speaker);
  out.k_speaker = core::ensure_backend_addressable_layout(
      ctx, head_rms_norm(
               ctx, reshape_heads(ctx, out.k_speaker, config.num_heads, dim),
               weights.k_norm, config.norm_eps));
  out.v_speaker = core::ensure_backend_addressable_layout(
      ctx, reshape_heads(ctx, out.v_speaker, config.num_heads, dim));

  if (config.use_caption_condition && caption_state.tensor != nullptr &&
      caption_state.shape.dims[1] > 0) {
    out.k_caption = modules::LinearModule(
                        binding::linear_config(config.caption_dim_resolved(),
                                               config.model_dim, false))
                        .build(ctx, caption_state, weights.wk_caption);
    out.v_caption = modules::LinearModule(
                        binding::linear_config(config.caption_dim_resolved(),
                                               config.model_dim, false))
                        .build(ctx, caption_state, weights.wv_caption);
    out.k_caption = core::ensure_backend_addressable_layout(
        ctx, head_rms_norm(
                 ctx, reshape_heads(ctx, out.k_caption, config.num_heads, dim),
                 weights.k_norm, config.norm_eps));
    out.v_caption = core::ensure_backend_addressable_layout(
        ctx, reshape_heads(ctx, out.v_caption, config.num_heads, dim));
  }
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
    auto packed_projection =
        modules::FastPackedProjection4Module(
            {config.model_dim, 4 * config.model_dim})
            .build(ctx, x, weights.qkvg);
    q = modules::SliceModule({2, 0, config.model_dim})
            .build(ctx, packed_projection);
    k_self = modules::SliceModule({2, config.model_dim, config.model_dim})
                 .build(ctx, packed_projection);
    v_self =
        modules::SliceModule({2, 2 * config.model_dim, config.model_dim})
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
    q = modules::LinearModule(binding::linear_config(
                                  config.model_dim, config.model_dim, false))
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
        ctx, modules::LinearModule(binding::linear_config(
                                       config.model_dim, config.model_dim,
                                       false))
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
  auto k = modules::ConcatModule({1}).build(ctx, k_self, context_kv->k_text);
  auto v = modules::ConcatModule({1}).build(ctx, v_self, context_kv->v_text);
  k = modules::ConcatModule({1}).build(ctx, k, context_kv->k_speaker);
  v = modules::ConcatModule({1}).build(ctx, v, context_kv->v_speaker);
  if (config.use_caption_condition && context_kv->k_caption.tensor != nullptr &&
      context_kv->k_caption.shape.dims[1] > 0) {
    k = modules::ConcatModule({1}).build(ctx, k, context_kv->k_caption);
    v = modules::ConcatModule({1}).build(ctx, v, context_kv->v_caption);
  }
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
    const IrodoriLayerContextKV *context_kv) {
  auto attn_adaln =
      low_rank_adaln(ctx, x, cond_embed, weights.attention_adaln, config);
  auto attn =
      build_joint_attention(ctx, attn_adaln.hidden, text_state, speaker_state,
                            caption_state, attention_mask, positions,
                            weights.attention, config, layer_index, context_kv);
  auto gated_attn = core::wrap_tensor(
      ggml_mul(ctx.ggml, attn.tensor, attn_adaln.gate.tensor), attn.shape,
      GGML_TYPE_F32);
  auto hidden =
      core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, gated_attn.tensor),
                        x.shape, GGML_TYPE_F32);
  auto mlp_adaln =
      low_rank_adaln(ctx, hidden, cond_embed, weights.mlp_adaln, config);
  auto mlp = build_swiglu(ctx, mlp_adaln.hidden, weights.mlp_w1, weights.mlp_w2,
                          weights.mlp_w3, config);
  auto gated_mlp = core::wrap_tensor(
      ggml_mul(ctx.ggml, mlp.tensor, mlp_adaln.gate.tensor), mlp.shape,
      GGML_TYPE_F32);
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
    weights.blocks.push_back(load_block(*weights.store, source,
                                        "blocks." + std::to_string(layer),
                                        weight_storage_type, config,
                                        backend_type));
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

core::TensorValue build_irodori_rf_dit(
    core::ModuleBuildContext &ctx, const core::TensorValue &x_t,
    const core::TensorValue &t, const core::TensorValue &text_state,
    const core::TensorValue &speaker_state,
    const core::TensorValue &caption_state,
    const core::TensorValue &attention_mask, const core::TensorValue &positions,
    const IrodoriRfDitWeights &weights, const IrodoriModelConfig &config,
    const std::vector<IrodoriLayerContextKV> *context_kv_cache) {
  if (context_kv_cache != nullptr &&
      context_kv_cache->size() != weights.blocks.size()) {
    throw std::runtime_error(
        "Irodori-TTS RF context KV cache layer count mismatch");
  }
  auto cond_embed = build_condition_embedding(ctx, t, weights, config);
  auto hidden =
      modules::LinearModule(binding::linear_config(config.patched_latent_dim(),
                                                   config.model_dim, true))
          .build(ctx, x_t, weights.in_proj);
  for (size_t layer = 0; layer < weights.blocks.size(); ++layer) {
    const IrodoriLayerContextKV *context_kv =
        context_kv_cache == nullptr ? nullptr : &(*context_kv_cache)[layer];
    hidden = build_block(ctx, hidden, cond_embed, text_state, speaker_state,
                         caption_state, attention_mask, positions,
                         weights.blocks[layer], config,
                         static_cast<int64_t>(layer), context_kv);
  }
  hidden =
      modules::RMSNormModule({config.model_dim, config.norm_eps, true, false})
          .build(ctx, hidden, {weights.out_norm, std::nullopt});
  return modules::LinearModule(
             binding::linear_config(config.model_dim,
                                    config.patched_latent_dim(), true))
      .build(ctx, hidden, weights.out_proj);
}

} // namespace engine::models::irodori_tts
