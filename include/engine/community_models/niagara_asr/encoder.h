#pragma once

#include "engine/community_models/niagara_asr/assets.h"
#include "engine/community_models/niagara_asr/weights.h"
#include "engine/framework/core/module.h"

namespace engine::community_models::niagara_asr {

core::TensorValue build_niagara_l1_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const engine::modules::NormWeights & weights,
    const NiagaraEncoderConfig & config);

core::TensorValue build_niagara_feed_forward(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const NiagaraFeedForwardWeights & weights,
    const NiagaraEncoderConfig & config);

core::TensorValue build_niagara_subsampling(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const NiagaraSubsamplingWeights & weights,
    const NiagaraEncoderConfig & config);

core::TensorValue build_niagara_self_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const NiagaraSelfAttentionWeights & weights,
    const NiagaraEncoderConfig & config);

core::TensorValue build_niagara_state_space(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const NiagaraStateSpaceWeights & weights,
    const NiagaraEncoderConfig & config);

core::TensorValue build_niagara_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const NiagaraLayerWeights & weights,
    const NiagaraEncoderConfig & config);

core::TensorValue build_niagara_encoder_logits(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const NiagaraWeights & weights,
    const NiagaraAsrConfig & config);

}  // namespace engine::community_models::niagara_asr
