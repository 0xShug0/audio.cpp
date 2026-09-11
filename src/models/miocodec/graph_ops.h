#pragma once

#include "engine/framework/core/module.h"
#include "engine/models/miocodec/weights.h"
#include "engine/framework/core/constant_tensor_cache.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace engine::models::miocodec::graphs {

struct TimeMaskConstants {
    core::TensorValue mask;
    core::TensorValue inv_valid_group_count;
};

struct InterpolationConstants {
    core::TensorValue left_indices;
    core::TensorValue right_indices;
    core::TensorValue right_weight;
};

core::TensorValue mask_bct(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & time_mask);

std::vector<float> local_attention_mask(int64_t heads, int64_t frames, int64_t window_size, int64_t valid_frames);

core::TensorValue make_i32_constant(
    core::ConstantTensorCache & constants,
    const core::TensorShape & shape,
    const std::vector<int32_t> & values);

core::TensorValue make_ones_i32(
    core::ConstantTensorCache & constants,
    const core::TensorShape & shape);

TimeMaskConstants make_full_time_mask(
    core::ConstantTensorCache & constants,
    int64_t frames,
    int64_t channels_per_group);

InterpolationConstants make_interpolation_constants(
    core::ConstantTensorCache & constants,
    int64_t content_frames,
    int64_t stft_frames,
    const MioCodecConvTranspose1dWeights & upsample);

core::TensorValue transformer(
    core::ModuleBuildContext & ctx,
    core::ConstantTensorCache & constants,
    const core::TensorValue & input,
    const MioCodecTransformerWeights & weights,
    const std::optional<core::TensorValue> & condition = std::nullopt,
    const std::optional<core::TensorValue> & attention_mask = std::nullopt,
    bool exact_rope = false);

core::TensorValue fsq_quantized(
    core::ModuleBuildContext & ctx,
    core::ConstantTensorCache & constants,
    const core::TensorValue & input,
    const MioCodecQuantizerWeights & weights);

core::TensorValue global_encoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_btc,
    const MioCodecGlobalEncoderWeights & weights);

core::TensorValue dynamic_linear_interpolate_bct(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bct,
    const core::TensorValue & left_indices,
    const core::TensorValue & right_indices,
    const core::TensorValue & right_weight);

core::TensorValue resnet_stack(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bct,
    const MioCodecResNetStackWeights & weights,
    int64_t groups,
    const std::optional<core::TensorValue> & time_mask = std::nullopt,
    const std::optional<core::TensorValue> & inv_valid_group_values = std::nullopt);

core::TensorValue upsampler(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bct,
    const MioCodecUpsamplerWeights & weights,
    int64_t groups,
    const std::vector<core::TensorValue> * stage_time_masks = nullptr,
    const std::vector<core::TensorValue> * stage_inv_valid_group_values = nullptr);

}  // namespace engine::models::miocodec::graphs
