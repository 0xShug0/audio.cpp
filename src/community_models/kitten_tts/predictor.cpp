#include "engine/community_models/kitten_tts/predictor.h"

#include "engine/community_models/kitten_tts/assets.h"
#include "engine/community_models/kitten_tts/plbert.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/recurrent_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::kitten_tts {

namespace core = engine::core;
namespace modules = engine::modules;

namespace {

using engine::debug::measure_ms;

int64_t checked_capacity(int64_t value, int64_t max_value) {
    if (value <= 0 || max_value <= 0) {
        throw std::runtime_error("kitten capacity must be positive");
    }
    if (value > max_value) {
        throw std::runtime_error("kitten requested capacity exceeds maximum");
    }
    return value;
}

struct TimeMaskInputs {
    ggml_tensor *keep = nullptr;
    ggml_tensor *norm = nullptr;
    int64_t frame_capacity = 0;
};

const TimeMaskInputs &add_time_mask_inputs(ggml_context *ctx, std::vector<TimeMaskInputs> &masks,
                                           int64_t frame_capacity) {
    if (frame_capacity <= 0) {
        throw std::runtime_error("kitten predictor mask capacity must be positive");
    }
    TimeMaskInputs mask = {};
    mask.keep = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, frame_capacity, 1);
    mask.norm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, frame_capacity, 1);
    mask.frame_capacity = frame_capacity;
    ggml_set_input(mask.keep);
    ggml_set_input(mask.norm);
    masks.push_back(mask);
    return masks.back();
}

const TimeMaskInputs &add_keep_mask_input(ggml_context *ctx, std::vector<TimeMaskInputs> &masks,
                                          int64_t frame_capacity) {
    if (frame_capacity <= 0) {
        throw std::runtime_error("kitten predictor keep-mask capacity must be positive");
    }
    TimeMaskInputs mask = {};
    mask.keep = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, frame_capacity, 1);
    mask.frame_capacity = frame_capacity;
    ggml_set_input(mask.keep);
    masks.push_back(mask);
    return masks.back();
}

ggml_tensor *build_masked_adain_ct(core::ModuleBuildContext &build_ctx, ggml_tensor *x, const core::TensorValue &gamma,
                                   const core::TensorValue &beta, int64_t channels, float eps,
                                   std::vector<TimeMaskInputs> &masks) {
    const TimeMaskInputs &mask = add_time_mask_inputs(build_ctx.ggml, masks, x->ne[0]);
    const auto x_value = core::wrap_tensor(x, core::TensorShape::from_dims({channels, x->ne[0]}), GGML_TYPE_F32);
    const auto mask_shape = core::TensorShape::from_dims({1, x->ne[0]});
    const auto keep = modules::RepeatModule({x_value.shape})
                          .build(build_ctx, core::wrap_tensor(mask.keep, mask_shape, GGML_TYPE_F32));
    const auto norm = modules::RepeatModule({x_value.shape})
                          .build(build_ctx, core::wrap_tensor(mask.norm, mask_shape, GGML_TYPE_F32));
    const auto masked = modules::MulModule{}.build(build_ctx, x_value, norm);
    const auto mean = modules::ReduceMeanModule({1}).build(build_ctx, masked);
    const auto mean_rep = modules::RepeatModule({x_value.shape}).build(build_ctx, mean);
    const auto centered =
        core::wrap_tensor(ggml_sub(build_ctx.ggml, x_value.tensor, mean_rep.tensor), x_value.shape, GGML_TYPE_F32);
    const auto centered_for_variance = modules::MulModule{}.build(build_ctx, centered, norm);
    const auto squared = modules::MulModule{}.build(build_ctx, centered, centered_for_variance);
    const auto variance = modules::ReduceMeanModule({1}).build(build_ctx, squared);
    const auto stddev = modules::SqrtModule{}.build(
        build_ctx,
        core::wrap_tensor(ggml_scale_bias(build_ctx.ggml, variance.tensor, 1.0f, eps), variance.shape, GGML_TYPE_F32));
    const auto stddev_rep = modules::RepeatModule({x_value.shape}).build(build_ctx, stddev);
    auto normalized =
        core::wrap_tensor(ggml_div(build_ctx.ggml, centered.tensor, stddev_rep.tensor), x_value.shape, GGML_TYPE_F32);
    normalized = modules::MulModule{}.build(build_ctx, normalized, keep);
    core::validate_shape(gamma, core::TensorShape::from_dims({channels}), "kitten predictor adain gamma");
    core::validate_shape(beta, core::TensorShape::from_dims({channels}), "kitten predictor adain beta");
    const auto gamma_bc = core::reshape_tensor(build_ctx, gamma, core::TensorShape::from_dims({channels, 1}));
    const auto beta_bc = core::reshape_tensor(build_ctx, beta, core::TensorShape::from_dims({channels, 1}));
    const auto scaled = modules::MulModule{}.build(build_ctx, normalized, gamma_bc);
    const auto out = modules::AddModule{}.build(build_ctx, scaled, beta_bc);
    return modules::MulModule{}.build(build_ctx, out, keep).tensor;
}

void upload_time_masks(const std::vector<TimeMaskInputs> &masks, int64_t valid_base_frames,
                       int64_t base_frame_capacity) {
    if (valid_base_frames <= 0 || valid_base_frames > base_frame_capacity) {
        throw std::runtime_error("kitten predictor valid frame count exceeds prepared capacity");
    }
    for (const TimeMaskInputs &mask : masks) {
        if (mask.frame_capacity <= 0 || (valid_base_frames * mask.frame_capacity) % base_frame_capacity != 0) {
            throw std::runtime_error("kitten predictor mask capacity is not aligned with graph capacity");
        }
        const int64_t valid_frames = (valid_base_frames * mask.frame_capacity) / base_frame_capacity;
        if (valid_frames <= 0 || valid_frames > mask.frame_capacity) {
            throw std::runtime_error("kitten predictor mask valid frame count is invalid");
        }
        std::vector<float> keep(static_cast<size_t>(mask.frame_capacity), 0.0f);
        std::vector<float> norm(static_cast<size_t>(mask.frame_capacity), 0.0f);
        std::fill(keep.begin(), keep.begin() + valid_frames, 1.0f);
        const float norm_value = static_cast<float>(mask.frame_capacity) / static_cast<float>(valid_frames);
        std::fill(norm.begin(), norm.begin() + valid_frames, norm_value);
        core::write_tensor_f32(
            core::wrap_tensor(mask.keep, core::TensorShape::from_dims({1, mask.frame_capacity}), GGML_TYPE_F32), keep);
        if (mask.norm != nullptr) {
            core::write_tensor_f32(
                core::wrap_tensor(mask.norm, core::TensorShape::from_dims({1, mask.frame_capacity}), GGML_TYPE_F32),
                norm);
        }
    }
}

std::vector<float> expand_tc_by_durations(const std::vector<float> &values, int64_t rows, int64_t cols,
                                          const std::vector<int32_t> &durations, int64_t total_rows) {
    if (rows != static_cast<int64_t>(durations.size())) {
        throw std::runtime_error("kitten predictor tc expansion shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(total_rows * cols), 0.0f);
    int64_t dst_row = 0;
    for (int64_t src_row = 0; src_row < rows; ++src_row) {
        const int32_t repeat = durations[static_cast<size_t>(src_row)];
        const float *src = values.data() + static_cast<ptrdiff_t>(src_row * cols);
        for (int32_t i = 0; i < repeat; ++i) {
            std::memcpy(out.data() + static_cast<ptrdiff_t>(dst_row * cols), src,
                        static_cast<size_t>(cols) * sizeof(float));
            ++dst_row;
        }
    }
    if (dst_row != total_rows) {
        throw std::runtime_error("kitten predictor tc expansion produced unexpected frame count");
    }
    return out;
}

std::vector<float> expand_ct_by_durations(const std::vector<float> &values, int64_t rows, int64_t cols,
                                          const std::vector<int32_t> &durations, int64_t total_cols) {
    if (cols != static_cast<int64_t>(durations.size())) {
        throw std::runtime_error("kitten predictor ct expansion shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(rows * total_cols), 0.0f);
    for (int64_t row = 0; row < rows; ++row) {
        const float *src_row = values.data() + static_cast<ptrdiff_t>(row * cols);
        float *dst_row = out.data() + static_cast<ptrdiff_t>(row * total_cols);
        int64_t dst_col = 0;
        for (int64_t src_col = 0; src_col < cols; ++src_col) {
            const float value = src_row[src_col];
            const int32_t repeat = durations[static_cast<size_t>(src_col)];
            for (int32_t i = 0; i < repeat; ++i) {
                dst_row[dst_col++] = value;
            }
        }
        if (dst_col != total_cols) {
            throw std::runtime_error("kitten predictor ct expansion produced unexpected frame count");
        }
    }
    return out;
}

core::TensorValue make_zero_lstm_state(core::ModuleBuildContext &ctx, int64_t hidden_size,
                                       std::vector<ggml_tensor *> &zero_state_inputs) {
    auto tensor = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, hidden_size}));
    ggml_set_input(tensor.tensor);
    zero_state_inputs.push_back(tensor.tensor);
    return tensor;
}

void upload_zero_state_inputs(const std::vector<ggml_tensor *> &zero_state_inputs) {
    for (ggml_tensor *tensor : zero_state_inputs) {
        std::vector<float> zeros(static_cast<size_t>(ggml_nelements(tensor)), 0.0f);
        core::write_tensor_f32(
            core::wrap_tensor(tensor, core::TensorShape::from_dims({ggml_nelements(tensor)}), GGML_TYPE_F32), zeros);
    }
}

modules::LSTMSequenceOutputs
build_lstm_sequence_combined_input_bias(core::ModuleBuildContext &ctx, const core::TensorValue &input,
                                        const KittenWeights::LstmWeights &lstm, bool reverse,
                                        const core::TensorValue &initial_hidden, const core::TensorValue &initial_cell,
                                        const core::TensorValue *valid_mask, bool preserve_inactive_state) {
    core::validate_shape(input, core::TensorShape::from_dims({input.shape.dims[0], lstm.input_size}), "input");
    if (valid_mask != nullptr) {
        core::validate_shape(*valid_mask, core::TensorShape::from_dims({input.shape.dims[0], 1}), "valid_mask");
    }
    const int64_t frames = input.shape.dims[0];
    const int64_t gates = 4 * lstm.hidden_size;
    const auto &weight_ih = reverse ? lstm.weight_ih_l0_reverse : lstm.weight_ih_l0;
    const auto &weight_hh = reverse ? lstm.weight_hh_l0_reverse : lstm.weight_hh_l0;
    const auto &combined_bias = reverse ? lstm.combined_bias_l0_reverse : lstm.combined_bias_l0;
    const auto projected_inputs = modules::LinearModule({lstm.input_size, gates, true})
                                      .build(ctx, input,
                                             {
                                                 weight_ih,
                                                 combined_bias,
                                             });

    std::vector<core::TensorValue> steps(static_cast<size_t>(frames));
    auto hidden = initial_hidden;
    auto cell = initial_cell;
    const modules::ConcatModule concat_rows({0});

    for (int64_t step = 0; step < frames; ++step) {
        const int64_t t = reverse ? (frames - 1 - step) : step;
        const auto previous_hidden = hidden;
        const auto previous_cell = cell;
        const auto projected_x_t = modules::SliceModule({0, t, 1}).build(ctx, projected_inputs);
        const auto projected_hidden = modules::LinearModule({lstm.hidden_size, gates, false})
                                          .build(ctx, hidden,
                                                 {
                                                     weight_hh,
                                                     core::TensorValue{},
                                                 });
        const auto gate_values = modules::AddModule{}.build(ctx, projected_x_t, projected_hidden);

        const auto input_gate =
            modules::SigmoidModule().build(ctx, modules::SliceModule({1, 0, lstm.hidden_size}).build(ctx, gate_values));
        const auto forget_gate = modules::SigmoidModule().build(
            ctx, modules::SliceModule({1, lstm.hidden_size, lstm.hidden_size}).build(ctx, gate_values));
        const auto candidate = modules::TanhModule().build(
            ctx, modules::SliceModule({1, 2 * lstm.hidden_size, lstm.hidden_size}).build(ctx, gate_values));
        const auto output_gate = modules::SigmoidModule().build(
            ctx, modules::SliceModule({1, 3 * lstm.hidden_size, lstm.hidden_size}).build(ctx, gate_values));
        const auto kept_cell = modules::MulModule{}.build(ctx, forget_gate, cell);
        const auto written_cell = modules::MulModule{}.build(ctx, input_gate, candidate);
        cell = modules::AddModule{}.build(ctx, kept_cell, written_cell);
        const auto activated_cell = modules::TanhModule().build(ctx, cell);
        hidden = modules::MulModule{}.build(ctx, output_gate, activated_cell);
        if (valid_mask != nullptr) {
            const auto active = modules::SliceModule({0, t, 1}).build(ctx, *valid_mask);
            const auto active_hidden = modules::RepeatModule({hidden.shape}).build(ctx, active);
            if (preserve_inactive_state) {
                const auto inactive_hidden = core::wrap_tensor(
                    ggml_scale_bias(ctx.ggml, active_hidden.tensor, -1.0f, 1.0f), hidden.shape, GGML_TYPE_F32);
                const auto kept_hidden = modules::MulModule{}.build(ctx, previous_hidden, inactive_hidden);
                const auto written_hidden = modules::MulModule{}.build(ctx, hidden, active_hidden);
                hidden = modules::AddModule{}.build(ctx, written_hidden, kept_hidden);

                const auto active_cell = modules::RepeatModule({cell.shape}).build(ctx, active);
                const auto inactive_cell = core::wrap_tensor(ggml_scale_bias(ctx.ggml, active_cell.tensor, -1.0f, 1.0f),
                                                             cell.shape, GGML_TYPE_F32);
                const auto kept_cell = modules::MulModule{}.build(ctx, previous_cell, inactive_cell);
                const auto written_cell = modules::MulModule{}.build(ctx, cell, active_cell);
                cell = modules::AddModule{}.build(ctx, written_cell, kept_cell);
            } else {
                hidden = modules::MulModule{}.build(ctx, hidden, active_hidden);
                cell = modules::MulModule{}.build(ctx, cell, modules::RepeatModule({cell.shape}).build(ctx, active));
            }
        }
        steps[static_cast<size_t>(t)] = hidden;
    }

    auto sequence = steps[0];
    for (int64_t t = 1; t < frames; ++t) {
        sequence = concat_rows.build(ctx, sequence, steps[static_cast<size_t>(t)]);
    }
    return modules::LSTMSequenceOutputs{sequence, hidden, cell};
}

modules::LSTMSequenceOutputs build_lstm_sequence_combined_input_bias(core::ModuleBuildContext &ctx,
                                                                     const core::TensorValue &input,
                                                                     const KittenWeights::LstmWeights &lstm,
                                                                     bool reverse, const core::TensorValue *valid_mask,
                                                                     std::vector<ggml_tensor *> &zero_state_inputs) {
    return build_lstm_sequence_combined_input_bias(
        ctx, input, lstm, reverse, make_zero_lstm_state(ctx, lstm.hidden_size, zero_state_inputs),
        make_zero_lstm_state(ctx, lstm.hidden_size, zero_state_inputs), valid_mask, false);
}

modules::BidirectionalLSTMOutputs build_bidirectional_lstm_combined_input_bias(
    core::ModuleBuildContext &ctx, const core::TensorValue &input, const KittenWeights::LstmWeights &lstm,
    const core::TensorValue *valid_mask, std::vector<ggml_tensor *> &zero_state_inputs) {
    const auto forward =
        build_lstm_sequence_combined_input_bias(ctx, input, lstm, false, valid_mask, zero_state_inputs);
    const auto reverse = build_lstm_sequence_combined_input_bias(ctx, input, lstm, true, valid_mask, zero_state_inputs);
    const auto sequence = modules::ConcatModule({1}).build(ctx, forward.sequence, reverse.sequence);
    return {sequence, forward.hidden, forward.cell, reverse.hidden, reverse.cell};
}

core::TensorValue apply_time_mask(core::ModuleBuildContext &ctx, const core::TensorValue &value,
                                  const core::TensorValue &mask) {
    return modules::MulModule{}.build(ctx, value, modules::RepeatModule({value.shape}).build(ctx, mask));
}

template <typename ConvWeightsT> modules::Conv1dWeights make_conv1d_weights(const ConvWeightsT &conv) {
    modules::Conv1dWeights weights = {};
    weights.weight = conv.weight;
    if (conv.use_bias) {
        weights.bias = conv.bias;
    }
    return weights;
}

modules::ConvTranspose1dWeights
make_conv_transpose1d_weights(const KittenWeights::WeightNormConvTranspose1dWeights &conv) {
    modules::ConvTranspose1dWeights weights = {};
    weights.weight = conv.groups == 1 ? conv.weight : conv.dense_weight;
    if (conv.use_bias) {
        weights.bias = conv.bias;
    }
    return weights;
}

template <typename ConvWeightsT>
ggml_tensor *build_standard_conv1d_ct_predictor(ggml_context *ctx, ggml_tensor *input, const ConvWeightsT &conv) {
    if (conv.groups != 1) {
        throw std::runtime_error("kitten predictor conv1d requires groups == 1");
    }
    core::ModuleBuildContext build_ctx = {};
    build_ctx.ggml = ctx;
    const auto input_ct = core::wrap_tensor(ggml_cont(ctx, input),
                                            core::TensorShape::from_dims({input->ne[1], input->ne[0]}), GGML_TYPE_F32);
    const auto input_bct = modules::ReshapeModule({core::TensorShape::from_dims({1, conv.in_channels, input->ne[0]})})
                               .build(build_ctx, input_ct);
    const auto output_bct =
        modules::Conv1dModule({conv.in_channels, conv.out_channels, conv.kernel, static_cast<int>(conv.stride),
                               static_cast<int>(conv.padding), static_cast<int>(conv.dilation), conv.use_bias})
            .build(build_ctx, input_bct, make_conv1d_weights(conv));
    return modules::ReshapeModule({core::TensorShape::from_dims({conv.out_channels, output_bct.shape.dims[2]})})
        .build(build_ctx, output_bct)
        .tensor;
}

ggml_tensor *build_grouped_conv_transpose1d_ct_predictor(ggml_context *ctx, ggml_tensor *input,
                                                         const KittenWeights::WeightNormConvTranspose1dWeights &conv) {
    const int64_t grouped_in = conv.in_channels / conv.groups;
    const int64_t grouped_out = conv.out_channels / conv.groups;
    if (conv.groups == conv.in_channels && conv.in_channels == conv.out_channels && grouped_in == 1 &&
        grouped_out == 1) {
        core::ModuleBuildContext build_ctx = {};
        build_ctx.ggml = ctx;
        const auto input_ct = core::wrap_tensor(
            ggml_cont(ctx, input), core::TensorShape::from_dims({input->ne[1], input->ne[0]}), GGML_TYPE_F32);
        const auto input_bct =
            modules::ReshapeModule({core::TensorShape::from_dims({1, conv.in_channels, input->ne[0]})})
                .build(build_ctx, input_ct);
        modules::ConvTranspose1dWeights weights = {};
        weights.weight = conv.dense_weight;
        if (conv.use_bias) {
            weights.bias = conv.bias;
        }
        auto output_bct = modules::ConvTranspose1dModule({conv.in_channels, conv.out_channels, conv.kernel,
                                                          static_cast<int>(conv.stride), 0, 1, conv.use_bias})
                              .build(build_ctx, input_bct, weights);
        const int64_t cropped_len =
            (input->ne[0] - 1) * conv.stride - 2 * conv.padding + conv.kernel + conv.output_padding;
        const auto cropped =
            core::wrap_tensor(ggml_cont(ctx, ggml_view_3d(ctx, output_bct.tensor, cropped_len, conv.out_channels, 1,
                                                          output_bct.tensor->nb[1], output_bct.tensor->nb[2],
                                                          static_cast<size_t>(conv.padding) * sizeof(float))),
                              core::TensorShape::from_dims({1, conv.out_channels, cropped_len}), GGML_TYPE_F32);
        return modules::ReshapeModule({core::TensorShape::from_dims({conv.out_channels, cropped.shape.dims[2]})})
            .build(build_ctx, cropped)
            .tensor;
    }
    throw std::runtime_error("kitten predictor grouped conv_transpose1d supports "
                             "only depthwise layout");
}

ggml_tensor *build_conv_transpose1d_ct_predictor(ggml_context *ctx, ggml_tensor *input,
                                                 const KittenWeights::WeightNormConvTranspose1dWeights &conv) {
    if (conv.groups > 1) {
        return build_grouped_conv_transpose1d_ct_predictor(ctx, input, conv);
    }
    core::ModuleBuildContext build_ctx = {};
    build_ctx.ggml = ctx;
    const auto input_ct = core::wrap_tensor(ggml_cont(ctx, input),
                                            core::TensorShape::from_dims({input->ne[1], input->ne[0]}), GGML_TYPE_F32);
    const auto input_bct = modules::ReshapeModule({core::TensorShape::from_dims({1, conv.in_channels, input->ne[0]})})
                               .build(build_ctx, input_ct);
    auto output_bct = modules::ConvTranspose1dModule({conv.in_channels, conv.out_channels, conv.kernel,
                                                      static_cast<int>(conv.stride), 0, 1, conv.use_bias})
                          .build(build_ctx, input_bct, make_conv_transpose1d_weights(conv));
    const int64_t cropped_len = (input->ne[0] - 1) * conv.stride - 2 * conv.padding + conv.kernel + conv.output_padding;
    const auto cropped = core::wrap_tensor(
        ggml_cont(ctx, ggml_view_3d(ctx, output_bct.tensor, cropped_len, conv.out_channels, 1, output_bct.tensor->nb[1],
                                    output_bct.tensor->nb[2], static_cast<size_t>(conv.padding) * sizeof(float))),
        core::TensorShape::from_dims({1, conv.out_channels, cropped_len}), GGML_TYPE_F32);
    return modules::ReshapeModule({core::TensorShape::from_dims({conv.out_channels, cropped.shape.dims[2]})})
        .build(build_ctx, cropped)
        .tensor;
}

ggml_tensor *build_adaptive_instance_norm_ct_predictor(ggml_context *ctx, ggml_tensor *x,
                                                       const KittenWeights::AdaIn1dWeights &weights,
                                                       const core::TensorValue &style,
                                                       std::vector<TimeMaskInputs> &masks, bool use_time_masks) {
    const int64_t channels = x->ne[1];
    core::ModuleBuildContext build_ctx = {};
    build_ctx.ggml = ctx;
    const auto affine = modules::LinearModule({weights.fc.in_features, weights.fc.out_features, weights.fc.use_bias})
                            .build(build_ctx, style,
                                   {
                                       weights.fc.weight,
                                       weights.fc.bias,
                                   });
    const auto scale_delta = modules::SliceModule({1, 0, channels}).build(build_ctx, affine);
    const auto shift = modules::SliceModule({1, channels, channels}).build(build_ctx, affine);
    const auto gamma_2d =
        core::wrap_tensor(ggml_scale_bias(ctx, scale_delta.tensor, 1.0f, 1.0f), scale_delta.shape, GGML_TYPE_F32);
    const auto gamma = core::wrap_tensor(ggml_reshape_1d(ctx, gamma_2d.tensor, channels),
                                         core::TensorShape::from_dims({channels}), GGML_TYPE_F32);
    const auto beta = core::wrap_tensor(ggml_reshape_1d(ctx, shift.tensor, channels),
                                        core::TensorShape::from_dims({channels}), GGML_TYPE_F32);
    if (use_time_masks) {
        return build_masked_adain_ct(build_ctx, x, gamma, beta, channels, weights.eps, masks);
    }
    const auto input = core::wrap_tensor(x, core::TensorShape::from_dims({channels, x->ne[0]}), GGML_TYPE_F32);
    return modules::AdaptiveInstanceNorm1dModule({channels, weights.eps}).build(build_ctx, input, {gamma, beta}).tensor;
}

ggml_tensor *build_adain_resblock_ct_predictor(ggml_context *ctx, ggml_tensor *x,
                                               const KittenWeights::AdainResBlock1dWeights &block,
                                               const core::TensorValue &style_predictor,
                                               const core::TensorValue &style_decoder, bool use_decoder_style,
                                               std::vector<TimeMaskInputs> &masks, bool use_time_masks) {
    const auto &style = use_decoder_style ? style_decoder : style_predictor;
    ggml_tensor *shortcut = x;
    if (block.upsample) {
        core::ModuleBuildContext build_ctx = {};
        build_ctx.ggml = ctx;
        const auto input = core::wrap_tensor(x, core::TensorShape::from_dims({x->ne[1], x->ne[0]}), GGML_TYPE_F32);
        const auto upsampled =
            modules::Interpolate1dModule({x->ne[0] * 2, modules::Interpolate1dMode::Nearest}).build(build_ctx, input);
        shortcut = upsampled.tensor;
    }
    if (block.learned_sc) {
        shortcut = build_standard_conv1d_ct_predictor(ctx, shortcut, block.conv1x1);
    }

    ggml_tensor *residual =
        build_adaptive_instance_norm_ct_predictor(ctx, x, block.norm1, style, masks, use_time_masks);
    residual = ggml_leaky_relu(ctx, residual, 0.2f, false);
    if (block.use_pool) {
        residual = build_conv_transpose1d_ct_predictor(ctx, residual, block.pool);
    }
    residual = build_standard_conv1d_ct_predictor(ctx, residual, block.conv1);
    residual = build_adaptive_instance_norm_ct_predictor(ctx, residual, block.norm2, style, masks, use_time_masks);
    residual = ggml_leaky_relu(ctx, residual, 0.2f, false);
    residual = build_standard_conv1d_ct_predictor(ctx, residual, block.conv2);
    core::ModuleBuildContext add_ctx = {};
    add_ctx.ggml = ctx;
    const auto residual_value =
        core::wrap_tensor(residual, core::TensorShape::from_dims({residual->ne[1], residual->ne[0]}), GGML_TYPE_F32);
    const auto shortcut_value =
        core::wrap_tensor(shortcut, core::TensorShape::from_dims({shortcut->ne[1], shortcut->ne[0]}), GGML_TYPE_F32);
    ggml_tensor *out = ggml_scale(ctx, modules::AddModule{}.build(add_ctx, residual_value, shortcut_value).tensor,
                                  0.7071067811865475f);
    if (!use_time_masks) {
        return out;
    }
    const TimeMaskInputs &mask = add_keep_mask_input(ctx, masks, out->ne[0]);
    core::ModuleBuildContext mask_ctx = {};
    mask_ctx.ggml = ctx;
    const auto out_value =
        core::wrap_tensor(out, core::TensorShape::from_dims({out->ne[1], out->ne[0]}), GGML_TYPE_F32);
    const auto keep = modules::RepeatModule({out_value.shape})
                          .build(mask_ctx, core::wrap_tensor(mask.keep, core::TensorShape::from_dims({1, out->ne[0]}),
                                                             GGML_TYPE_F32));
    return modules::MulModule{}.build(mask_ctx, out_value, keep).tensor;
}

struct PredictorPreTailOutputs {
    std::vector<int32_t> durations;
    std::vector<float> expanded_encoder_tc;
    int64_t expanded_encoder_rows = 0;
    int64_t expanded_encoder_cols = 0;
    std::vector<float> asr_ct;
    int64_t asr_rows = 0;
    int64_t asr_cols = 0;
};

class PredictorPreTailGraphRuntime {
  public:
    PredictorPreTailGraphRuntime(const KittenWeights &weights, ggml_backend_t backend, int n_threads,
                                 bool use_device_backend, KittenPredictorGraphConfig graph_config)
        : weights_(&weights), backend_(backend), duration_backend_(backend), text_backend_(backend),
          n_threads_(std::max(1, n_threads)), graph_config_(graph_config) {
        (void)use_device_backend;
    }

    ~PredictorPreTailGraphRuntime() = default;

    PredictorPreTailOutputs run(const std::vector<float> &plbert_hidden_tc, const std::vector<int32_t> &input_ids,
                                const std::vector<float> &style_predictor, float speed, int64_t token_capacity) {
        const int64_t valid_token_count = static_cast<int64_t>(input_ids.size());
        if (valid_token_count <= 0 || valid_token_count > token_capacity) {
            throw std::runtime_error("kitten predictor pre-tail input length exceeds prepared capacity");
        }
        auto &prepared = session_for_capacity(token_capacity, token_capacity != valid_token_count);
        const int64_t graph_token_count = prepared.token_count;
        const int64_t predictor_feature_cols = weights_->predictor.lstm.input_size;
        const int64_t text_feature_rows = weights_->text_encoder.lstm.hidden_size * 2;

        std::vector<float> padded_hidden;
        const std::vector<float> *hidden_for_graph = &plbert_hidden_tc;
        if (valid_token_count != graph_token_count) {
            padded_hidden.assign(static_cast<size_t>(graph_token_count * weights_->hidden_dim), 0.0f);
            std::memcpy(padded_hidden.data(), plbert_hidden_tc.data(),
                        static_cast<size_t>(valid_token_count * weights_->hidden_dim) * sizeof(float));
            hidden_for_graph = &padded_hidden;
        }
        std::vector<int32_t> padded_ids;
        const std::vector<int32_t> *ids_for_graph = &input_ids;
        if (valid_token_count != graph_token_count) {
            padded_ids.assign(static_cast<size_t>(graph_token_count), 0);
            std::memcpy(padded_ids.data(), input_ids.data(), static_cast<size_t>(valid_token_count) * sizeof(int32_t));
            ids_for_graph = &padded_ids;
        }

        PredictorPreTailOutputs out = {};
        std::vector<float> duration_features_tc;
        auto &duration = duration_session(prepared);
        const double duration_ms = measure_ms([&]() {
            duration.run(*hidden_for_graph, valid_token_count, style_predictor, speed, out.durations,
                         duration_features_tc);
        });
        engine::debug::timing_log_scalar("kitten.predictor.duration_compute_ms", duration_ms);
        auto &text = text_session(prepared);
        std::vector<float> text_features_ct;
        const double text_ms = measure_ms([&]() { text_features_ct = text.run(*ids_for_graph, valid_token_count); });
        engine::debug::timing_log_scalar("kitten.predictor.text_compute_ms", text_ms);
        if (valid_token_count != graph_token_count) {
            out.durations.resize(static_cast<size_t>(valid_token_count));
            std::vector<float> valid_duration_features(static_cast<size_t>(valid_token_count * predictor_feature_cols),
                                                       0.0f);
            for (int64_t row = 0; row < valid_token_count; ++row) {
                std::memcpy(valid_duration_features.data() + static_cast<ptrdiff_t>(row * predictor_feature_cols),
                            duration_features_tc.data() + static_cast<ptrdiff_t>(row * predictor_feature_cols),
                            static_cast<size_t>(predictor_feature_cols) * sizeof(float));
            }
            duration_features_tc = std::move(valid_duration_features);
            std::vector<float> valid_text_features(static_cast<size_t>(text_feature_rows * valid_token_count), 0.0f);
            for (int64_t row = 0; row < text_feature_rows; ++row) {
                std::memcpy(valid_text_features.data() + static_cast<ptrdiff_t>(row * valid_token_count),
                            text_features_ct.data() + static_cast<ptrdiff_t>(row * graph_token_count),
                            static_cast<size_t>(valid_token_count) * sizeof(float));
            }
            text_features_ct = std::move(valid_text_features);
        }

        int64_t total_frames = 0;
        for (int32_t duration : out.durations) {
            total_frames += duration;
        }
        const int64_t max_output_frames = valid_token_count * weights_->max_dur;
        if (total_frames > max_output_frames) {
            throw std::runtime_error("kitten predictor produced durations beyond prepared capacity");
        }

        const double expand_ms = measure_ms([&]() {
            out.expanded_encoder_tc =
                expand_tc_by_durations(duration_features_tc, valid_token_count, weights_->predictor.lstm.input_size,
                                       out.durations, total_frames);
            out.asr_ct = expand_ct_by_durations(text_features_ct, text_feature_rows, valid_token_count, out.durations,
                                                total_frames);
        });
        engine::debug::timing_log_scalar("kitten.predictor.expand_ms", expand_ms);
        out.expanded_encoder_rows = total_frames;
        out.expanded_encoder_cols = predictor_feature_cols;
        out.asr_rows = text_feature_rows;
        out.asr_cols = total_frames;
        return out;
    }

    int64_t token_count() const { return session_.token_count; }

    void prepare_capacity(int64_t token_count) {
        auto &prepared = session_for_capacity(token_count, true);
        (void)duration_session(prepared);
        (void)text_session(prepared);
    }

  private:
    struct DurationSession {
        const KittenWeights *weights = nullptr;
        int64_t token_count = 0;
        bool use_valid_mask = false;
        int n_threads = 1;
        KittenPredictorGraphConfig graph_config = {};
        ggml_backend_t backend = nullptr;
        ggml_context *ctx = nullptr;
        ggml_cgraph *graph = nullptr;
        ggml_gallocr_t gallocr = nullptr;
        ggml_backend_graph_plan_t plan = nullptr;
        core::TensorValue plbert_hidden_tc = {};
        core::TensorValue valid_mask_tc = {};
        core::TensorValue style_predictor = {};
        core::TensorValue speed = {};
        core::TensorValue durations_scaled = {};
        core::TensorValue duration_features_tc = {};

        DurationSession(const KittenWeights &weights_in, ggml_backend_t backend_in, int64_t token_count_in,
                        bool use_valid_mask_in, int n_threads_in, KittenPredictorGraphConfig graph_config_in)
            : weights(&weights_in), token_count(token_count_in), use_valid_mask(use_valid_mask_in),
              n_threads(n_threads_in), graph_config(graph_config_in), backend(backend_in) {
            ggml_init_params params = {};
            params.mem_size = graph_config.duration_graph_bytes;
            params.mem_buffer = nullptr;
            params.no_alloc = true;
            ctx = ggml_init(params);
            if (!ctx) {
                throw std::runtime_error("failed to initialize ggml context for Kitten predictor pre-tail");
            }

            try {
                std::vector<ggml_tensor *> zero_state_inputs;

                core::ModuleBuildContext build_ctx = {};
                build_ctx.ggml = ctx;
                build_ctx.module_instance_name = "kitten_predictor_pretail";

                plbert_hidden_tc = core::make_tensor(build_ctx, GGML_TYPE_F32,
                                                     core::TensorShape::from_dims({token_count, weights->hidden_dim}));
                if (use_valid_mask) {
                    valid_mask_tc =
                        core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({token_count, 1}));
                    ggml_set_input(valid_mask_tc.tensor);
                }
                style_predictor =
                    core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, weights->style_dim}));
                speed = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
                ggml_set_input(plbert_hidden_tc.tensor);
                ggml_set_input(style_predictor.tensor);
                ggml_set_input(speed.tensor);

                const auto concat_with_style = [&](const core::TensorValue &x_tc) {
                    const auto repeated_style =
                        modules::RepeatModule(
                            {core::TensorShape::from_dims({x_tc.shape.dims[0], style_predictor.shape.dims[1]})})
                            .build(build_ctx, style_predictor);
                    return modules::ConcatModule({1}).build(build_ctx, x_tc, repeated_style);
                };

                const auto apply_adaptive_layer_norm = [&](const core::TensorValue &x_tc,
                                                           const KittenWeights::AdaLayerNormWeights &ada_weights) {
                    const int64_t hidden = ada_weights.fc.out_features / 2;
                    const int64_t cond = ada_weights.fc.in_features;
                    const auto normed =
                        modules::LayerNormModule({hidden, ada_weights.eps, false, false}).build(build_ctx, x_tc, {});
                    const auto affine =
                        modules::LinearModule({cond, hidden * 2, ada_weights.fc.use_bias})
                            .build(build_ctx, style_predictor, {ada_weights.fc.weight, ada_weights.fc.bias});
                    const auto scale_delta = modules::SliceModule({1, 0, hidden}).build(build_ctx, affine);
                    const auto shift_vec = modules::SliceModule({1, hidden, hidden}).build(build_ctx, affine);
                    const auto scale_vec = core::wrap_tensor(ggml_scale_bias(ctx, scale_delta.tensor, 1.0f, 1.0f),
                                                             scale_delta.shape, GGML_TYPE_F32);
                    const auto scale =
                        modules::RepeatModule({core::TensorShape::from_dims({x_tc.shape.dims[0], hidden})})
                            .build(build_ctx, scale_vec);
                    const auto shift =
                        modules::RepeatModule({core::TensorShape::from_dims({x_tc.shape.dims[0], hidden})})
                            .build(build_ctx, shift_vec);
                    const auto scaled = modules::MulModule{}.build(build_ctx, normed, scale);
                    return modules::AddModule{}.build(build_ctx, scaled, shift);
                };

                const auto build_duration_bidir_lstm = [&](const core::TensorValue &input,
                                                           const KittenWeights::LstmWeights &lstm) {
                    return build_bidirectional_lstm_combined_input_bias(
                        build_ctx, input, lstm, use_valid_mask ? &valid_mask_tc : nullptr, zero_state_inputs);
                };

                duration_features_tc = concat_with_style(plbert_hidden_tc);
                for (size_t i = 0; i < weights->predictor.duration_encoder.lstms.size(); ++i) {
                    const auto &lstm = weights->predictor.duration_encoder.lstms[i];
                    const auto lstm_outs = build_duration_bidir_lstm(duration_features_tc, lstm);
                    duration_features_tc = concat_with_style(apply_adaptive_layer_norm(
                        lstm_outs.sequence, weights->predictor.duration_encoder.ada_layer_norms[i]));
                }

                const auto predictor_hidden_tc =
                    build_duration_bidir_lstm(duration_features_tc, weights->predictor.lstm).sequence;

                auto logits =
                    modules::LinearModule({weights->predictor.duration_proj.in_features,
                                           weights->predictor.duration_proj.out_features,
                                           weights->predictor.duration_proj.use_bias})
                        .build(build_ctx, predictor_hidden_tc,
                               {
                                   weights->predictor.duration_proj.weight,
                                   weights->predictor.duration_proj.use_bias ? *weights->predictor.duration_proj.bias
                                                                             : core::TensorValue{},
                               });

                auto sig = modules::SigmoidModule().build(build_ctx, logits);
                auto flat =
                    core::reshape_tensor(build_ctx, sig, core::TensorShape::from_dims({token_count, weights->max_dur}));
                auto summed = modules::ReduceSumModule({-1}).build(build_ctx, flat);
                auto scaled =
                    core::wrap_tensor(ggml_div(ctx, summed.tensor, speed.tensor), summed.shape, GGML_TYPE_F32);
                durations_scaled = scaled;
                ggml_set_output(durations_scaled.tensor);
                if (durations_scaled.tensor->view_src != nullptr) {
                    ggml_set_output(durations_scaled.tensor->view_src);
                }
                ggml_set_output(duration_features_tc.tensor);
                if (duration_features_tc.tensor->view_src != nullptr) {
                    ggml_set_output(duration_features_tc.tensor->view_src);
                }

                graph = ggml_new_graph_custom(ctx, graph_config.graph_node_capacity, false);
                ggml_build_forward_expand(graph, durations_scaled.tensor);
                ggml_build_forward_expand(graph, duration_features_tc.tensor);

                core::set_backend_threads(backend, n_threads);
                const double alloc_ms = measure_ms([&]() {
                    gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
                    if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) ||
                        !ggml_gallocr_alloc_graph(gallocr, graph)) {
                        throw std::runtime_error("failed to allocate Kitten predictor pre-tail graph tensors");
                    }
                });
                if (engine::core::uses_host_graph_plan(backend)) {
                    plan = engine::core::create_backend_graph_plan_if_host(backend, graph);
                    if (!plan) {
                        throw std::runtime_error("failed to create Kitten predictor pre-tail plan");
                    }
                }
                const double zero_state_upload_ms = measure_ms([&]() { upload_zero_state_inputs(zero_state_inputs); });
                const double input_init_ms = measure_ms([&]() {
                    std::vector<float> hidden(static_cast<size_t>(token_count * weights->hidden_dim), 0.0f);
                    std::vector<float> style(static_cast<size_t>(weights->style_dim), 0.0f);
                    float speed_value = 1.0f;
                    core::write_tensor_f32(plbert_hidden_tc, hidden);
                    if (use_valid_mask) {
                        std::vector<float> valid_mask(static_cast<size_t>(token_count), 1.0f);
                        core::write_tensor_f32(valid_mask_tc, valid_mask);
                    }
                    core::write_tensor_f32(style_predictor, style);
                    core::write_tensor_f32(speed, &speed_value, 1);
                });
                engine::debug::timing_log_scalar("kitten.graph.build.predictor_duration_alloc_ms", alloc_ms);
                engine::debug::timing_log_scalar("kitten.graph.build.predictor_duration_zero_state_upload_ms",
                                                 zero_state_upload_ms);
                engine::debug::timing_log_scalar("kitten.graph.build.predictor_duration_input_init_ms", input_init_ms);
            } catch (...) {
                core::release_backend_graph_resources(backend, graph);
                if (gallocr) {
                    ggml_gallocr_free(gallocr);
                    gallocr = nullptr;
                }
                if (ctx) {
                    ggml_free(ctx);
                }
                ctx = nullptr;
                throw;
            }
        }

        ~DurationSession() {
            if (plan) {
                engine::core::free_backend_graph_plan(backend, plan);
            }
            core::release_backend_graph_resources(backend, graph);
            if (gallocr) {
                ggml_gallocr_free(gallocr);
            }
            if (ctx) {
                ggml_free(ctx);
            }
        }

        void run(const std::vector<float> &plbert_hidden_tc_host, int64_t valid_token_count,
                 const std::vector<float> &style_predictor_host, float speed_host, std::vector<int32_t> &durations_out,
                 std::vector<float> &duration_features_out) {
            if (valid_token_count <= 0 || valid_token_count > token_count) {
                throw std::runtime_error("kitten predictor duration valid token count "
                                         "exceeds prepared capacity");
            }
            const double input_upload_ms = measure_ms([&]() {
                core::write_tensor_f32(plbert_hidden_tc, plbert_hidden_tc_host);
                if (use_valid_mask) {
                    std::vector<float> valid_mask(static_cast<size_t>(token_count), 0.0f);
                    std::fill(valid_mask.begin(), valid_mask.begin() + valid_token_count, 1.0f);
                    core::write_tensor_f32(valid_mask_tc, valid_mask);
                }
                core::write_tensor_f32(style_predictor, style_predictor_host);
                core::write_tensor_f32(speed, &speed_host, 1);
            });
            core::set_backend_threads(backend, n_threads);
            ggml_status status = GGML_STATUS_SUCCESS;
            const double graph_compute_ms =
                measure_ms([&]() { status = core::compute_backend_graph(backend, graph, plan); });
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error(std::string("kitten predictor pre-tail compute failed: ") +
                                         ggml_status_to_string(status));
            }

            std::vector<float> durations_f;
            const double duration_read_ms =
                measure_ms([&]() { durations_f = core::read_tensor_f32(durations_scaled.tensor); });
            durations_out.resize(static_cast<size_t>(token_count), 1);
            for (int64_t i = 0; i < token_count; ++i) {
                const auto rounded = static_cast<int32_t>(std::nearbyint(durations_f[static_cast<size_t>(i)]));
                durations_out[static_cast<size_t>(i)] =
                    std::clamp<int32_t>(rounded, 1, static_cast<int32_t>(weights->max_dur));
            }
            const double feature_read_ms =
                measure_ms([&]() { duration_features_out = core::read_tensor_f32(duration_features_tc.tensor); });
            engine::debug::timing_log_scalar("kitten.predictor_duration.input_upload_ms", input_upload_ms);
            engine::debug::timing_log_scalar("kitten.predictor_duration.graph.compute_ms", graph_compute_ms);
            engine::debug::timing_log_scalar("kitten.predictor_duration.duration_read_ms", duration_read_ms);
            engine::debug::timing_log_scalar("kitten.predictor_duration.feature_read_ms", feature_read_ms);
        }
    };

    struct TextSession {
        const KittenWeights *weights = nullptr;
        int64_t token_count = 0;
        bool use_valid_mask = false;
        int n_threads = 1;
        KittenPredictorGraphConfig graph_config = {};
        ggml_backend_t backend = nullptr;
        ggml_context *ctx = nullptr;
        ggml_cgraph *graph = nullptr;
        ggml_gallocr_t gallocr = nullptr;
        ggml_backend_graph_plan_t plan = nullptr;
        core::TensorValue input_ids = {};
        core::TensorValue valid_mask_tc = {};
        core::TensorValue valid_mask_bct = {};
        core::TensorValue text_features_ct = {};

        TextSession(const KittenWeights &weights_in, ggml_backend_t backend_in, int64_t token_count_in,
                    bool use_valid_mask_in, int n_threads_in, KittenPredictorGraphConfig graph_config_in)
            : weights(&weights_in), token_count(token_count_in), use_valid_mask(use_valid_mask_in),
              n_threads(n_threads_in), graph_config(graph_config_in), backend(backend_in) {
            ggml_init_params params = {};
            params.mem_size = graph_config.text_graph_bytes;
            params.mem_buffer = nullptr;
            params.no_alloc = true;
            ctx = ggml_init(params);
            if (!ctx) {
                throw std::runtime_error("failed to initialize ggml context for Kitten predictor text path");
            }

            try {
                std::vector<ggml_tensor *> zero_state_inputs;

                core::ModuleBuildContext build_ctx = {};
                build_ctx.ggml = ctx;
                build_ctx.module_instance_name = "kitten_predictor_text";

                input_ids = core::make_tensor(build_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({token_count}));
                ggml_set_input(input_ids.tensor);
                if (use_valid_mask) {
                    valid_mask_tc =
                        core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({token_count, 1}));
                    valid_mask_bct =
                        core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, token_count}));
                    ggml_set_input(valid_mask_tc.tensor);
                    ggml_set_input(valid_mask_bct.tensor);
                }

                auto text_x_tc = modules::EmbeddingModule({weights->text_encoder.embedding.num_embeddings,
                                                           weights->text_encoder.embedding.embedding_dim})
                                     .build(build_ctx, input_ids, weights->text_encoder.embedding.weight);
                if (use_valid_mask) {
                    text_x_tc = apply_time_mask(build_ctx, text_x_tc, valid_mask_tc);
                }
                auto text_x_ct = modules::TransposeModule({{1, 0, 2, 3}, 2}).build(build_ctx, text_x_tc);
                text_x_ct = core::wrap_tensor(ggml_cont(ctx, text_x_ct.tensor), text_x_ct.shape, GGML_TYPE_F32);
                auto text_x_bct =
                    modules::ReshapeModule(
                        {core::TensorShape::from_dims({1, text_x_ct.shape.dims[0], text_x_ct.shape.dims[1]})})
                        .build(build_ctx, text_x_ct);
                for (size_t i = 0; i < weights->text_encoder.cnn.size(); ++i) {
                    const auto &block = weights->text_encoder.cnn[i];
                    text_x_bct = modules::Conv1dModule({block.conv.in_channels, block.conv.out_channels,
                                                        block.conv.kernel, static_cast<int>(block.conv.stride),
                                                        static_cast<int>(block.conv.padding),
                                                        static_cast<int>(block.conv.dilation), block.conv.use_bias})
                                     .build(build_ctx, text_x_bct, make_conv1d_weights(block.conv));
                    auto norm_in = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(build_ctx, text_x_bct);
                    auto normed =
                        modules::LayerNormModule({block.layer_norm.channels, block.layer_norm.eps, true, true})
                            .build(build_ctx, norm_in,
                                   {
                                       block.layer_norm.weight,
                                       block.layer_norm.bias,
                                   });
                    text_x_bct = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(build_ctx, normed);
                    text_x_bct = core::wrap_tensor(ggml_cont(ctx, text_x_bct.tensor), text_x_bct.shape, GGML_TYPE_F32);
                    text_x_bct = core::wrap_tensor(ggml_leaky_relu(ctx, text_x_bct.tensor, 0.2f, false),
                                                   text_x_bct.shape, GGML_TYPE_F32);
                    if (use_valid_mask) {
                        text_x_bct = apply_time_mask(build_ctx, text_x_bct, valid_mask_bct);
                    }
                }
                auto text_x_btc = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(build_ctx, text_x_bct);
                text_x_btc = core::wrap_tensor(ggml_cont(ctx, text_x_btc.tensor), text_x_btc.shape, GGML_TYPE_F32);
                auto text_x_tc_for_lstm =
                    modules::ReshapeModule({core::TensorShape::from_dims({token_count, text_x_btc.shape.dims[2]})})
                        .build(build_ctx, text_x_btc);
                const auto text_lstm_sequence = build_bidirectional_lstm_combined_input_bias(
                                                    build_ctx, text_x_tc_for_lstm, weights->text_encoder.lstm,
                                                    use_valid_mask ? &valid_mask_tc : nullptr, zero_state_inputs)
                                                    .sequence;
                text_features_ct = modules::TransposeModule({{1, 0, 2, 3}, 2}).build(build_ctx, text_lstm_sequence);
                ggml_set_output(text_features_ct.tensor);
                if (text_features_ct.tensor->view_src != nullptr) {
                    ggml_set_output(text_features_ct.tensor->view_src);
                }

                graph = ggml_new_graph_custom(ctx, graph_config.graph_node_capacity, false);
                ggml_build_forward_expand(graph, text_features_ct.tensor);

                core::set_backend_threads(backend, n_threads);
                const double alloc_ms = measure_ms([&]() {
                    gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
                    if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) ||
                        !ggml_gallocr_alloc_graph(gallocr, graph)) {
                        throw std::runtime_error("failed to allocate Kitten predictor text graph tensors");
                    }
                });
                if (engine::core::uses_host_graph_plan(backend)) {
                    plan = engine::core::create_backend_graph_plan_if_host(backend, graph);
                    if (!plan) {
                        throw std::runtime_error("failed to create Kitten predictor text plan");
                    }
                }
                const double zero_state_upload_ms = measure_ms([&]() { upload_zero_state_inputs(zero_state_inputs); });
                const double input_init_ms = measure_ms([&]() {
                    std::vector<int32_t> ids(static_cast<size_t>(token_count), 0);
                    core::write_tensor_i32(input_ids, ids);
                    if (use_valid_mask) {
                        std::vector<float> valid_mask(static_cast<size_t>(token_count), 1.0f);
                        core::write_tensor_f32(valid_mask_tc, valid_mask);
                        core::write_tensor_f32(valid_mask_bct, valid_mask);
                    }
                });
                engine::debug::timing_log_scalar("kitten.graph.build.predictor_text_alloc_ms", alloc_ms);
                engine::debug::timing_log_scalar("kitten.graph.build.predictor_text_zero_state_upload_ms",
                                                 zero_state_upload_ms);
                engine::debug::timing_log_scalar("kitten.graph.build.predictor_text_input_init_ms", input_init_ms);
            } catch (...) {
                core::release_backend_graph_resources(backend, graph);
                if (gallocr) {
                    ggml_gallocr_free(gallocr);
                    gallocr = nullptr;
                }
                if (ctx) {
                    ggml_free(ctx);
                }
                ctx = nullptr;
                throw;
            }
        }

        ~TextSession() {
            if (plan) {
                engine::core::free_backend_graph_plan(backend, plan);
            }
            core::release_backend_graph_resources(backend, graph);
            if (gallocr) {
                ggml_gallocr_free(gallocr);
            }
            if (ctx) {
                ggml_free(ctx);
            }
        }

        std::vector<float> run(const std::vector<int32_t> &input_ids_host, int64_t valid_token_count) {
            if (valid_token_count <= 0 || valid_token_count > token_count) {
                throw std::runtime_error("kitten predictor text valid token count "
                                         "exceeds prepared capacity");
            }
            const double input_upload_ms = measure_ms([&]() {
                core::write_tensor_i32(input_ids, input_ids_host);
                if (use_valid_mask) {
                    std::vector<float> valid_mask(static_cast<size_t>(token_count), 0.0f);
                    std::fill(valid_mask.begin(), valid_mask.begin() + valid_token_count, 1.0f);
                    core::write_tensor_f32(valid_mask_tc, valid_mask);
                    core::write_tensor_f32(valid_mask_bct, valid_mask);
                }
            });
            core::set_backend_threads(backend, n_threads);
            ggml_status status = GGML_STATUS_SUCCESS;
            const double graph_compute_ms =
                measure_ms([&]() { status = core::compute_backend_graph(backend, graph, plan); });
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error(std::string("kitten predictor text compute failed: ") +
                                         ggml_status_to_string(status));
            }
            std::vector<float> text_features;
            const double output_read_ms =
                measure_ms([&]() { text_features = core::read_tensor_f32(text_features_ct.tensor); });
            engine::debug::timing_log_scalar("kitten.predictor_text.input_upload_ms", input_upload_ms);
            engine::debug::timing_log_scalar("kitten.predictor_text.graph.compute_ms", graph_compute_ms);
            engine::debug::timing_log_scalar("kitten.predictor_text.output_read_ms", output_read_ms);
            return text_features;
        }
    };

    const KittenWeights *weights_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    ggml_backend_t duration_backend_ = nullptr;
    ggml_backend_t text_backend_ = nullptr;
    int n_threads_ = 1;
    KittenPredictorGraphConfig graph_config_ = {};

    struct PreparedPreTailSession {
        int64_t token_count = 0;
        bool use_valid_mask = false;
        std::unique_ptr<DurationSession> duration;
        std::unique_ptr<TextSession> text;
    };

    PreparedPreTailSession session_;

    PreparedPreTailSession &session_for_capacity(int64_t token_count, bool use_valid_mask) {
        if (token_count <= 0) {
            throw std::runtime_error("kitten predictor pre-tail capacity must be positive");
        }
        if (session_.token_count == token_count && session_.use_valid_mask == use_valid_mask) {
            return session_;
        }
        PreparedPreTailSession session;
        session.token_count = token_count;
        session.use_valid_mask = use_valid_mask;
        session_ = std::move(session);
        return session_;
    }

    DurationSession &duration_session(PreparedPreTailSession &prepared) {
        if (!prepared.duration) {
            const double build_ms = measure_ms([&]() {
                prepared.duration =
                    std::make_unique<DurationSession>(*weights_, duration_backend_, prepared.token_count,
                                                      prepared.use_valid_mask, n_threads_, graph_config_);
            });
            engine::debug::timing_log_scalar("kitten.graph.build.predictor_duration_ms", build_ms);
        }
        return *prepared.duration;
    }

    TextSession &text_session(PreparedPreTailSession &prepared) {
        if (!prepared.text) {
            const double build_ms = measure_ms([&]() {
                prepared.text = std::make_unique<TextSession>(*weights_, text_backend_, prepared.token_count,
                                                              prepared.use_valid_mask, n_threads_, graph_config_);
            });
            engine::debug::timing_log_scalar("kitten.graph.build.predictor_text_ms", build_ms);
        }
        return *prepared.text;
    }
};

class TailSharedLstmBlockRuntime {
  public:
    TailSharedLstmBlockRuntime(const KittenWeights &weights, ggml_backend_t backend, int n_threads,
                               KittenPredictorGraphConfig graph_config)
        : weights_(&weights), backend_(backend), n_threads_(std::max(1, n_threads)), graph_config_(graph_config) {}

    std::vector<float> run(const std::vector<float> &encoder_tc, int64_t frames, int64_t cols) {
        if (frames <= 0) {
            throw std::runtime_error("kitten tail shared LSTM requires positive frame count");
        }
        if (cols != weights_->predictor.shared.input_size) {
            throw std::runtime_error("kitten tail shared LSTM input width changed");
        }
        const int64_t hidden_size = weights_->predictor.shared.hidden_size;
        std::vector<float> forward(static_cast<size_t>(frames * hidden_size), 0.0f);
        std::vector<float> reverse(static_cast<size_t>(frames * hidden_size), 0.0f);
        run_direction(false, encoder_tc, frames, cols, forward);
        run_direction(true, encoder_tc, frames, cols, reverse);

        std::vector<float> shared_ct(static_cast<size_t>(2 * hidden_size * frames), 0.0f);
        for (int64_t t = 0; t < frames; ++t) {
            for (int64_t h = 0; h < hidden_size; ++h) {
                shared_ct[static_cast<size_t>(h * frames + t)] = forward[static_cast<size_t>(t * hidden_size + h)];
                shared_ct[static_cast<size_t>((hidden_size + h) * frames + t)] =
                    reverse[static_cast<size_t>(t * hidden_size + h)];
            }
        }
        return shared_ct;
    }

  private:
    static constexpr int64_t kBlockFrames = 64;

    struct DirectionResult {
        std::vector<float> sequence;
        std::vector<float> hidden;
        std::vector<float> cell;
    };

    struct DirectionSession {
        const KittenWeights *weights = nullptr;
        ggml_backend_t backend = nullptr;
        int n_threads = 1;
        bool reverse = false;
        KittenPredictorGraphConfig graph_config = {};
        ggml_context *ctx = nullptr;
        ggml_gallocr_t gallocr = nullptr;
        ggml_backend_graph_plan_t plan = nullptr;
        ggml_cgraph *graph = nullptr;
        core::TensorValue input = {};
        core::TensorValue valid_mask = {};
        core::TensorValue hidden_in = {};
        core::TensorValue cell_in = {};
        core::TensorValue sequence_out = {};
        core::TensorValue hidden_out = {};
        core::TensorValue cell_out = {};

        DirectionSession(const KittenWeights &weights_in, ggml_backend_t backend_in, int n_threads_in, bool reverse_in,
                         KittenPredictorGraphConfig graph_config_in)
            : weights(&weights_in), backend(backend_in), n_threads(n_threads_in), reverse(reverse_in),
              graph_config(graph_config_in) {
            ggml_init_params params{
                /*.mem_size   =*/graph_config.tail_graph_bytes,
                /*.mem_buffer =*/nullptr,
                /*.no_alloc   =*/true,
            };
            ctx = ggml_init(params);
            if (!ctx) {
                throw std::runtime_error("failed to initialize Kitten tail shared LSTM block context");
            }
            try {
                core::ModuleBuildContext build_ctx = {};
                build_ctx.ggml = ctx;
                input = core::make_tensor(
                    build_ctx, GGML_TYPE_F32,
                    core::TensorShape::from_dims({kBlockFrames, weights->predictor.shared.input_size}));
                valid_mask =
                    core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({kBlockFrames, 1}));
                hidden_in = core::make_tensor(build_ctx, GGML_TYPE_F32,
                                              core::TensorShape::from_dims({1, weights->predictor.shared.hidden_size}));
                cell_in = core::make_tensor(build_ctx, GGML_TYPE_F32,
                                            core::TensorShape::from_dims({1, weights->predictor.shared.hidden_size}));
                ggml_set_input(input.tensor);
                ggml_set_input(valid_mask.tensor);
                ggml_set_input(hidden_in.tensor);
                ggml_set_input(cell_in.tensor);

                const auto outputs = build_lstm_sequence_combined_input_bias(
                    build_ctx, input, weights->predictor.shared, reverse, hidden_in, cell_in, &valid_mask, true);
                sequence_out = outputs.sequence;
                hidden_out = outputs.hidden;
                cell_out = outputs.cell;
                ggml_set_output(sequence_out.tensor);
                if (sequence_out.tensor->view_src != nullptr) {
                    ggml_set_output(sequence_out.tensor->view_src);
                }
                ggml_set_output(hidden_out.tensor);
                if (hidden_out.tensor->view_src != nullptr) {
                    ggml_set_output(hidden_out.tensor->view_src);
                }
                ggml_set_output(cell_out.tensor);
                if (cell_out.tensor->view_src != nullptr) {
                    ggml_set_output(cell_out.tensor->view_src);
                }
                graph = ggml_new_graph_custom(ctx, graph_config.graph_node_capacity, false);
                ggml_build_forward_expand(graph, sequence_out.tensor);
                ggml_build_forward_expand(graph, hidden_out.tensor);
                ggml_build_forward_expand(graph, cell_out.tensor);

                core::set_backend_threads(backend, n_threads);
                gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
                if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) ||
                    !ggml_gallocr_alloc_graph(gallocr, graph)) {
                    throw std::runtime_error("failed to allocate Kitten tail shared LSTM block graph tensors");
                }
                if (engine::core::uses_host_graph_plan(backend)) {
                    plan = engine::core::create_backend_graph_plan_if_host(backend, graph);
                    if (!plan) {
                        throw std::runtime_error("failed to create Kitten tail shared LSTM block plan");
                    }
                }
                std::vector<float> zeros_input(static_cast<size_t>(kBlockFrames * weights->predictor.shared.input_size),
                                               0.0f);
                std::vector<float> zeros_mask(static_cast<size_t>(kBlockFrames), 0.0f);
                std::vector<float> zeros_state(static_cast<size_t>(weights->predictor.shared.hidden_size), 0.0f);
                core::write_tensor_f32(input, zeros_input);
                core::write_tensor_f32(valid_mask, zeros_mask);
                core::write_tensor_f32(hidden_in, zeros_state);
                core::write_tensor_f32(cell_in, zeros_state);
            } catch (...) {
                if (plan) {
                    engine::core::free_backend_graph_plan(backend, plan);
                    plan = nullptr;
                }
                core::release_backend_graph_resources(backend, graph);
                if (gallocr) {
                    ggml_gallocr_free(gallocr);
                    gallocr = nullptr;
                }
                if (ctx) {
                    ggml_free(ctx);
                    ctx = nullptr;
                }
                throw;
            }
        }

        ~DirectionSession() {
            if (plan) {
                engine::core::free_backend_graph_plan(backend, plan);
            }
            core::release_backend_graph_resources(backend, graph);
            if (gallocr) {
                ggml_gallocr_free(gallocr);
            }
            if (ctx) {
                ggml_free(ctx);
            }
        }

        DirectionResult run(const std::vector<float> &block, int64_t valid_frames, const std::vector<float> &hidden,
                            const std::vector<float> &cell) {
            if (valid_frames <= 0 || valid_frames > kBlockFrames) {
                throw std::runtime_error("kitten tail shared LSTM block valid frame count is out of range");
            }
            std::vector<float> mask(static_cast<size_t>(kBlockFrames), 0.0f);
            std::fill(mask.begin(), mask.begin() + valid_frames, 1.0f);
            const double input_upload_ms = measure_ms([&]() {
                core::write_tensor_f32(input, block);
                core::write_tensor_f32(valid_mask, mask);
                core::write_tensor_f32(hidden_in, hidden);
                core::write_tensor_f32(cell_in, cell);
            });

            core::set_backend_threads(backend, n_threads);
            ggml_status status = GGML_STATUS_SUCCESS;
            const double compute_ms = measure_ms([&]() { status = core::compute_backend_graph(backend, graph, plan); });
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error(std::string("kitten tail shared LSTM block compute failed: ") +
                                         ggml_status_to_string(status));
            }
            DirectionResult result;
            const double output_read_ms = measure_ms([&]() {
                result = {
                    core::read_tensor_f32(sequence_out.tensor),
                    core::read_tensor_f32(hidden_out.tensor),
                    core::read_tensor_f32(cell_out.tensor),
                };
            });
            engine::debug::timing_log_scalar(reverse ? "kitten.predictor_tail.shared_lstm.reverse.input_upload_ms"
                                                     : "kitten.predictor_tail.shared_lstm.forward.input_upload_ms",
                                             input_upload_ms);
            engine::debug::timing_log_scalar(reverse ? "kitten.predictor_tail.shared_lstm.reverse.compute_ms"
                                                     : "kitten.predictor_tail.shared_lstm.forward.compute_ms",
                                             compute_ms);
            engine::debug::timing_log_scalar(reverse ? "kitten.predictor_tail.shared_lstm.reverse.output_read_ms"
                                                     : "kitten.predictor_tail.shared_lstm.forward.output_read_ms",
                                             output_read_ms);
            return result;
        }
    };

    DirectionSession &direction_session(bool reverse) {
        auto &session = reverse ? reverse_session_ : forward_session_;
        if (!session) {
            const double build_ms = measure_ms([&]() {
                session = std::make_unique<DirectionSession>(*weights_, backend_, n_threads_, reverse, graph_config_);
            });
            engine::debug::timing_log_scalar(reverse ? "kitten.graph.build.predictor_tail_shared_lstm_reverse_ms"
                                                     : "kitten.graph.build.predictor_tail_shared_lstm_forward_ms",
                                             build_ms);
        }
        return *session;
    }

    void run_direction(bool reverse, const std::vector<float> &encoder_tc, int64_t frames, int64_t cols,
                       std::vector<float> &output_tc) {
        DirectionSession &session = direction_session(reverse);
        const int64_t hidden_size = weights_->predictor.shared.hidden_size;
        std::vector<float> hidden(static_cast<size_t>(hidden_size), 0.0f);
        std::vector<float> cell(static_cast<size_t>(hidden_size), 0.0f);
        const int64_t chunks = (frames + kBlockFrames - 1) / kBlockFrames;

        for (int64_t chunk_index = 0; chunk_index < chunks; ++chunk_index) {
            const int64_t logical_chunk = reverse ? (chunks - 1 - chunk_index) : chunk_index;
            const int64_t start = logical_chunk * kBlockFrames;
            const int64_t valid = std::min<int64_t>(kBlockFrames, frames - start);
            std::vector<float> block(static_cast<size_t>(kBlockFrames * cols), 0.0f);
            for (int64_t t = 0; t < valid; ++t) {
                std::memcpy(block.data() + static_cast<ptrdiff_t>(t * cols),
                            encoder_tc.data() + static_cast<ptrdiff_t>((start + t) * cols),
                            static_cast<size_t>(cols) * sizeof(float));
            }
            DirectionResult result = session.run(block, valid, hidden, cell);
            hidden = std::move(result.hidden);
            cell = std::move(result.cell);
            for (int64_t t = 0; t < valid; ++t) {
                std::memcpy(output_tc.data() + static_cast<ptrdiff_t>((start + t) * hidden_size),
                            result.sequence.data() + static_cast<ptrdiff_t>(t * hidden_size),
                            static_cast<size_t>(hidden_size) * sizeof(float));
            }
        }
    }

    const KittenWeights *weights_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    int n_threads_ = 1;
    KittenPredictorGraphConfig graph_config_ = {};
    std::unique_ptr<DirectionSession> forward_session_;
    std::unique_ptr<DirectionSession> reverse_session_;
};

class PredictorTailGraphRuntime {
  public:
    PredictorTailGraphRuntime(const KittenWeights &weights, ggml_backend_t backend, int n_threads,
                              bool use_device_backend, KittenPredictorGraphConfig graph_config)
        : weights_(&weights), backend_(backend), n_threads_(std::max(1, n_threads)),
          use_device_backend_(use_device_backend), graph_config_(graph_config),
          shared_lstm_(weights, backend, n_threads, graph_config) {}

    PredictorOutputs run(const PredictorPreTailOutputs &pre_tail, const std::vector<float> &style_predictor,
                         const std::vector<float> &style_decoder) {
        std::vector<float> shared_ct;
        const double shared_lstm_ms = measure_ms([&]() {
            shared_ct = shared_lstm_.run(pre_tail.expanded_encoder_tc, pre_tail.expanded_encoder_rows,
                                         pre_tail.expanded_encoder_cols);
        });
        engine::debug::timing_log_scalar("kitten.predictor_tail.shared_lstm_block_ms", shared_lstm_ms);
        return session_for(pre_tail.expanded_encoder_rows)
            .run(shared_ct, pre_tail.expanded_encoder_rows, pre_tail.asr_ct, pre_tail.asr_rows, pre_tail.asr_cols,
                 pre_tail.durations, style_predictor, style_decoder);
    }

  private:
    struct Session {
        const KittenWeights *weights = nullptr;
        int64_t frames = 0;
        int n_threads = 1;
        bool use_device_backend = false;
        KittenPredictorGraphConfig graph_config = {};
        ggml_context *ctx = nullptr;
        ggml_tensor *shared_in = nullptr;
        ggml_tensor *asr_in = nullptr;
        ggml_tensor *style_predictor_in = nullptr;
        ggml_tensor *style_decoder_in = nullptr;
        ggml_tensor *f0_out = nullptr;
        ggml_tensor *decoder_x_out = nullptr;
        std::vector<TimeMaskInputs> time_masks;
        ggml_cgraph *graph = nullptr;
        ggml_backend_t backend = nullptr;
        ggml_gallocr_t gallocr = nullptr;
        ggml_backend_graph_plan_t plan = nullptr;

        Session(const KittenWeights &weights_in, ggml_backend_t backend_in, int64_t frames_in, int n_threads_in,
                bool use_device_backend_in, KittenPredictorGraphConfig graph_config_in)
            : weights(&weights_in), frames(frames_in), n_threads(n_threads_in),
              use_device_backend(use_device_backend_in), graph_config(graph_config_in), backend(backend_in) {
            ggml_init_params params{
                /*.mem_size   =*/graph_config.tail_graph_bytes,
                /*.mem_buffer =*/nullptr,
                /*.no_alloc   =*/true,
            };
            ctx = ggml_init(params);
            if (!ctx) {
                throw std::runtime_error("failed to initialize ggml context for Kitten predictor tail");
            }

            try {
                double define_prosody_ms = 0.0;
                double define_decoder_ms = 0.0;
                double define_expand_ms = 0.0;
                const double define_ms = measure_ms([&]() {
                    shared_in =
                        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, frames, weights->predictor.shared.hidden_size * 2);
                    asr_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, frames, 512);
                    style_predictor_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, weights->style_dim, 1);
                    style_decoder_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, weights->style_dim, 1);
                    ggml_set_input(shared_in);
                    ggml_set_input(asr_in);
                    ggml_set_input(style_predictor_in);
                    ggml_set_input(style_decoder_in);
                    core::ModuleBuildContext build_ctx = {};
                    build_ctx.ggml = ctx;
                    const auto style_predictor = core::wrap_tensor(
                        style_predictor_in, core::TensorShape::from_dims({1, weights->style_dim}), GGML_TYPE_F32);
                    const auto style_decoder = core::wrap_tensor(
                        style_decoder_in, core::TensorShape::from_dims({1, weights->style_dim}), GGML_TYPE_F32);

                    ggml_tensor *shared_ct = shared_in;

                    ggml_tensor *f0 = shared_ct;
                    ggml_tensor *n = shared_ct;
                    ggml_tensor *f0_down = nullptr;
                    ggml_tensor *n_down = nullptr;
                    define_prosody_ms = measure_ms([&]() {
                        for (size_t i = 0; i < weights->predictor.f0_blocks.size(); ++i) {
                            f0 = build_adain_resblock_ct_predictor(ctx, f0, weights->predictor.f0_blocks[i],
                                                                   style_predictor, style_decoder, false, time_masks,
                                                                   false);
                        }
                        for (size_t i = 0; i < weights->predictor.n_blocks.size(); ++i) {
                            n = build_adain_resblock_ct_predictor(ctx, n, weights->predictor.n_blocks[i],
                                                                  style_predictor, style_decoder, false, time_masks,
                                                                  false);
                        }

                        f0_out = build_standard_conv1d_ct_predictor(ctx, f0, weights->predictor.f0_proj);
                        ggml_tensor *n_out = build_standard_conv1d_ct_predictor(ctx, n, weights->predictor.n_proj);
                        f0_down = build_standard_conv1d_ct_predictor(ctx, f0_out, weights->decoder.f0_conv);
                        n_down = build_standard_conv1d_ct_predictor(ctx, n_out, weights->decoder.n_conv);
                    });
                    const auto asr_in_tv = core::wrap_tensor(
                        asr_in, core::TensorShape::from_dims({asr_in->ne[1], asr_in->ne[0]}), GGML_TYPE_F32);
                    const auto f0_down_tv = core::wrap_tensor(
                        f0_down, core::TensorShape::from_dims({f0_down->ne[1], f0_down->ne[0]}), GGML_TYPE_F32);
                    const auto n_down_tv = core::wrap_tensor(
                        n_down, core::TensorShape::from_dims({n_down->ne[1], n_down->ne[0]}), GGML_TYPE_F32);
                    define_decoder_ms = measure_ms([&]() {
                        auto asr_f0 = modules::ConcatModule({0}).build(build_ctx, asr_in_tv, f0_down_tv);
                        auto pre_encode = modules::ConcatModule({0}).build(build_ctx, asr_f0, n_down_tv);
                        ggml_tensor *x = pre_encode.tensor;
                        x = build_adain_resblock_ct_predictor(ctx, x, weights->decoder.encode, style_predictor,
                                                              style_decoder, true, time_masks, false);
                        ggml_tensor *asr_res =
                            build_standard_conv1d_ct_predictor(ctx, asr_in, weights->decoder.asr_res);
                        const auto asr_res_tv = core::wrap_tensor(
                            asr_res, core::TensorShape::from_dims({asr_res->ne[1], asr_res->ne[0]}), GGML_TYPE_F32);
                        auto f0_n = modules::ConcatModule({0}).build(build_ctx, f0_down_tv, n_down_tv);
                        auto cond = modules::ConcatModule({0}).build(build_ctx, asr_res_tv, f0_n);
                        bool use_residual_conditioning = true;
                        for (size_t i = 0; i < weights->decoder.decode.size(); ++i) {
                            if (use_residual_conditioning) {
                                const auto x_tv = core::wrap_tensor(
                                    x, core::TensorShape::from_dims({x->ne[1], x->ne[0]}), GGML_TYPE_F32);
                                auto decode_in = modules::ConcatModule({0}).build(build_ctx, x_tv, cond);
                                x = decode_in.tensor;
                            }
                            x = build_adain_resblock_ct_predictor(ctx, x, weights->decoder.decode[i], style_predictor,
                                                                  style_decoder, true, time_masks, false);
                            if (weights->decoder.decode[i].upsample) {
                                use_residual_conditioning = false;
                            }
                        }
                        decoder_x_out = ggml_cont(ctx, x);
                    });
                    define_expand_ms = measure_ms([&]() {
                        for (ggml_tensor *output_tensor : {f0_out, decoder_x_out}) {
                            ggml_set_output(output_tensor);
                            for (ggml_tensor *backing = output_tensor->view_src; backing != nullptr;
                                 backing = backing->view_src) {
                                ggml_set_output(backing);
                            }
                        }
                        graph = ggml_new_graph_custom(ctx, graph_config.graph_node_capacity, false);
                        ggml_build_forward_expand(graph, f0_out);
                        ggml_build_forward_expand(graph, decoder_x_out);
                    });
                });

                core::set_backend_threads(backend, n_threads);
                const double alloc_ms = measure_ms([&]() {
                    gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
                    if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) ||
                        !ggml_gallocr_alloc_graph(gallocr, graph)) {
                        throw std::runtime_error("failed to allocate Kitten predictor tail graph tensors");
                    }
                });
                if (engine::core::uses_host_graph_plan(backend)) {
                    plan = engine::core::create_backend_graph_plan_if_host(backend, graph);
                    if (!plan) {
                        throw std::runtime_error("failed to create Kitten predictor tail plan");
                    }
                }
                const double materialize_ms = measure_ms([&]() {
                    std::vector<float> shared(static_cast<size_t>(weights->predictor.shared.hidden_size * 2 * frames),
                                              0.0f);
                    std::vector<float> asr(static_cast<size_t>(512 * frames), 0.0f);
                    std::vector<float> style_predictor(static_cast<size_t>(weights->style_dim), 0.0f);
                    std::vector<float> style_decoder(static_cast<size_t>(weights->style_dim), 0.0f);
                    core::write_tensor_f32(core::wrap_tensor(shared_in,
                                                             core::TensorShape::from_dims(
                                                                 {weights->predictor.shared.hidden_size * 2, frames}),
                                                             GGML_TYPE_F32),
                                           shared);
                    core::write_tensor_f32(
                        core::wrap_tensor(asr_in, core::TensorShape::from_dims({512, frames}), GGML_TYPE_F32), asr);
                    core::write_tensor_f32(core::wrap_tensor(style_predictor_in,
                                                             core::TensorShape::from_dims({weights->style_dim}),
                                                             GGML_TYPE_F32),
                                           style_predictor);
                    core::write_tensor_f32(core::wrap_tensor(style_decoder_in,
                                                             core::TensorShape::from_dims({weights->style_dim}),
                                                             GGML_TYPE_F32),
                                           style_decoder);
                    upload_time_masks(time_masks, frames, frames);
                });
                engine::debug::timing_log_scalar("kitten.graph.build.predictor_tail_define_ms", define_ms);
                engine::debug::timing_log_scalar("kitten.graph.build.predictor_tail_define_prosody_ms",
                                                 define_prosody_ms);
                engine::debug::timing_log_scalar("kitten.graph.build.predictor_tail_define_decoder_ms",
                                                 define_decoder_ms);
                engine::debug::timing_log_scalar("kitten.graph.build.predictor_tail_define_expand_ms",
                                                 define_expand_ms);
                engine::debug::timing_log_scalar("kitten.graph.build.predictor_tail_alloc_ms", alloc_ms);
                engine::debug::timing_log_scalar("kitten.graph.build.predictor_tail_materialize_ms", materialize_ms);
            } catch (...) {
                core::release_backend_graph_resources(backend, graph);
                if (gallocr) {
                    ggml_gallocr_free(gallocr);
                    gallocr = nullptr;
                }
                if (ctx) {
                    ggml_free(ctx);
                }
                ctx = nullptr;
                throw;
            }
        }

        ~Session() {
            if (plan) {
                engine::core::free_backend_graph_plan(backend, plan);
            }
            core::release_backend_graph_resources(backend, graph);
            if (gallocr) {
                ggml_gallocr_free(gallocr);
            }
            if (ctx) {
                ggml_free(ctx);
            }
        }

        PredictorOutputs run(const std::vector<float> &shared_ct, int64_t expanded_encoder_rows,
                             const std::vector<float> &asr_ct, int64_t asr_rows, int64_t asr_cols,
                             const std::vector<int32_t> &durations, const std::vector<float> &style_predictor,
                             const std::vector<float> &style_decoder) {
            if (expanded_encoder_rows <= 0 || expanded_encoder_rows > frames || asr_cols != expanded_encoder_rows) {
                throw std::runtime_error("kitten predictor tail input frame count "
                                         "exceeds prepared capacity");
            }
            if (static_cast<int64_t>(shared_ct.size()) !=
                    weights->predictor.shared.hidden_size * 2 * expanded_encoder_rows ||
                asr_rows != 512) {
                throw std::runtime_error("kitten predictor tail input shape changed");
            }
            const double upload_ms = measure_ms([&]() {
                double pad_ms = 0.0;
                double tensor_upload_ms = 0.0;
                std::vector<float> padded_shared(
                    static_cast<size_t>(weights->predictor.shared.hidden_size * 2 * frames), 0.0f);
                const int64_t shared_rows = weights->predictor.shared.hidden_size * 2;
                std::vector<float> padded_asr(static_cast<size_t>(512 * frames), 0.0f);
                pad_ms = measure_ms([&]() {
                    for (int64_t row = 0; row < shared_rows; ++row) {
                        std::memcpy(padded_shared.data() + static_cast<ptrdiff_t>(row * frames),
                                    shared_ct.data() + static_cast<ptrdiff_t>(row * expanded_encoder_rows),
                                    static_cast<size_t>(expanded_encoder_rows) * sizeof(float));
                    }
                    for (int64_t row = 0; row < 512; ++row) {
                        std::memcpy(padded_asr.data() + static_cast<ptrdiff_t>(row * frames),
                                    asr_ct.data() + static_cast<ptrdiff_t>(row * expanded_encoder_rows),
                                    static_cast<size_t>(expanded_encoder_rows) * sizeof(float));
                    }
                });
                tensor_upload_ms = measure_ms([&]() {
                    core::write_tensor_f32(core::wrap_tensor(shared_in,
                                                             core::TensorShape::from_dims(
                                                                 {weights->predictor.shared.hidden_size * 2, frames}),
                                                             GGML_TYPE_F32),
                                           padded_shared);
                    core::write_tensor_f32(
                        core::wrap_tensor(asr_in, core::TensorShape::from_dims({512, frames}), GGML_TYPE_F32),
                        padded_asr);
                    core::write_tensor_f32(core::wrap_tensor(style_predictor_in,
                                                             core::TensorShape::from_dims({weights->style_dim}),
                                                             GGML_TYPE_F32),
                                           style_predictor);
                    core::write_tensor_f32(core::wrap_tensor(style_decoder_in,
                                                             core::TensorShape::from_dims({weights->style_dim}),
                                                             GGML_TYPE_F32),
                                           style_decoder);
                    upload_time_masks(time_masks, expanded_encoder_rows, frames);
                });
                engine::debug::timing_log_scalar("kitten.predictor_tail.input_pad_ms", pad_ms);
                engine::debug::timing_log_scalar("kitten.predictor_tail.tensor_upload_ms", tensor_upload_ms);
            });
            engine::debug::timing_log_scalar("kitten.predictor_tail.input_upload_ms", upload_ms);
            core::set_backend_threads(backend, n_threads);
            ggml_status status = GGML_STATUS_SUCCESS;
            const double compute_ms = measure_ms([&]() { status = core::compute_backend_graph(backend, graph, plan); });
            engine::debug::timing_log_scalar("kitten.predictor_tail.compute_ms", compute_ms);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error(std::string("kitten predictor tail compute failed: ") +
                                         ggml_status_to_string(status));
            }
            std::vector<float> f0_curve;
            const int64_t decoder_x_capacity_cols = decoder_x_out->ne[0];
            if (decoder_x_capacity_cols <= 0 || decoder_x_capacity_cols % frames != 0) {
                throw std::runtime_error("kitten predictor tail decoder output capacity is inconsistent");
            }
            const int64_t decoder_frame_scale = decoder_x_capacity_cols / frames;
            const int64_t decoder_x_cols = expanded_encoder_rows * decoder_frame_scale;
            if (decoder_x_cols <= 0 || decoder_x_cols > decoder_x_capacity_cols) {
                throw std::runtime_error("kitten predictor tail decoder output length is inconsistent");
            }
            const double output_read_ms = measure_ms([&]() { f0_curve = core::read_tensor_f32(f0_out); });
            if (static_cast<int64_t>(f0_curve.size()) < decoder_x_cols) {
                throw std::runtime_error("kitten predictor tail f0 output is shorter than decoder output");
            }
            if (static_cast<int64_t>(f0_curve.size()) > decoder_x_cols) {
                f0_curve.resize(static_cast<size_t>(decoder_x_cols));
            }
            engine::debug::timing_log_scalar("kitten.predictor_tail.output_read_ms", output_read_ms);
            PredictorOutputs result;
            result.durations = durations;
            result.f0_curve = std::move(f0_curve);
            result.decoder_x = std::vector<float>{};
            result.decoder_x_rows = decoder_x_out->ne[1];
            result.decoder_x_cols = decoder_x_cols;
            result.borrowed_decoder_x_tensor = decoder_x_out;
            result.decoder_x_on_backend = true;
            return result;
        }
    };

    Session &session_for(int64_t frames) {
        if (session_ && session_->weights == weights_ && session_->frames == frames &&
            session_->n_threads == n_threads_ && session_->use_device_backend == use_device_backend_) {
            return *session_;
        }
        session_.reset();
        const double build_ms = measure_ms([&]() {
            session_ =
                std::make_unique<Session>(*weights_, backend_, frames, n_threads_, use_device_backend_, graph_config_);
        });
        engine::debug::timing_log_scalar("kitten.graph.build.predictor_tail_ms", build_ms);
        return *session_;
    }

    const KittenWeights *weights_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    int n_threads_ = 1;
    bool use_device_backend_ = false;
    KittenPredictorGraphConfig graph_config_ = {};
    TailSharedLstmBlockRuntime shared_lstm_;
    std::unique_ptr<Session> session_;
};

} // namespace

struct KittenPredictorRuntime::Impl {
    std::shared_ptr<const KittenWeights> weights;
    int n_threads = 1;
    bool use_device_backend = false;
    ggml_backend_t backend = nullptr;
    int64_t pre_tail_token_capacity = 0;
    KittenPredictorGraphConfig graph_config = {};
    KittenPlbertRuntime plbert;
    PredictorPreTailGraphRuntime pre_tail_graph;
    PredictorTailGraphRuntime tail_graph;

    Impl(std::shared_ptr<const KittenWeights> weights_in, ggml_backend_t backend_in, int n_threads_in,
         bool use_device_backend_in, int64_t plbert_fixed_token_capacity, int64_t pre_tail_token_capacity_in,
         KittenPredictorGraphConfig graph_config_in)
        : weights(std::move(weights_in)), n_threads(std::max(1, n_threads_in)),
          use_device_backend(use_device_backend_in), backend(backend_in),
          pre_tail_token_capacity(pre_tail_token_capacity_in), graph_config(graph_config_in),
          plbert(weights, backend, n_threads, use_device_backend, plbert_fixed_token_capacity),
          pre_tail_graph(*weights, backend, n_threads, use_device_backend, graph_config),
          tail_graph(*weights, backend, n_threads, use_device_backend, graph_config) {
        plbert.prepare(true);
        if (pre_tail_token_capacity > 0) {
            pre_tail_graph.prepare_capacity(pre_tail_token_capacity);
        }
    }
};

KittenPredictorRuntime::KittenPredictorRuntime(std::shared_ptr<const KittenWeights> weights, ggml_backend_t backend,
                                               int n_threads, bool use_device_backend,
                                               int64_t plbert_fixed_token_capacity, int64_t pre_tail_token_capacity,
                                               KittenPredictorGraphConfig graph_config)
    : impl_(std::make_unique<Impl>(std::move(weights), backend, n_threads, use_device_backend,
                                   plbert_fixed_token_capacity, pre_tail_token_capacity, graph_config)) {}

KittenPredictorRuntime::~KittenPredictorRuntime() = default;

PredictorOutputs KittenPredictorRuntime::predict(const std::vector<int32_t> &input_ids, const std::vector<float> &ref_s,
                                                 float speed) {
    if (input_ids.empty()) {
        throw std::runtime_error("kitten_predict requires non-empty input_ids");
    }
    if (static_cast<int64_t>(ref_s.size()) != 256) {
        throw std::runtime_error("kitten_predict requires ref_s with 256 elements");
    }
    const int64_t token_count = static_cast<int64_t>(input_ids.size());
    const std::vector<float> style_decoder(ref_s.begin(), ref_s.begin() + 128);
    const std::vector<float> style_predictor(ref_s.begin() + 128, ref_s.end());
    std::vector<float> plbert_hidden_tc;
    const double plbert_ms = measure_ms([&]() { plbert_hidden_tc = impl_->plbert.encode(input_ids, true); });
    engine::debug::timing_log_scalar("kitten.predictor.plbert_ms", plbert_ms);

    const int64_t pre_tail_capacity =
        checked_capacity(std::max(token_count, impl_->pre_tail_token_capacity), impl_->weights->context_length);
    PredictorPreTailOutputs pre_tail;
    const double pretail_ms = measure_ms([&]() {
        pre_tail = impl_->pre_tail_graph.run(plbert_hidden_tc, input_ids, style_predictor, speed, pre_tail_capacity);
    });
    engine::debug::timing_log_scalar("kitten.predictor.pretail_ms", pretail_ms);
    PredictorOutputs output;
    const double tail_ms =
        measure_ms([&]() { output = impl_->tail_graph.run(pre_tail, style_predictor, style_decoder); });
    engine::debug::timing_log_scalar("kitten.predictor.tail_ms", tail_ms);
    return output;
}

} // namespace engine::models::kitten_tts
