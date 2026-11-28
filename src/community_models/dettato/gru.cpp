#include "engine/community_models/dettato/gru.h"

#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace engine::community_models::dettato::detail {
using namespace modules;
namespace {
// PyTorch reset-after recurrence, using only the supplied tensors. Each task
// owns independent batch lanes; output rows also hold the preceding state.
// No allocation, host readback, or mutable callback data during execution.
void host_recurrence(ggml_tensor *output, int task, int tasks, void *) {
    const int64_t hidden = output->ne[0], batch = output->ne[1], frames = output->src[0]->ne[2];
    const int64_t directions = output->src[0]->ne[3];
    for (int64_t lane = task; lane < batch * directions; lane += tasks) {
        const int64_t d = lane / batch, b = lane % batch;
        const bool reverse = d != 0;
        const auto *gates = static_cast<const float *>(output->src[0]->data) + d * frames * batch * 3 * hidden;
        const auto *weight = static_cast<const float *>(output->src[1]->data) + d * hidden * 3 * hidden;
        const auto *bias = static_cast<const float *>(output->src[2]->data) + d * 3 * hidden;
        const auto *initial = static_cast<const float *>(output->src[3]->data) + d * batch * hidden;
        auto *result = static_cast<float *>(output->data) + d * (frames + 3) * batch * hidden;
        auto *sums = result + frames * batch * hidden + b * 3 * hidden;
        for (int64_t step = 0; step < frames; ++step) {
            const int64_t t = reverse ? frames - step - 1 : step;
            const auto *previous =
                step == 0 ? initial + b * hidden : result + ((t + (reverse ? 1 : -1)) * batch + b) * hidden;
            const auto *input = gates + (t * batch + b) * 3 * hidden;
            auto *next = result + (t * batch + b) * hidden;
            std::fill(sums, sums + 3 * hidden, 0.0f);
            // Transposed rows let the compiler vectorize independent output
            // gates without changing the reduction order for any gate.
            for (int64_t c = 0; c < hidden; ++c) {
                const float value = previous[c];
                const auto *row = weight + c * 3 * hidden;
                for (int64_t j = 0; j < 3 * hidden; ++j)
                    sums[j] += row[j] * value;
            }
            for (int64_t h = 0; h < hidden; ++h) {
                const float reset = 1.f / (1.f + std::exp(-(input[h] + (sums[h] + bias[h]))));
                const float update =
                    1.f / (1.f + std::exp(-(input[hidden + h] + (sums[hidden + h] + bias[hidden + h]))));
                const float candidate =
                    std::tanh(input[2 * hidden + h] + reset * (sums[2 * hidden + h] + bias[2 * hidden + h]));
                next[h] = candidate + update * (previous[h] - candidate);
            }
        }
    }
}
} // namespace

BidirectionalGRU::BidirectionalGRU(BidirectionalGRUConfig config) : config_(config) {
    if (config.input_size <= 0 || config.hidden_size <= 0)
        throw std::runtime_error("BidirectionalGRU dimensions must be positive");
}
BidirectionalGRUOutputs BidirectionalGRU::build(core::ModuleBuildContext &ctx, const core::TensorValue &input,
                                                const core::TensorValue &forward_initial,
                                                const core::TensorValue &reverse_initial,
                                                const GRUWeights &forward_weights,
                                                const GRUWeights &reverse_weights) const {
    using core::TensorShape;
    core::validate_rank_between(input, 3, 3, "GRU input");
    core::validate_last_dim(input, config_.input_size, "GRU input");
    const auto frames = input.shape.dims[0], batch = input.shape.dims[1], width = config_.hidden_size;
    if (frames <= 0 || batch <= 0)
        throw std::runtime_error("BidirectionalGRU requires nonempty time and batch dimensions");
    const auto gates = 3 * width;
    for (const auto &initial : {forward_initial, reverse_initial})
        core::validate_shape(initial, TensorShape::from_dims({batch, width}), "GRU initial hidden");
    for (const auto &weights : {forward_weights, reverse_weights}) {
        core::validate_shape(weights.weight_hh, TensorShape::from_dims({gates, width}), "GRU weight_hh");
        core::validate_shape(weights.bias_hh, TensorShape::from_dims({gates}), "GRU bias_hh");
    }
    const auto forward = LinearModule({config_.input_size, gates, true, GGML_PREC_F32})
                             .build(ctx, input, {forward_weights.weight_ih, forward_weights.bias_ih});
    const auto reverse = LinearModule({config_.input_size, gates, true, GGML_PREC_F32})
                             .build(ctx, input, {reverse_weights.weight_ih, reverse_weights.bias_ih});
    auto *forward_hh = forward_weights.weight_hh.tensor;
    auto *reverse_hh = reverse_weights.weight_hh.tensor;
    // Vulkan's CONCAT dispatch accepts only four-byte elements. F16 -> F32 ->
    // F16 is exact; retain the original matmul weight type after concatenation.
    const bool widen_concat = ctx.backend_type == core::BackendType::Vulkan && forward_hh->type == GGML_TYPE_F16;
    if (widen_concat) {
        forward_hh = ggml_cast(ctx.ggml, forward_hh, GGML_TYPE_F32);
        reverse_hh = ggml_cast(ctx.ggml, reverse_hh, GGML_TYPE_F32);
    }
    auto *weight = ggml_concat(ctx.ggml, ggml_reshape_3d(ctx.ggml, forward_hh, width, gates, 1),
                               ggml_reshape_3d(ctx.ggml, reverse_hh, width, gates, 1), 2);
    if (widen_concat)
        weight = ggml_cast(ctx.ggml, weight, GGML_TYPE_F16);
    auto *bias = ggml_concat(ctx.ggml, ggml_reshape_3d(ctx.ggml, forward_weights.bias_hh.tensor, gates, 1, 1),
                             ggml_reshape_3d(ctx.ggml, reverse_weights.bias_hh.tensor, gates, 1, 1), 2);
    auto *hidden = ggml_concat(ctx.ggml, ggml_reshape_3d(ctx.ggml, forward_initial.tensor, width, batch, 1),
                               ggml_reshape_3d(ctx.ggml, reverse_initial.tensor, width, batch, 1), 2);
    if (config_.lowering == GRURecurrenceLowering::HostRecurrence ||
        config_.lowering == GRURecurrenceLowering::DeviceScan) {
        auto *projected = ggml_concat(ctx.ggml, ggml_reshape_4d(ctx.ggml, forward.tensor, gates, batch, frames, 1),
                                      ggml_reshape_4d(ctx.ggml, reverse.tensor, gates, batch, frames, 1), 3);
        auto *transposed = ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, weight, 1, 0, 2, 3));
        const bool scan = config_.lowering == GRURecurrenceLowering::DeviceScan;
        core::TensorValue combined;
        if (scan) {
            combined = core::wrap_tensor(ggml_gru_scan(ctx.ggml, projected, transposed, bias, hidden),
                                         TensorShape::from_dims({2, frames, batch, width}));
        } else {
            ggml_tensor *arguments[]{projected, transposed, bias, hidden};
            for (const auto *value : arguments)
                if (value->type != GGML_TYPE_F32 || !ggml_is_contiguous(value))
                    throw std::invalid_argument("Host GRU recurrence requires contiguous F32 tensors");
            combined = core::wrap_tensor(ggml_custom_4d(ctx.ggml, GGML_TYPE_F32, width, batch, frames + 3, 2, arguments,
                                                        4, host_recurrence, GGML_N_TASKS_MAX, nullptr),
                                         TensorShape::from_dims({2, frames + 3, batch, width}));
        }
        std::array<core::TensorValue, 2> sequences;
        for (int d = 0; d < 2; ++d) {
            auto sequence = SliceModule({0, d, 1}).build(ctx, combined);
            sequence =
                core::reshape_tensor(ctx, sequence, TensorShape::from_dims({scan ? frames : frames + 3, batch, width}));
            sequences[d] = SliceModule({0, 0, frames}).build(ctx, sequence);
        }
        const auto forward_hidden = core::reshape_tensor(ctx, SliceModule({0, frames - 1, 1}).build(ctx, sequences[0]),
                                                         TensorShape::from_dims({batch, width}));
        const auto reverse_hidden = core::reshape_tensor(ctx, SliceModule({0, 0, 1}).build(ctx, sequences[1]),
                                                         TensorShape::from_dims({batch, width}));
        return {ConcatModule({2}).build(ctx, sequences[0], sequences[1]), forward_hidden, reverse_hidden};
    }
    const auto state_shape = TensorShape::from_dims({2, batch, width});
    const auto positions =
        core::wrap_tensor(ggml_arange(ctx.ggml, 0, static_cast<float>(frames), 1), TensorShape::from_dims({frames}));
    auto *rows = ggml_cast(ctx.ggml, positions.tensor, GGML_TYPE_I32);
    const auto reversed_positions = ScaleBiasModule({-1, static_cast<float>(frames - 1)}).build(ctx, positions);
    auto *reverse_rows = ggml_cast(ctx.ggml, reversed_positions.tensor, GGML_TYPE_I32);
    auto *reverse_projected =
        ggml_get_rows(ctx.ggml, ggml_reshape_2d(ctx.ggml, reverse.tensor, gates * batch, frames), reverse_rows);
    auto *projected = ggml_concat(ctx.ggml, ggml_reshape_4d(ctx.ggml, forward.tensor, gates, batch, frames, 1),
                                  ggml_reshape_4d(ctx.ggml, reverse_projected, gates, batch, frames, 1), 3);
    const bool store_cell = config_.lowering == GRURecurrenceLowering::DeviceCell;
    auto *sequence = store_cell ? ggml_new_tensor_4d(ctx.ggml, GGML_TYPE_F32, width, batch, 2, frames)
                                : ggml_new_tensor_3d(ctx.ggml, GGML_TYPE_F32, width, frames, batch * 2);
    for (int64_t t = 0; t < frames; ++t) {
        auto *input_gates = ggml_view_3d(ctx.ggml, projected, gates, batch, 2, projected->nb[1], projected->nb[3],
                                         t * projected->nb[2]);
        auto *recurrent = ggml_mul_mat(ctx.ggml, weight, hidden);
        ggml_mul_mat_set_prec(recurrent, GGML_PREC_F32);
        if (config_.lowering == GRURecurrenceLowering::DeviceCell) {
            sequence = ggml_gru_cell_store(ctx.ggml, input_gates, recurrent, bias, hidden, sequence, t);
            hidden = ggml_view_3d(ctx.ggml, sequence, width, batch, 2, sequence->nb[1], sequence->nb[2],
                                  t * sequence->nb[3]);
        } else {
            recurrent = ggml_add(ctx.ggml, recurrent, bias);
            auto *rz = ggml_sigmoid(ctx.ggml, ggml_add(ctx.ggml,
                                                       ggml_view_3d(ctx.ggml, input_gates, 2 * width, batch, 2,
                                                                    input_gates->nb[1], input_gates->nb[2], 0),
                                                       ggml_view_3d(ctx.ggml, recurrent, 2 * width, batch, 2,
                                                                    recurrent->nb[1], recurrent->nb[2], 0)));
            const auto reset =
                core::wrap_tensor(ggml_view_3d(ctx.ggml, rz, width, batch, 2, rz->nb[1], rz->nb[2], 0), state_shape);
            const auto update = core::wrap_tensor(
                ggml_view_3d(ctx.ggml, rz, width, batch, 2, rz->nb[1], rz->nb[2], width * sizeof(float)), state_shape);
            const auto recurrent_new =
                core::wrap_tensor(ggml_view_3d(ctx.ggml, recurrent, width, batch, 2, recurrent->nb[1], recurrent->nb[2],
                                               2 * width * sizeof(float)),
                                  state_shape);
            const auto reset_new = MulModule{false, true}.build(ctx, reset, recurrent_new);
            auto *candidate =
                ggml_tanh(ctx.ggml, ggml_add(ctx.ggml,
                                             ggml_view_3d(ctx.ggml, input_gates, width, batch, 2, input_gates->nb[1],
                                                          input_gates->nb[2], 2 * width * sizeof(float)),
                                             reset_new.tensor));
            const auto delta = core::wrap_tensor(ggml_sub(ctx.ggml, hidden, candidate), state_shape);
            hidden = ggml_add(ctx.ggml, candidate, MulModule{false, true}.build(ctx, update, delta).tensor);
        }
        if (!store_cell)
            sequence = ggml_set_rows(ctx.ggml, sequence, ggml_reshape_3d(ctx.ggml, hidden, width, 1, batch * 2),
                                     ggml_view_1d(ctx.ggml, rows, 1, t * sizeof(int32_t)));
    }
    // Bind the same chronological output views to either bank layout. No
    // full-bank transpose is needed after direct cell stores.
    const size_t time_stride = store_cell ? sequence->nb[3] : sequence->nb[1];
    const size_t batch_stride = store_cell ? sequence->nb[1] : sequence->nb[2];
    const size_t direction_stride = store_cell ? sequence->nb[2] : batch * sequence->nb[2];
    auto *f = ggml_view_3d(ctx.ggml, sequence, width, frames, batch, time_stride, batch_stride, 0);
    auto *r = ggml_view_3d(ctx.ggml, sequence, width, frames, batch, time_stride, batch_stride, direction_stride);
    // Repeat the exact integral-valued F32 positions before casting: CUDA's
    // REPEAT supports F32, but not I32 index tensors.
    auto *reverse_batch_rows =
        ggml_cast(ctx.ggml, ggml_repeat_4d(ctx.ggml, reversed_positions.tensor, frames, batch, 1, 1), GGML_TYPE_I32);
    r = ggml_get_rows(ctx.ggml, ggml_cont(ctx.ggml, r), reverse_batch_rows);
    const auto sequence_shape = TensorShape::from_dims({frames, batch, width});
    const auto forward_sequence =
        core::wrap_tensor(ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, f, 0, 2, 1, 3)), sequence_shape);
    const auto reverse_sequence =
        core::wrap_tensor(ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, r, 0, 2, 1, 3)), sequence_shape);
    const auto forward_hidden = core::wrap_tensor(ggml_view_2d(ctx.ggml, hidden, width, batch, hidden->nb[1], 0),
                                                  TensorShape::from_dims({batch, width}));
    const auto reverse_hidden =
        core::wrap_tensor(ggml_view_2d(ctx.ggml, hidden, width, batch, hidden->nb[1], hidden->nb[2]),
                          TensorShape::from_dims({batch, width}));
    return {ConcatModule({2}).build(ctx, forward_sequence, reverse_sequence), forward_hidden, reverse_hidden};
}

} // namespace engine::community_models::dettato::detail
