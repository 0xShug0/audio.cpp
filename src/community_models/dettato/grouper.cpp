// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
/*
Adapted from SciPy rectangular_lsap, via the existing Nemotron alignment port.
Copyright (c) 2001-2002 Enthought, Inc.
Copyright (c) 2003, SciPy Developers.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include "engine/community_models/dettato/grouper.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/community_models/dettato/frontend.h"
#include "engine/community_models/dettato/transcriber.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <future>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace engine::community_models::dettato::detail {

using namespace core;
using namespace modules;

GrouperNetwork::GrouperNetwork(GrouperWeights weights, const ExecutionContext &execution)
    : execution_(execution), weights_(std::move(weights)), graphs_(32) {}

TensorValue GrouperNetwork::linear(ModuleBuildContext &ctx, const TensorValue &x, const LinearWeights &weights) {
    return LinearModule({weights.weight.shape.dims[1], weights.weight.shape.dims[0], true, GGML_PREC_F32})
        .build(ctx, x, weights);
}

void GrouperNetwork::build_lanes() {
    if (frames_ < 8)
        throw std::runtime_error("Dettato grouping frontend requires at least eight frames");
    auto &ctx = lane_graph_->build_context();
    const int64_t T = frames_ / 8, P = 96, C = 128;
    lane_input_ = make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 9, frames_, 288}));
    raster_input_ = make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 4, T, P}));
    ggml_set_input(lane_input_.tensor);
    ggml_set_input(raster_input_.tensor);
    const auto convolution = [&](const TensorValue &value, const Conv2dWeights &parameters, int dilation = 1,
                                 bool depthwise = false, int stride_height = 1, int stride_width = 1,
                                 int padding_width = 0) {
        const auto weight = parameters.weight;
        const auto kh = weight.shape.dims[2], kw = weight.shape.dims[3];
        if (depthwise)
            return DepthwiseConv2dModule({C, kh, kw, 1, 1, static_cast<int>(3 * dilation), 0, dilation, 1, true})
                .build(ctx, value, parameters);
        Conv2dConfig cfg{weight.shape.dims[1], weight.shape.dims[0], kh, kw};
        cfg.stride_height = stride_height;
        cfg.stride_width = stride_width;
        cfg.padding_width = padding_width;
        return Conv2dModule(cfg).build(ctx, value, parameters);
    };
    auto value = AddModule{}.build(ctx, convolution(lane_input_, weights_.lane_stem, 1, false, 8, 3, 1),
                                   convolution(raster_input_, weights_.occupancy));
    for (int layer = 0; layer < 6; ++layer) {
        const auto &block = weights_.lanes[layer];
        auto y = convolution(value, block.depthwise, 1 << (layer % 4), true);
        TensorValue mixed;
        if (ctx.backend_type == BackendType::Cuda) {
            mixed = wrap_tensor(
                ggml_conv_2d_dw_sparse(ctx.ggml, block.pitch.coefficients.tensor, y.tensor, block.pitch.offsets.tensor),
                y.shape);
        } else {
            mixed = TransposeModule({{1, 0, 2, 3}, 4}).build(ctx, y);
            mixed =
                reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, mixed), TensorShape::from_dims({C, T, P}));
            const auto matrix = TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, block.pitch.matrix);
            mixed = MatMulModule{}.build(ctx, mixed, matrix);
            mixed = reshape_tensor(ctx, mixed, TensorShape::from_dims({C, 1, T, P}));
            mixed = TransposeModule({{1, 0, 2, 3}, 4}).build(ctx, mixed);
        }
        y = AddModule{}.build(ctx, y, mixed);
        y = TransposeModule({{0, 2, 3, 1}, 4}).build(ctx, y);
        y = LayerNormModule({C, 1e-6f, true, true}).build(ctx, y, block.norm);
        y = TransposeModule({{0, 3, 1, 2}, 4}).build(ctx, y);
        y = convolution(GeluModule{}.build(ctx, convolution(y, block.ffn_in)), block.ffn_out);
        const auto gamma = reshape_tensor(ctx, block.gamma, TensorShape::from_dims({1, C, 1, 1}));
        value = AddModule{}.build(ctx, value, MulModule{true}.build(ctx, y, gamma));
        if (layer == 2 || layer == 5) {
            const auto &attention = weights_.lane_attention[layer == 2 ? 0 : 1];
            auto sequence = TransposeModule({{0, 2, 3, 1}, 4}).build(ctx, value);
            sequence = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, sequence),
                                      TensorShape::from_dims({T, P, C}));
            auto normalized = LayerNormModule({C, 1e-5f, true, true}).build(ctx, sequence, attention.norm);
            const auto packed = linear(ctx, normalized, attention.qkv);
            std::array<TensorValue, 3> heads;
            for (int i = 0; i < 3; ++i) {
                auto part = SliceModule({2, i * C, C}).build(ctx, packed);
                part = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, part),
                                      TensorShape::from_dims({T, P, 4, 32}));
                heads[i] = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, part);
            }
            const auto bias = make_tensor(lane_graph_->persistent_build_context(), GGML_TYPE_F32,
                                          TensorShape::from_dims({1, 4, P, P}));
            constants_.push_back({bias, attention.relative_bias});
            auto attended =
                ScaledDotProductAttentionModule({32, ScaledDotProductAttentionLowering::Explicit, GGML_PREC_F32})
                    .build(ctx, heads[0], heads[1], heads[2], bias);
            attended = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, attended),
                                      TensorShape::from_dims({T, P, C}));
            attended = linear(ctx, attended, attention.output);
            attended = MulModule{true}.build(ctx, attended, attention.gamma);
            attended = reshape_tensor(ctx, attended, TensorShape::from_dims({1, T, P, C}));
            value = AddModule{}.build(ctx, value, TransposeModule({{0, 3, 1, 2}, 4}).build(ctx, attended));
        }
    }
    lane_output_ = value;
    ggml_set_output(value.tensor);
    ggml_build_forward_expand(lane_graph_->graph(), value.tensor);
}

std::vector<float> GrouperNetwork::lane_features(const HarmonicFeatures &features, const std::vector<float> &notes) {
    if (features.batch != 1 || notes.size() % 12 != 0)
        throw std::runtime_error("Dettato lane reader expects one aligned note window");
    const int64_t count = static_cast<int64_t>(notes.size() / 12), T = features.frames / 8;
    prepare_lanes(features.frames);
    std::vector<int64_t> onset(count), end(count), lane(count);
    std::vector<float> raster(static_cast<size_t>(4 * (T + 1) * 96), 0);
    for (int64_t i = 0; i < count; ++i) {
        const auto *note = notes.data() + i * 12;
        onset[i] = std::clamp<int64_t>(static_cast<int64_t>((note[0] + .5f) * 12.5f), 0, T - 1);
        end[i] = std::max<int64_t>(std::min<int64_t>(static_cast<int64_t>((note[1] + .5f) * 12.5f), T), onset[i] + 1);
        lane[i] = std::clamp<int64_t>(static_cast<int64_t>(std::nearbyint(note[2])) - 21, 0, 95);
        const auto a = static_cast<size_t>(onset[i] * 96 + lane[i]), b = static_cast<size_t>(end[i] * 96 + lane[i]);
        const size_t plane = static_cast<size_t>((T + 1) * 96);
        raster[a] += 1;
        raster[plane + a] += 1;
        raster[plane + b] -= 1;
        raster[2 * plane + a] += note[note[3] > .5f ? 5 : 4];
        raster[3 * plane + a] += note[3] > .5f ? 1 : 0;
    }
    const auto plane = static_cast<size_t>((T + 1) * 96);
    for (int64_t p = 0; p < 96; ++p) {
        float active = 0;
        for (int64_t t = 0; t < T; ++t) {
            active += raster[plane + static_cast<size_t>(t * 96 + p)];
            raster[plane + static_cast<size_t>(t * 96 + p)] = active;
        }
    }
    std::vector<float> compact(static_cast<size_t>(4 * T * 96));
    for (size_t ch = 0; ch < 4; ++ch)
        std::copy_n(raster.begin() + ch * plane, static_cast<size_t>(T * 96), compact.begin() + ch * T * 96);
    write_tensor_f32(lane_input_, features.values);
    write_tensor_f32(raster_input_, compact);
    const auto start = std::chrono::steady_clock::now();
    lane_graph_->compute();
    const auto output = read_tensor_f32(lane_output_.tensor);
    std::vector<float> reads(static_cast<size_t>(count * 256));
    std::vector<float> prefix(static_cast<size_t>((T + 1) * 128));
    for (int64_t p = 0; p < 96; ++p) {
        if (std::find(lane.begin(), lane.end(), p) == lane.end())
            continue;
        std::fill_n(prefix.begin(), 128, 0);
        for (int64_t c = 0; c < 128; ++c) {
            double sum = 0;
            for (int64_t t = 0; t < T; ++t) {
                sum += output[static_cast<size_t>((c * T + t) * 96 + p)];
                prefix[static_cast<size_t>((t + 1) * 128 + c)] = static_cast<float>(sum);
            }
        }
        for (int64_t i = 0; i < count; ++i)
            if (lane[i] == p)
                for (int64_t c = 0; c < 128; ++c) {
                    reads[static_cast<size_t>(i * 256 + c)] = output[static_cast<size_t>((c * T + onset[i]) * 96 + p)];
                    reads[static_cast<size_t>(i * 256 + 128 + c)] = (prefix[static_cast<size_t>(end[i] * 128 + c)] -
                                                                     prefix[static_cast<size_t>(onset[i] * 128 + c)]) /
                                                                    static_cast<float>(end[i] - onset[i]);
                }
    }
    debug::timing_log_scalar("dettato.grouper.lanes_ms", debug::elapsed_ms(start));
    return reads;
}

void GrouperNetwork::prepare_lanes(int64_t frames) {
    if (lane_graph_ && frames_ == frames)
        return;
    lane_graph_.reset();
    constants_.clear();
    frames_ = frames;
    lane_graph_ = std::make_unique<PreparedGraph>(
        execution_, PreparedGraphConfig{"dettato.grouper.lanes", 4 * 1024 * 1024, 4096, 1024 * 1024});
    build_lanes();
    lane_graph_->allocate_persistent();
    lane_graph_->allocate();
    for (const auto &constant : constants_)
        write_tensor_f32(constant.first, constant.second);
}

GrouperNetwork::GraphState &GrouperNetwork::prepare(int64_t notes) {
    if (notes <= 0 || notes > 768)
        throw std::runtime_error("Dettato grouper graph expects 1..768 notes");
    if (auto *cached = graphs_.find(notes))
        return **cached;
    auto state = std::make_unique<GraphState>();
    auto &graph_ = state->graph;
    auto &inputs_ = state->inputs;
    auto &outputs_ = state->outputs;
    graph_ = std::make_unique<PreparedGraph>(
        execution_, PreparedGraphConfig{"dettato.grouper", size_t{2} * 1024 * 1024, 8192, 1024 * 1024});
    auto &ctx = graph_->build_context();
    inputs_[0] = make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({1, notes}));
    const std::array<int64_t, 4> widths{11, 150, 36, 256};
    for (size_t i = 0; i < widths.size(); ++i)
        inputs_[i + 1] = make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, notes, widths[i]}));
    inputs_[5] = make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, notes, 3, 368}));
    for (size_t i = 6; i < 8; ++i)
        inputs_[i] = make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 4, notes, 22}));
    for (const auto &input : inputs_)
        ggml_set_input(input.tensor);
    auto x = EmbeddingModule({128, 176}).build(ctx, inputs_[0], weights_.pitch_embedding);
    x = AddModule{}.build(
        ctx, x,
        linear(ctx, GeluModule{}.build(ctx, linear(ctx, inputs_[1], weights_.features[0])), weights_.features[1]));
    auto audio =
        linear(ctx, GeluModule{}.build(ctx, linear(ctx, inputs_[2], weights_.harmonics[0])), weights_.harmonics[1]);
    audio = AddModule{}.build(
        ctx, audio,
        linear(ctx, GeluModule{}.build(ctx, linear(ctx, inputs_[3], weights_.profile[0])), weights_.profile[1]));
    audio = AddModule{}.build(ctx, audio, linear(ctx, inputs_[4], weights_.lane_read));
    x = AddModule{}.build(ctx, x, audio);
    const auto mean = ReduceMeanModule({1}).build(ctx, inputs_[5]);
    auto f = wrap_tensor(ggml_sub(ctx.ggml, inputs_[5].tensor, mean.tensor), inputs_[5].shape);
    const auto variance = ReduceMeanModule({1}).build(ctx, MulModule{}.build(ctx, f, f));
    const auto sd = SqrtModule{}.build(ctx, variance);
    auto *denominator = ggml_clamp(ctx.ggml, sd.tensor, 1e-3f, INFINITY);
    f = wrap_tensor(ggml_div(ctx.ggml, f.tensor, denominator), f.shape);
    f = LayerNormModule({368, 1e-5f, true, true}).build(ctx, f, weights_.tap_norm);
    f = reshape_tensor(ctx, f, TensorShape::from_dims({1, notes, 1104}));
    f = linear(ctx, GeluModule{}.build(ctx, linear(ctx, f, weights_.taps[0])), weights_.taps[1]);
    x = AddModule{}.build(ctx, x, f);
    RopeTransformerEncoderConfig encoder;
    encoder.hidden_size = 176;
    encoder.intermediate_size = 352;
    encoder.heads = 4;
    encoder.rotary_dimensions = 44;
    encoder.rotary = EncoderRoPE::SplitTables;
    encoder.projection_precision = GGML_PREC_F32;
    encoder.flash_attention = false;
    RopeTransformerEncoderInputs position;
    position.cosine = inputs_[6];
    position.sine = inputs_[7];
    x = RopeTransformerEncoderModule(encoder).build(ctx, x, position, weights_.encoder);
    outputs_[0] = linear(ctx, x, weights_.heads[0]);
    outputs_[1] = linear(ctx, x, weights_.heads[1]);
    outputs_[1] = wrap_tensor(ggml_l2_norm(ctx.ggml, outputs_[1].tensor, 1e-12f), outputs_[1].shape);
    outputs_[2] = linear(ctx, x, weights_.heads[2]);
    outputs_[3] = x;
    outputs_[4] = SoftmaxModule{}.build(ctx, outputs_[0]);
    for (const auto &output : outputs_) {
        ggml_set_output(output.tensor);
        if (output.tensor->view_src)
            ggml_set_output(output.tensor->view_src);
        ggml_build_forward_expand(graph_->graph(), output.tensor);
    }
    debug::timing_log_context_reservation("dettato.grouper.graph", ctx.ggml);
    graph_->allocate();
    size_t retained = graph_->workspace_bytes() + graph_->persistent_bytes();
    graphs_.for_each_value(
        [&](const auto &entry) { retained += entry->graph->workspace_bytes() + entry->graph->persistent_bytes(); });
    // Exact note counts preserve attention math. Bound activation retention;
    // a single oversized shape remains usable without retaining other shapes.
    if (retained > 32 * 1024 * 1024)
        graphs_.clear();
    auto *result = state.get();
    graphs_.put(notes, std::move(state));
    return *result;
}

std::array<std::vector<float>, 5> GrouperNetwork::run(const GrouperInputs &inputs) {
    auto &state = prepare(static_cast<int64_t>(inputs.pitch.size()));
    write_tensor_i32(state.inputs[0], inputs.pitch);
    const std::array<const std::vector<float> *, 7> values{
        &inputs.scalars, &inputs.harmonics, &inputs.profile, &inputs.lanes, &inputs.taps, &inputs.cosine, &inputs.sine};
    for (size_t i = 0; i < values.size(); ++i)
        write_tensor_f32(state.inputs[i + 1], *values[i]);
    const auto start = std::chrono::steady_clock::now();
    state.graph->compute();
    std::array<std::vector<float>, 5> result;
    for (size_t i = 0; i < result.size(); ++i)
        result[i] = read_tensor_f32(state.outputs[i].tensor);
    debug::timing_log_scalar("dettato.grouper.forward_ms", debug::elapsed_ms(start));
    return result;
}

GroupingFeatures collect_grouping_features(const WindowTranscription &transcription, const std::vector<Note> &notes,
                                           const NoteThresholds &thresholds) {
    const auto T = transcription.frames;
    if (T <= 0 ||
        transcription.grouping_hidden->size() != (transcription.compact_grouping
                                                      ? transcription.grouping_rows.size() * 368
                                                      : static_cast<size_t>(T * 88 * 368)) ||
        transcription.grouping_stem->size() != transcription.grouping_hidden->size())
        throw std::runtime_error("Dettato grouping requires aligned hidden/stem sub-averages");
    const auto row_offset = [&](int64_t t, int64_t lane) {
        const auto row = t * 88 + lane;
        if (!transcription.compact_grouping)
            return static_cast<size_t>(row * 368);
        const auto found =
            std::lower_bound(transcription.grouping_rows.begin(), transcription.grouping_rows.end(), row);
        if (found == transcription.grouping_rows.end() || *found != row)
            throw std::runtime_error("Dettato grouping readback omitted a required feature row");
        return static_cast<size_t>(found - transcription.grouping_rows.begin()) * 368;
    };
    const auto own = decode_notes(transcription.grouping_maps, T, .02, thresholds);
    auto sorted = notes;
    std::stable_sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) {
        return std::tie(a.onset, a.pitch) < std::tie(b.onset, b.pitch);
    });
    const size_t count = sorted.size();
    GroupingFeatures out;
    out.notes.resize(count * 12);
    out.taps.resize(count * 1104);
    std::vector<int64_t> a(count), b(count), lanes(count);
    std::array<std::vector<size_t>, 88> selected;
    const auto clip = [&](int64_t frame) { return std::clamp<int64_t>(frame, 0, T - 1); };
    for (size_t i = 0; i < count; ++i) {
        const auto &note = sorted[i];
        const double off = std::max(note.offset, note.onset + .02);
        double closest = INFINITY;
        int slot = 0;
        for (const auto &candidate : own)
            if (candidate.pitch == note.pitch) {
                const double distance = std::abs(candidate.onset - note.onset);
                if (distance < closest) {
                    closest = distance;
                    slot = candidate.voice - 1;
                }
            }
        if (closest > .1)
            slot = 0;
        a[i] = clip(static_cast<int64_t>(std::nearbyint(note.onset / .02)));
        b[i] = std::min<int64_t>(std::max<int64_t>(static_cast<int64_t>(std::ceil(off / .02)), a[i] + 1), T);
        lanes[i] = std::clamp<int64_t>(note.pitch - 21, 0, 87);
        selected[static_cast<size_t>(lanes[i])].push_back(i);
        auto *meta = out.notes.data() + i * 12;
        meta[0] = static_cast<float>(note.onset);
        meta[1] = static_cast<float>(off);
        meta[2] = static_cast<float>(note.pitch);
        meta[3] = static_cast<float>(slot);
        const auto get = [&](int head, int64_t t) {
            return transcription.grouping_maps[head][static_cast<size_t>(clip(t) * 88 + lanes[i])];
        };
        const auto maximum = [&](int head, int64_t t) {
            return std::max({get(head, t - 1), get(head, t), get(head, t + 1)});
        };
        meta[4] = maximum(0, a[i]);
        meta[5] = maximum(3, a[i]);
        meta[6] = get(1, a[i]);
        meta[7] = get(4, a[i]);
        meta[10] = maximum(2, b[i] - 1);
        meta[11] = maximum(5, b[i] - 1);
        std::array<float, 1104> values{};
        const auto limit = std::min<int64_t>(b[i], a[i] + 50), stem_limit = std::min<int64_t>(b[i], a[i] + 10);
        ggml_fp16_to_fp32_row(transcription.grouping_hidden->data() + row_offset(a[i], lanes[i]), values.data(), 368);
        std::array<float, 368> row;
        for (int64_t t = a[i]; t < limit; ++t) {
            ggml_fp16_to_fp32_row(transcription.grouping_hidden->data() + row_offset(t, lanes[i]), row.data(), 368);
            for (size_t channel = 0; channel < 368; ++channel)
                values[368 + channel] += row[channel];
        }
        for (int64_t t = a[i]; t < stem_limit; ++t) {
            ggml_fp16_to_fp32_row(transcription.grouping_stem->data() + row_offset(t, lanes[i]), row.data(), 368);
            for (size_t channel = 0; channel < 368; ++channel)
                values[736 + channel] += row[channel];
        }
        for (size_t channel = 0; channel < 368; ++channel) {
            values[368 + channel] /= static_cast<float>(limit - a[i]);
            values[736 + channel] /= static_cast<float>(stem_limit - a[i]);
        }
        for (size_t tap = 0; tap < 3; ++tap) {
            float maximum = 0;
            for (size_t c = 0; c < 368; ++c)
                maximum = std::max(maximum, std::abs(values[tap * 368 + c]));
            const float scale = std::max(maximum, 1e-6f) / 127;
            const float stored_scale = ggml_fp16_to_fp32(ggml_fp32_to_fp16(scale));
            for (size_t c = 0; c < 368; ++c) {
                const float value = std::clamp(std::nearbyint(values[tap * 368 + c] / scale), -127.0f, 127.0f);
                out.taps[(i * 3 + tap) * 368 + c] = ggml_fp32_to_fp16(value * stored_scale);
            }
        }
    }
    std::vector<float> prefix(static_cast<size_t>((T + 1) * 2));
    for (size_t lane = 0; lane < selected.size(); ++lane) {
        if (selected[lane].empty())
            continue;
        for (size_t voice = 0; voice < 2; ++voice) {
            double sum = 0;
            prefix[voice * (T + 1)] = 0;
            for (int64_t t = 0; t < T; ++t) {
                sum += transcription.grouping_maps[1 + voice * 3][static_cast<size_t>(t * 88 + lane)];
                prefix[voice * (T + 1) + t + 1] = static_cast<float>(sum);
            }
            for (const auto i : selected[lane])
                out.notes[i * 12 + 8 + voice] =
                    (prefix[voice * (T + 1) + b[i]] - prefix[voice * (T + 1) + a[i]]) / static_cast<float>(b[i] - a[i]);
        }
    }
    return out;
}

GrouperInputs make_grouper_inputs(const HarmonicFeatures &features, const std::vector<float> &notes,
                                  const std::vector<float> &taps, const std::vector<float> &harmonic_prior,
                                  const std::vector<float> &lane_reads) {
    const int64_t T = features.frames, E = features.extended_bins;
    if (features.batch != 1 || T <= 0 || E <= 0 || notes.size() % 12 || harmonic_prior.size() != 10 ||
        features.values.size() != static_cast<size_t>(9 * T * 288) ||
        features.power.size() != static_cast<size_t>(E * T))
        throw std::runtime_error("Dettato grouping readers require aligned single-window features and notes");
    const size_t count = notes.size() / 12;
    if (taps.size() != count * 1104 || lane_reads.size() != count * 256)
        throw std::runtime_error("Dettato grouping tap/read rows differ from notes");
    GrouperInputs out;
    out.taps = taps;
    out.lanes = lane_reads;
    out.pitch.resize(count);
    out.scalars.resize(count * 11);
    out.harmonics.resize(count * 150);
    out.profile.resize(count * 36);
    out.cosine.resize(count * 4 * 22);
    out.sine.resize(out.cosine.size());
    std::vector<int64_t> a(count), b(count), lane(count);
    std::vector<int64_t> bins(count * 10);
    std::array<float, 10> prior;
    std::array<int64_t, 10> shifts;
    for (size_t h = 0; h < 10; ++h) {
        const float value = harmonic_prior[h];
        prior[h] = std::max(value, 0.0f) + std::log1p(std::exp(-std::abs(value)));
        shifts[h] = static_cast<int64_t>(std::nearbyint(36 * std::log2(static_cast<double>(h + 1))));
    }
    std::vector<float> occupancy(static_cast<size_t>(E * (T + 1)), 0);
    for (size_t i = 0; i < count; ++i) {
        const auto *note = notes.data() + i * 12;
        out.pitch[i] = std::clamp<int32_t>(static_cast<int32_t>(std::nearbyint(note[2])), 0, 127);
        lane[i] = std::clamp<int64_t>(out.pitch[i] - 21, 0, 87);
        a[i] = std::clamp<int64_t>(static_cast<int64_t>(std::nearbyint((note[0] + .5f) * 100)), 0, T - 1);
        b[i] = std::max<int64_t>(std::min<int64_t>(static_cast<int64_t>(std::nearbyint((note[1] + .5f) * 100)), T),
                                 a[i] + 1);
        out.scalars[i * 11] = std::log(std::max(note[1] - note[0], .01f));
        std::copy_n(note + 3, 9, out.scalars.begin() + i * 11 + 1);
        out.scalars[i * 11 + 10] = note[2] / 64 - 1;
        for (size_t h = 0; h < 10; ++h) {
            bins[i * 10 + h] = 3 * lane[i] + 36 + shifts[h];
            if (bins[i * 10 + h] < E)
                occupancy[static_cast<size_t>(bins[i * 10 + h] * (T + 1) + a[i])] += prior[h];
        }
        for (int64_t head = 0; head < 4; ++head)
            for (int64_t d = 0; d < 22; ++d) {
                const float frequency = 1 / std::pow(10000.0f, static_cast<float>(d) / 22);
                const float angle = (note[0] * 10) * frequency;
                const auto index = static_cast<size_t>((head * count + i) * 22 + d);
                out.cosine[index] = std::cos(angle);
                out.sine[index] = std::sin(angle);
            }
    }
    // Match the two scatter operations: all starts, then all ends.
    for (size_t i = 0; i < count; ++i)
        for (size_t h = 0; h < 10; ++h)
            if (bins[i * 10 + h] < E)
                occupancy[static_cast<size_t>(bins[i * 10 + h] * (T + 1) + b[i])] -= prior[h];
    for (int64_t bin = 0; bin < E; ++bin) {
        double sum = 0;
        for (int64_t t = 0; t < T; ++t) {
            sum += occupancy[static_cast<size_t>(bin * (T + 1) + t)];
            occupancy[static_cast<size_t>(bin * (T + 1) + t)] = static_cast<float>(sum);
        }
    }
    const std::array<int, 10> attacks{-2, 0, 2, 4, 8, 12, 16, 24, 32, 48};
    const std::array<float, 4> sustain{.25f, .5f, .75f, .95f};
    for (size_t i = 0; i < count; ++i) {
        std::array<int64_t, 14> time;
        for (size_t k = 0; k < attacks.size(); ++k)
            time[k] = std::clamp<int64_t>(a[i] + attacks[k], 0, T - 1);
        for (size_t k = 0; k < sustain.size(); ++k)
            time[10 + k] = std::clamp<int64_t>(a[i] + static_cast<int64_t>((b[i] - a[i]) * sustain[k]), 0, T - 1);
        float maximum = -INFINITY;
        for (size_t h = 0; h < 10; ++h) {
            const bool valid = bins[i * 10 + h] < E;
            const auto bin = std::clamp<int64_t>(bins[i * 10 + h], 0, E - 1);
            out.harmonics[i * 150 + 140 + h] = valid ? 1.0f : 0.0f;
            for (size_t k = 0; k < time.size(); ++k) {
                const float active = time[k] >= a[i] && time[k] < b[i] ? 1.0f : 0.0f;
                const float total = occupancy[static_cast<size_t>(bin * (T + 1) + time[k])] + prior[h] * (1 - active);
                const float share = std::min(prior[h] / std::max(total, 1e-6f), 1.0f);
                const float value = std::log(features.power[static_cast<size_t>(bin * T + time[k])] * share + 1e-7f) *
                                    (valid ? 1.0f : 0.0f);
                out.harmonics[i * 150 + h * 14 + k] = value;
                maximum = std::max(maximum, value);
            }
        }
        for (size_t j = 0; j < 140; ++j)
            out.harmonics[i * 150 + j] = std::max(out.harmonics[i * 150 + j] - maximum, -20.0f) / 10;
    }
    std::array<std::vector<std::pair<size_t, size_t>>, 288> reads;
    for (size_t i = 0; i < count; ++i)
        for (size_t offset = 0; offset < 2; ++offset)
            reads[static_cast<size_t>(std::min<int64_t>(3 * lane[i] + offset, 287))].push_back({i, offset});
    std::vector<float> sum(static_cast<size_t>(T + 1)), square(sum.size());
    for (size_t channel = 0; channel < 9; ++channel)
        for (size_t bin = 0; bin < reads.size(); ++bin) {
            if (reads[bin].empty())
                continue;
            double value_sum = 0, square_sum = 0;
            sum[0] = square[0] = 0;
            for (int64_t t = 0; t < T; ++t) {
                const float value = features.values[static_cast<size_t>((channel * T + t) * 288 + bin)];
                value_sum += value;
                square_sum += static_cast<float>(value * value);
                sum[static_cast<size_t>(t + 1)] = static_cast<float>(value_sum);
                square[static_cast<size_t>(t + 1)] = static_cast<float>(square_sum);
            }
            for (const auto &read : reads[bin]) {
                const auto i = read.first, offset = read.second;
                const float duration = static_cast<float>(b[i] - a[i]);
                const float mean = (sum[static_cast<size_t>(b[i])] - sum[static_cast<size_t>(a[i])]) / duration;
                const float second = (square[static_cast<size_t>(b[i])] - square[static_cast<size_t>(a[i])]) / duration;
                out.profile[i * 36 + offset * 18 + channel] = mean;
                out.profile[i * 36 + offset * 18 + 9 + channel] =
                    std::sqrt(std::max(second - mean * mean, 0.0f) + 1e-4f);
            }
        }
    return out;
}

struct WindowGrouper::Impl {
    HarmonicFrontend frontend;
    std::vector<float> prior;
    GroupingDecoder decoder;
    GrouperNetwork network;
    double window_seconds, hop;
    size_t maximum_notes;

    Impl(GrouperWeights weights, const io::json::Value &config, const core::ExecutionContext &execution)
        : frontend(std::move(weights.frontend), execution.config().threads), prior(std::move(weights.harmonic_prior)),
          decoder(std::move(weights.decoder), config), network(std::move(weights), execution),
          window_seconds(config.require("window_s").as_number()),
          hop(window_seconds * config.require("window_hop").as_number()),
          maximum_notes(static_cast<size_t>(config.require("max_notes").as_i64())) {
        if (!(window_seconds > 0) || !(hop > 0) || maximum_notes < 4 || maximum_notes > 768)
            throw std::invalid_argument("Dettato grouper requires positive windows/hops and 4..768 notes per window");
    }
};

WindowGrouper::WindowGrouper(GrouperWeights weights, const io::json::Value &config,
                             const core::ExecutionContext &execution)
    : impl_(std::make_unique<Impl>(std::move(weights), config, execution)) {}
WindowGrouper::~WindowGrouper() = default;

std::vector<Track> WindowGrouper::group(const std::vector<float> &audio16, const WindowTranscription &transcription,
                                        const std::vector<Note> &notes, const NoteThresholds &thresholds,
                                        double time_ratio) {
    const auto features = collect_grouping_features(transcription, notes, thresholds);
    // Canonical F32 note rows define the same ordering and confidence source
    // for neural grouping and the host track decoder.
    std::vector<Note> grouped_notes(features.notes.size() / 12);
    for (size_t i = 0; i < grouped_notes.size(); ++i) {
        const auto *row = features.notes.data() + i * 12;
        auto &note = grouped_notes[i];
        note.onset = row[0];
        note.offset = row[1];
        note.pitch = static_cast<int>(row[2]);
        note.confidence = row[row[3] > .5f ? 5 : 4];
    }
    auto tracks = impl_->decoder.decode(predict(audio16, features), grouped_notes, time_ratio);
    for (auto &track : tracks)
        std::replace(track.name.begin(), track.name.end(), '_', ' ');
    return tracks;
}

GroupingPrediction WindowGrouper::predict(const std::vector<float> &audio16, const GroupingFeatures &features) {
    if (features.notes.size() % 12 || features.taps.size() != features.notes.size() / 12 * 1104)
        throw std::invalid_argument("Dettato grouping requires aligned note and tap rows");
    const auto start = std::chrono::steady_clock::now();
    const size_t count = features.notes.size() / 12;
    auto &state = *impl_;
    GroupingPrediction out;
    out.window_seconds = state.window_seconds;
    out.embeddings.resize(count * 32);
    out.probabilities.resize(count * 73);
    out.hidden.resize(count * 176);
    if (count == 0)
        return out;
    std::vector<double> onsets(count);
    for (size_t i = 0; i < count; ++i)
        onsets[i] = features.notes[i * 12];
    if (!std::is_sorted(onsets.begin(), onsets.end()))
        throw std::invalid_argument("Dettato grouping notes must be sorted by onset");
    // Quantize once at the song boundary, matching _Song's PCM16 storage.
    std::vector<int16_t> pcm(audio16.size());
    for (size_t i = 0; i < pcm.size(); ++i)
        pcm[i] = static_cast<int16_t>(std::clamp(std::nearbyint(audio16[i] * 32768.f), -32768.f, 32767.f));
    const size_t samples = static_cast<size_t>(std::nearbyint((state.window_seconds + 1.5) * 16000));
    std::vector<float> notes, taps;
    double frontend_ms = 0;
    std::vector<double> mass(count * 73), embeddings(count * 32), weights(count), best(count, -1);
    const auto prepare_frontend = [&](double t0) {
        std::vector<float> audio(samples, 0);
        const int64_t first = static_cast<int64_t>(std::nearbyint((t0 - .5) * 16000));
        const int64_t lo = std::max<int64_t>(first, 0);
        const int64_t hi = std::min(first + static_cast<int64_t>(samples), static_cast<int64_t>(pcm.size()));
        for (int64_t i = lo; i < hi; ++i)
            audio[static_cast<size_t>(i - first)] = pcm[static_cast<size_t>(i)] / 32768.f;
        const auto frontend_start = std::chrono::steady_clock::now();
        auto front = state.frontend.extract(audio, 1, true);
        return std::make_pair(std::move(front), debug::elapsed_ms(frontend_start));
    };
    // One frontend job at a time; it owns its next-window feature arrays.
    // Current graph consumers read only the already-returned private features.
    std::future<std::pair<HarmonicFeatures, double>> next_features;
    for (double t0 = 0;;) {
        const size_t a = static_cast<size_t>(std::lower_bound(onsets.begin(), onsets.end(), t0) - onsets.begin());
        const size_t b = static_cast<size_t>(std::lower_bound(onsets.begin(), onsets.end(), t0 + state.window_seconds) -
                                             onsets.begin());
        const size_t end = std::min(b, a + state.maximum_notes);
        const bool done = b >= count && a + state.maximum_notes >= b;
        const double next_t0 = b - a > state.maximum_notes
                                   ? std::max(t0 + .25 * state.hop, onsets[a + state.maximum_notes / 2])
                                   : t0 + state.hop;
        if (end > a) {
            const size_t real = end - a, used = std::max<size_t>(real, 4);
            notes.resize(used * 12);
            taps.resize(used * 1104);
            for (size_t i = 0; i < used; ++i) {
                const size_t source = a + i % real;
                std::copy_n(features.notes.data() + source * 12, 12, notes.data() + i * 12);
                notes[i * 12] -= static_cast<float>(t0);
                notes[i * 12 + 1] =
                    std::min(notes[i * 12 + 1] - static_cast<float>(t0), static_cast<float>(state.window_seconds + 1));
                for (size_t c = 0; c < 1104; ++c)
                    taps[i * 1104 + c] = ggml_fp16_to_fp32(features.taps[source * 1104 + c]);
            }
            auto prepared = next_features.valid() ? next_features.get() : prepare_frontend(t0);
            // Avoid preparing an empty window. The next iteration may advance
            // across silence without ever invoking its neural graphs.
            const auto next_note = std::lower_bound(onsets.begin(), onsets.end(), next_t0);
            if (!done && next_t0 <= onsets.back() && next_note != onsets.end() &&
                *next_note < next_t0 + state.window_seconds)
                next_features = std::async(std::launch::async, prepare_frontend, next_t0);
            frontend_ms += prepared.second;
            const auto &front = prepared.first;
            const auto lanes = state.network.lane_features(front, notes);
            const auto input = make_grouper_inputs(front, notes, taps, state.prior, lanes);
            const auto output = state.network.run(input);
            GroupingWindow window;
            window.start = t0;
            window.embeddings.resize(real * 32);
            window.probabilities.resize(real * 73);
            window.notes.reserve(real);
            for (size_t i = 0; i < real; ++i) {
                const size_t id = a + i;
                window.notes.push_back(static_cast<int>(id));
                const double weight =
                    std::clamp(1. - std::abs((onsets[id] - t0) / state.window_seconds - .5) * 2., .05, 1.);
                weights[id] += weight;
                for (size_t c = 0; c < 73; ++c) {
                    mass[id * 73 + c] += output[4][i * 73 + c] * weight;
                    window.probabilities[i * 73 + c] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(output[4][i * 73 + c]));
                }
                for (size_t c = 0; c < 32; ++c) {
                    embeddings[id * 32 + c] += output[1][i * 32 + c] * weight;
                    window.embeddings[i * 32 + c] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(output[1][i * 32 + c]));
                }
                if (weight > best[id]) {
                    best[id] = weight;
                    for (size_t c = 0; c < 176; ++c)
                        out.hidden[id * 176 + c] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(output[3][i * 176 + c]));
                }
            }
            out.windows.push_back(std::move(window));
        }
        if (done)
            break;
        t0 = next_t0;
        if (t0 > onsets.back())
            break;
    }
    for (size_t i = 0; i < count; ++i) {
        double norm = 0;
        for (size_t c = 0; c < 32; ++c)
            norm += embeddings[i * 32 + c] * embeddings[i * 32 + c];
        norm = std::max(std::sqrt(norm), 1e-9);
        for (size_t c = 0; c < 32; ++c)
            out.embeddings[i * 32 + c] = static_cast<float>(embeddings[i * 32 + c] / norm);
        for (size_t c = 0; c < 73; ++c)
            out.probabilities[i * 73 + c] = static_cast<float>(mass[i * 73 + c] / std::max(weights[i], 1e-9));
    }
    debug::timing_log_scalar("dettato.grouper.frontend_ms", frontend_ms);
    debug::timing_log_scalar("dettato.grouper.windows", out.windows.size());
    debug::timing_log_scalar("dettato.grouper.wall_ms", debug::elapsed_ms(start));
    return out;
}

namespace {
std::vector<std::pair<int, int>> linear_sum_assignment(const std::vector<double> &cost_in, int nr, int nc) {
    // Port of SciPy's scipy/optimize/rectangular_lsap/rectangular_lsap.cpp (Crouse's shortest
    // augmenting path; BSD 3-Clause, notice above).
    std::vector<std::pair<int, int>> out;
    if (nr < 0 || nc < 0 || cost_in.size() != static_cast<size_t>(nr) * nc)
        throw std::invalid_argument("linear_sum_assignment: cost size");
    if (nr == 0 || nc == 0)
        return out;
    const bool transpose = nc < nr;
    std::vector<double> cost(cost_in);
    if (transpose) {
        for (int i = 0; i < nr; ++i)
            for (int j = 0; j < nc; ++j)
                cost[static_cast<size_t>(j) * nr + i] = cost_in[static_cast<size_t>(i) * nc + j];
        std::swap(nr, nc);
    }
    for (double v : cost)
        if (std::isnan(v) || v == -std::numeric_limits<double>::infinity())
            throw std::invalid_argument("matrix contains invalid numeric entries");
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> u(nr, 0.0), v(nc, 0.0), shortest(nc);
    std::vector<int> path(nc, -1), col4row(nr, -1), row4col(nc, -1), remaining(nc);
    std::vector<char> SR(nr), SC(nc);
    for (int cur = 0; cur < nr; ++cur) {
        double min_val = 0.0;
        int i = cur, sink = -1, num_remaining = nc;
        for (int it = 0; it < nc; ++it)
            remaining[it] = nc - it - 1;
        std::fill(SR.begin(), SR.end(), 0);
        std::fill(SC.begin(), SC.end(), 0);
        std::fill(shortest.begin(), shortest.end(), inf);
        while (sink == -1) {
            int index = -1;
            double lowest = inf;
            SR[i] = 1;
            for (int it = 0; it < num_remaining; ++it) {
                const int j = remaining[it];
                const double r = min_val + cost[static_cast<size_t>(i) * nc + j] - u[i] - v[j];
                if (r < shortest[j]) {
                    path[j] = i;
                    shortest[j] = r;
                }
                if (shortest[j] < lowest || (shortest[j] == lowest && row4col[j] == -1)) {
                    lowest = shortest[j];
                    index = it;
                }
            }
            min_val = lowest;
            if (min_val == inf)
                throw std::invalid_argument("cost matrix is infeasible");
            const int j = remaining[index];
            if (row4col[j] == -1)
                sink = j;
            else
                i = row4col[j];
            SC[j] = 1;
            remaining[index] = remaining[--num_remaining];
        }
        u[cur] += min_val;
        for (int r = 0; r < nr; ++r)
            if (SR[r] && r != cur)
                u[r] += min_val - shortest[col4row[r]];
        for (int c = 0; c < nc; ++c)
            if (SC[c])
                v[c] -= min_val - shortest[c];
        for (int j = sink;;) {
            const int r = path[j];
            row4col[j] = r;
            std::swap(col4row[r], j);
            if (r == cur)
                break;
        }
    }
    if (transpose) {
        // argsort of col4row: original rows ascending.
        std::vector<int> order(nr);
        for (int k = 0; k < nr; ++k)
            order[k] = k;
        std::sort(order.begin(), order.end(), [&](int a, int b) { return col4row[a] < col4row[b]; });
        for (int k : order)
            out.emplace_back(col4row[k], k);
    } else {
        for (int r = 0; r < nr; ++r)
            out.emplace_back(r, col4row[r]);
    }
    return out;
}

constexpr int embedding_channels = 32;
constexpr int instrument_count = 73;

// Nearest-neighbour-chain average linkage, followed by SciPy's stable merge
// ordering and distance-cut traversal. Unlike scanning every remaining pair
// after every merge, this requires quadratic work for a dense note window.
std::vector<int> cluster_window(const GroupingWindow &window, double threshold, float lambda) {
    const int n = static_cast<int>(window.notes.size());
    if (n == 1)
        return {0};
    std::vector<float> probabilities = window.probabilities;
    for (int i = 0; i < n; ++i) {
        float norm = 0;
        for (int c = 0; c < instrument_count; ++c) {
            const auto p = probabilities[i * instrument_count + c];
            norm += p * p;
        }
        norm = std::max(std::sqrt(norm), 1e-9f);
        for (int c = 0; c < instrument_count; ++c)
            probabilities[i * instrument_count + c] /= norm;
    }
    std::vector<double> distances(static_cast<size_t>(n) * n);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            float e = 0, p = 0;
            for (int c = 0; c < embedding_channels; ++c)
                e += window.embeddings[i * embedding_channels + c] * window.embeddings[j * embedding_channels + c];
            for (int c = 0; c < instrument_count; ++c)
                p += probabilities[i * instrument_count + c] * probabilities[j * instrument_count + c];
            const float d = std::max(0.f, (1.f - e) + (lambda > 0 ? lambda * (1.f - p) : 0.f));
            distances[i * n + j] = distances[j * n + i] = d;
        }
    struct Merge {
        int left, right, size;
        double distance;
    };
    std::vector<Merge> merges;
    std::vector<int> size(n, 1), chain;
    for (int k = 0; k < n - 1; ++k) {
        if (chain.empty()) {
            int first = 0;
            while (!size[first])
                ++first;
            chain.push_back(first);
        }
        int x, y;
        double minimum;
        while (true) {
            x = chain.back();
            y = chain.size() > 1 ? chain[chain.size() - 2] : -1;
            minimum = y >= 0 ? distances[x * n + y] : std::numeric_limits<double>::infinity();
            for (int i = 0; i < n; ++i)
                if (i != x && size[i] && distances[x * n + i] < minimum) {
                    minimum = distances[x * n + i];
                    y = i;
                }
            if (chain.size() > 1 && y == chain[chain.size() - 2])
                break;
            chain.push_back(y);
        }
        chain.resize(chain.size() - 2);
        if (x > y)
            std::swap(x, y);
        const int nx = size[x], ny = size[y];
        merges.push_back({x, y, nx + ny, minimum});
        size[x] = 0;
        size[y] = nx + ny;
        for (int i = 0; i < n; ++i)
            if (i != y && size[i])
                distances[i * n + y] = distances[y * n + i] =
                    (nx * distances[i * n + x] + ny * distances[i * n + y]) / (nx + ny);
    }
    std::stable_sort(merges.begin(), merges.end(),
                     [](const auto &a, const auto &b) { return a.distance < b.distance; });
    std::vector<int> parent(2 * n - 1);
    std::iota(parent.begin(), parent.end(), 0);
    const auto root = [&](int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    std::vector<double> maximum(n - 1);
    for (int k = 0; k < n - 1; ++k) {
        auto &m = merges[k];
        m.left = root(m.left);
        m.right = root(m.right);
        if (m.left > m.right)
            std::swap(m.left, m.right);
        parent[m.left] = parent[m.right] = n + k;
        maximum[k] = m.distance;
        if (m.left >= n)
            maximum[k] = std::max(maximum[k], maximum[m.left - n]);
        if (m.right >= n)
            maximum[k] = std::max(maximum[k], maximum[m.right - n]);
    }
    std::vector<int> labels(n, -1);
    int next = 0;
    std::function<void(int, int)> visit = [&](int node, int active) {
        const auto &m = merges[node - n];
        if (active < 0 && maximum[node - n] <= threshold)
            active = next++;
        if (m.left >= n)
            visit(m.left, active);
        if (m.right >= n)
            visit(m.right, active);
        if (m.left < n)
            labels[m.left] = active < 0 ? next++ : active;
        if (m.right < n)
            labels[m.right] = active < 0 ? next++ : active;
    };
    visit(2 * n - 2, -1);
    return labels;
}
} // namespace

std::vector<int> chain_grouping_windows(const std::vector<GroupingWindow> &windows, const std::vector<double> &onsets,
                                        double window_seconds, double threshold, float lambda) {
    if (!(window_seconds > 0))
        throw std::invalid_argument("Dettato grouping window duration must be positive");
    std::vector<int> track(onsets.size(), -1), owner(onsets.size(), -1);
    std::vector<double> best(onsets.size(), -1);
    std::vector<size_t> order(windows.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b) { return windows[a].start < windows[b].start; });
    int global = 0;
    for (const auto wi : order) {
        const auto &w = windows[wi];
        if (w.embeddings.size() != w.notes.size() * embedding_channels ||
            w.probabilities.size() != w.notes.size() * instrument_count)
            throw std::invalid_argument("Dettato grouping window feature shape mismatch");
        if (w.notes.empty())
            continue;
        for (const int id : w.notes)
            if (id < 0 || static_cast<size_t>(id) >= onsets.size())
                throw std::invalid_argument("Dettato grouping window note index out of range");
        const auto labels = cluster_window(w, threshold, lambda);
        const int count = *std::max_element(labels.begin(), labels.end()) + 1;
        std::vector<int> mapping(count, -1), previous;
        for (const int id : w.notes)
            if (owner[id] >= 0)
                previous.push_back(owner[id]);
        std::sort(previous.begin(), previous.end());
        previous.erase(std::unique(previous.begin(), previous.end()), previous.end());
        if (!previous.empty()) {
            std::vector<double> costs(static_cast<size_t>(count) * previous.size(), 0);
            for (size_t i = 0; i < w.notes.size(); ++i)
                if (owner[w.notes[i]] >= 0) {
                    const auto col =
                        std::lower_bound(previous.begin(), previous.end(), owner[w.notes[i]]) - previous.begin();
                    costs[labels[i] * previous.size() + col] -= 1;
                }
            for (const auto &pair : linear_sum_assignment(costs, count, static_cast<int>(previous.size())))
                if (costs[pair.first * previous.size() + pair.second] < 0)
                    mapping[pair.first] = previous[pair.second];
        }
        for (auto &id : mapping)
            if (id < 0)
                id = global++;
        for (size_t i = 0; i < w.notes.size(); ++i) {
            const int id = w.notes[i], assigned = mapping[labels[i]];
            owner[id] = assigned;
            const double weight = std::clamp(1. - std::abs((onsets[id] - w.start) / window_seconds - .5) * 2., .05, 1.);
            if (weight > best[id]) {
                track[id] = assigned;
                best[id] = weight;
            }
        }
    }
    return track;
}

std::vector<int> merge_grouping_tracklets(const std::vector<int> &tracklets, const std::vector<float> &embeddings,
                                          const std::vector<float> &probabilities, double threshold, double lambda) {
    if (embeddings.size() != tracklets.size() * embedding_channels ||
        probabilities.size() != tracklets.size() * instrument_count)
        throw std::invalid_argument("Dettato song grouping feature shape mismatch");
    std::vector<int> ids;
    for (const auto id : tracklets)
        if (id >= 0)
            ids.push_back(id);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    const int count = static_cast<int>(ids.size());
    if (count < 2)
        return tracklets;
    std::vector<double> sizes(count, 0), e(static_cast<size_t>(count) * embedding_channels),
        p(static_cast<size_t>(count) * instrument_count);
    std::vector<int> inverse(tracklets.size(), -1), labels(count);
    std::iota(labels.begin(), labels.end(), 0);
    for (size_t i = 0; i < tracklets.size(); ++i)
        if (tracklets[i] >= 0) {
            const int t = static_cast<int>(std::lower_bound(ids.begin(), ids.end(), tracklets[i]) - ids.begin());
            inverse[i] = t;
            sizes[t] += 1;
            double norm = 0;
            for (int c = 0; c < instrument_count; ++c) {
                const double x = probabilities[i * instrument_count + c];
                norm += x * x;
            }
            norm = std::max(std::sqrt(norm), 1e-9);
            for (int c = 0; c < embedding_channels; ++c)
                e[t * embedding_channels + c] += embeddings[i * embedding_channels + c];
            for (int c = 0; c < instrument_count; ++c)
                p[t * instrument_count + c] += probabilities[i * instrument_count + c] / norm;
        }
    for (int t = 0; t < count; ++t) {
        for (int c = 0; c < embedding_channels; ++c)
            e[t * embedding_channels + c] /= sizes[t];
        for (int c = 0; c < instrument_count; ++c)
            p[t * instrument_count + c] /= sizes[t];
    }
    const auto distance = [&](int i, int j) {
        double ed = 0, pd = 0;
        for (int c = 0; c < embedding_channels; ++c)
            ed += e[i * embedding_channels + c] * e[j * embedding_channels + c];
        for (int c = 0; c < instrument_count; ++c)
            pd += p[i * instrument_count + c] * p[j * instrument_count + c];
        return 1. - ed + (lambda > 0 ? lambda * (1. - pd) : 0.);
    };
    const auto infinity = std::numeric_limits<double>::infinity();
    std::vector<double> distances(static_cast<size_t>(count) * count, infinity);
    std::vector<bool> alive(count, true), eligible(count, true);
    if (count > 600) {
        std::vector<int> largest(count);
        std::iota(largest.begin(), largest.end(), 0);
        std::stable_sort(largest.begin(), largest.end(), [&](int a, int b) { return sizes[a] > sizes[b]; });
        for (int i = 600; i < count; ++i)
            eligible[largest[i]] = false;
    }
    for (int i = 0; i < count; ++i)
        for (int j = i + 1; j < count; ++j)
            if (eligible[i] && eligible[j])
                distances[i * count + j] = distances[j * count + i] = distance(i, j);
    while (true) {
        const auto found = std::min_element(distances.begin(), distances.end());
        if (!std::isfinite(*found) || *found > threshold)
            break;
        const int i = static_cast<int>((found - distances.begin()) / count),
                  j = static_cast<int>((found - distances.begin()) % count);
        const double si = sizes[i], sj = sizes[j], total = si + sj;
        for (int c = 0; c < embedding_channels; ++c)
            e[i * embedding_channels + c] =
                (si * e[i * embedding_channels + c] + sj * e[j * embedding_channels + c]) / total;
        for (int c = 0; c < instrument_count; ++c)
            p[i * instrument_count + c] = (si * p[i * instrument_count + c] + sj * p[j * instrument_count + c]) / total;
        sizes[i] = total;
        alive[j] = false;
        for (auto &label : labels)
            if (label == j)
                label = i;
        for (int k = 0; k < count; ++k) {
            distances[j * count + k] = distances[k * count + j] = infinity;
            distances[i * count + k] = distances[k * count + i] = alive[k] && k != i ? distance(i, k) : infinity;
        }
    }
    auto result = inverse;
    for (auto &label : result)
        if (label >= 0)
            label = labels[label];
    return result;
}

GroupingDecoder::GroupingDecoder(GroupingDecoderWeights weights, const io::json::Value &config)
    : head_(std::move(weights.head)), credit_(std::move(weights.credit)) {
    const auto &decode = config.require("decode");
    threshold_ = decode.require("tau").as_number();
    merge_threshold_ = decode.require("tau2").as_number();
    lambda_ = decode.require("lam").as_number();
    alpha_ = decode.require("alpha").as_number();
    absorb_ = decode.require("absorb").as_number();
    absorb_fraction_ = decode.require("absorb_frac").as_number();
    minimum_notes_ = decode.require("min_notes").as_i64();
    refine_ = decode.require("refine").as_i64();
    expected_credit_ = decode.require("ec").as_bool();
    merge_ = decode.require("merge").as_bool();
    for (const auto &name : config.require("instruments").as_array())
        names_.push_back(name.as_string());
    for (const auto &program : config.require("programs").as_array())
        programs_.push_back(static_cast<int>(program.as_i64()));
    if (names_.size() != instrument_count || programs_.size() != instrument_count)
        throw std::invalid_argument("Dettato grouping instrument table must contain 73 entries");
}

std::vector<Track> GroupingDecoder::decode(const GroupingPrediction &prediction, const std::vector<Note> &notes,
                                           double time_ratio) const {
    const size_t n = notes.size();
    if (prediction.embeddings.size() != n * embedding_channels ||
        prediction.probabilities.size() != n * instrument_count || prediction.hidden.size() != n * 176)
        throw std::invalid_argument("Dettato grouping prediction shape mismatch");
    std::vector<double> onsets;
    onsets.reserve(n);
    for (const auto &note : notes)
        onsets.push_back(note.onset);
    auto track = chain_grouping_windows(prediction.windows, onsets, prediction.window_seconds, threshold_,
                                        static_cast<float>(lambda_));
    track = merge_grouping_tracklets(track, prediction.embeddings, prediction.probabilities, merge_threshold_, lambda_);
    std::vector<size_t> selected;
    std::vector<int> ids;
    for (size_t i = 0; i < n; ++i)
        if (track[i] >= 0) {
            selected.push_back(i);
            ids.push_back(track[i]);
        }
    if (selected.empty())
        return {};
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    const int count = static_cast<int>(ids.size());
    std::vector<int> labels(selected.size());
    std::vector<double> e(selected.size() * embedding_channels), p(selected.size() * instrument_count);
    for (size_t i = 0; i < selected.size(); ++i) {
        labels[i] = static_cast<int>(std::lower_bound(ids.begin(), ids.end(), track[selected[i]]) - ids.begin());
        for (int c = 0; c < embedding_channels; ++c)
            e[i * embedding_channels + c] = prediction.embeddings[selected[i] * embedding_channels + c];
        for (int c = 0; c < instrument_count; ++c)
            p[i * instrument_count + c] = prediction.probabilities[selected[i] * instrument_count + c];
    }
    std::vector<int> sizes(count);
    std::vector<double> centres(static_cast<size_t>(count) * embedding_channels),
        mass(static_cast<size_t>(count) * instrument_count);
    const auto accumulate = [&] {
        std::fill(sizes.begin(), sizes.end(), 0);
        std::fill(centres.begin(), centres.end(), 0);
        std::fill(mass.begin(), mass.end(), 0);
        for (size_t i = 0; i < selected.size(); ++i) {
            const int t = labels[i];
            ++sizes[t];
            for (int c = 0; c < embedding_channels; ++c)
                centres[t * embedding_channels + c] += e[i * embedding_channels + c];
            for (int c = 0; c < instrument_count; ++c)
                mass[t * instrument_count + c] += p[i * instrument_count + c];
        }
    };
    const auto normalize = [](std::vector<double> &values, int width) {
        for (size_t row = 0; row < values.size(); row += width) {
            double norm = 0;
            for (int c = 0; c < width; ++c)
                norm += values[row + c] * values[row + c];
            norm = std::max(std::sqrt(norm), 1e-9);
            for (int c = 0; c < width; ++c)
                values[row + c] /= norm;
        }
    };
    auto pn = p;
    normalize(pn, instrument_count);
    accumulate();
    for (int pass = 0; pass < refine_; ++pass) {
        std::vector<int> big;
        for (int t = 0; t < count; ++t)
            if (sizes[t] >= minimum_notes_)
                big.push_back(t);
        if (big.size() < 2)
            break;
        auto ce = centres, cp = mass;
        normalize(ce, embedding_channels);
        normalize(cp, instrument_count);
        for (size_t i = 0; i < selected.size(); ++i) {
            double best = -std::numeric_limits<double>::infinity();
            int chosen = -1;
            for (const auto t : big) {
                double score = 0, probability = 0;
                for (int c = 0; c < embedding_channels; ++c)
                    score += e[i * embedding_channels + c] * ce[t * embedding_channels + c];
                for (int c = 0; c < instrument_count; ++c)
                    probability += pn[i * instrument_count + c] * cp[t * instrument_count + c];
                score += lambda_ > 0 ? lambda_ * probability : 0;
                if (score > best) {
                    best = score;
                    chosen = t;
                }
            }
            labels[i] = chosen;
        }
        accumulate();
    }
    std::vector<int> big;
    for (int t = 0; t < count; ++t)
        if (sizes[t] >= minimum_notes_)
            big.push_back(t);
    if (big.empty())
        big.push_back(static_cast<int>(std::max_element(sizes.begin(), sizes.end()) - sizes.begin()));
    auto normalized_mass = mass;
    normalize(normalized_mass, instrument_count);
    for (int t = 0; t < count; ++t)
        if (sizes[t] && std::find(big.begin(), big.end(), t) == big.end()) {
            double best = -std::numeric_limits<double>::infinity();
            int chosen = -1;
            for (const auto b : big) {
                double score = 0;
                for (int c = 0; c < instrument_count; ++c)
                    score += normalized_mass[t * instrument_count + c] * normalized_mass[b * instrument_count + c];
                if (score > best) {
                    best = score;
                    chosen = b;
                }
            }
            for (auto &label : labels)
                if (label == t)
                    label = chosen;
        }
    if (absorb_ > 0 && absorb_fraction_ > 0) {
        auto unit = e;
        normalize(unit, embedding_channels);
        while (true) {
            accumulate();
            std::vector<int> order;
            for (int t = 0; t < count; ++t)
                if (sizes[t])
                    order.push_back(t);
            if (order.size() < 2)
                break;
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return sizes[a] < sizes[b]; });
            // Absorption uses means of per-note unit embeddings, not normalized
            // mean directions; only the final note reassignment uses directions.
            std::fill(centres.begin(), centres.end(), 0);
            for (size_t i = 0; i < selected.size(); ++i)
                for (int c = 0; c < embedding_channels; ++c)
                    centres[labels[i] * embedding_channels + c] += unit[i * embedding_channels + c];
            for (int t = 0; t < count; ++t)
                if (sizes[t])
                    for (int c = 0; c < embedding_channels; ++c)
                        centres[t * embedding_channels + c] /= sizes[t];
            bool changed = false;
            for (const auto t : order) {
                const double share = static_cast<double>(sizes[t]) / selected.size();
                if (share >= absorb_fraction_)
                    break;
                std::vector<int> larger;
                for (int b = 0; b < count; ++b)
                    if (sizes[b] > sizes[t])
                        larger.push_back(b);
                bool dissolve = false;
                for (const auto b : larger) {
                    double dot = 0;
                    for (int c = 0; c < embedding_channels; ++c)
                        dot += centres[t * embedding_channels + c] * centres[b * embedding_channels + c];
                    if (1. - dot < absorb_ * (1. - share / absorb_fraction_))
                        dissolve = true;
                }
                if (!dissolve)
                    continue;
                auto directions = centres;
                normalize(directions, embedding_channels);
                for (size_t i = 0; i < selected.size(); ++i)
                    if (labels[i] == t) {
                        double best = -std::numeric_limits<double>::infinity();
                        int chosen = -1;
                        for (const auto b : larger) {
                            double score = 0;
                            for (int c = 0; c < embedding_channels; ++c)
                                score += unit[i * embedding_channels + c] * directions[b * embedding_channels + c];
                            if (score > best) {
                                best = score;
                                chosen = b;
                            }
                        }
                        labels[i] = chosen;
                    }
                changed = true;
                break;
            }
            if (!changed)
                break;
        }
    }
    std::vector<std::pair<int, std::vector<size_t>>> groups;
    for (int t = 0; t < count; ++t) {
        std::vector<size_t> members;
        for (size_t i = 0; i < selected.size(); ++i)
            if (labels[i] == t)
                members.push_back(selected[i]);
        if (members.empty())
            continue;
        std::vector<double> logits(instrument_count, 0);
        for (const auto i : members)
            for (int c = 0; c < instrument_count; ++c)
                logits[c] += prediction.probabilities[i * instrument_count + c];
        const double total = std::accumulate(logits.begin(), logits.end(), 0.);
        for (auto &x : logits)
            x = std::log(std::max(x / total, 1e-9));
        if (alpha_ > 0) {
            std::vector<double> state(182, 0);
            double pitch_mean = 0, pitch_square = 0, low = std::numeric_limits<double>::infinity(), high = -low,
                   polyphony = 0;
            for (const auto i : members) {
                for (int c = 0; c < 176; ++c)
                    state[c] += prediction.hidden[i * 176 + c];
                const double pitch = notes[i].pitch / 64. - 1.;
                pitch_mean += pitch;
                pitch_square += pitch * pitch;
                low = std::min(low, pitch);
                high = std::max(high, pitch);
                if (members.size() <= 3000) {
                    int overlaps = 0;
                    for (const auto j : members)
                        if (std::min(std::max(notes[i].offset, notes[i].onset + .02),
                                     std::max(notes[j].offset, notes[j].onset + .02)) >
                            std::max(notes[i].onset, notes[j].onset) + .03)
                            ++overlaps;
                    if (overlaps >= 2)
                        polyphony += 1;
                }
            }
            for (int c = 0; c < 176; ++c)
                state[c] /= members.size();
            pitch_mean /= members.size();
            state[176] = pitch_mean;
            state[177] = std::sqrt(std::max(0., pitch_square / members.size() - pitch_mean * pitch_mean));
            state[178] = low;
            state[179] = high;
            state[180] = std::log1p(members.size()) / 5.;
            state[181] = polyphony / members.size();
            // Upstream deliberately evaluates this small track classifier in F64
            // on the host, outside its neural graph/autocast context.
            for (size_t l = 0; l < head_.size(); ++l) {
                const auto &layer = head_[l];
                std::vector<double> next(layer.output);
                for (int row = 0; row < layer.output; ++row) {
                    double x = layer.bias[row];
                    for (int c = 0; c < layer.input; ++c)
                        x += layer.weight[row * layer.input + c] * state[c];
                    next[row] = l + 1 < head_.size() ? .5 * x * (1. + std::erf(x / std::sqrt(2.))) : x;
                }
                state = std::move(next);
            }
            const double maximum = *std::max_element(state.begin(), state.end());
            double sum = 0;
            for (const auto x : state)
                sum += std::exp(x - maximum);
            const double log_sum = std::log(sum);
            for (int c = 0; c < instrument_count; ++c)
                logits[c] = alpha_ * (state[c] - maximum - log_sum) + (1. - alpha_) * logits[c];
        }
        int instrument;
        if (expected_credit_) {
            const double maximum = *std::max_element(logits.begin(), logits.end());
            double total_probability = 0;
            for (auto &x : logits) {
                x = std::exp(x - maximum);
                total_probability += x;
            }
            std::vector<double> score(instrument_count, 0);
            for (int row = 0; row < instrument_count; ++row)
                for (int c = 0; c < instrument_count; ++c)
                    score[row] += credit_[row * instrument_count + c] * logits[c] / total_probability;
            instrument = static_cast<int>(std::max_element(score.begin(), score.end()) - score.begin());
        } else
            instrument = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        auto found = std::find_if(groups.begin(), groups.end(),
                                  [&](const auto &group) { return merge_ && group.first == instrument; });
        if (found == groups.end())
            groups.emplace_back(instrument, std::move(members));
        else {
            found->second.insert(found->second.end(), members.begin(), members.end());
            std::sort(found->second.begin(), found->second.end());
        }
    }
    std::stable_sort(groups.begin(), groups.end(),
                     [](const auto &a, const auto &b) { return a.second.size() > b.second.size(); });
    std::vector<Track> tracks;
    for (auto &group : groups) {
        auto &members = group.second;
        std::stable_sort(members.begin(), members.end(), [&](size_t a, size_t b) {
            return notes[a].pitch == notes[b].pitch ? notes[a].onset < notes[b].onset : notes[a].pitch < notes[b].pitch;
        });
        std::vector<size_t> kept;
        for (const auto i : members)
            if (kept.empty() || notes[i].pitch != notes[kept.back()].pitch ||
                (notes[i].onset - notes[kept.back()].onset) * time_ratio > .03)
                kept.push_back(i);
        std::stable_sort(kept.begin(), kept.end(), [&](size_t a, size_t b) { return notes[a].onset < notes[b].onset; });
        Track out;
        out.name = names_[group.first];
        out.program = programs_[group.first];
        for (const auto i : kept) {
            auto note = notes[i];
            note.onset *= time_ratio;
            note.offset = std::max(note.offset * time_ratio, note.onset + .02);
            note.velocity = static_cast<int>(std::clamp(20. + 107. * note.confidence, 1., 127.));
            out.notes.push_back(note);
        }
        tracks.push_back(std::move(out));
    }
    return tracks;
}
} // namespace engine::community_models::dettato::detail
