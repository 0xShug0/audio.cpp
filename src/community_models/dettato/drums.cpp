#include "engine/community_models/dettato/drums.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/community_models/dettato/gru.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace engine::community_models::dettato::detail {

using namespace core;
using namespace modules;

DrumEmbedder::DrumEmbedder(DrumEmbedderWeights weights, const io::json::Value &config,
                           const ExecutionContext &execution, GraphWorkspace *workspace)
    : execution_(execution), workspace_(workspace), weights_(std::move(weights)),
      activation_(config.require("act").as_string()), width_(config.require("w").as_i64()),
      band_hidden_(config.require("collapse_m").as_i64()) {}

void DrumEmbedder::prepare(const std::vector<int64_t> &shapes) {
    if (graph_ && graph_->graph() && shapes_ == shapes)
        return;
    graph_.reset();
    shapes_ = shapes;
    ports_.clear();
    ports_.resize(shapes.size());
    graph_ = std::make_unique<PreparedGraph>(
        execution_, PreparedGraphConfig{"dettato.drums.embed", 8 * 1024 * 1024, 8192, 1024 * 1024});
    for (size_t shape = 0; shape < shapes.size(); ++shape) {
        const auto frames = shapes[shape];
        auto &ports = ports_[shape];
        const auto id = shape == 0 ? 0 : graph_->add_graph("window_" + std::to_string(frames));
        auto &ctx = graph_->build_context();
        ports.input = make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 1, frames, 160}));
        ggml_set_input(ports.input.tensor);
        auto x = ScaleBiasModule({1.f / 3, 0}).build(ctx, ScaleBiasModule({1, 5}).build(ctx, ports.input));
        const auto activate = [&](const TensorValue &value) {
            if (activation_ == "silu")
                return SiluModule{}.build(ctx, value);
            if (activation_ == "relu")
                return ReluModule{}.build(ctx, value);
            if (activation_ == "gelu")
                return GeluModule{}.build(ctx, value);
            throw std::invalid_argument("Unknown Dettato drum stem activation: " + activation_);
        };
        for (const auto &layer : weights_.stem) {
            if (layer.kind == DrumStemLayer::Kind::Convolution) {
                const auto &weight = layer.conv.weight;
                Conv2dConfig conv;
                conv.in_channels = weight.shape.dims[1];
                conv.out_channels = weight.shape.dims[0];
                conv.kernel_height = conv.kernel_width = 3;
                conv.padding_height = conv.padding_width = 1;
                x = Conv2dModule(conv).build(ctx, x, layer.conv);
                x = BatchNorm2dEvalModule({conv.out_channels}).build(ctx, x, layer.norm);
                x = activate(x);
            } else if (layer.kind == DrumStemLayer::Kind::Residual) {
                const auto residual = x;
                const auto channels = x.shape.dims[1];
                DepthwiseConv2dConfig dw;
                dw.channels = channels;
                dw.kernel_height = dw.kernel_width = 3;
                dw.padding_height = dw.padding_width = 1;
                x = DepthwiseConv2dModule(dw).build(ctx, x, layer.conv);
                x = BatchNorm2dEvalModule({channels}).build(ctx, x, layer.norm);
                x = activate(x);
                Conv2dConfig pointwise;
                pointwise.in_channels = pointwise.out_channels = channels;
                pointwise.kernel_height = pointwise.kernel_width = 1;
                x = Conv2dModule(pointwise).build(ctx, x, layer.pointwise);
                x = BatchNorm2dEvalModule({channels}).build(ctx, x, layer.residual_norm);
                x = AddModule{}.build(ctx, residual, x);
            } else {
                const auto time = layer.time_pool, frequency = layer.frequency_pool;
                auto shape = x.shape;
                shape.dims[2] /= time;
                shape.dims[3] /= frequency;
                x = wrap_tensor(
                    ggml_pool_2d(ctx.ggml, x.tensor, GGML_OP_POOL_MAX, frequency, time, frequency, time, 0, 0), shape);
            }
        }
        const int64_t time = x.shape.dims[2], bands = x.shape.dims[3], channels = x.shape.dims[1];
        if (time != frames / 2)
            throw std::invalid_argument("Dettato drum stem must halve its time dimension");
        x = TransposeModule({{3, 0, 2, 1}, 4}).build(ctx, x);
        x = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, x),
                           TensorShape::from_dims({bands, time, channels}));
        ports.initial = make_tensor(graph_->persistent_build_context(), GGML_TYPE_F32,
                                    TensorShape::from_dims({time, band_hidden_}));
        ggml_set_input(ports.initial.tensor);
        const auto &parameters = weights_.recurrent;
        const auto recurrent =
            BidirectionalGRU({channels, band_hidden_,
                              ctx.backend_type == BackendType::Cuda ? GRURecurrenceLowering::DeviceCell
                                                                    : GRURecurrenceLowering::Unrolled})
                .build(ctx, x, ports.initial, ports.initial, parameters[0], parameters[1]);
        const auto hidden = ConcatModule({1}).build(ctx, recurrent.forward_hidden, recurrent.reverse_hidden);
        auto mean = ReduceMeanModule({0}).build(ctx, recurrent.sequence);
        mean = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, mean),
                              TensorShape::from_dims({time, band_hidden_ * 2}));
        const auto collapsed = ConcatModule({1}).build(ctx, hidden, mean);
        ports.output =
            LinearModule({4 * band_hidden_, width_, true, GGML_PREC_F32}).build(ctx, collapsed, weights_.projection);
        ports.output = reshape_tensor(ctx, ports.output, TensorShape::from_dims({1, time, width_}));
        ggml_set_output(ports.output.tensor);
        ggml_build_forward_expand(graph_->graph(id), ports.output.tensor);
    }
    graph_->allocate_persistent();
    if (workspace_)
        workspace_->add(*graph_);
    else
        graph_->allocate();
    for (size_t i = 0; i < shapes.size(); ++i)
        write_tensor_f32(ports_[i].initial, std::vector<float>(static_cast<size_t>(shapes[i] / 2 * band_hidden_)));
}

void DrumEmbedder::prepare(int64_t frames, int64_t long_above) {
    constexpr int64_t chunk = 8000, margin = 200;
    const auto step = frames > long_above ? chunk : frames;
    std::vector<int64_t> shapes;
    for (int64_t start = 0; start < frames; start += step) {
        const auto begin = std::max<int64_t>(0, start - margin);
        const auto end = std::min(frames, start + step + margin);
        const auto size = end - begin;
        if (std::find(shapes.begin(), shapes.end(), size) == shapes.end())
            shapes.push_back(size);
    }
    prepare(shapes);
}

std::vector<float> DrumEmbedder::run(const std::vector<float> &mel, int64_t long_above) {
    if (mel.size() % 160 || mel.size() < 320)
        throw std::invalid_argument("Dettato drum embedding expects at least two 160-band frames");
    const auto frames = static_cast<int64_t>(mel.size() / 160);
    // These are the upstream DrumNet/OrchestralAdapter windows, including the
    // different direct-path thresholds supplied by their callers.
    constexpr int64_t chunk = 8000, margin = 200;
    const auto step = frames > long_above ? chunk : frames;
    prepare(frames, long_above);
    std::vector<float> output;
    output.reserve(static_cast<size_t>(frames / 2 * width_));
    for (int64_t start = 0; start < frames; start += step) {
        const auto begin = std::max<int64_t>(0, start - margin);
        const auto end = std::min(frames, start + step + margin);
        const auto id = static_cast<size_t>(std::find(shapes_.begin(), shapes_.end(), end - begin) - shapes_.begin());
        window_.assign(mel.begin() + begin * 160, mel.begin() + end * 160);
        write_tensor_f32(ports_[id].input, window_);
        const auto phase = std::chrono::steady_clock::now();
        graph_->compute(id);
        debug::timing_log_scalar("dettato.drums.embed_ms", debug::elapsed_ms(phase));
        const auto embeddings = read_tensor_f32(ports_[id].output.tensor);
        const auto offset = (start - begin) / 2 * width_;
        const auto count = (std::min(start + step, frames) - start) / 2 * width_;
        output.insert(output.end(), embeddings.begin() + offset, embeddings.begin() + offset + count);
    }
    return output;
}

Track decode_drum_hits(const DrumProbabilities &probabilities, const io::json::Value &config) {
    const auto &thresholds = config.require("thresholds");
    const auto &decode = config.require("decode");
    const bool orchestral = probabilities.router >= thresholds.require("router").as_number();
    const auto &onset = orchestral ? probabilities.orchestral : probabilities.kit;
    const auto &velocity = orchestral ? probabilities.velocity : probabilities.kit_velocity;
    if (onset.size() % 61 || onset.size() != velocity.size())
        throw std::invalid_argument("Dettato drum onset/velocity shapes differ");
    const size_t frames = onset.size() / 61;
    const int gap = static_cast<int>(decode.require("peak_gap_frames").as_i64());
    const double repeat = decode.require("timpani_repeat_s").as_number();
    const double length = decode.require("hit_length_s").as_number();
    Track out;
    out.name = "Drums";
    out.is_drum = true;
    const auto peaks = [&](const std::vector<float> &values, int lanes, float threshold, const auto &accept) {
        std::vector<int64_t> previous(lanes, -10);
        for (size_t t = 0; t < frames; ++t)
            for (int lane = 0; lane < lanes; ++lane) {
                const float p = values[t * lanes + lane];
                const float left = values[(t ? t - 1 : t) * lanes + lane];
                const float right = values[(t + 1 < frames ? t + 1 : t) * lanes + lane];
                if (p >= threshold && p >= left && p >= right && static_cast<int64_t>(t) - previous[lane] >= gap) {
                    accept(t, lane);
                    previous[lane] = static_cast<int64_t>(t);
                }
            }
    };
    const auto append = [&](size_t frame, int pitch, float value) {
        Note note;
        note.onset = std::nearbyint((frame / 50.) * 10000.) / 10000.;
        note.offset = std::nearbyint((frame / 50. + length) * 10000.) / 10000.;
        note.pitch = pitch;
        const double rounded = std::nearbyint(static_cast<double>(std::clamp(value, .05f, 1.f)) * 1000.) / 1000.;
        note.confidence = static_cast<float>(rounded);
        note.velocity = static_cast<int>(std::clamp(std::nearbyint(rounded * 127.), 1., 127.));
        out.notes.push_back(note);
    };
    const float threshold = thresholds.require(orchestral ? "orchestral_drum" : "drum_kit").as_f32();
    peaks(onset, 61, threshold, [&](size_t frame, int lane) { append(frame, lane + 27, velocity[frame * 61 + lane]); });
    if (orchestral) {
        if (probabilities.timpani.size() != frames * 13)
            throw std::invalid_argument("Dettato timpani shape differs from orchestral drums");
        const size_t drum_hits = out.notes.size();
        peaks(probabilities.timpani, 13, thresholds.require("timpani").as_f32(), [&](size_t frame, int lane) {
            const double onset = frame / 50.;
            const int pitch = lane + 41;
            for (size_t i = 0; i < drum_hits; ++i)
                if (out.notes[i].pitch == pitch && std::abs(out.notes[i].onset - onset) <= repeat)
                    return;
            append(frame, pitch, std::clamp(probabilities.timpani[frame * 13 + lane], .3f, 1.f));
        });
        std::stable_sort(out.notes.begin(), out.notes.end(),
                         [](const auto &a, const auto &b) { return a.onset < b.onset; });
    }
    return out;
}

DrumTrunk::DrumTrunk(DrumTrunkWeights weights, const io::json::Value &config, const ExecutionContext &execution,
                     GraphWorkspace *workspace)
    : execution_(execution),
      workspace_(execution.backend_type() == BackendType::Cuda || execution.backend_type() == BackendType::Cpu
                     ? workspace
                     : nullptr),
      weights_(std::move(weights)), width_(config.require("w").as_i64()),
      family_hidden_(io::json::optional_i64(config, "tower_h", 24)) {
    for (const auto &group : config.require("towers").as_array()) {
        groups_.emplace_back();
        for (const auto &key : group.as_array())
            groups_.back().push_back(static_cast<int>(key.as_i64()) - 27);
    }
    if (execution.backend_type() != BackendType::Cpu && execution.backend_type() != BackendType::Cuda)
        host_execution_ =
            std::make_unique<ExecutionContext>(BackendConfig{BackendType::Cpu, 0, execution.config().threads});
}

void DrumTrunk::prepare(int64_t frames) {
    if (frames <= 0)
        throw std::invalid_argument("Dettato drum trunk requires nonempty frame embeddings");
    if (graph_ && graph_->graph() && frames_ == frames)
        return;
    graph_.reset();
    zero_states_.clear();
    frames_ = frames;
    graph_ = std::make_unique<PreparedGraph>(
        execution_, PreparedGraphConfig{"dettato.drums.trunk", 4 * 1024 * 1024, 4096, 1024 * 1024});
    auto &ctx = graph_->build_context();
    input_ = make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, frames, width_}));
    ggml_set_input(input_.tensor);
    const auto project = [&](const TensorValue &x, const LinearWeights &weights) {
        const auto weight = weights.weight;
        return LinearModule({weight.shape.dims[1], weight.shape.dims[0], true, GGML_PREC_F32}).build(ctx, x, weights);
    };
    const auto recurrent = [&](const TensorValue &x, const BidirectionalWeights &parameters, int64_t hidden) {
        auto it = zero_states_.find(hidden);
        if (it == zero_states_.end()) {
            const auto zero =
                make_tensor(graph_->persistent_build_context(), GGML_TYPE_F32, TensorShape::from_dims({1, hidden}));
            ggml_set_input(zero.tensor);
            it = zero_states_.emplace(hidden, zero).first;
        }
        auto sequence = TransposeModule({{1, 0, 2}, 3}).build(ctx, x);
        sequence = ensure_backend_addressable_layout(ctx, sequence);
        const auto output =
            BidirectionalGRU({x.shape.dims[2], hidden,
                              ctx.backend_type == BackendType::Cuda ? GRURecurrenceLowering::DeviceScan
                                                                    : GRURecurrenceLowering::HostRecurrence})
                .build(ctx, sequence, it->second, it->second, parameters[0], parameters[1]);
        return TransposeModule({{1, 0, 2}, 3}).build(ctx, output.sequence);
    };
    auto h = input_;
    for (const auto &block : weights_.blocks) {
        if (block.kind == DrumTemporalBlock::Kind::Convolution) {
            const int64_t kernel = block.kernel, dilation = block.dilation;
            auto x = TransposeModule({{0, 2, 1}, 3}).build(ctx, h);
            x = DepthwiseConv1dModule(
                    {width_, kernel, 1, static_cast<int>(dilation * (kernel / 2)), static_cast<int>(dilation), true})
                    .build(ctx, x, block.depthwise);
            x = BatchNorm1dEvalModule({width_}).build(ctx, x, block.affine);
            x = SiluModule{}.build(ctx, x);
            x = PointwiseConv1dModule({width_, width_, true}).build(ctx, x, block.pointwise);
            h = AddModule{}.build(ctx, h, TransposeModule({{0, 2, 1}, 3}).build(ctx, x));
        } else if (block.kind == DrumTemporalBlock::Kind::Recurrent) {
            const auto norm = LayerNormModule({width_, 1e-5f, true, true}).build(ctx, h, block.norm);
            h = AddModule{}.build(ctx, h, project(recurrent(norm, block.recurrent, block.hidden), block.output));
        } else if (block.kind == DrumTemporalBlock::Kind::Attention) {
            const int64_t channels = block.channels, heads = block.heads;
            const auto norm = LayerNormModule({width_, 1e-5f, true, true}).build(ctx, h, block.norm);
            const auto packed = project(norm, block.qkv);
            std::array<TensorValue, 3> qkv;
            for (int j = 0; j < 3; ++j) {
                auto x = SliceModule({2, j * channels, channels}).build(ctx, packed);
                x = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, x),
                                   TensorShape::from_dims({1, frames, heads, channels / heads}));
                qkv[j] = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
            }
            TensorValue x;
            if (execution_.backend_type() == BackendType::Vulkan) {
                x = ScaledDotProductAttentionModule(
                        {channels / heads, ScaledDotProductAttentionLowering::Flash, GGML_PREC_F32})
                        .build(ctx, qkv[0], qkv[1], qkv[2]);
            } else {
                // Attention rows are independent. Bound the F32 score workspace
                // without changing key coverage, softmax or K/V precision.
                constexpr int64_t query_rows = 128;
                const ScaledDotProductAttentionModule attention(
                    {channels / heads, ScaledDotProductAttentionLowering::Explicit, GGML_PREC_F32});
                for (int64_t first = 0; first < frames; first += query_rows) {
                    const auto query = SliceModule({2, first, std::min(query_rows, frames - first)}).build(ctx, qkv[0]);
                    const auto block = attention.build(ctx, query, qkv[1], qkv[2]);
                    x = first == 0 ? block : ConcatModule({1}).build(ctx, x, block);
                }
            }
            x = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, x),
                               TensorShape::from_dims({1, frames, channels}));
            h = AddModule{}.build(ctx, h, project(x, block.output));
        } else
            throw std::logic_error("Invalid bound Dettato temporal block");
    }
    outputs_[0] = h;
    auto packed = project(h, weights_.onset);
    std::vector<int32_t> selection(61);
    std::iota(selection.begin(), selection.end(), 0);
    int64_t channels = 61;
    for (size_t i = 0; i < groups_.size(); ++i) {
        auto x = recurrent(h, weights_.towers[i], family_hidden_);
        x = project(ConcatModule({2}).build(ctx, x, h), weights_.tower_heads[i]);
        packed = ConcatModule({2}).build(ctx, packed, x);
        for (size_t j = 0; j < groups_[i].size(); ++j)
            selection.at(static_cast<size_t>(groups_[i][j])) = static_cast<int32_t>(channels + j);
        channels += static_cast<int64_t>(groups_[i].size());
    }
    indices_ = make_tensor(graph_->persistent_build_context(), GGML_TYPE_I32, TensorShape::from_dims({1, 61}));
    ggml_set_input(indices_.tensor);
    packed = TransposeModule({{0, 2, 1}, 3}).build(ctx, packed);
    packed =
        reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, packed), TensorShape::from_dims({channels, frames}));
    auto onset = EmbeddingModule({channels, frames}).build(ctx, indices_, packed);
    onset = TransposeModule({{0, 2, 1}, 3}).build(ctx, onset);
    outputs_[1] = wrap_tensor(ggml_cont(ctx.ggml, onset.tensor), onset.shape);
    outputs_[2] = project(project(h, weights_.velocity_in), weights_.velocity_out);
    // The adapter reads these results after this trunk's shared scratch is reused.
    for (auto &output : outputs_) {
        const auto saved = make_tensor(graph_->persistent_build_context(), output.type, output.shape);
        auto *copy = ggml_cpy(ctx.ggml, output.tensor, saved.tensor);
        ggml_set_output(copy);
        ggml_build_forward_expand(graph_->graph(), copy);
        output = saved;
    }
    const auto scheduler =
        host_execution_ ? graph_->create_scheduler({execution_.backend(), host_execution_->backend()}) : nullptr;
    graph_->allocate_persistent();
    if (workspace_)
        workspace_->add(*graph_);
    else
        graph_->allocate();
    if (scheduler) {
        debug::timing_log_scalar("dettato.drums.trunk.device_workspace_mb",
                                 ggml_backend_sched_get_buffer_size(scheduler, execution_.backend()) / (1024. * 1024));
        debug::timing_log_scalar("dettato.drums.trunk.host_workspace_mb",
                                 ggml_backend_sched_get_buffer_size(scheduler, host_execution_->backend()) /
                                     (1024. * 1024));
    }
    write_tensor_i32(indices_, selection);
    for (const auto &state : zero_states_)
        write_tensor_f32(state.second, std::vector<float>(static_cast<size_t>(state.first)));
}

std::array<TensorValue, 3> DrumTrunk::run(const std::vector<float> &embeddings) {
    if (embeddings.empty() || embeddings.size() % width_)
        throw std::invalid_argument("Dettato drum trunk expects frame embeddings of its configured width");
    prepare(static_cast<int64_t>(embeddings.size() / width_));
    write_tensor_f32(input_, embeddings);
    const auto start = std::chrono::steady_clock::now();
    graph_->compute();
    debug::timing_log_scalar("dettato.drums.trunk_ms", debug::elapsed_ms(start));
    return outputs_;
}

OrchestralDrumExpert::OrchestralDrumExpert(DrumAdapterWeights weights, const io::json::Value &config,
                                           const ExecutionContext &execution, GraphWorkspace *workspace)
    : execution_(execution),
      workspace_(execution.backend_type() == BackendType::Cuda || execution.backend_type() == BackendType::Cpu
                     ? workspace
                     : nullptr),
      embedder_(std::move(weights.embedder), config.require("drum_kit"), execution, workspace),
      trunk_(std::move(weights.trunk), config.require("drum_kit"), execution, workspace), weights_(std::move(weights)),
      hidden_(config.require("orchestral").require("adapter").require("h").as_i64()),
      layers_(config.require("orchestral").require("adapter").require("layers").as_i64()),
      heads_(config.require("orchestral").require("adapter").require("heads").as_i64()),
      base_width_(config.require("drum_kit").require("w").as_i64()) {
    if (execution.backend_type() != BackendType::Cpu && execution.backend_type() != BackendType::Cuda)
        host_execution_ =
            std::make_unique<ExecutionContext>(BackendConfig{BackendType::Cpu, 0, execution.config().threads});
}

void OrchestralDrumExpert::prepare(int64_t frames) {
    if (frames < 2 || frames % 2)
        throw std::invalid_argument("Dettato orchestral expert needs a positive even feature-frame count");
    if (graph_ && graph_->graph() && frames_ == frames)
        return;
    embedder_.prepare(frames);
    trunk_.prepare(frames / 2);
    graph_.reset();
    frames_ = frames;
    graph_ = std::make_unique<PreparedGraph>(
        execution_, PreparedGraphConfig{"dettato.drums.expert", 4 * 1024 * 1024, 4096, 1024 * 1024});
    auto &ctx = graph_->build_context();
    const int64_t time = frames / 2;
    cq_input_ = make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 3, frames, 120}));
    hidden_input_ = make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, time, base_width_}));
    base_input_ = make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, time, 61}));
    for (const auto &input : {cq_input_, hidden_input_, base_input_})
        ggml_set_input(input.tensor);
    const auto project = [&](const TensorValue &x, const LinearWeights &weights) {
        const auto weight = weights.weight;
        return LinearModule({weight.shape.dims[1], weight.shape.dims[0], true, GGML_PREC_F32}).build(ctx, x, weights);
    };
    // The shifted input and its minimum padding were packed on the host from
    // the already-host-resident features. Mean/normalization remain in graph.
    const auto base_cq = SliceModule({1, 0, 1}).build(ctx, cq_input_);
    const auto mean = ReduceMeanModule({2}).build(ctx, ReduceMeanModule({3}).build(ctx, base_cq));
    auto x = wrap_tensor(ggml_sub(ctx.ggml, cq_input_.tensor, mean.tensor), cq_input_.shape);
    x = ScaleBiasModule({.25f, 0}).build(ctx, x);
    TensorValue residual;
    for (int layer = 1; layer <= 3; ++layer) {
        const auto &parameters = weights_.convolutions[layer - 1];
        const auto weight = parameters.weight;
        Conv2dConfig conv;
        conv.in_channels = weight.shape.dims[1];
        conv.out_channels = weight.shape.dims[0];
        conv.kernel_height = weight.shape.dims[2];
        conv.kernel_width = weight.shape.dims[3];
        conv.padding_height = static_cast<int>(conv.kernel_height / 2);
        conv.padding_width = static_cast<int>(conv.kernel_width / 2);
        conv.stride_height = layer == 2 ? 2 : 1;
        if (execution_.backend_type() == BackendType::Cpu) {
            x = Conv2dModule(conv).build(ctx, x, parameters);
        } else {
            // Keep CQ workspace bounded with the existing native direct op.
            // Activations are F32; the package supplies contiguous F16/F32 kernels.
            const auto input = ensure_backend_addressable_layout(ctx, x);
            const auto kernel = ensure_backend_addressable_layout(ctx, weight);
            auto *output =
                ggml_conv_2d_direct(ctx.ggml, kernel.tensor, input.tensor, conv.stride_width, conv.stride_height,
                                    conv.padding_width, conv.padding_height, conv.dilation_width, conv.dilation_height);
            const auto bias =
                reshape_tensor(ctx, *parameters.bias, TensorShape::from_dims({1, conv.out_channels, 1, 1}));
            x = wrap_tensor(output,
                            TensorShape::from_dims({output->ne[3], output->ne[2], output->ne[1], output->ne[0]}),
                            GGML_TYPE_F32);
            x = AddModule{false, true}.build(ctx, x, bias);
        }
        const auto &affine = weights_.norms[layer - 1];
        x = GroupNormModule({conv.out_channels, 4, 1e-5f, false, false}).build(ctx, x, {});
        if (ctx.backend_type == BackendType::Cuda) {
            x = wrap_tensor(ggml_affine(ctx.ggml, x.tensor, affine.scale.tensor, affine.bias.tensor, 2), x.shape);
        } else {
            const auto affine_shape = TensorShape::from_dims({1, conv.out_channels, 1, 1});
            const auto scale = reshape_tensor(ctx, affine.scale, affine_shape);
            const auto bias = reshape_tensor(ctx, affine.bias, affine_shape);
            x = AddModule{false, true}.build(ctx, MulModule{true, true}.build(ctx, x, scale), bias);
        }
        x = SiluModule{}.build(ctx, x);
        if (layer == 2)
            residual = x;
    }
    x = AddModule{}.build(ctx, x, residual);
    const int64_t cq_channels = x.shape.dims[1];
    x = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
    x = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, x),
                       TensorShape::from_dims({1, time, cq_channels * 120}));
    x = SiluModule{}.build(ctx, project(x, weights_.cq_projection));
    x = ConcatModule({2}).build(ctx, ConcatModule({2}).build(ctx, hidden_input_, x),
                                SigmoidModule{}.build(ctx, base_input_));
    x = SiluModule{}.build(ctx, project(x, weights_.input));
    initial_ = make_tensor(graph_->persistent_build_context(), GGML_TYPE_F32, TensorShape::from_dims({1, hidden_}));
    ggml_set_input(initial_.tensor);
    x = TransposeModule({{1, 0, 2}, 3}).build(ctx, x);
    for (int64_t layer = 0; layer < layers_; ++layer) {
        const auto &parameters = weights_.recurrent[layer];
        x = BidirectionalGRU({x.shape.dims[2], hidden_,
                              ctx.backend_type == BackendType::Cuda ? GRURecurrenceLowering::DeviceScan
                                                                    : GRURecurrenceLowering::HostRecurrence})
                .build(ctx, x, initial_, initial_, parameters[0], parameters[1])
                .sequence;
    }
    const auto channels = hidden_ * 2;
    x = TransposeModule({{1, 0, 2}, 3}).build(ctx, x);
    const auto packed = project(x, weights_.qkv);
    std::array<TensorValue, 3> qkv;
    for (int i = 0; i < 3; ++i) {
        auto part = SliceModule({2, i * channels, channels}).build(ctx, packed);
        part = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, part),
                              TensorShape::from_dims({1, time, heads_, channels / heads_}));
        qkv[i] = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, part);
    }
    const auto attention_lowering = execution_.backend_type() == BackendType::Cpu
                                        ? ScaledDotProductAttentionLowering::Explicit
                                        : ScaledDotProductAttentionLowering::Flash;
    auto attention = ScaledDotProductAttentionModule({channels / heads_, attention_lowering, GGML_PREC_F32})
                         .build(ctx, qkv[0], qkv[1], qkv[2]);
    attention = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, attention),
                               TensorShape::from_dims({1, time, channels}));
    x = LayerNormModule({channels, 1e-5f, true, true})
            .build(ctx, AddModule{}.build(ctx, x, project(attention, weights_.attention_output)), weights_.norm);
    outputs_[0] = AddModule{}.build(ctx, base_input_, project(x, weights_.delta));
    outputs_[1] = project(x, weights_.timpani);
    auto pooled = ReduceMeanModule({1}).build(ctx, x);
    pooled = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, pooled), TensorShape::from_dims({1, channels}));
    outputs_[2] = project(SiluModule{}.build(ctx, project(pooled, weights_.gate_in)), weights_.gate_out);
    for (const auto &output : outputs_) {
        ggml_set_output(output.tensor);
        if (output.tensor->view_src)
            ggml_set_output(output.tensor->view_src);
        ggml_build_forward_expand(graph_->graph(), output.tensor);
    }
    const auto scheduler =
        host_execution_ ? graph_->create_scheduler({execution_.backend(), host_execution_->backend()}) : nullptr;
    graph_->allocate_persistent();
    if (workspace_)
        workspace_->add(*graph_);
    else
        graph_->allocate();
    if (scheduler) {
        debug::timing_log_scalar("dettato.drums.expert.device_workspace_mb",
                                 ggml_backend_sched_get_buffer_size(scheduler, execution_.backend()) / (1024. * 1024));
        debug::timing_log_scalar("dettato.drums.expert.host_workspace_mb",
                                 ggml_backend_sched_get_buffer_size(scheduler, host_execution_->backend()) /
                                     (1024. * 1024));
    }
    write_tensor_f32(initial_, std::vector<float>(static_cast<size_t>(hidden_)));
}

std::array<TensorValue, 5> OrchestralDrumExpert::run(const std::vector<float> &features) {
    if (features.size() % 280)
        throw std::invalid_argument("Dettato orchestral features must have 280 bands");
    const int64_t frames = static_cast<int64_t>(features.size() / 280) / 2 * 2;
    prepare(frames);
    std::vector<float> mel(static_cast<size_t>(frames * 160)), cq(static_cast<size_t>(frames * 120));
    for (int64_t t = 0; t < frames; ++t) {
        std::copy_n(features.data() + t * 280, 160, mel.data() + t * 160);
        std::copy_n(features.data() + t * 280 + 160, 120, cq.data() + t * 120);
    }
    const auto trunk = trunk_.run(embedder_.run(mel));
    const float minimum = *std::min_element(cq.begin(), cq.end());
    const std::array<int, 3> offsets{0, 14, 24};
    std::vector<float> stacked(static_cast<size_t>(3 * frames * 120));
    for (int ch = 0; ch < 3; ++ch)
        for (int64_t t = 0; t < frames; ++t)
            for (int f = 0; f < 120; ++f)
                stacked[static_cast<size_t>((ch * frames + t) * 120 + f)] =
                    f + offsets[ch] < 120 ? cq[static_cast<size_t>(t * 120 + f + offsets[ch])] : minimum;
    write_tensor_f32(cq_input_, stacked);
    ggml_backend_tensor_copy(trunk[0].tensor, hidden_input_.tensor);
    ggml_backend_tensor_copy(trunk[1].tensor, base_input_.tensor);
    const auto start = std::chrono::steady_clock::now();
    graph_->compute();
    debug::timing_log_scalar("dettato.drums.expert_ms", debug::elapsed_ms(start));
    return {trunk[1], outputs_[0], outputs_[1], trunk[2], outputs_[2]};
}

struct DrumTranscriber::Impl {
    const ExecutionContext &execution;
    io::json::Value config;
    GraphWorkspace workspace;
    DrumEmbedder kit_embedder;
    DrumTrunk kit_trunk;
    std::vector<std::unique_ptr<OrchestralDrumExpert>> experts;
    std::unique_ptr<PreparedGraph> graph;
    std::array<TensorValue, 2> kit_inputs;
    std::vector<std::array<TensorValue, 5>> expert_inputs;
    std::array<TensorValue, 6> outputs;
    TensorValue divisor;
    int64_t frames = 0;

    Impl(DrumWeights weights, const io::json::Value &cfg, const ExecutionContext &ctx)
        : execution(ctx), config(cfg), workspace(ctx.backend()),
          kit_embedder(std::move(weights.kit_embedder), cfg.require("drum_kit"), ctx, &workspace),
          kit_trunk(std::move(weights.kit_trunk), cfg.require("drum_kit"), ctx, &workspace) {
        const auto count = cfg.require("orchestral").require("n_models").as_i64();
        if (count <= 0)
            throw std::invalid_argument("Dettato drum routing requires orchestral experts");
        for (int64_t i = 0; i < count; ++i)
            experts.push_back(
                std::make_unique<OrchestralDrumExpert>(std::move(weights.experts[i]), cfg, ctx, &workspace));
    }
    void prepare(int64_t count) {
        if (graph && frames == count)
            return;
        workspace.reset();
        graph.reset();
        frames = 0;
        graph = std::make_unique<PreparedGraph>(
            execution, PreparedGraphConfig{"dettato.drums.probabilities", 1024 * 1024, 1024, 64 * 1024});
        auto &ctx = graph->build_context();
        for (auto &input : kit_inputs) {
            input = make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, count, 61}));
            ggml_set_input(input.tensor);
        }
        outputs[0] = SigmoidModule{}.build(ctx, kit_inputs[0]);
        outputs[1] = SigmoidModule{}.build(ctx, kit_inputs[1]);
        expert_inputs.resize(experts.size());
        std::array<TensorValue, 3> sums;
        TensorValue routers;
        for (size_t i = 0; i < experts.size(); ++i) {
            auto &inputs = expert_inputs[i];
            for (size_t j = 0; j < inputs.size(); ++j) {
                inputs[j] = make_tensor(ctx, GGML_TYPE_F32,
                                        j == 4 ? TensorShape::from_dims({1, 1})
                                               : TensorShape::from_dims({1, count, j == 2 ? 13 : 61}));
                ggml_set_input(inputs[j].tensor);
            }
            // Preserve upstream's base + (corrected - base) rounding boundary.
            const auto difference =
                wrap_tensor(ggml_sub(ctx.ggml, inputs[1].tensor, inputs[0].tensor), inputs[0].shape);
            const std::array<TensorValue, 3> p{
                SigmoidModule{}.build(ctx, AddModule{}.build(ctx, inputs[0], difference)),
                SigmoidModule{}.build(ctx, inputs[2]), SigmoidModule{}.build(ctx, inputs[3])};
            for (size_t k = 0; k < sums.size(); ++k)
                sums[k] = i == 0 ? p[k] : AddModule{}.build(ctx, sums[k], p[k]);
            const auto router = SigmoidModule{}.build(ctx, inputs[4]);
            routers = i == 0 ? router : ConcatModule({1}).build(ctx, routers, router);
        }
        divisor = make_tensor(graph->persistent_build_context(), GGML_TYPE_F32, TensorShape::from_dims({1}));
        ggml_set_input(divisor.tensor);
        for (size_t i = 0; i < sums.size(); ++i)
            outputs[i + 2] = wrap_tensor(ggml_div(ctx.ggml, sums[i].tensor, divisor.tensor), sums[i].shape);
        outputs[5] = routers;
        for (const auto &output : outputs) {
            ggml_set_output(output.tensor);
            ggml_build_forward_expand(graph->graph(), output.tensor);
        }
        graph->allocate_persistent();
        graph->allocate();
        write_tensor_f32(divisor, {static_cast<float>(experts.size())});
        kit_embedder.prepare(2 * count, 9000);
        kit_trunk.prepare(count);
        for (auto &expert : experts)
            expert->prepare(2 * count);
        workspace.prepare();
        frames = count;
        debug::timing_log_scalar("dettato.drums.shared_workspace_mb", workspace.bytes() / (1024. * 1024));
    }
};

DrumTranscriber::DrumTranscriber(DrumWeights weights, const io::json::Value &config, const ExecutionContext &execution)
    : impl_(std::make_unique<Impl>(std::move(weights), config, execution)) {}
DrumTranscriber::~DrumTranscriber() = default;

DrumProbabilities DrumTranscriber::probabilities(const std::vector<float> &features) {
    if (features.size() % 280 || features.size() < 560)
        throw std::invalid_argument("Dettato drum network requires at least two 280-band frames");
    const auto start = std::chrono::steady_clock::now();
    auto &state = *impl_;
    const int64_t frames = static_cast<int64_t>(features.size() / 280) / 2 * 2;
    state.prepare(frames / 2);
    std::vector<float> mel(static_cast<size_t>(frames * 160));
    for (int64_t t = 0; t < frames; ++t)
        std::copy_n(features.data() + t * 280, 160, mel.data() + t * 160);
    const auto kit = state.kit_trunk.run(state.kit_embedder.run(mel, 9000));
    ggml_backend_tensor_copy(kit[1].tensor, state.kit_inputs[0].tensor);
    ggml_backend_tensor_copy(kit[2].tensor, state.kit_inputs[1].tensor);
    for (size_t i = 0; i < state.experts.size(); ++i) {
        const auto output = state.experts[i]->run(features);
        for (size_t j = 0; j < output.size(); ++j)
            ggml_backend_tensor_copy(output[j].tensor, state.expert_inputs[i][j].tensor);
    }
    state.graph->compute();
    const auto routers = read_tensor_f32(state.outputs[5].tensor);
    DrumProbabilities result;
    // Upstream converts each F32 router to a Python float before averaging.
    result.router = std::accumulate(routers.begin(), routers.end(), 0.) / routers.size();
    result.kit = read_tensor_f32(state.outputs[0].tensor);
    result.kit_velocity = read_tensor_f32(state.outputs[1].tensor);
    result.orchestral = read_tensor_f32(state.outputs[2].tensor);
    result.timpani = read_tensor_f32(state.outputs[3].tensor);
    result.velocity = read_tensor_f32(state.outputs[4].tensor);
    debug::timing_log_scalar("dettato.drums.network_ms", debug::elapsed_ms(start));
    return result;
}

DrumResult DrumTranscriber::transcribe_features(const std::vector<float> &features) {
    const auto start = std::chrono::steady_clock::now();
    DrumResult result;
    if (!features.empty()) {
        const auto p = probabilities(features);
        result.router = p.router;
        result.model =
            p.router < impl_->config.require("thresholds").require("router").as_number() ? "drum-kit" : "orchestral";
        result.track = decode_drum_hits(p, impl_->config);
    }
    debug::timing_log_scalar("dettato.drums.wall_ms", debug::elapsed_ms(start));
    return result;
}
} // namespace engine::community_models::dettato::detail
