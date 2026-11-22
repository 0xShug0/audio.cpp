#include "drums.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "gru.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include <numeric>
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include <algorithm>

namespace engine::models::dettato::detail {
using namespace core;
using namespace modules;

DrumTrunk::DrumTrunk(const assets::TensorSource & source, std::string prefix,
    const io::json::Value & config, const ExecutionContext & execution, GraphWorkspace *workspace)
    : execution_(execution), workspace_(execution.backend_type() == BackendType::Cuda || execution.backend_type() == BackendType::Cpu ? workspace : nullptr), store_(execution.backend(), execution.backend_type(), "dettato.drums." + prefix + "trunk.weights", 1024 * 1024),
      width_(config.require("w").as_i64()), family_hidden_(io::json::optional_i64(config, "tower_h", 24)) {
    for (const auto & spec : config.require("temporal").as_array())
        for (int64_t i = 0; i < io::json::optional_i64(spec, "n", 1); ++i) blocks_.push_back(spec);
    for (const auto & group : config.require("towers").as_array()) {
        groups_.emplace_back();
        for (const auto & key : group.as_array()) groups_.back().push_back(static_cast<int>(key.as_i64()) - 27);
    }
    for (const auto & metadata : source.tensors()) {
        if (metadata.name.rfind(prefix, 0) != 0) continue;
        const auto name = metadata.name.substr(prefix.size());
        if (name == "on.idx" || (name.rfind("temporal.", 0) != 0 && name.rfind("on.", 0) != 0 && name.rfind("vel.", 0) != 0)) continue;
        weights_.emplace(name, store_.load_tensor(source, metadata.name, assets::TensorStorageType::Native, metadata.shape));
        const std::string suffix = ".running_var";
        if (name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            const auto bn = name.substr(0, name.size() - suffix.size());
            const auto affine = binding::batch_norm_eval_from_source(store_, source, prefix + bn, metadata.shape[0], 1e-5f);
            weights_.emplace(bn + ".eval_scale", affine.scale); weights_.emplace(bn + ".eval_bias", affine.bias);
        }
    }
    store_.upload();
    if (execution.backend_type() != BackendType::Cpu && execution.backend_type() != BackendType::Cuda)
        host_execution_ = std::make_unique<ExecutionContext>(BackendConfig{BackendType::Cpu, 0, execution.config().threads});
}

void DrumTrunk::prepare(int64_t frames) {
    if (frames <= 0) throw std::invalid_argument("Dettato drum trunk requires nonempty frame embeddings");
    if (graph_ && graph_->graph() && frames_ == frames) return;
    graph_.reset(); zero_states_.clear(); frames_ = frames;
    graph_ = std::make_unique<PreparedGraph>(execution_, PreparedGraphConfig{"dettato.drums.trunk", 4 * 1024 * 1024, 4096, 1024 * 1024});
    auto & ctx = graph_->build_context();
    input_ = make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, frames, width_})); ggml_set_input(input_.tensor);
    const auto project = [&](const TensorValue & x, const std::string & prefix) {
        const auto weight = weights_.at(prefix + ".weight");
        return LinearModule({weight.shape.dims[1], weight.shape.dims[0], true, GGML_PREC_F32})
            .build(ctx, x, {weight, weights_.at(prefix + ".bias")});
    };
    const auto recurrent = [&](const TensorValue & x, const std::string & prefix, int64_t hidden) {
        auto it = zero_states_.find(hidden);
        if (it == zero_states_.end()) {
            const auto zero = make_tensor(graph_->persistent_build_context(), GGML_TYPE_F32, TensorShape::from_dims({1, hidden}));
            ggml_set_input(zero.tensor); it = zero_states_.emplace(hidden, zero).first;
        }
        auto sequence = TransposeModule({{1, 0, 2}, 3}).build(ctx, x);
        sequence = ensure_backend_addressable_layout(ctx, sequence);
        std::array<GRUWeights, 2> parameters;
        for (int d = 0; d < 2; ++d) {
            const auto suffix = std::string(d ? "_l0_reverse" : "_l0");
            parameters[d] = {weights_.at(prefix + ".weight_ih" + suffix), weights_.at(prefix + ".weight_hh" + suffix),
                 weights_.at(prefix + ".bias_ih" + suffix), weights_.at(prefix + ".bias_hh" + suffix)};
        }
        const auto output = BidirectionalGRU({x.shape.dims[2], hidden, ctx.backend_type == BackendType::Cuda ? GRURecurrenceLowering::DeviceScan : GRURecurrenceLowering::HostRecurrence})
            .build(ctx, sequence, it->second, it->second, parameters[0], parameters[1]);
        return TransposeModule({{1, 0, 2}, 3}).build(ctx, output.sequence);
    };
    auto h = input_;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        const auto & block = blocks_[i]; const auto type = block.require("t").as_string();
        const auto prefix = "temporal." + std::to_string(i);
        if (type == "tb") {
            const int64_t kernel = io::json::optional_i64(block, "k", 15), dilation = io::json::optional_i64(block, "d", 1);
            auto x = TransposeModule({{0, 2, 1}, 3}).build(ctx, h);
            x = DepthwiseConv1dModule({width_, kernel, 1, static_cast<int>(dilation * (kernel / 2)), static_cast<int>(dilation), true})
                .build(ctx, x, {weights_.at(prefix + ".dw.weight"), weights_.at(prefix + ".dw.bias")});
            x = BatchNorm1dEvalModule({width_}).build(ctx, x, {weights_.at(prefix + ".bn.eval_scale"), weights_.at(prefix + ".bn.eval_bias")});
            x = SiluModule{}.build(ctx, x);
            x = PointwiseConv1dModule({width_, width_, true}).build(ctx, x, {weights_.at(prefix + ".pw.weight"), weights_.at(prefix + ".pw.bias")});
            h = AddModule{}.build(ctx, h, TransposeModule({{0, 2, 1}, 3}).build(ctx, x));
        } else if (type == "gru") {
            const auto norm = LayerNormModule({width_, 1e-5f, true, true}).build(ctx, h,
                {weights_.at(prefix + ".norm.weight"), weights_.at(prefix + ".norm.bias")});
            h = AddModule{}.build(ctx, h, project(recurrent(norm, prefix + ".rnn", io::json::optional_i64(block, "h", 48)), prefix + ".out"));
        } else if (type == "attn") {
            const int64_t channels = io::json::optional_i64(block, "da", 32), heads = io::json::optional_i64(block, "heads", 2);
            const auto norm = LayerNormModule({width_, 1e-5f, true, true}).build(ctx, h,
                {weights_.at(prefix + ".n.weight"), weights_.at(prefix + ".n.bias")});
            const auto packed = project(norm, prefix + ".qkv");
            std::array<TensorValue, 3> qkv;
            for (int j = 0; j < 3; ++j) {
                auto x = SliceModule({2, j * channels, channels}).build(ctx, packed);
                x = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, x), TensorShape::from_dims({1, frames, heads, channels / heads}));
                qkv[j] = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
            }
            TensorValue x;
            if (execution_.backend_type() == BackendType::Vulkan) {
                x = ScaledDotProductAttentionModule({channels / heads, ScaledDotProductAttentionLowering::Flash, GGML_PREC_F32})
                    .build(ctx, qkv[0], qkv[1], qkv[2]);
            } else {
                // Attention rows are independent. Bound the F32 score workspace
                // without changing key coverage, softmax or K/V precision.
                constexpr int64_t query_rows = 128;
                const ScaledDotProductAttentionModule attention({channels / heads,
                    ScaledDotProductAttentionLowering::Explicit, GGML_PREC_F32});
                for (int64_t first = 0; first < frames; first += query_rows) {
                    const auto query = SliceModule({2, first, std::min(query_rows, frames - first)}).build(ctx, qkv[0]);
                    const auto block = attention.build(ctx, query, qkv[1], qkv[2]);
                    x = first == 0 ? block : ConcatModule({1}).build(ctx, x, block);
                }
            }
            x = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, x), TensorShape::from_dims({1, frames, channels}));
            h = AddModule{}.build(ctx, h, project(x, prefix + ".o"));
        } else throw std::invalid_argument("Unknown Dettato drum temporal block: " + type);
    }
    outputs_[0] = h;
    auto packed = project(h, "on.rest");
    std::vector<int32_t> selection(61); std::iota(selection.begin(), selection.end(), 0);
    int64_t channels = 61;
    for (size_t i = 0; i < groups_.size(); ++i) {
        auto x = recurrent(h, "on.towers." + std::to_string(i), family_hidden_);
        x = project(ConcatModule({2}).build(ctx, x, h), "on.heads." + std::to_string(i));
        packed = ConcatModule({2}).build(ctx, packed, x);
        for (size_t j = 0; j < groups_[i].size(); ++j) selection.at(static_cast<size_t>(groups_[i][j])) = static_cast<int32_t>(channels + j);
        channels += static_cast<int64_t>(groups_[i].size());
    }
    indices_ = make_tensor(graph_->persistent_build_context(), GGML_TYPE_I32, TensorShape::from_dims({1, 61})); ggml_set_input(indices_.tensor);
    packed = TransposeModule({{0, 2, 1}, 3}).build(ctx, packed);
    packed = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, packed), TensorShape::from_dims({channels, frames}));
    auto onset = EmbeddingModule({channels, frames}).build(ctx, indices_, packed);
    onset = TransposeModule({{0, 2, 1}, 3}).build(ctx, onset);
    outputs_[1] = wrap_tensor(ggml_cont(ctx.ggml, onset.tensor), onset.shape);
    outputs_[2] = project(project(h, "vel.0"), "vel.1");
    // The adapter reads these results after this trunk's shared scratch is reused.
    for (auto &output : outputs_) {
        const auto saved = make_tensor(graph_->persistent_build_context(), output.type, output.shape);
        auto *copy = ggml_cpy(ctx.ggml, output.tensor, saved.tensor);
        ggml_set_output(copy);
        ggml_build_forward_expand(graph_->graph(), copy);
        output = saved;
    }
    const auto scheduler = host_execution_
        ? graph_->create_scheduler({execution_.backend(), host_execution_->backend()}) : nullptr;
    graph_->allocate_persistent();
    if (workspace_) workspace_->add(*graph_);
    else graph_->allocate();
    if (scheduler) {
        debug::timing_log_scalar("dettato.drums.trunk.device_workspace_mb",
            ggml_backend_sched_get_buffer_size(scheduler, execution_.backend()) / (1024. * 1024));
        debug::timing_log_scalar("dettato.drums.trunk.host_workspace_mb",
            ggml_backend_sched_get_buffer_size(scheduler, host_execution_->backend()) / (1024. * 1024));
    }
    write_tensor_i32(indices_, selection);
    for (const auto & state : zero_states_) write_tensor_f32(state.second, std::vector<float>(static_cast<size_t>(state.first)));
}

std::array<TensorValue, 3> DrumTrunk::run(const std::vector<float> & embeddings) {
    if (embeddings.empty() || embeddings.size() % width_)
        throw std::invalid_argument("Dettato drum trunk expects frame embeddings of its configured width");
    prepare(static_cast<int64_t>(embeddings.size() / width_)); write_tensor_f32(input_, embeddings);
    const auto start = std::chrono::steady_clock::now(); graph_->compute();
    debug::timing_log_scalar("dettato.drums.trunk_ms", debug::elapsed_ms(start));
    return outputs_;
}

OrchestralDrumExpert::OrchestralDrumExpert(const assets::TensorSource & source, size_t index,
    const io::json::Value & config, const ExecutionContext & execution, GraphWorkspace *workspace)
    : execution_(execution), workspace_(execution.backend_type() == BackendType::Cuda || execution.backend_type() == BackendType::Cpu ? workspace : nullptr),
      embedder_(source, "orchestral." + std::to_string(index) + ".base.", config.require("drum_kit"), execution, workspace),
      trunk_(source, "orchestral." + std::to_string(index) + ".base.", config.require("drum_kit"), execution, workspace),
      store_(execution.backend(), execution.backend_type(), "dettato.drums.expert." + std::to_string(index) + ".weights", 1024 * 1024),
      hidden_(config.require("orchestral").require("adapter").require("h").as_i64()),
      layers_(config.require("orchestral").require("adapter").require("layers").as_i64()),
      heads_(config.require("orchestral").require("adapter").require("heads").as_i64()),
      base_width_(config.require("drum_kit").require("w").as_i64()) {
    const auto prefix = "orchestral." + std::to_string(index) + ".";
    for (const auto & metadata : source.tensors()) {
        if (metadata.name.rfind(prefix, 0) != 0) continue;
        const auto name = metadata.name.substr(prefix.size());
        if (name.rfind("base.", 0) == 0) continue;
        weights_.emplace(name, store_.load_tensor(source, metadata.name, assets::TensorStorageType::Native, metadata.shape));
    }
    store_.upload();
    if (execution.backend_type() != BackendType::Cpu && execution.backend_type() != BackendType::Cuda)
        host_execution_ = std::make_unique<ExecutionContext>(BackendConfig{BackendType::Cpu, 0, execution.config().threads});
}

void OrchestralDrumExpert::prepare(int64_t frames) {
    if (frames < 2 || frames % 2) throw std::invalid_argument("Dettato orchestral expert needs a positive even feature-frame count");
    if (graph_ && graph_->graph() && frames_ == frames) return;
    embedder_.prepare(frames);
    trunk_.prepare(frames / 2);
    graph_.reset(); frames_ = frames;
    graph_ = std::make_unique<PreparedGraph>(execution_, PreparedGraphConfig{"dettato.drums.expert", 4 * 1024 * 1024, 4096, 1024 * 1024});
    auto & ctx = graph_->build_context(); const int64_t time = frames / 2;
    cq_input_ = make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 3, frames, 120}));
    hidden_input_ = make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, time, base_width_}));
    base_input_ = make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, time, 61}));
    for (const auto & input : {cq_input_, hidden_input_, base_input_}) ggml_set_input(input.tensor);
    const auto project = [&](const TensorValue & x, const std::string & name) {
        const auto weight = weights_.at(name + ".weight");
        return LinearModule({weight.shape.dims[1], weight.shape.dims[0], true, GGML_PREC_F32})
            .build(ctx, x, {weight, weights_.at(name + ".bias")});
    };
    // The shifted input and its minimum padding were packed on the host from
    // the already-host-resident features. Mean/normalization remain in graph.
    const auto base_cq = SliceModule({1, 0, 1}).build(ctx, cq_input_);
    const auto mean = ReduceMeanModule({2}).build(ctx, ReduceMeanModule({3}).build(ctx, base_cq));
    auto x = wrap_tensor(ggml_sub(ctx.ggml, cq_input_.tensor, mean.tensor), cq_input_.shape);
    x = ScaleBiasModule({.25f, 0}).build(ctx, x);
    TensorValue residual;
    for (int layer = 1; layer <= 3; ++layer) {
        const auto name = "cq.c" + std::to_string(layer);
        const auto weight = weights_.at(name + ".weight");
        Conv2dConfig conv;
        conv.in_channels = weight.shape.dims[1]; conv.out_channels = weight.shape.dims[0];
        conv.kernel_height = weight.shape.dims[2]; conv.kernel_width = weight.shape.dims[3];
        conv.padding_height = static_cast<int>(conv.kernel_height / 2); conv.padding_width = static_cast<int>(conv.kernel_width / 2);
        conv.stride_height = layer == 2 ? 2 : 1;
        if (execution_.backend_type() == BackendType::Cpu) {
            x = Conv2dModule(conv).build(ctx, x, {weight, weights_.at(name + ".bias")});
        } else {
            // Keep CQ workspace bounded with the existing native direct op.
            // Activations are F32; the package supplies contiguous F16/F32 kernels.
            const auto input = ensure_backend_addressable_layout(ctx, x);
            const auto kernel = ensure_backend_addressable_layout(ctx, weight);
            auto * output = ggml_conv_2d_direct(ctx.ggml, kernel.tensor, input.tensor,
                conv.stride_width, conv.stride_height, conv.padding_width, conv.padding_height,
                conv.dilation_width, conv.dilation_height);
            const auto bias = reshape_tensor(ctx, weights_.at(name + ".bias"),
                TensorShape::from_dims({1, conv.out_channels, 1, 1}));
            x = wrap_tensor(output,
                TensorShape::from_dims({output->ne[3], output->ne[2], output->ne[1], output->ne[0]}), GGML_TYPE_F32);
            x = AddModule{false, true}.build(ctx, x, bias);
        }
        const auto norm = "cq.n" + std::to_string(layer);
        x = GroupNormModule({conv.out_channels, 4, 1e-5f, false, false}).build(ctx, x, {});
        if (ctx.backend_type == BackendType::Cuda) {
            x = wrap_tensor(ggml_affine(ctx.ggml, x.tensor, weights_.at(norm + ".weight").tensor,
                weights_.at(norm + ".bias").tensor, 2), x.shape);
        } else {
            const auto affine_shape = TensorShape::from_dims({1, conv.out_channels, 1, 1});
            const auto scale = reshape_tensor(ctx, weights_.at(norm + ".weight"), affine_shape);
            const auto bias = reshape_tensor(ctx, weights_.at(norm + ".bias"), affine_shape);
            x = AddModule{false, true}.build(ctx, MulModule{true, true}.build(ctx, x, scale), bias);
        }
        x = SiluModule{}.build(ctx, x);
        if (layer == 2) residual = x;
    }
    x = AddModule{}.build(ctx, x, residual);
    const int64_t cq_channels = x.shape.dims[1];
    x = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
    x = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, x), TensorShape::from_dims({1, time, cq_channels * 120}));
    x = SiluModule{}.build(ctx, project(x, "cq.proj"));
    x = ConcatModule({2}).build(ctx, ConcatModule({2}).build(ctx, hidden_input_, x), SigmoidModule{}.build(ctx, base_input_));
    x = SiluModule{}.build(ctx, project(x, "inp"));
    initial_ = make_tensor(graph_->persistent_build_context(), GGML_TYPE_F32, TensorShape::from_dims({1, hidden_})); ggml_set_input(initial_.tensor);
    x = TransposeModule({{1, 0, 2}, 3}).build(ctx, x);
    for (int64_t layer = 0; layer < layers_; ++layer) {
        std::array<GRUWeights,2> parameters;
        for(int d=0;d<2;++d) {
            const auto suffix="_l"+std::to_string(layer)+(d ? "_reverse" : "");
            parameters[d]={weights_.at("rnn.weight_ih"+suffix),weights_.at("rnn.weight_hh"+suffix),
                weights_.at("rnn.bias_ih"+suffix),weights_.at("rnn.bias_hh"+suffix)};
        }
        x = BidirectionalGRU({x.shape.dims[2], hidden_, ctx.backend_type == BackendType::Cuda ? GRURecurrenceLowering::DeviceScan : GRURecurrenceLowering::HostRecurrence})
            .build(ctx, x, initial_, initial_, parameters[0], parameters[1]).sequence;
    }
    const auto channels = hidden_ * 2;
    x = TransposeModule({{1, 0, 2}, 3}).build(ctx, x);
    const auto packed = project(x, "qkv"); std::array<TensorValue, 3> qkv;
    for (int i = 0; i < 3; ++i) {
        auto part = SliceModule({2, i * channels, channels}).build(ctx, packed);
        part = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, part), TensorShape::from_dims({1, time, heads_, channels / heads_}));
        qkv[i] = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, part);
    }
    const auto attention_lowering = execution_.backend_type() == BackendType::Cpu
        ? ScaledDotProductAttentionLowering::Explicit : ScaledDotProductAttentionLowering::Flash;
    auto attention = ScaledDotProductAttentionModule({channels / heads_, attention_lowering, GGML_PREC_F32})
        .build(ctx, qkv[0], qkv[1], qkv[2]);
    attention = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, attention), TensorShape::from_dims({1, time, channels}));
    x = LayerNormModule({channels, 1e-5f, true, true}).build(ctx, AddModule{}.build(ctx, x, project(attention, "ao")),
        {weights_.at("ln.weight"), weights_.at("ln.bias")});
    outputs_[0] = AddModule{}.build(ctx, base_input_, project(x, "delta"));
    outputs_[1] = project(x, "timp");
    auto pooled = ReduceMeanModule({1}).build(ctx, x);
    pooled = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, pooled), TensorShape::from_dims({1, channels}));
    outputs_[2] = project(SiluModule{}.build(ctx, project(pooled, "gate.0")), "gate.2");
    for (const auto & output : outputs_) {
        ggml_set_output(output.tensor); if (output.tensor->view_src) ggml_set_output(output.tensor->view_src);
        ggml_build_forward_expand(graph_->graph(), output.tensor);
    }
    const auto scheduler = host_execution_
        ? graph_->create_scheduler({execution_.backend(), host_execution_->backend()}) : nullptr;
    graph_->allocate_persistent();
    if (workspace_) workspace_->add(*graph_);
    else graph_->allocate();
    if (scheduler) {
        debug::timing_log_scalar("dettato.drums.expert.device_workspace_mb",
            ggml_backend_sched_get_buffer_size(scheduler, execution_.backend()) / (1024. * 1024));
        debug::timing_log_scalar("dettato.drums.expert.host_workspace_mb",
            ggml_backend_sched_get_buffer_size(scheduler, host_execution_->backend()) / (1024. * 1024));
    }
    write_tensor_f32(initial_, std::vector<float>(static_cast<size_t>(hidden_)));
}

std::array<TensorValue, 5> OrchestralDrumExpert::run(const std::vector<float> & features) {
    if (features.size() % 280) throw std::invalid_argument("Dettato orchestral features must have 280 bands");
    const int64_t frames = static_cast<int64_t>(features.size() / 280) / 2 * 2;
    prepare(frames);
    std::vector<float> mel(static_cast<size_t>(frames * 160)), cq(static_cast<size_t>(frames * 120));
    for (int64_t t = 0; t < frames; ++t) {
        std::copy_n(features.data() + t * 280, 160, mel.data() + t * 160);
        std::copy_n(features.data() + t * 280 + 160, 120, cq.data() + t * 120);
    }
    const auto trunk = trunk_.run(embedder_.run(mel));
    const float minimum = *std::min_element(cq.begin(), cq.end());
    const std::array<int, 3> offsets{0, 14, 24}; std::vector<float> stacked(static_cast<size_t>(3 * frames * 120));
    for (int ch = 0; ch < 3; ++ch) for (int64_t t = 0; t < frames; ++t) for (int f = 0; f < 120; ++f)
        stacked[static_cast<size_t>((ch * frames + t) * 120 + f)] = f + offsets[ch] < 120 ? cq[static_cast<size_t>(t * 120 + f + offsets[ch])] : minimum;
    write_tensor_f32(cq_input_, stacked); ggml_backend_tensor_copy(trunk[0].tensor, hidden_input_.tensor); ggml_backend_tensor_copy(trunk[1].tensor, base_input_.tensor);
    const auto start = std::chrono::steady_clock::now(); graph_->compute();
    debug::timing_log_scalar("dettato.drums.expert_ms", debug::elapsed_ms(start));
    return {trunk[1], outputs_[0], outputs_[1], trunk[2], outputs_[2]};
}

struct DrumTranscriber::Impl {
    const ExecutionContext & execution;
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

    Impl(const assets::TensorSource & source, const io::json::Value & cfg, const ExecutionContext & ctx)
        : execution(ctx), config(cfg), workspace(ctx.backend()),
          kit_embedder(source, "drum_kit.", cfg.require("drum_kit"), ctx, &workspace),
          kit_trunk(source, "drum_kit.", cfg.require("drum_kit"), ctx, &workspace) {
        const auto count = cfg.require("orchestral").require("n_models").as_i64();
        if (count <= 0) throw std::invalid_argument("Dettato drum routing requires orchestral experts");
        for (int64_t i = 0; i < count; ++i) experts.push_back(std::make_unique<OrchestralDrumExpert>(source, static_cast<size_t>(i), cfg, ctx, &workspace));
    }
    void prepare(int64_t count) {
        if (graph && frames == count) return;
        workspace.reset();
        graph.reset(); frames = 0;
        graph = std::make_unique<PreparedGraph>(execution, PreparedGraphConfig{"dettato.drums.probabilities", 1024 * 1024, 1024, 64 * 1024});
        auto & ctx = graph->build_context();
        for (auto & input : kit_inputs) {
            input = make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, count, 61})); ggml_set_input(input.tensor);
        }
        outputs[0] = SigmoidModule{}.build(ctx, kit_inputs[0]); outputs[1] = SigmoidModule{}.build(ctx, kit_inputs[1]);
        expert_inputs.resize(experts.size()); std::array<TensorValue, 3> sums;
        TensorValue routers;
        for (size_t i = 0; i < experts.size(); ++i) {
            auto & inputs = expert_inputs[i];
            for (size_t j = 0; j < inputs.size(); ++j) {
                inputs[j] = make_tensor(ctx, GGML_TYPE_F32, j == 4 ? TensorShape::from_dims({1, 1})
                    : TensorShape::from_dims({1, count, j == 2 ? 13 : 61}));
                ggml_set_input(inputs[j].tensor);
            }
            // Preserve upstream's base + (corrected - base) rounding boundary.
            const auto difference = wrap_tensor(ggml_sub(ctx.ggml, inputs[1].tensor, inputs[0].tensor), inputs[0].shape);
            const std::array<TensorValue, 3> p{
                SigmoidModule{}.build(ctx, AddModule{}.build(ctx, inputs[0], difference)),
                SigmoidModule{}.build(ctx, inputs[2]), SigmoidModule{}.build(ctx, inputs[3])};
            for (size_t k = 0; k < sums.size(); ++k) sums[k] = i == 0 ? p[k] : AddModule{}.build(ctx, sums[k], p[k]);
            const auto router = SigmoidModule{}.build(ctx, inputs[4]);
            routers = i == 0 ? router : ConcatModule({1}).build(ctx, routers, router);
        }
        divisor = make_tensor(graph->persistent_build_context(), GGML_TYPE_F32, TensorShape::from_dims({1})); ggml_set_input(divisor.tensor);
        for (size_t i = 0; i < sums.size(); ++i) outputs[i + 2] = wrap_tensor(ggml_div(ctx.ggml, sums[i].tensor, divisor.tensor), sums[i].shape);
        outputs[5] = routers;
        for (const auto & output : outputs) { ggml_set_output(output.tensor); ggml_build_forward_expand(graph->graph(), output.tensor); }
        graph->allocate_persistent(); graph->allocate(); write_tensor_f32(divisor, {static_cast<float>(experts.size())});
        kit_embedder.prepare(2 * count, 9000);
        kit_trunk.prepare(count);
        for (auto &expert : experts) expert->prepare(2 * count);
        workspace.prepare();
        frames = count;
        debug::timing_log_scalar("dettato.drums.shared_workspace_mb", workspace.bytes() / (1024. * 1024));
    }
};

DrumTranscriber::DrumTranscriber(const assets::TensorSource & source, const io::json::Value & config,
    const ExecutionContext & execution) : impl_(std::make_unique<Impl>(source, config, execution)) {}
DrumTranscriber::~DrumTranscriber() = default;

DrumProbabilities DrumTranscriber::probabilities(const std::vector<float> & features) {
    if (features.size() % 280 || features.size() < 560) throw std::invalid_argument("Dettato drum network requires at least two 280-band frames");
    const auto start = std::chrono::steady_clock::now();
    auto & state = *impl_; const int64_t frames = static_cast<int64_t>(features.size() / 280) / 2 * 2;
    state.prepare(frames / 2);
    std::vector<float> mel(static_cast<size_t>(frames * 160));
    for (int64_t t = 0; t < frames; ++t) std::copy_n(features.data() + t * 280, 160, mel.data() + t * 160);
    const auto kit = state.kit_trunk.run(state.kit_embedder.run(mel, 9000));
    ggml_backend_tensor_copy(kit[1].tensor, state.kit_inputs[0].tensor); ggml_backend_tensor_copy(kit[2].tensor, state.kit_inputs[1].tensor);
    for (size_t i = 0; i < state.experts.size(); ++i) {
        const auto output = state.experts[i]->run(features);
        for (size_t j = 0; j < output.size(); ++j) ggml_backend_tensor_copy(output[j].tensor, state.expert_inputs[i][j].tensor);
    }
    state.graph->compute();
    const auto routers = read_tensor_f32(state.outputs[5].tensor);
    DrumProbabilities result;
    // Upstream converts each F32 router to a Python float before averaging.
    result.router = std::accumulate(routers.begin(), routers.end(), 0.) / routers.size();
    result.kit = read_tensor_f32(state.outputs[0].tensor); result.kit_velocity = read_tensor_f32(state.outputs[1].tensor);
    result.orchestral = read_tensor_f32(state.outputs[2].tensor); result.timpani = read_tensor_f32(state.outputs[3].tensor);
    result.velocity = read_tensor_f32(state.outputs[4].tensor);
    debug::timing_log_scalar("dettato.drums.network_ms", debug::elapsed_ms(start));
    return result;
}

DrumResult DrumTranscriber::transcribe_features(const std::vector<float> & features) {
    const auto start = std::chrono::steady_clock::now();
    DrumResult result;
    if (!features.empty()) {
        const auto p = probabilities(features);
        result.router = p.router;
        result.model = p.router < impl_->config.require("thresholds").require("router").as_number() ? "drum-kit" : "orchestral";
        result.track = decode_drum_hits(p, impl_->config);
    }
    debug::timing_log_scalar("dettato.drums.wall_ms", debug::elapsed_ms(start));
    return result;
}
} // namespace engine::models::dettato::detail
