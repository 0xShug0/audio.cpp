#include "drums.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "gru.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include <algorithm>

namespace engine::models::dettato::detail {
using namespace core;
using namespace modules;

OrchestralDrumExpert::OrchestralDrumExpert(const assets::TensorSource & source, size_t index,
    const io::json::Value & config, const ExecutionContext & execution)
    : execution_(execution),
      embedder_(source, "orchestral." + std::to_string(index) + ".base.", config.require("drum_kit"), execution),
      trunk_(source, "orchestral." + std::to_string(index) + ".base.", config.require("drum_kit"), execution),
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
    if (execution.backend_type() != BackendType::Cpu)
        host_execution_ = std::make_unique<ExecutionContext>(BackendConfig{BackendType::Cpu, 0, execution.config().threads});
}

void OrchestralDrumExpert::prepare(int64_t frames) {
    if (frames < 2 || frames % 2) throw std::invalid_argument("Dettato orchestral expert needs a positive even feature-frame count");
    if (graph_ && frames_ == frames) return;
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
            x = wrap_tensor(ggml_add(ctx.ggml, output, ggml_repeat(ctx.ggml, bias.tensor, output)),
                TensorShape::from_dims({output->ne[3], output->ne[2], output->ne[1], output->ne[0]}), GGML_TYPE_F32);
        }
        const auto norm = "cq.n" + std::to_string(layer);
        x = GroupNormModule({conv.out_channels, 4, 1e-5f, false, false}).build(ctx, x, {});
        const auto affine_shape = TensorShape::from_dims({1, conv.out_channels, 1, 1});
        const auto scale = reshape_tensor(ctx, weights_.at(norm + ".weight"), affine_shape);
        const auto bias = reshape_tensor(ctx, weights_.at(norm + ".bias"), affine_shape);
        x = AddModule{false, true}.build(ctx, MulModule{true, true}.build(ctx, x, scale), bias);
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
        x = BidirectionalGRU({x.shape.dims[2], hidden_, GRURecurrenceLowering::HostRecurrence})
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
    if (host_execution_) graph_->create_scheduler({execution_.backend(), host_execution_->backend()});
    graph_->allocate_persistent(); graph_->allocate(); write_tensor_f32(initial_, std::vector<float>(static_cast<size_t>(hidden_)));
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
} // namespace engine::models::dettato::detail
