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

namespace engine::models::dettato::detail {
using namespace core;
using namespace modules;

DrumTrunk::DrumTrunk(const assets::TensorSource & source, std::string prefix,
    const io::json::Value & config, const ExecutionContext & execution)
    : execution_(execution), store_(execution.backend(), execution.backend_type(), "dettato.drums." + prefix + "trunk.weights", 1024 * 1024),
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
    if (execution.backend_type() != BackendType::Cpu)
        host_execution_ = std::make_unique<ExecutionContext>(BackendConfig{BackendType::Cpu, 0, execution.config().threads});
}

void DrumTrunk::prepare(int64_t frames) {
    if (frames <= 0) throw std::invalid_argument("Dettato drum trunk requires nonempty frame embeddings");
    if (graph_ && frames_ == frames) return;
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
        const auto output = BidirectionalGRU({x.shape.dims[2], hidden, GRURecurrenceLowering::HostRecurrence})
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
            const auto attention_lowering = execution_.backend_type() == BackendType::Vulkan
                ? ScaledDotProductAttentionLowering::Flash : ScaledDotProductAttentionLowering::Explicit;
            auto x = ScaledDotProductAttentionModule({channels / heads, attention_lowering, GGML_PREC_F32})
                .build(ctx, qkv[0], qkv[1], qkv[2]);
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
    for (const auto & output : outputs_) {
        ggml_set_output(output.tensor); if (output.tensor->view_src) ggml_set_output(output.tensor->view_src);
        ggml_build_forward_expand(graph_->graph(), output.tensor);
    }
    if (host_execution_) graph_->create_scheduler({execution_.backend(), host_execution_->backend()});
    graph_->allocate_persistent(); graph_->allocate(); write_tensor_i32(indices_, selection);
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
} // namespace engine::models::dettato::detail
