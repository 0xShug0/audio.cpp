#include "engine/models/mossformer2/runtime.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/runtime/graph_optimizer.h"

#include <ggml-alloc.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace engine::models::mossformer2 {
namespace {
using core::TensorValue;
using core::TensorShape;
using TensorMap = std::unordered_map<std::string, TensorValue>;
constexpr const char * kBackbonePrefix = "mask_net.mdl.intra_mdl.mossformerM.";
constexpr int64_t kAttentionGroupSize = 256;
constexpr int64_t kQueryKeyWidth = 128;
constexpr int64_t kRotaryWidth = 32;
constexpr int64_t kRotaryPairs = kRotaryWidth / 2;
constexpr size_t kWeightContextBytes = 4 * 1024 * 1024;

class MossFormer2Graph {
public:
    MossFormer2Graph(core::ExecutionContext & execution, const MossFormer2Config & c,
        const TensorMap & weights, int64_t samples)
        : backend_(execution.backend()), weights_(weights) {
        const size_t nodes = 1024 + 512 * static_cast<size_t>(c.layers) + 64 * static_cast<size_t>(c.speakers);
        context_.reset(ggml_init({nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true}));
        if (!context_) throw std::runtime_error("MossFormer2 graph context allocation failed");
        ctx_ = {context_.get(), "mossformer2", execution.backend_type()};
        auto input = core::make_tensor(ctx_, GGML_TYPE_F32, TensorShape::from_dims({1, 1, samples}));
        input_ = input.tensor;
        ggml_set_input(input_);
        auto encoded = modules::Conv1dModule({1, c.channels, c.kernel, c.kernel / 2, 0, 1, false})
            .build(ctx_, input, {weights_.at("enc.conv1d.weight"), std::nullopt});
        encoded = modules::ReluModule().build(ctx_, encoded);
        frames_ = encoded.shape.dims[2];
        auto x = modules::GroupNormModule({c.channels, 1, 1e-8f}).build(ctx_, encoded,
            {weights_.at("mask_net.norm.weight"), weights_.at("mask_net.norm.bias")});
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, x);
        x = core::reshape_tensor(ctx_, core::ensure_backend_addressable_layout(ctx_, x),
            TensorShape::from_dims({frames_, c.channels}));
        x = linear(x, "mask_net.conv1d_encoder");
        position_ = core::make_tensor(ctx_, GGML_TYPE_F32, TensorShape::from_dims({frames_, c.hidden})).tensor;
        ggml_set_input(position_);
        // This constant is uploaded once and must survive in-place graph reuse.
        ggml_set_output(position_);
        x = modules::AddModule().build(ctx_, x, core::wrap_tensor(position_, x.shape));
        auto residual = x;
        cos_ = core::make_tensor(ctx_, GGML_TYPE_F32, TensorShape::from_dims({frames_, kRotaryPairs, 1})).tensor;
        sin_ = core::make_tensor(ctx_, GGML_TYPE_F32, TensorShape::from_dims({frames_, kRotaryPairs, 1})).tensor;
        ggml_set_input(cos_);
        ggml_set_input(sin_);
        ggml_set_output(cos_);
        ggml_set_output(sin_);
        for (int layer = 0; layer < c.layers; ++layer) {
            x = attention(x, std::string(kBackbonePrefix) + "layers." + std::to_string(layer));
            x = fsmn(x, std::string(kBackbonePrefix) + "fsmn." + std::to_string(layer));
        }
        x = modules::LayerNormModule({c.hidden, 1e-6f}).build(ctx_, x,
            {weights_.at("mask_net.mdl.intra_mdl.norm.weight"), weights_.at("mask_net.mdl.intra_mdl.norm.bias")});
        x = modules::TransposeModule({{1, 0, 2, 3}, 2}).build(ctx_, x);
        x = core::reshape_tensor(ctx_, core::ensure_backend_addressable_layout(ctx_, x), TensorShape::from_dims({1, c.hidden, frames_}));
        x = modules::GroupNormModule({c.hidden, 1, 1e-8f}).build(ctx_, x,
            {weights_.at("mask_net.mdl.intra_norm.weight"), weights_.at("mask_net.mdl.intra_norm.bias")});
        x = core::reshape_tensor(ctx_, x, TensorShape::from_dims({c.hidden, frames_}));
        x = modules::TransposeModule({{1, 0, 2, 3}, 2}).build(ctx_, x);
        x = modules::AddModule().build(ctx_, x, residual);
        x = modules::PReluModule({1})
            .build(ctx_, x, weights_.at("mask_net.prelu.weight"));
        x = linear(x, "mask_net.conv1d_out");
        for (int speaker = 0; speaker < c.speakers; ++speaker) {
            auto mask = modules::SliceModule({1, speaker * c.hidden, c.hidden}).build(ctx_, x);
            auto value = modules::TanhModule().build(ctx_, linear(mask, "mask_net.output.0"));
            auto gate = modules::SigmoidModule().build(ctx_, linear(mask, "mask_net.output_gate.0"));
            mask = linear(modules::MulModule().build(ctx_, value, gate), "mask_net.conv1_decoder");
            mask = modules::ReluModule().build(ctx_, mask);
            mask = modules::TransposeModule({{1, 0, 2, 3}, 2}).build(ctx_, mask);
            mask = core::reshape_tensor(ctx_, core::ensure_backend_addressable_layout(ctx_, mask), encoded.shape);
            auto separated = modules::MulModule().build(ctx_, encoded, mask);
            auto waveform = modules::ConvTranspose1dModule({c.channels, 1, c.kernel, c.kernel / 2, 0, 1, false})
                .build(ctx_, separated, {weights_.at("dec.weight"), std::nullopt});
            waveform = core::ensure_backend_addressable_layout(ctx_, waveform);
            outputs_.push_back(waveform.tensor);
            ggml_set_output(waveform.tensor);
            if (waveform.tensor->view_src) ggml_set_output(waveform.tensor->view_src);
        }
        graph_ = ggml_new_graph_custom(ctx_.ggml, nodes, false);
        for (auto output : outputs_) ggml_build_forward_expand(graph_, output);
        if (execution.backend_type() == core::BackendType::Cpu) {
            auto options = runtime::graph_optimization_options_for_backend(runtime::GraphOptimizationBackend::Cpu);
            // Keep view lifetimes visible to the graph allocator.
            options.elide_metadata_only_ops = false;
            runtime::optimize_graph(*graph_, options);
        }
        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        if (!ggml_gallocr_alloc_graph(allocator_.get(), graph_)) throw std::runtime_error("MossFormer2 graph allocation failed");
        core::prepare_host_graph_plan(execution, graph_, host_plan_);
        debug::timing_log_context_reservation("mossformer2.graph", context_.get());
        debug::timing_log_scalar("mossformer2.graph.workspace_bytes", ggml_gallocr_get_buffer_size(allocator_.get(), 0));
    }

    ~MossFormer2Graph() { core::release_backend_graph_resources(backend_, graph_, true); }

    void set_positions(const std::vector<float> & frequencies, float scale, const std::vector<float> & rotary) {
        const auto channels = position_->ne[0];
        std::vector<float> values(frames_ * channels);
        for (int64_t t = 0; t < frames_; ++t) {
            for (int64_t f = 0; f < channels / 2; ++f) {
                const float angle = static_cast<float>(t) * frequencies[f];
                values[t * channels + f] = std::sin(angle) * scale;
                values[t * channels + channels / 2 + f] = std::cos(angle) * scale;
            }
        }
        ggml_backend_tensor_set(position_, values.data(), 0, values.size() * sizeof(float));
        values.resize(frames_ * kRotaryPairs);
        std::vector<float> sine(values.size());
        for (int64_t t = 0; t < frames_; ++t) {
            for (int64_t f = 0; f < kRotaryPairs; ++f) {
                const float angle = static_cast<float>(t) * rotary[f];
                values[t * kRotaryPairs + f] = std::cos(angle);
                sine[t * kRotaryPairs + f] = std::sin(angle);
            }
        }
        ggml_backend_tensor_set(cos_, values.data(), 0, values.size() * sizeof(float));
        ggml_backend_tensor_set(sin_, sine.data(), 0, sine.size() * sizeof(float));
    }

    std::vector<std::vector<float>> run(const std::vector<float> & input) {
        ggml_backend_tensor_set(input_, input.data(), 0, input.size() * sizeof(float));
        if (core::compute_backend_graph(backend_, graph_, host_plan_.plan) != GGML_STATUS_SUCCESS) throw std::runtime_error("MossFormer2 graph execution failed");
        std::vector<std::vector<float>> result;
        for (auto output : outputs_) {
            result.emplace_back(input.size(), 0.0f);
            ggml_backend_tensor_get(output, result.back().data(), 0, ggml_nbytes(output));
        }
        return result;
    }

private:
    TensorValue linear(TensorValue x, const std::string & name) {
        const auto & source = weights_.at(name + ".weight");
        auto weight = core::reshape_tensor(ctx_, source, TensorShape::from_dims({source.shape.dims[0], source.shape.dims[1]}));
        auto bias = weights_.find(name + ".bias");
        return modules::LinearModule({weight.shape.dims[1], weight.shape.dims[0], bias != weights_.end()})
            .build(ctx_, x, {weight, bias == weights_.end() ? std::nullopt : std::optional<TensorValue>(bias->second)});
    }

    TensorValue feed_forward(TensorValue x, const std::string & name, bool scale_norm) {
        const auto channels = x.shape.last_dim();
        if (scale_norm) {
            // Upstream clamps the RMS itself; adding epsilon inside RMSNorm differs.
            auto square = modules::MulModule().build(ctx_, x, x);
            auto rms = modules::SqrtModule().build(ctx_, modules::ReduceMeanModule({1}).build(ctx_, square));
            rms = core::wrap_tensor(ggml_clamp(ctx_.ggml, rms.tensor, 1e-5f, std::numeric_limits<float>::max()), rms.shape);
            rms = modules::RepeatModule({x.shape}).build(ctx_, rms);
            x = core::wrap_tensor(ggml_div(ctx_.ggml, x.tensor, rms.tensor), x.shape);
            auto gain = core::reshape_tensor(ctx_, weights_.at(name + ".mdl.0.g"), TensorShape::from_dims({1, 1}));
            x = modules::MulModule().build(ctx_, x, modules::RepeatModule({x.shape}).build(ctx_, gain));
        } else {
            x = modules::LayerNormModule({channels, 1e-5f}).build(ctx_, x,
                {weights_.at(name + ".mdl.0.weight"), weights_.at(name + ".mdl.0.bias")});
        }
        x = modules::SiluModule().build(ctx_, linear(x, name + ".mdl.1"));
        const auto shape = x.shape;
        auto temporal = modules::TransposeModule({{1, 0, 2, 3}, 2}).build(ctx_, x);
        temporal = core::reshape_tensor(ctx_, core::ensure_backend_addressable_layout(ctx_, temporal),
            TensorShape::from_dims({1, shape.last_dim(), frames_}));
        auto weight = weights_.at(name + ".mdl.3.sequential.1.conv.weight");
        weight = core::reshape_tensor(ctx_, weight, TensorShape::from_dims({shape.last_dim(), 1, 17}));
        temporal = modules::DepthwiseConv1dModule({shape.last_dim(), 17, 1, 8, 1, false})
            .build(ctx_, temporal, {weight, std::nullopt});
        temporal = core::reshape_tensor(ctx_, temporal, TensorShape::from_dims({shape.last_dim(), frames_}));
        temporal = modules::TransposeModule({{1, 0, 2, 3}, 2}).build(ctx_, temporal);
        return modules::AddModule().build(ctx_, x, temporal);
    }

    TensorValue rotary(TensorValue x) {
        auto leading = modules::SliceModule({1, 0, kRotaryWidth}).build(ctx_, x);
        leading = core::reshape_tensor(ctx_, core::ensure_backend_addressable_layout(ctx_, leading), TensorShape::from_dims({frames_, kRotaryPairs, 2}));
        auto even = modules::SliceModule({2, 0, 1}).build(ctx_, leading);
        auto odd = modules::SliceModule({2, 1, 1}).build(ctx_, leading);
        const auto cosine = core::wrap_tensor(cos_, even.shape);
        const auto sine = core::wrap_tensor(sin_, even.shape);
        TensorValue rotated;
        if (core::uses_ggml_cuda_or_hip_backend(ctx_.backend_type)) {
            rotated = modules::RopeInterleavedPairsModule().build(ctx_, even, odd, cosine, sine);
        } else {
            const auto pairs = core::reshape_tensor(ctx_, leading, TensorShape::from_dims({frames_, 1, kRotaryPairs, 2}));
            const auto cos = core::reshape_tensor(ctx_, cosine, TensorShape::from_dims({frames_, 1, kRotaryPairs, 1}));
            const auto sin = core::reshape_tensor(ctx_, sine, cos.shape);
            rotated = modules::SplitRoPEModule({2}).build(ctx_, pairs, cos, sin);
        }
        rotated = core::reshape_tensor(ctx_, rotated, TensorShape::from_dims({frames_, kRotaryWidth}));
        return modules::ConcatModule({1}).build(ctx_, rotated,
            modules::SliceModule({1, kRotaryWidth, x.shape.last_dim() - kRotaryWidth}).build(ctx_, x));
    }

    TensorValue attention(TensorValue x, const std::string & name) {
        auto residual = x;
        const int64_t channels = x.shape.last_dim();
        auto shift = modules::SliceModule({1, 0, channels / 2}).build(ctx_, x);
        shift = core::reshape_tensor(ctx_, core::ensure_backend_addressable_layout(ctx_, shift),
            TensorShape::from_dims({1, 1, frames_, channels / 2}));
        shift = modules::Pad2dModule({0, 0, 1, 0}).build(ctx_, shift);
        shift = core::reshape_tensor(ctx_, shift, TensorShape::from_dims({frames_ + 1, channels / 2}));
        shift = modules::SliceModule({0, 0, frames_}).build(ctx_, shift);
        x = modules::ConcatModule({1}).build(ctx_, shift, modules::SliceModule({1, channels / 2, channels / 2}).build(ctx_, x));
        auto vu = feed_forward(x, name + ".to_hidden", true);
        auto qk = feed_forward(x, name + ".to_qk", true);
        std::vector<TensorValue> qks;
        for (int i = 0; i < 4; ++i) {
            auto gamma = modules::SliceModule({0, i, 1}).build(ctx_, weights_.at(name + ".qk_offset_scale.gamma"));
            auto beta = modules::SliceModule({0, i, 1}).build(ctx_, weights_.at(name + ".qk_offset_scale.beta"));
            auto v = modules::MulModule().build(ctx_, qk, modules::RepeatModule({qk.shape}).build(ctx_, gamma));
            v = modules::AddModule().build(ctx_, v, modules::RepeatModule({qk.shape}).build(ctx_, beta));
            qks.push_back(rotary(v));
        }
        const int64_t padded = (frames_ + kAttentionGroupSize - 1) / kAttentionGroupSize * kAttentionGroupSize;
        auto q = core::reshape_tensor(ctx_, qks[0], TensorShape::from_dims({1, 1, frames_, kQueryKeyWidth}));
        auto k = core::reshape_tensor(ctx_, qks[2], TensorShape::from_dims({1, 1, frames_, kQueryKeyWidth}));
        auto values = core::reshape_tensor(ctx_, vu, TensorShape::from_dims({1, 1, frames_, vu.shape.last_dim()}));
        q = modules::Pad2dModule({0, 0, 0, padded - frames_}).build(ctx_, q);
        k = modules::Pad2dModule({0, 0, 0, padded - frames_}).build(ctx_, k);
        values = modules::Pad2dModule({0, 0, 0, padded - frames_}).build(ctx_, values);
        q = core::reshape_tensor(ctx_, q, TensorShape::from_dims({padded / kAttentionGroupSize, kAttentionGroupSize, kQueryKeyWidth}));
        k = core::reshape_tensor(ctx_, k, q.shape);
        k = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, k);
        auto scores = modules::MatMulModule().build(ctx_, q, k);
        scores = core::wrap_tensor(ggml_scale(ctx_.ggml, scores.tensor, 1.0f / kAttentionGroupSize), scores.shape);
        scores = modules::ReluModule().build(ctx_, scores);
        scores = modules::MulModule().build(ctx_, scores, scores);
        values = core::reshape_tensor(ctx_, values, TensorShape::from_dims({padded / kAttentionGroupSize, kAttentionGroupSize, vu.shape.last_dim()}));
        auto local = modules::MatMulModule().build(ctx_, scores, values);
        local = core::reshape_tensor(ctx_, local, TensorShape::from_dims({padded, vu.shape.last_dim()}));
        local = modules::SliceModule({0, 0, frames_}).build(ctx_, local);
        auto linear_key = modules::TransposeModule({{1, 0, 2, 3}, 2}).build(ctx_, qks[3]);
        linear_key = core::ensure_backend_addressable_layout(ctx_, linear_key);
        auto kv = modules::MatMulModule().build(ctx_, linear_key, vu);
        kv = core::wrap_tensor(ggml_scale(ctx_.ggml, kv.tensor, 1.0f / frames_), kv.shape);
        auto global = modules::MatMulModule().build(ctx_, qks[1], kv);
        auto attended = modules::AddModule().build(ctx_, local, global);
        const int64_t width = vu.shape.last_dim() / 2;
        auto v = modules::SliceModule({1, 0, width}).build(ctx_, vu);
        auto u = modules::SliceModule({1, width, width}).build(ctx_, vu);
        auto av = modules::SliceModule({1, 0, width}).build(ctx_, attended);
        auto au = modules::SliceModule({1, width, width}).build(ctx_, attended);
        auto gate = modules::SigmoidModule().build(ctx_, modules::MulModule().build(ctx_, av, u));
        auto out = modules::MulModule().build(ctx_, modules::MulModule().build(ctx_, au, v), gate);
        return modules::AddModule().build(ctx_, residual, feed_forward(out, name + ".to_out", true));
    }

    TensorValue fsmn(TensorValue input, const std::string & name) {
        auto x = modules::PReluModule({1})
            .build(ctx_, linear(input, name + ".conv1.0"), weights_.at(name + ".conv1.1.weight"));
        const auto channels = x.shape.last_dim();
        x = modules::LayerNormModule({channels, 1e-5f}).build(ctx_, x,
            {weights_.at(name + ".norm1.weight"), weights_.at(name + ".norm1.bias")});
        auto u = feed_forward(x, name + ".gated_fsmn.to_u", false);
        auto v = feed_forward(x, name + ".gated_fsmn.to_v", false);
        const auto p = name + ".gated_fsmn.fsmn";
        auto dense = linear(modules::ReluModule().build(ctx_, linear(u, p + ".linear")), p + ".project");
        dense = modules::TransposeModule({{1, 0, 2, 3}, 2}).build(ctx_, dense);
        dense = core::reshape_tensor(ctx_, core::ensure_backend_addressable_layout(ctx_, dense), TensorShape::from_dims({1, channels, frames_}));
        auto skip = dense;
        for (int depth = 0; depth < 2; ++depth) {
            const auto base = p + ".conv.";
            auto weight = weights_.at(base + "conv" + std::to_string(depth + 1) + ".weight");
            if (depth == 0) {
                weight = core::reshape_tensor(ctx_, weight, TensorShape::from_dims({channels, 1, 39}));
                dense = modules::DepthwiseConv1dModule({channels, 39, 1, 19, 1, false})
                    .build(ctx_, skip, {weight, std::nullopt});
            } else {
                // Grouped convolution consumes adjacent pairs in the concatenated
                // channel order, not matching channels from each dense branch.
                auto grouped = core::reshape_tensor(ctx_, skip, TensorShape::from_dims({channels, 2, frames_}));
                std::vector<TensorValue> terms;
                for (int tap = 0; tap < 2; ++tap) {
                    auto lane = modules::SliceModule({1, tap, 1}).build(ctx_, grouped);
                    lane = core::reshape_tensor(ctx_, core::ensure_backend_addressable_layout(ctx_, lane), TensorShape::from_dims({1, channels, frames_}));
                    auto w = modules::SliceModule({1, tap, 1}).build(ctx_, weight);
                    if (ctx_.backend_type == core::BackendType::Metal) {
                        // Metal-only copy workaround; remove this gate and rewrite once
                        // https://github.com/0xShug0/audio.cpp/pull/831 is merged.
                        w = modules::TransposeModule({{1, 3, 0, 2}, 4}).build(ctx_, w);
                    }
                    w = core::reshape_tensor(ctx_, core::ensure_backend_addressable_layout(ctx_, w), TensorShape::from_dims({channels, 1, 39}));
                    terms.push_back(modules::DepthwiseConv1dModule({channels, 39, 1, 38, 2, false})
                        .build(ctx_, lane, {w, std::nullopt}));
                }
                dense = modules::AddModule().build(ctx_, terms[0], terms[1]);
            }
            dense = modules::GroupNormModule({channels, channels, 1e-5f}).build(ctx_, dense,
                {weights_.at(base + "norm" + std::to_string(depth + 1) + ".weight"),
                 weights_.at(base + "norm" + std::to_string(depth + 1) + ".bias")});
            dense = modules::PReluModule({1})
                .build(ctx_, dense, weights_.at(base + "prelu" + std::to_string(depth + 1) + ".weight"));
            if (depth == 0) skip = modules::ConcatModule({1}).build(ctx_, dense, skip);
        }
        dense = core::reshape_tensor(ctx_, dense, TensorShape::from_dims({channels, frames_}));
        dense = modules::TransposeModule({{1, 0, 2, 3}, 2}).build(ctx_, dense);
        u = modules::AddModule().build(ctx_, u, dense);
        x = modules::AddModule().build(ctx_, x, modules::MulModule().build(ctx_, v, u));
        x = modules::LayerNormModule({channels, 1e-5f}).build(ctx_, x,
            {weights_.at(name + ".norm2.weight"), weights_.at(name + ".norm2.bias")});
        return modules::AddModule().build(ctx_, input, linear(x, name + ".conv2"));
    }

    ggml_backend_t backend_;
    const TensorMap & weights_;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context_{nullptr, ggml_free};
    core::ModuleBuildContext ctx_;
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator_{nullptr, ggml_gallocr_free};
    ggml_cgraph * graph_ = nullptr;
    core::HostGraphPlan host_plan_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * position_ = nullptr;
    ggml_tensor * cos_ = nullptr;
    ggml_tensor * sin_ = nullptr;
    std::vector<ggml_tensor *> outputs_;
    int64_t frames_ = 0;
};
}  // namespace

class MossFormer2GatedFSMNRuntime::Impl {
public:
    Impl(std::shared_ptr<const MossFormer2Assets> assets, core::ExecutionContext & execution)
        : assets_(std::move(assets)), execution_(execution),
          store_(execution.backend(), execution.backend_type(), "mossformer2", kWeightContextBytes) {
        const auto & source = *assets_->tensors;
        frequencies_ = source.require_f32("mask_net.pos_enc.inv_freq", {assets_->config.hidden / 2});
        position_scale_ = source.require_f32("mask_net.pos_enc.scale", {1})[0];
        rotary_ = source.require_f32(std::string(kBackbonePrefix) + "layers.0.rotary_pos_emb.freqs", {kRotaryPairs});
        size_t bytes = 0;
        for (const auto & tensor : source.tensors()) {
            if (tensor.name.find("rotary_pos_emb") != std::string::npos || tensor.name.find("mask_net.pos_enc.") == 0) continue;
            auto value = store_.load_tensor(source, tensor.name, assets::TensorStorageType::Native, tensor.shape);
            bytes += ggml_nbytes(value.tensor);
            weights_.emplace(tensor.name, value);
        }
        store_.upload();
        source.release_storage();
        core::set_backend_threads(execution.backend(), execution.config().threads);
        debug::timing_log_scalar("mossformer2.weights_bytes", bytes);
    }

    std::vector<std::vector<float>> separate(const std::vector<float> & input) {
        if (!graph_ || samples_ != input.size()) {
            graph_.reset();
            const auto started = std::chrono::steady_clock::now();
            graph_ = std::make_unique<MossFormer2Graph>(execution_, assets_->config, weights_, input.size());
            graph_->set_positions(frequencies_, position_scale_, rotary_);
            samples_ = input.size();
            debug::timing_log_scalar("mossformer2.graph.build_ms", debug::elapsed_ms(started));
        }
        const auto started = std::chrono::steady_clock::now();
        auto result = graph_->run(input);
        debug::timing_log_scalar("mossformer2.graph.compute_ms", debug::elapsed_ms(started));
        return result;
    }

private:
    std::shared_ptr<const MossFormer2Assets> assets_;
    core::ExecutionContext & execution_;
    core::BackendWeightStore store_;
    TensorMap weights_;
    std::vector<float> frequencies_, rotary_;
    float position_scale_ = 1;
    std::unique_ptr<MossFormer2Graph> graph_;
    size_t samples_ = 0;
};

MossFormer2GatedFSMNRuntime::MossFormer2GatedFSMNRuntime(std::shared_ptr<const MossFormer2Assets> assets,
    core::ExecutionContext & execution) : impl_(std::make_unique<Impl>(std::move(assets), execution)) {}
MossFormer2GatedFSMNRuntime::~MossFormer2GatedFSMNRuntime() = default;
std::vector<std::vector<float>> MossFormer2GatedFSMNRuntime::separate(const std::vector<float> & input) {
    return impl_->separate(input);
}
}  // namespace engine::models::mossformer2
