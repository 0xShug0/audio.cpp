#include "engine/models/tf_gridnet/runtime.h"

#include "engine/framework/audio/dsp.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/recurrent_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace engine::models::tf_gridnet {
namespace {

using core::TensorShape;
using core::TensorValue;
using TensorMap = std::unordered_map<std::string, TensorValue>;

// ESPnet's GridNet block: spectral BiLSTM, temporal BiLSTM, then full-band attention.
class TFGridNetBlock {
public:
    TFGridNetBlock(const TFGridNetConfig & config, const TensorMap & weights, int index)
        : config_(config), weights_(weights), prefix_("blocks." + std::to_string(index)) {}

    TensorValue build(core::ModuleBuildContext & ctx, TensorValue x) const {
        const int64_t frames = x.shape.dims[2];
        const int64_t bins = x.shape.dims[3];
        const int64_t kernel = config_.embedding_kernel;
        const int64_t stride = config_.embedding_stride;
        const auto padded = [kernel, stride](int64_t n) {
            return std::max<int64_t>(0, (n - kernel + stride - 1) / stride) * stride + kernel;
        };
        x = modules::Pad2dModule({0, padded(bins) - bins, 0, padded(frames) - frames}).build(ctx, x);
        x = recurrent(ctx, x, false);
        x = recurrent(ctx, x, true);
        x = modules::SliceModule({2, 0, frames}).build(ctx, x);
        x = modules::SliceModule({3, 0, bins}).build(ctx, x);
        x = core::ensure_backend_addressable_layout(ctx, x);
        std::vector<TensorValue> heads;
        for (int head = 0; head < config_.attention_heads; ++head) {
            auto q = attention_projection(ctx, x, prefix_ + ".attn_conv_Q_" + std::to_string(head));
            auto k = attention_projection(ctx, x, prefix_ + ".attn_conv_K_" + std::to_string(head));
            auto v = attention_projection(ctx, x, prefix_ + ".attn_conv_V_" + std::to_string(head));
            const int64_t value_channels = v.shape.dims[1];
            q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
            k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
            v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
            q = core::ensure_backend_addressable_layout(ctx, q);
            k = core::ensure_backend_addressable_layout(ctx, k);
            v = core::ensure_backend_addressable_layout(ctx, v);
            q = core::reshape_tensor(ctx, q, TensorShape::from_dims({1, 1, frames, q.shape.dims[2] * bins}));
            k = core::reshape_tensor(ctx, k, TensorShape::from_dims({1, 1, frames, k.shape.dims[2] * bins}));
            v = core::reshape_tensor(ctx, v, TensorShape::from_dims({1, 1, frames, value_channels * bins}));
            auto context = modules::ScaledDotProductAttentionModule({q.shape.dims[3],
                modules::ScaledDotProductAttentionLowering::Explicit, GGML_PREC_DEFAULT})
                .build(ctx, q, k, v, std::nullopt, weights_.at("attention_scale"));
            context = core::reshape_tensor(ctx, context, TensorShape::from_dims({1, frames, value_channels, bins}));
            heads.push_back(modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, context));
        }
        auto attention = heads.front();
        for (size_t i = 1; i < heads.size(); ++i) {
            attention = modules::ConcatModule({1}).build(ctx, attention, heads[i]);
        }
        attention = attention_projection(ctx, attention, prefix_ + ".attn_concat_proj");
        return modules::AddModule().build(ctx, x, attention);
    }

private:
    TensorValue recurrent(core::ModuleBuildContext & ctx, TensorValue x, bool temporal) const {
        const std::string prefix = prefix_ + (temporal ? ".inter" : ".intra");
        const int64_t channels = config_.embedding_dim;
        const int64_t frames = x.shape.dims[2];
        const int64_t bins = x.shape.dims[3];
        auto norm = modules::TransposeModule({{0, 2, 3, 1}, 4}).build(ctx, x);
        norm = modules::LayerNormModule({channels, config_.epsilon}).build(ctx, norm, {
            core::reshape_tensor(ctx, weights_.at(prefix + "_norm.gamma"), TensorShape::from_dims({channels})),
            core::reshape_tensor(ctx, weights_.at(prefix + "_norm.beta"), TensorShape::from_dims({channels}))});
        // Canonical recurrent layout is [sequence, batch, channels*kernel].
        auto axes = temporal ? std::array<int, 4>{1, 2, 3, 0} : std::array<int, 4>{2, 1, 3, 0};
        if (ctx.backend_type == core::BackendType::Metal) {
            // Metal-only copy workaround; remove this gate and rewrite once
            // https://github.com/0xShug0/audio.cpp/pull/831 is merged.
            axes = temporal ? std::array<int, 4>{0, 1, 2, 3} : std::array<int, 4>{0, 2, 1, 3};
        }
        norm = modules::TransposeModule({axes, 4}).build(ctx, norm);
        const int64_t length = temporal ? frames : bins;
        const int64_t batch = temporal ? bins : frames;
        const int64_t steps = (length - config_.embedding_kernel) / config_.embedding_stride + 1;
        norm = core::ensure_backend_addressable_layout(ctx, norm);
        norm = core::reshape_tensor(ctx, norm, TensorShape::from_dims({length, batch, channels}));
        std::vector<TensorValue> windows;
        for (int tap = 0; tap < config_.embedding_kernel; ++tap) {
            auto selected = modules::SliceModule({0, tap, (steps - 1) * config_.embedding_stride + 1}).build(ctx, norm);
            if (config_.embedding_stride != 1) {
                std::vector<TensorValue> rows;
                for (int64_t step = 0; step < steps; ++step) {
                    rows.push_back(modules::SliceModule({0, step * config_.embedding_stride, 1}).build(ctx, selected));
                }
                while (rows.size() > 1) {
                    std::vector<TensorValue> merged;
                    for (size_t i = 0; i < rows.size(); i += 2) {
                        merged.push_back(i + 1 < rows.size() ? modules::ConcatModule({0}).build(ctx, rows[i], rows[i + 1]) : rows[i]);
                    }
                    rows = std::move(merged);
                }
                selected = rows.front();
            }
            selected = core::ensure_backend_addressable_layout(ctx, selected);
            windows.push_back(core::reshape_tensor(ctx, selected, TensorShape::from_dims({steps, batch, channels, 1})));
        }
        auto unfolded = windows.front();
        for (size_t i = 1; i < windows.size(); ++i) {
            unfolded = modules::ConcatModule({3}).build(ctx, unfolded, windows[i]);
        }
        unfolded = core::reshape_tensor(ctx, unfolded,
            TensorShape::from_dims({steps, batch, channels * config_.embedding_kernel}));
        auto zero = modules::RepeatModule({TensorShape::from_dims({batch, config_.lstm_hidden})})
            .build(ctx, core::reshape_tensor(ctx, weights_.at("zero"), TensorShape::from_dims({1, 1})));
        modules::BidirectionalLSTMWeights weights;
        weights.forward = {weights_.at(prefix + "_rnn.weight_ih_l0"), weights_.at(prefix + "_rnn.weight_hh_l0"),
                           weights_.at(prefix + "_rnn.bias_ih_l0"), weights_.at(prefix + "_rnn.bias_hh_l0")};
        weights.reverse = {weights_.at(prefix + "_rnn.weight_ih_l0_reverse"), weights_.at(prefix + "_rnn.weight_hh_l0_reverse"),
                           weights_.at(prefix + "_rnn.bias_ih_l0_reverse"), weights_.at(prefix + "_rnn.bias_hh_l0_reverse")};
        auto recurrent = modules::BidirectionalLSTMModule({channels * config_.embedding_kernel,
            config_.lstm_hidden, false, true, true}).build(ctx, unfolded, zero, zero, zero, zero, weights).sequence;
        recurrent = modules::TransposeModule({{1, 2, 0, 3}, 3}).build(ctx, recurrent);
        recurrent = modules::ConvTranspose1dModule({2 * config_.lstm_hidden, channels, config_.embedding_kernel,
            config_.embedding_stride, 0, 1, true}).build(ctx, recurrent,
                {weights_.at(prefix + "_linear.weight"), weights_.at(prefix + "_linear.bias")});
        recurrent = core::reshape_tensor(ctx, recurrent, TensorShape::from_dims({1, batch, channels, length}));
        recurrent = modules::TransposeModule({temporal ? std::array<int, 4>{0, 2, 3, 1}
                                                       : std::array<int, 4>{0, 2, 1, 3}, 4}).build(ctx, recurrent);
        return modules::AddModule().build(ctx, x, recurrent);
    }

    TensorValue attention_projection(core::ModuleBuildContext & ctx, TensorValue x, const std::string & prefix) const {
        const auto & weight = weights_.at(prefix + ".0.weight");
        const int64_t channels = weight.shape.dims[0];
        x = modules::Conv2dModule({x.shape.dims[1], channels, 1, 1}).build(ctx, x,
            {weight, weights_.at(prefix + ".0.bias")});
        const auto positive = modules::ReluModule().build(ctx, x);
        auto negative = modules::MulModule().build(ctx, x,
            modules::RepeatModule({x.shape}).build(ctx,
                core::reshape_tensor(ctx, weights_.at("negative_one"), TensorShape::from_dims({1, 1, 1, 1}))));
        negative = modules::ReluModule().build(ctx, negative);
        auto slope = modules::MulModule().build(ctx, weights_.at(prefix + ".1.weight"), weights_.at("negative_one"));
        negative = modules::MulModule().build(ctx, negative, modules::RepeatModule({x.shape}).build(ctx,
            core::reshape_tensor(ctx, slope, TensorShape::from_dims({1, 1, 1, 1}))));
        x = modules::AddModule().build(ctx, positive, negative);
        const auto shape = x.shape;
        x = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
        const int64_t features = channels * shape.dims[3];
        x = core::ensure_backend_addressable_layout(ctx, x);
        x = core::reshape_tensor(ctx, x, TensorShape::from_dims({shape.dims[2], features}));
        x = modules::LayerNormModule({features, config_.epsilon}).build(ctx, x, {
            core::reshape_tensor(ctx, weights_.at(prefix + ".2.gamma"), TensorShape::from_dims({features})),
            core::reshape_tensor(ctx, weights_.at(prefix + ".2.beta"), TensorShape::from_dims({features}))});
        x = core::reshape_tensor(ctx, x, TensorShape::from_dims({1, shape.dims[2], channels, shape.dims[3]}));
        return modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
    }

    const TFGridNetConfig & config_;
    const TensorMap & weights_;
    std::string prefix_;
};

class TFGridNetStageGraph {
public:
    TFGridNetStageGraph(core::ExecutionContext & execution, const TFGridNetConfig & config,
        const TensorMap & weights, int stage, int64_t frames) : backend_(execution.backend()) {
        const int64_t bins = config.n_fft / 2 + 1;
        const size_t nodes = stage >= 0 && stage < config.layers
            ? static_cast<size_t>((frames + bins + config.embedding_kernel) * 80 + 8192) : 8192;
        ctx_.reset(ggml_init({nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true}));
        if (!ctx_) throw std::runtime_error("TF-GridNet graph context allocation failed");
        core::ModuleBuildContext ctx{ctx_.get(), "tf_gridnet", execution.backend_type()};
        auto x = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1,
            stage < 0 ? 2 * config.microphones : config.embedding_dim, frames, bins}));
        input = x.tensor;
        ggml_set_input(input);
        if (stage < 0) {
            x = modules::Conv2dModule({2 * config.microphones, config.embedding_dim, 3, 3, 1, 1, 1, 1})
                .build(ctx, x, {weights.at("conv.0.weight"), weights.at("conv.0.bias")});
            x = modules::GroupNormModule({config.embedding_dim, 1, config.epsilon})
                .build(ctx, x, {weights.at("conv.1.weight"), weights.at("conv.1.bias")});
        } else if (stage < config.layers) {
            x = TFGridNetBlock(config, weights, stage).build(ctx, x);
        } else {
            // Stride-one transpose convolution equals convolution with reversed spatial kernels.
            x = modules::Conv2dModule({config.embedding_dim, 2 * config.speakers, 3, 3, 1, 1, 1, 1})
                .build(ctx, x, {weights.at("output_conv.weight"), weights.at("deconv.bias")});
        }
        x = core::ensure_backend_addressable_layout(ctx, x);
        output = x.tensor;
        ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx.ggml, nodes, false);
        ggml_build_forward_expand(graph, output);
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> planner(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)), ggml_gallocr_free);
        ggml_gallocr_reserve_n_size(planner.get(), graph, nullptr, nullptr, &workspace_bytes);
    }

    ~TFGridNetStageGraph() { core::release_backend_graph_resources(backend_, graph, true); }
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    size_t workspace_bytes = 0;

private:
    ggml_backend_t backend_;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx_{nullptr, ggml_free};
};

}  // namespace

class TFGridNetBiLSTMAttentionRuntime::Impl {
public:
    Impl(std::shared_ptr<const TFGridNetAssets> assets, core::ExecutionContext & execution, assets::TensorStorageType storage)
        : assets_(std::move(assets)), execution_(execution),
          store_(execution.backend(), execution.backend_type(), "tf_gridnet", 4 * 1024 * 1024) {
        const auto & source = *assets_->tensors;
        for (const auto & metadata : source.tensors()) {
            if (metadata.name == "deconv.weight") continue;
            weights_.emplace(metadata.name, store_.load_tensor(source, metadata.name,
                metadata.shape.size() >= 2 ? storage : assets::TensorStorageType::F32, metadata.shape));
        }
        const auto & c = assets_->config;
        const auto original = source.require_f32("deconv.weight", {c.embedding_dim, c.speakers * 2, 3, 3});
        std::vector<float> reversed(original.size());
        for (int in = 0; in < c.embedding_dim; ++in) {
            for (int out = 0; out < 2 * c.speakers; ++out) {
                for (int tap = 0; tap < 9; ++tap) {
                    reversed[(out * c.embedding_dim + in) * 9 + tap] = original[(in * 2 * c.speakers + out) * 9 + 8 - tap];
                }
            }
        }
        weights_.emplace("output_conv.weight", store_.make_f32(TensorShape::from_dims({2 * c.speakers, c.embedding_dim, 3, 3}), reversed));
        weights_.emplace("zero", store_.make_f32(TensorShape::from_dims({1}), {0}));
        weights_.emplace("negative_one", store_.make_f32(TensorShape::from_dims({1}), {-1}));
        const int bins = c.n_fft / 2 + 1;
        const int qk = (c.attention_qk_dim + bins - 1) / bins * bins;
        weights_.emplace("attention_scale", store_.make_f32(TensorShape::from_dims({1}), {1.0f / std::sqrt(float(qk))}));
        store_.upload();
        // All GridNet blocks share a topology. Stable staging addresses let one
        // compiled graph serve every block without retaining per-block CUDA captures.
        block_store_ = std::make_unique<core::BackendWeightStore>(execution.backend(),
            execution.backend_type(), "tf_gridnet.block", 4 * 1024 * 1024);
        block_weights_ = weights_;
        block_copies_.resize(c.layers);
        for (const auto & metadata : source.tensors()) {
            constexpr std::string_view prefix = "blocks.0.";
            if (metadata.name.compare(0, prefix.size(), prefix) != 0) continue;
            auto staged = block_store_->load_tensor(source, metadata.name,
                metadata.shape.size() >= 2 ? storage : assets::TensorStorageType::F32, metadata.shape);
            block_weights_[metadata.name] = staged;
            for (int layer = 0; layer < c.layers; ++layer) {
                const auto name = "blocks." + std::to_string(layer) + "." + metadata.name.substr(prefix.size());
                block_copies_[layer].emplace_back(weights_.at(name).tensor, staged.tensor);
            }
        }
        block_store_->upload();
        core::set_backend_threads(execution.backend(), execution.config().threads);
    }

    std::vector<std::vector<float>> separate(const std::vector<float> & input) {
        const auto started = std::chrono::steady_clock::now();
        const auto & c = assets_->config;
        const int64_t samples = static_cast<int64_t>(input.size()) / c.microphones;
        const int64_t frames = samples / c.hop_length + 1;
        const int64_t bins = c.n_fft / 2 + 1;
        const double mean = std::accumulate(input.begin(), input.end(), 0.0) / input.size();
        double variance = 0;
        for (float value : input) variance += (value - mean) * (value - mean);
        const float scale = static_cast<float>(std::sqrt(variance / (input.size() - 1)));
        if (scale == 0) throw std::runtime_error("TF-GridNet input has zero variance");
        std::vector<float> planar(input.size());
        for (int channel = 0; channel < c.microphones; ++channel) {
            for (int64_t t = 0; t < samples; ++t) planar[channel * samples + t] = input[t * c.microphones + channel] / scale;
        }
        const audio::STFTConfig stft{c.n_fft, c.hop_length, c.n_fft, true, audio::STFTPadMode::Reflect};
        auto spectrum = audio::STFT().compute_complex(planar, assets_->window, c.microphones, samples, stft, execution_.config().threads);
        std::vector<float> features(static_cast<size_t>(2 * c.microphones * frames * bins));
        for (int channel = 0; channel < c.microphones; ++channel) {
            for (int64_t t = 0; t < frames; ++t) {
                for (int64_t f = 0; f < bins; ++f) {
                    const size_t offset = ((channel * bins + f) * frames + t) * 2;
                    features[(channel * frames + t) * bins + f] = spectrum.values[offset];
                    features[((channel + c.microphones) * frames + t) * bins + f] = spectrum.values[offset + 1];
                }
            }
        }
        debug::timing_log_scalar("tf_gridnet.frontend_ms", debug::elapsed_ms(started));
        prepare_graphs(frames);
        const auto graph_started = std::chrono::steady_clock::now();
        for (int index = 0; index < c.layers + 2; ++index) {
            auto & stage = *graphs_[index == 0 ? 0 : index == c.layers + 1 ? 2 : 1];
            if (index > 0 && index <= c.layers) {
                for (const auto & [source, destination] : block_copies_[index - 1]) {
                    ggml_backend_tensor_copy_async(execution_.backend(), execution_.backend(), source, destination);
                }
            }
            if (index == 0) {
                ggml_backend_tensor_set(stage.input, features.data(), 0, features.size() * sizeof(float));
            } else {
                ggml_backend_tensor_copy(handoff_, stage.input);
            }
            if (core::compute_backend_graph(execution_.backend(), stage.graph) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("TF-GridNet graph execution failed");
            }
            if (index < c.layers + 1) {
                ggml_backend_tensor_copy(stage.output, handoff_);
            } else {
                features.resize(static_cast<size_t>(ggml_nelements(stage.output)));
                ggml_backend_tensor_get(stage.output, features.data(), 0, features.size() * sizeof(float));
            }
        }
        debug::timing_log_scalar("tf_gridnet.graph.compute_ms", debug::elapsed_ms(graph_started));
        const auto decode_started = std::chrono::steady_clock::now();
        std::vector<float> complex(static_cast<size_t>(c.speakers * bins * frames * 2));
        for (int speaker = 0; speaker < c.speakers; ++speaker) {
            for (int64_t f = 0; f < bins; ++f) {
                for (int64_t t = 0; t < frames; ++t) {
                    const size_t offset = ((speaker * bins + f) * frames + t) * 2;
                    complex[offset] = features[(2 * speaker * frames + t) * bins + f];
                    complex[offset + 1] = features[((2 * speaker + 1) * frames + t) * bins + f];
                }
            }
        }
        auto waveform = audio::ISTFT().compute(complex, assets_->window, c.speakers, bins, frames, samples, stft, execution_.config().threads);
        std::vector<std::vector<float>> result(c.speakers, std::vector<float>(samples));
        for (int speaker = 0; speaker < c.speakers; ++speaker) {
            for (int64_t t = 0; t < samples; ++t) result[speaker][t] = waveform.values[speaker * samples + t] * scale;
        }
        debug::timing_log_scalar("tf_gridnet.istft_ms", debug::elapsed_ms(decode_started));
        return result;
    }

private:
    void prepare_graphs(int64_t frames) {
        if (frames == frames_) return;
        graphs_.clear();
        allocator_.reset();
        handoff_buffer_.reset();
        handoff_context_.reset();
        const auto started = std::chrono::steady_clock::now();
        handoff_context_.reset(ggml_init({ggml_tensor_overhead(), nullptr, true}));
        core::ModuleBuildContext ctx{handoff_context_.get(), "tf_gridnet", execution_.backend_type()};
        handoff_ = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1,
            assets_->config.embedding_dim, frames, assets_->config.n_fft / 2 + 1})).tensor;
        handoff_buffer_.reset(ggml_backend_alloc_ctx_tensors(handoff_context_.get(), execution_.backend()));
        if (!handoff_buffer_) throw std::runtime_error("TF-GridNet stage buffer allocation failed");
        size_t largest = 0;
        for (int stage : {-1, 0, assets_->config.layers}) {
            graphs_.push_back(std::make_unique<TFGridNetStageGraph>(execution_, assets_->config,
                stage == 0 ? block_weights_ : weights_, stage, frames));
            if (graphs_.back()->workspace_bytes > graphs_[largest]->workspace_bytes) largest = graphs_.size() - 1;
        }
        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend())));
        if (!ggml_gallocr_reserve(allocator_.get(), graphs_[largest]->graph)) {
            throw std::runtime_error("TF-GridNet workspace reservation failed");
        }
        for (auto & stage : graphs_) {
            if (!ggml_gallocr_alloc_graph(allocator_.get(), stage->graph)) {
                throw std::runtime_error("TF-GridNet graph allocation failed");
            }
        }
        frames_ = frames;
        debug::timing_log_scalar("tf_gridnet.graph.build_ms", debug::elapsed_ms(started));
        debug::timing_log_scalar("tf_gridnet.graph.workspace_bytes", ggml_gallocr_get_buffer_size(allocator_.get(), 0));
    }

    std::shared_ptr<const TFGridNetAssets> assets_;
    core::ExecutionContext & execution_;
    core::BackendWeightStore store_;
    TensorMap weights_;
    std::unique_ptr<core::BackendWeightStore> block_store_;
    TensorMap block_weights_;
    std::vector<std::vector<std::pair<ggml_tensor *, ggml_tensor *>>> block_copies_;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> handoff_context_{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> handoff_buffer_{nullptr, ggml_backend_buffer_free};
    ggml_tensor * handoff_ = nullptr;
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator_{nullptr, ggml_gallocr_free};
    std::vector<std::unique_ptr<TFGridNetStageGraph>> graphs_;
    int64_t frames_ = 0;
};

TFGridNetBiLSTMAttentionRuntime::TFGridNetBiLSTMAttentionRuntime(std::shared_ptr<const TFGridNetAssets> assets,
    core::ExecutionContext & execution, assets::TensorStorageType storage)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, storage)) {}
TFGridNetBiLSTMAttentionRuntime::~TFGridNetBiLSTMAttentionRuntime() = default;
std::vector<std::vector<float>> TFGridNetBiLSTMAttentionRuntime::separate(const std::vector<float> & input) {
    return impl_->separate(input);
}

}  // namespace engine::models::tf_gridnet
