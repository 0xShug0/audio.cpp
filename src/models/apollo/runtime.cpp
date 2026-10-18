#include "engine/models/apollo/runtime.h"

#include "engine/framework/audio/dsp.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/graph_optimizer.h"

#include <ggml-alloc.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace engine::models::apollo {
namespace {

using core::TensorShape;
using core::TensorValue;
using TensorMap = std::unordered_map<std::string, TensorValue>;

class ApolloGraph {
public:
    ApolloGraph(core::ExecutionContext & execution, const ApolloConfig & config,
                const TensorMap & weights, const TensorValue & positions, int64_t frames)
        : backend_(execution.backend()) {
        const auto started = std::chrono::steady_clock::now();
        constexpr size_t nodes = 16384;
        ctx_.reset(ggml_init({nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true}));
        if (!ctx_) {
            throw std::runtime_error("Apollo graph context allocation failed");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "apollo", execution.backend_type()};
        const int64_t bands = static_cast<int64_t>(config.band_widths.size());
        const int64_t dim = config.dim;
        const int64_t input_dim = 2 * (config.n_fft / 2 + 1) + bands;
        auto input = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({frames, input_dim}));
        input_ = input.tensor;
        ggml_set_input(input_);
        std::vector<TensorValue> encoded;
        int64_t offset = 0;
        for (int64_t band = 0; band < bands; ++band) {
            const int64_t width = config.band_widths[band] * 2 + 1;
            const std::string prefix = "BN." + std::to_string(band);
            auto x = modules::SliceModule({1, offset, width}).build(ctx, input);
            x = modules::RMSNormModule({width, 1e-5f, true, false}).build(ctx, x,
                {weights.at(prefix + ".0.weight"), std::nullopt});
            x = modules::LinearModule({width, dim, true}).build(ctx, x,
                {weights.at(prefix + ".1.weight"), weights.at(prefix + ".1.bias")});
            encoded.push_back(core::reshape_tensor(ctx, x, TensorShape::from_dims({frames, 1, dim})));
            offset += width;
        }
        // A balanced concatenation keeps the 80 independent band projections linear in data movement per level.
        while (encoded.size() > 1) {
            std::vector<TensorValue> merged;
            for (size_t i = 0; i < encoded.size(); i += 2) {
                merged.push_back(i + 1 < encoded.size()
                    ? modules::ConcatModule({1}).build(ctx, encoded[i], encoded[i + 1]) : encoded[i]);
            }
            encoded = std::move(merged);
        }
        auto x = encoded.front();
        // A zero bias keeps attention unmasked while folding scaling into the framework softmax path.
        auto attention_bias = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({bands, bands}));
        attention_bias = core::wrap_tensor(ggml_fill(ctx.ggml, attention_bias.tensor, 0.0f), attention_bias.shape);
        for (int layer = 0; layer < config.layers; ++layer) {
            const std::string prefix = "net." + std::to_string(layer) + ".band_net";
            auto norm = modules::RMSNormModule({dim, 1e-5f, true, false}).build(ctx, x,
                {weights.at(prefix + ".input_norm.weight"), std::nullopt});
            auto qkv = modules::LinearModule({dim, 3 * dim, false}).build(ctx, norm,
                {weights.at(prefix + ".weight.weight"), std::nullopt});
            qkv = core::reshape_tensor(ctx, qkv, TensorShape::from_dims({frames, bands, 8, 96}));
            auto q = modules::SliceModule({3, 0, 32}).build(ctx, qkv);
            auto k = modules::SliceModule({3, 32, 32}).build(ctx, qkv);
            auto v = modules::SliceModule({3, 64, 32}).build(ctx, qkv);
            q = modules::RoPEModule({32}).build(ctx, q, positions);
            k = modules::RoPEModule({32}).build(ctx, k, positions);
            q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
            k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
            v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
            // ggml CUDA flash attention does not instantiate 32-wide heads.
            auto attention = modules::ScaledDotProductAttentionModule(
                {32, modules::ScaledDotProductAttentionLowering::Explicit, GGML_PREC_F32})
                .build(ctx, q, k, v, attention_bias);
            attention = core::ensure_backend_addressable_layout(ctx, attention);
            attention = core::reshape_tensor(ctx, attention, TensorShape::from_dims({frames, bands, dim}));
            attention = modules::LinearModule({dim, dim, false}).build(ctx, attention,
                {weights.at(prefix + ".output.weight"), std::nullopt});
            x = modules::AddModule().build(ctx, x, attention);
            norm = modules::RMSNormModule({dim, 1e-5f, true, false}).build(ctx, x,
                {weights.at(prefix + ".MLP.0.weight"), std::nullopt});
            auto mlp = modules::LinearModule({dim, 8 * dim, false}).build(ctx, norm,
                {weights.at(prefix + ".MLP.1.weight"), std::nullopt});
            // Upstream applies SiLU before splitting, then applies SiLU again to the gate.
            mlp = modules::SiluModule().build(ctx, mlp);
            mlp = core::wrap_tensor(ggml_swiglu(ctx.ggml, mlp.tensor),
                TensorShape::from_dims({frames, bands, 4 * dim}));
            mlp = modules::LinearModule({4 * dim, dim, false}).build(ctx, mlp,
                {weights.at(prefix + ".MLP_output.weight"), std::nullopt});
            x = modules::AddModule().build(ctx, x, mlp);

            x = modules::TransposeModule({{1, 0, 2, 3}, 3}).build(ctx, x);
            x = core::ensure_backend_addressable_layout(ctx, x);
            for (int block = 0; block < 3; ++block) {
                const std::string conv = "net." + std::to_string(layer) + ".seq_net.blocks." + std::to_string(block) + ".conv";
                // Preserve channel-contiguous storage across the depthwise convolution and RMSNorm.
                auto y = core::reshape_tensor(ctx, x, TensorShape::from_dims({bands, 1, frames, dim}));
                y = modules::TransposeModule({{0, 3, 2, 1}, 4}).build(ctx, y);
                y = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, y);
                const auto kernel = modules::TransposeModule({{3, 1, 2, 0}, 4})
                    .build(ctx, weights.at(conv + ".0.weight"));
                y = core::wrap_tensor(ggml_conv_2d_dw_direct(ctx.ggml, kernel.tensor, y.tensor,
                    1, 1, 3, 0, 1, 1), y.shape);
                y = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, y);
                y = modules::TransposeModule({{0, 3, 2, 1}, 4}).build(ctx, y);
                y = core::reshape_tensor(ctx, y, TensorShape::from_dims({bands, frames, dim}));
                const auto bias = core::reshape_tensor(ctx, weights.at(conv + ".0.bias"),
                    TensorShape::from_dims({1, 1, dim}));
                y = modules::AddModule().build(ctx, y, modules::RepeatModule({y.shape}).build(ctx, bias));
                y = modules::RMSNormModule({dim, 1e-5f, true, false}).build(ctx, y,
                    {weights.at(conv + ".1.weight"), std::nullopt});
                y = modules::LinearModule({dim, 4 * dim, true}).build(ctx, y,
                    {weights.at(conv + ".2.weight"), weights.at(conv + ".2.bias")});
                y = modules::SiluModule().build(ctx, y);
                y = modules::LinearModule({4 * dim, dim, true}).build(ctx, y,
                    {weights.at(conv + ".4.weight"), weights.at(conv + ".4.bias")});
                x = modules::AddModule().build(ctx, x, y);
            }
            x = modules::TransposeModule({{1, 0, 2, 3}, 3}).build(ctx, x);
        }

        std::vector<TensorValue> decoded;
        for (int64_t band = 0; band < bands; ++band) {
            const int64_t width = config.band_widths[band];
            const std::string prefix = "output." + std::to_string(band);
            auto y = modules::SliceModule({1, band, 1}).build(ctx, x);
            y = core::ensure_backend_addressable_layout(ctx, y);
            y = core::reshape_tensor(ctx, y, TensorShape::from_dims({frames, dim}));
            y = modules::RMSNormModule({dim, 1e-5f, true, false}).build(ctx, y,
                {weights.at(prefix + ".0.weight"), std::nullopt});
            y = modules::LinearModule({dim, width * 4, true}).build(ctx, y,
                {weights.at(prefix + ".1.weight"), weights.at(prefix + ".1.bias")});
            decoded.push_back(modules::GLUModule().build(ctx, y));
        }
        while (decoded.size() > 1) {
            std::vector<TensorValue> merged;
            for (size_t i = 0; i < decoded.size(); i += 2) {
                merged.push_back(i + 1 < decoded.size()
                    ? modules::ConcatModule({1}).build(ctx, decoded[i], decoded[i + 1]) : decoded[i]);
            }
            decoded = std::move(merged);
        }
        output_ = decoded.front().tensor;
        ggml_set_output(output_);
        allocation_graph = ggml_new_graph_custom(ctx.ggml, nodes, false);
        ggml_build_forward_expand(allocation_graph, output_);
        // Fold explicit module broadcasts without changing other graph lowerings.
        runtime::GraphOptimizationOptions optimization;
        optimization.fold_commutative_lhs_repeats = false;
        optimization.fold_two_sided_broadcast_repeats = false;
        optimization.fold_unary_broadcast_repeats = false;
        optimization.fold_identity_materializations = false;
        optimization.elide_noop_nodes = false;
        optimization.elide_metadata_only_ops = false;
        runtime::optimize_graph(*allocation_graph, optimization);
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> planner(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)), ggml_gallocr_free);
        if (!planner) {
            throw std::runtime_error("Apollo graph planner allocation failed");
        }
        ggml_gallocr_reserve_n_size(planner.get(), allocation_graph, nullptr, nullptr, &workspace_bytes);
        debug::timing_log_scalar("apollo.graph.build_ms", debug::elapsed_ms(started));
        debug::timing_log_scalar("apollo.graph.frames", frames);
    }

    ~ApolloGraph() {
        if (allocation_graph) {
            core::release_backend_graph_resources(backend_, allocation_graph, true);
        }
    }

    ggml_cgraph * allocation_graph = nullptr;
    size_t workspace_bytes = 0;

    std::vector<float> run(const std::vector<float> & features) {
        if (features.size() * sizeof(float) != ggml_nbytes(input_)) {
            throw std::runtime_error("Apollo graph feature size mismatch");
        }
        const auto started = std::chrono::steady_clock::now();
        ggml_backend_tensor_set(input_, features.data(), 0, features.size() * sizeof(float));
        if (core::compute_backend_graph(backend_, allocation_graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Apollo graph execution failed");
        }
        std::vector<float> output(static_cast<size_t>(ggml_nelements(output_)));
        ggml_backend_tensor_get(output_, output.data(), 0, output.size() * sizeof(float));
        debug::timing_log_scalar("apollo.graph.run_ms", debug::elapsed_ms(started));
        return output;
    }

private:
    ggml_backend_t backend_;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx_{nullptr, ggml_free};
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
};

}  // namespace

class ApolloRuntime::Impl {
public:
    Impl(std::shared_ptr<const ApolloAssets> assets, core::ExecutionContext & execution,
         assets::TensorStorageType storage)
        : assets_(std::move(assets)), execution_(execution),
          store_(execution.backend(), execution.backend_type(), "apollo.weights", 4 * 1024 * 1024) {
        const auto & source = *assets_->tensors;
        for (const auto & metadata : source.tensors()) {
            const auto & shape = metadata.shape;
            // RoPE tables are deterministic buffers; the framework's normal RoPE has the same adjacent-pair convention.
            if (metadata.name.find(".cos_freq") != std::string::npos || metadata.name.find(".sin_freq") != std::string::npos) {
                continue;
            }
            if (shape.size() == 3 && shape[2] == 1) {
                weights_.emplace(metadata.name, store_.load_tensor_as_shape(source, metadata.name, storage, shape,
                    TensorShape::from_dims({shape[0], shape[1]})));
            } else if (shape.size() == 3 && shape[1] == 1 && shape[2] == 7) {
                // The channel-contiguous depthwise kernel consumes [tap, channel] storage.
                const auto kernel = source.require_f32(metadata.name, shape);
                std::vector<float> packed(kernel.size());
                for (int64_t channel = 0; channel < shape[0]; ++channel) {
                    for (int64_t tap = 0; tap < shape[2]; ++tap) {
                        packed[static_cast<size_t>(tap * shape[0] + channel)] =
                            kernel[static_cast<size_t>(channel * shape[2] + tap)];
                    }
                }
                weights_.emplace(metadata.name, store_.make_tensor(
                    TensorShape::from_dims({shape[2], 1, 1, shape[0]}), GGML_TYPE_F32,
                    packed.data(), packed.size() * sizeof(float)));
            } else {
                weights_.emplace(metadata.name, store_.load_f32_tensor(source, metadata.name, shape));
            }
        }
        std::vector<int32_t> positions(assets_->config.band_widths.size());
        std::iota(positions.begin(), positions.end(), 0);
        positions_ = store_.make_tensor(TensorShape::from_dims({static_cast<int64_t>(positions.size())}),
            GGML_TYPE_I32, positions.data(), positions.size() * sizeof(int32_t));
        store_.upload();
        core::set_backend_threads(execution.backend(), execution.config().threads);
    }

    std::vector<float> restore(const std::vector<float> & waveform) {
        const auto started = std::chrono::steady_clock::now();
        const auto & config = assets_->config;
        const int64_t samples = static_cast<int64_t>(waveform.size());
        const int64_t frames = 1 + samples / config.hop_length;
        const int64_t bins = config.n_fft / 2 + 1;
        const int64_t input_dim = 2 * bins + static_cast<int64_t>(config.band_widths.size());
        const audio::STFTConfig stft_config{config.n_fft, config.hop_length, config.n_fft, true, audio::STFTPadMode::Reflect};
        const auto spectrum = audio::STFT().compute_complex(waveform, assets_->window, 1, samples, stft_config,
            execution_.config().threads);
        std::vector<float> features(static_cast<size_t>(frames * input_dim));
        int64_t bin_start = 0;
        int64_t feature_start = 0;
        for (const int64_t width : config.band_widths) {
            for (int64_t frame = 0; frame < frames; ++frame) {
                float power = 0.0f;
                for (int64_t bin = 0; bin < width; ++bin) {
                    const size_t index = static_cast<size_t>(((bin_start + bin) * frames + frame) * 2);
                    const float magnitude = std::hypot(spectrum.values[index], spectrum.values[index + 1]);
                    power += magnitude * magnitude;
                }
                power = std::sqrt(power + std::numeric_limits<float>::epsilon());
                const size_t output = static_cast<size_t>(frame * input_dim + feature_start);
                for (int64_t bin = 0; bin < width; ++bin) {
                    const size_t index = static_cast<size_t>(((bin_start + bin) * frames + frame) * 2);
                    features[output + bin] = spectrum.values[index] / power;
                    features[output + width + bin] = spectrum.values[index + 1] / power;
                }
                features[output + width * 2] = std::log(power);
            }
            bin_start += width;
            feature_start += 2 * width + 1;
        }
        debug::timing_log_scalar("apollo.frontend.ms", debug::elapsed_ms(started));
        ApolloGraph * graph = nullptr;
        if (auto * found = graphs_.find(frames)) {
            graph = found->get();
        } else {
            auto created = std::make_unique<ApolloGraph>(execution_, config, weights_, positions_, frames);
            const auto buffer_type = ggml_backend_get_default_buffer_type(execution_.backend());
            const bool single_buffer = created->workspace_bytes <= ggml_backend_buft_get_max_size(buffer_type);
            // Cached tensor addresses remain valid only while the shared buffer is unchanged.
            if (!single_buffer || !single_buffer_ || !allocator_ ||
                created->workspace_bytes > ggml_gallocr_get_buffer_size(allocator_.get(), 0)) {
                graphs_.clear();
            }
            if (!allocator_ || !single_buffer || !single_buffer_) {
                allocator_.reset(ggml_gallocr_new(buffer_type));
            }
            if (!allocator_ || !ggml_gallocr_reserve(allocator_.get(), created->allocation_graph) ||
                !ggml_gallocr_alloc_graph(allocator_.get(), created->allocation_graph)) {
                throw std::runtime_error("Apollo shared graph allocation failed");
            }
            single_buffer_ = single_buffer;
            graph = created.get();
            graphs_.put(frames, std::move(created));
            debug::timing_log_scalar("apollo.graph.workspace_bytes", ggml_gallocr_get_buffer_size(allocator_.get(), 0));
        }
        const auto estimated = graph->run(features);
        const auto decode_started = std::chrono::steady_clock::now();
        std::vector<float> complex_spec(static_cast<size_t>(bins * frames * 2));
        bin_start = 0;
        feature_start = 0;
        for (const int64_t width : config.band_widths) {
            for (int64_t frame = 0; frame < frames; ++frame) {
                const size_t input = static_cast<size_t>(frame * bins * 2 + feature_start);
                for (int64_t bin = 0; bin < width; ++bin) {
                    const size_t output = static_cast<size_t>(((bin_start + bin) * frames + frame) * 2);
                    complex_spec[output] = estimated[input + bin];
                    complex_spec[output + 1] = estimated[input + width + bin];
                }
            }
            bin_start += width;
            feature_start += 2 * width;
        }
        auto output = audio::ISTFT().compute(complex_spec, assets_->window, 1, bins, frames, samples,
            stft_config, execution_.config().threads);
        debug::timing_log_scalar("apollo.istft.ms", debug::elapsed_ms(decode_started));
        return std::move(output.values);
    }

private:
    std::shared_ptr<const ApolloAssets> assets_;
    core::ExecutionContext & execution_;
    core::BackendWeightStore store_;
    TensorMap weights_;
    TensorValue positions_;
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator_{nullptr, ggml_gallocr_free};
    runtime::CacheSlots<int64_t, std::unique_ptr<ApolloGraph>> graphs_{2};
    bool single_buffer_ = true;
};

ApolloRuntime::ApolloRuntime(std::shared_ptr<const ApolloAssets> assets, core::ExecutionContext & execution,
                             assets::TensorStorageType storage)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, storage)) {}

ApolloRuntime::~ApolloRuntime() = default;

std::vector<float> ApolloRuntime::restore(const std::vector<float> & waveform) {
    return impl_->restore(waveform);
}

}  // namespace engine::models::apollo
