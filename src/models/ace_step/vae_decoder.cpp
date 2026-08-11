#include "engine/models/ace_step/vae_decoder.h"

#include "vae_common.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-backend.h>
#include <ggml-alloc.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::ace_step {
namespace vae_common {

namespace modules = engine::modules;

void GgmlContextDeleter::operator()(ggml_context * ctx) const noexcept {
    if (ctx != nullptr) {
        ggml_free(ctx);
    }
}

std::vector<float> apply_weight_norm(
    const std::vector<float> & g,
    const std::vector<float> & v,
    int64_t leading,
    int64_t inner_size) {
    std::vector<float> weight(v.size(), 0.0F);
    for (int64_t i = 0; i < leading; ++i) {
        double norm = 0.0;
        for (int64_t j = 0; j < inner_size; ++j) {
            const float value = v[static_cast<size_t>(i * inner_size + j)];
            norm += static_cast<double>(value) * static_cast<double>(value);
        }
        const float scale = g[static_cast<size_t>(i)] / std::sqrt(static_cast<float>(norm) + 1.0e-12F);
        for (int64_t j = 0; j < inner_size; ++j) {
            weight[static_cast<size_t>(i * inner_size + j)] = v[static_cast<size_t>(i * inner_size + j)] * scale;
        }
    }
    return weight;
}

std::vector<float> snake_exp_transform(const std::vector<float> & values) {
    std::vector<float> out(values.size(), 0.0F);
    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = std::exp(values[i]);
    }
    return out;
}

std::vector<float> snake_inverse_beta_transform(const std::vector<float> & values) {
    std::vector<float> out(values.size(), 0.0F);
    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = 1.0F / (std::exp(values[i]) + 1.0e-9F);
    }
    return out;
}

core::TensorValue ensure_f32(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    if (value.type == GGML_TYPE_F32) {
        return value;
    }
    return core::wrap_tensor(ggml_cast(ctx.ggml, value.tensor, GGML_TYPE_F32), value.shape, GGML_TYPE_F32);
}

core::TensorValue ensure_contiguous_nontransposed(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value) {
    if (core::has_backend_addressable_layout(value.tensor) && !ggml_is_transposed(value.tensor)) {
        return value;
    }
    return core::wrap_tensor(ggml_cont(ctx.ggml, value.tensor), value.shape, value.type);
}

core::TensorValue require_contiguous_nontransposed_weight(
    const core::TensorValue & weight,
    const char * module_name) {
    if (!core::has_backend_addressable_layout(weight.tensor) || ggml_is_transposed(weight.tensor)) {
        throw std::runtime_error(
            std::string(module_name) + " requires uploaded weights to be contiguous and non-transposed");
    }
    return weight;
}

core::TensorValue require_f32_weight(
    const core::TensorValue & value,
    const char * module_name) {
    if (value.type != GGML_TYPE_F32) {
        throw std::runtime_error(std::string(module_name) + " requires uploaded F32 weights");
    }
    return require_contiguous_nontransposed_weight(value, module_name);
}

core::TensorValue regular_conv_weight(
    const core::TensorValue & weight,
    const char * module_name) {
    const auto contiguous = require_contiguous_nontransposed_weight(weight, module_name);
    if (contiguous.type == GGML_TYPE_F32 || contiguous.type == GGML_TYPE_F16) {
        return contiguous;
    }
    throw std::runtime_error(
        std::string(module_name) + " weight type is unsupported for direct graph binding: " +
        ggml_type_name(contiguous.type));
}

core::TensorValue conv_transpose1d_weight(
    const core::TensorValue & weight) {
    const auto contiguous = require_contiguous_nontransposed_weight(weight, "ACE-Step VAE ConvTranspose1d");
    if (contiguous.type == GGML_TYPE_F32) {
        return contiguous;
    }
    throw std::runtime_error(
        std::string("ACE-Step VAE ConvTranspose1d requires uploaded F32 weights, got: ") +
        ggml_type_name(contiguous.type));
}

assets::TensorStorageType compatible_regular_conv_storage_type(
    const assets::TensorSource & source,
    std::string_view tensor_name,
    assets::TensorStorageType requested) {
    const assets::TensorStorageType resolved = assets::resolve_tensor_storage_type(source, tensor_name, requested);
    if (resolved == assets::TensorStorageType::F32 || resolved == assets::TensorStorageType::F16) {
        return resolved;
    }
    if (resolved == assets::TensorStorageType::BF16) {
        return assets::TensorStorageType::F16;
    }
    return assets::TensorStorageType::F32;
}

assets::TensorStorageType compatible_conv_transpose_storage_type(
    const assets::TensorSource &,
    std::string_view,
    assets::TensorStorageType) {
    return assets::TensorStorageType::F32;
}

std::vector<float> conv_transpose_col_weight(
    const std::vector<float> & weight,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel) {
    std::vector<float> out(static_cast<size_t>(in_channels * out_channels * kernel), 0.0F);
    for (int64_t in = 0; in < in_channels; ++in) {
        for (int64_t out_channel = 0; out_channel < out_channels; ++out_channel) {
            for (int64_t k = 0; k < kernel; ++k) {
                const size_t src_index = static_cast<size_t>((in * out_channels + out_channel) * kernel + k);
                const size_t dst_index = static_cast<size_t>((out_channel * kernel + k) * in_channels + in);
                out[dst_index] = weight[src_index];
            }
        }
    }
    return out;
}

core::TensorValue view_batch_matrix(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t batch_index,
    int64_t channels,
    int64_t frames) {
    auto * view = ggml_view_2d(
        ctx.ggml,
        input.tensor,
        frames,
        channels,
        input.tensor->nb[1],
        static_cast<size_t>(batch_index) * input.tensor->nb[2]);
    return core::wrap_tensor(view, core::TensorShape::from_dims({channels, frames}), input.type);
}

core::TensorValue build_snake1d_exact_bct(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const SnakeExactWeights & weights,
    int64_t channels) {
    const auto input_f32 = ensure_f32(ctx, ensure_contiguous_nontransposed(ctx, input));
    auto alpha = ensure_f32(ctx, weights.alpha);
    auto beta_inv = ensure_f32(ctx, weights.beta_inv);
    if (alpha.shape.rank != 3 || alpha.shape.dims[0] != 1 || alpha.shape.dims[1] != channels || alpha.shape.dims[2] != 1 ||
        beta_inv.shape.rank != 3 || beta_inv.shape.dims[0] != 1 || beta_inv.shape.dims[1] != channels || beta_inv.shape.dims[2] != 1) {
        throw std::runtime_error("ACE-Step VAE Snake1d weight shape mismatch");
    }
    if (input_f32.shape.rank != 3) {
        throw std::runtime_error("ACE-Step VAE Snake1d expects rank-3 BCT input");
    }
    auto periodic = core::wrap_tensor(
        ggml_sqr(ctx.ggml, ggml_sin(ctx.ggml, ggml_mul(ctx.ggml, input_f32.tensor, alpha.tensor))),
        input_f32.shape,
        GGML_TYPE_F32);
    const auto beta_inv_broadcast = core::wrap_tensor(
        ggml_repeat(ctx.ggml, beta_inv.tensor, input_f32.tensor),
        input_f32.shape,
        GGML_TYPE_F32);
    auto frac = modules::MulModule{}.build(ctx, periodic, beta_inv_broadcast);
    return modules::AddModule{}.build(ctx, input_f32, frac);
}

SnakeExactWeights load_snake_exact(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels) {
    SnakeExactWeights out = {};
    const auto alpha = source.require_f32(prefix + ".alpha", {1, channels, 1});
    const auto beta = source.require_f32(prefix + ".beta", {1, channels, 1});
    out.alpha = store.make_from_f32(
        core::TensorShape::from_dims({1, channels, 1}),
        assets::TensorStorageType::F32,
        snake_exp_transform(alpha));
    out.beta_inv = store.make_from_f32(
        core::TensorShape::from_dims({1, channels, 1}),
        assets::TensorStorageType::F32,
        snake_inverse_beta_transform(beta));
    return out;
}

WeightNormConv1dWeights load_weight_norm_conv1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    int stride,
    int padding,
    int dilation,
    assets::TensorStorageType storage_type,
    bool use_bias) {
    const auto g = source.require_f32(prefix + ".weight_g", {out_channels, 1, 1});
    const auto v = source.require_f32(prefix + ".weight_v", {out_channels, in_channels, kernel});
    const auto graph_storage_type =
        compatible_regular_conv_storage_type(source, prefix + ".weight_v", storage_type);
    WeightNormConv1dWeights out = {};
    out.stride = stride;
    out.padding = padding;
    out.dilation = dilation;
    out.conv.weight = store.make_from_f32(
        core::TensorShape::from_dims({out_channels, in_channels, kernel}),
        graph_storage_type,
        apply_weight_norm(g, v, out_channels, in_channels * kernel));
    if (use_bias) {
        out.conv.bias = store.load_f32_tensor(source, prefix + ".bias", {out_channels});
    }
    return out;
}

WeightNormConvTranspose1dWeights load_weight_norm_conv_transpose1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    int stride,
    int padding,
    int dilation,
    assets::TensorStorageType storage_type,
    bool use_bias) {
    const auto g = source.require_f32(prefix + ".weight_g", {in_channels, 1, 1});
    const auto v = source.require_f32(prefix + ".weight_v", {in_channels, out_channels, kernel});
    const auto graph_storage_type =
        compatible_conv_transpose_storage_type(source, prefix + ".weight_v", storage_type);
    WeightNormConvTranspose1dWeights out = {};
    out.stride = stride;
    out.padding = padding;
    out.dilation = dilation;
    auto fused_weight = apply_weight_norm(g, v, in_channels, out_channels * kernel);
    auto col_weight = conv_transpose_col_weight(fused_weight, in_channels, out_channels, kernel);
    out.conv.weight = store.make_from_f32(
        core::TensorShape::from_dims({in_channels, out_channels, kernel}),
        graph_storage_type,
        std::move(fused_weight));
    out.col_weight = store.make_from_f32(
        core::TensorShape::from_dims({out_channels * kernel, in_channels}),
        graph_storage_type,
        std::move(col_weight));
    if (use_bias) {
        out.conv.bias = store.load_f32_tensor(source, prefix + ".bias", {out_channels});
    }
    return out;
}

ResidualUnitWeights load_residual_unit(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels,
    assets::TensorStorageType storage_type) {
    ResidualUnitWeights out = {};
    out.snake1 = load_snake_exact(store, source, prefix + ".snake1", channels);
    out.conv1 = load_weight_norm_conv1d(store, source, prefix + ".conv1", channels, channels, 7, 1, 3, 1, storage_type, true);
    out.snake2 = load_snake_exact(store, source, prefix + ".snake2", channels);
    out.conv2 = load_weight_norm_conv1d(store, source, prefix + ".conv2", channels, channels, 1, 1, 0, 1, storage_type, true);
    return out;
}

VAEDecoderWeights load_vae_decoder_weights(
    const AceStepAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    const auto & config = assets.config.vae;
    if (config.decoder_channels <= 0 ||
        config.decoder_input_channels <= 0 ||
        config.channel_multiples.empty() || config.downsampling_ratios.empty()) {
        throw std::runtime_error("ACE-Step VAE config is incomplete for decoder loading");
    }
    const auto & source = *assets.vae_weights;
    VAEDecoderWeights weights = {};
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "ace_step.vae.decoder.weights",
        weight_context_bytes);

    const int64_t deepest_channels = config.decoder_channels * config.channel_multiples.back();
    weights.conv1 = load_weight_norm_conv1d(
        *weights.store,
        source,
        "decoder.conv1",
        deepest_channels,
        config.decoder_input_channels,
        7,
        1,
        3,
        1,
        storage_type,
        true);

    const size_t num_blocks = config.downsampling_ratios.size();
    weights.blocks.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
        const int64_t stride = config.downsampling_ratios[num_blocks - 1 - i];
        const int64_t in_channels = config.decoder_channels * config.channel_multiples[num_blocks - i - 1];
        const int64_t out_channels = config.decoder_channels * (i + 1 < num_blocks ? config.channel_multiples[num_blocks - i - 2] : 1);
        const std::string prefix = "decoder.block." + std::to_string(i);
        DecoderBlockWeights block = {};
        block.snake = load_snake_exact(*weights.store, source, prefix + ".snake1", in_channels);
        block.conv_t = load_weight_norm_conv_transpose1d(
            *weights.store,
            source,
            prefix + ".conv_t1",
            in_channels,
            out_channels,
            2 * stride,
            static_cast<int>(stride),
            static_cast<int>((stride + 1) / 2),
            1,
            storage_type,
            true);
        block.res1 = load_residual_unit(*weights.store, source, prefix + ".res_unit1", out_channels, storage_type);
        block.res2 = load_residual_unit(*weights.store, source, prefix + ".res_unit2", out_channels, storage_type);
        block.res3 = load_residual_unit(*weights.store, source, prefix + ".res_unit3", out_channels, storage_type);
        weights.blocks.push_back(std::move(block));
    }

    weights.snake_out = load_snake_exact(*weights.store, source, "decoder.snake1", config.decoder_channels);
    weights.conv2 = load_weight_norm_conv1d(
        *weights.store,
        source,
        "decoder.conv2",
        config.audio_channels,
        config.decoder_channels,
        7,
        1,
        3,
        1,
        storage_type,
        false);
    weights.store->upload();
    return weights;
}

core::TensorValue build_conv1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const WeightNormConv1dWeights & weights,
    int64_t in_channels,
    int64_t out_channels,
    bool use_bias) {
    const auto input_contiguous = ensure_f32(ctx, ensure_contiguous_nontransposed(ctx, input));
    const auto weight_contiguous = regular_conv_weight(weights.conv.weight, "ACE-Step VAE Conv1d");
    core::TensorValue bias_matrix;
    if (use_bias) {
        if (!weights.conv.bias.has_value()) {
            throw std::runtime_error("ACE-Step VAE Conv1d requires bias");
        }
        const auto bias = require_f32_weight(*weights.conv.bias, "ACE-Step VAE Conv1d bias");
        bias_matrix = core::reshape_tensor(ctx, bias, core::TensorShape::from_dims({out_channels, 1}));
    }
    core::TensorValue output;
    for (int64_t batch_index = 0; batch_index < input.shape.dims[0]; ++batch_index) {
        const auto batch_input = view_batch_matrix(ctx, input_contiguous, batch_index, in_channels, input.shape.dims[2]);
        auto * batch_output = ggml_conv_1d(
            ctx.ggml,
            weight_contiguous.tensor,
            batch_input.tensor,
            weights.stride,
            weights.padding,
            weights.dilation);
        if (use_bias) {
            batch_output = ggml_add(ctx.ggml, batch_output, bias_matrix.tensor);
        }
        auto batch_value = core::wrap_tensor(
            ggml_reshape_3d(ctx.ggml, batch_output, batch_output->ne[0], batch_output->ne[1], 1),
            core::TensorShape::from_dims({1, out_channels, batch_output->ne[0]}),
            GGML_TYPE_F32);
        output = output.valid() ? modules::ConcatModule({0}).build(ctx, output, batch_value) : batch_value;
    }
    return output;
}

core::TensorValue build_conv_transpose1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const WeightNormConvTranspose1dWeights & weights,
    int64_t in_channels,
    int64_t out_channels,
    bool use_bias) {
    const auto input_contiguous = ensure_f32(ctx, ensure_contiguous_nontransposed(ctx, input));
    const auto weight_contiguous = conv_transpose1d_weight(weights.conv.weight);
    core::TensorValue bias_matrix;
    if (use_bias) {
        if (!weights.conv.bias.has_value()) {
            throw std::runtime_error("ACE-Step VAE ConvTranspose1d requires bias");
        }
        const auto bias = require_f32_weight(*weights.conv.bias, "ACE-Step VAE ConvTranspose1d bias");
        bias_matrix = core::reshape_tensor(ctx, bias, core::TensorShape::from_dims({out_channels, 1}));
    }
    core::TensorValue output;
    for (int64_t batch_index = 0; batch_index < input.shape.dims[0]; ++batch_index) {
        const auto batch_input = view_batch_matrix(ctx, input_contiguous, batch_index, in_channels, input.shape.dims[2]);
        auto * batch_output = ggml_conv_transpose_1d(
            ctx.ggml,
            weight_contiguous.tensor,
            batch_input.tensor,
            weights.stride,
            0,
            weights.dilation);
        if (weights.padding != 0) {
            const int64_t full_frames = batch_output->ne[0];
            const int64_t trimmed_frames = full_frames - 2 * weights.padding;
            if (trimmed_frames <= 0) {
                throw std::runtime_error("ACE-Step VAE transposed convolution trim produced non-positive frame count");
            }
            batch_output = ggml_cont(
                ctx.ggml,
                ggml_view_2d(
                    ctx.ggml,
                    batch_output,
                    trimmed_frames,
                    out_channels,
                    batch_output->nb[1],
                    static_cast<size_t>(weights.padding) * batch_output->nb[0]));
        }
        if (use_bias) {
            batch_output = ggml_add(ctx.ggml, batch_output, bias_matrix.tensor);
        }
        auto batch_value = core::wrap_tensor(
            ggml_reshape_3d(ctx.ggml, batch_output, batch_output->ne[0], batch_output->ne[1], 1),
            core::TensorShape::from_dims({1, out_channels, batch_output->ne[0]}),
            GGML_TYPE_F32);
        output = output.valid() ? modules::ConcatModule({0}).build(ctx, output, batch_value) : batch_value;
    }
    return output;
}

core::TensorValue build_conv_transpose1d_col2im(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const WeightNormConvTranspose1dWeights & weights,
    int64_t in_channels,
    int64_t out_channels,
    bool use_bias) {
    const auto input_contiguous = ensure_f32(ctx, ensure_contiguous_nontransposed(ctx, input));
    const auto weight_contiguous =
        require_contiguous_nontransposed_weight(weights.col_weight, "ACE-Step VAE ConvTranspose1d col2im weight");
    if (weight_contiguous.type != GGML_TYPE_F32) {
        throw std::runtime_error("ACE-Step VAE ConvTranspose1d col2im requires uploaded F32 weights");
    }
    core::TensorValue bias_matrix;
    if (use_bias) {
        if (!weights.conv.bias.has_value()) {
            throw std::runtime_error("ACE-Step VAE ConvTranspose1d requires bias");
        }
        const auto bias = require_f32_weight(*weights.conv.bias, "ACE-Step VAE ConvTranspose1d bias");
        bias_matrix = core::reshape_tensor(ctx, bias, core::TensorShape::from_dims({out_channels, 1}));
    }
    core::TensorValue output;
    for (int64_t batch_index = 0; batch_index < input.shape.dims[0]; ++batch_index) {
        const auto batch_input = view_batch_matrix(ctx, input_contiguous, batch_index, in_channels, input.shape.dims[2]);
        auto * transposed_input = ggml_cont(ctx.ggml, ggml_transpose(ctx.ggml, batch_input.tensor));
        auto * columns = ggml_mul_mat(ctx.ggml, weight_contiguous.tensor, transposed_input);
        auto * batch_output = ggml_col2im_1d(
            ctx.ggml,
            columns,
            weights.stride,
            static_cast<int>(out_channels),
            weights.padding);
        if (use_bias) {
            batch_output = ggml_add(ctx.ggml, batch_output, bias_matrix.tensor);
        }
        auto batch_value = core::wrap_tensor(
            ggml_reshape_3d(ctx.ggml, batch_output, batch_output->ne[0], batch_output->ne[1], 1),
            core::TensorShape::from_dims({1, out_channels, batch_output->ne[0]}),
            GGML_TYPE_F32);
        output = output.valid() ? modules::ConcatModule({0}).build(ctx, output, batch_value) : batch_value;
    }
    return output;
}

core::TensorValue build_residual_unit(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ResidualUnitWeights & weights,
    int64_t channels,
    int dilation) {
    auto hidden = build_snake1d_exact_bct(ctx, input, weights.snake1, channels);
    auto conv1 = weights.conv1;
    conv1.dilation = dilation;
    conv1.padding = 3 * dilation;
    hidden = build_conv1d(ctx, hidden, conv1, channels, channels, true);
    hidden = build_snake1d_exact_bct(ctx, hidden, weights.snake2, channels);
    hidden = build_conv1d(ctx, hidden, weights.conv2, channels, channels, true);
    return modules::AddModule{}.build(ctx, input, hidden);
}

int64_t audio_frames_per_latent(const AceStepVAEConfig & config) {
    if (config.downsampling_ratios.empty()) {
        throw std::runtime_error("ACE-Step VAE requires downsampling ratios");
    }
    int64_t factor = 1;
    for (const int64_t ratio : config.downsampling_ratios) {
        if (ratio <= 0) {
            throw std::runtime_error("ACE-Step VAE downsampling ratio must be positive");
        }
        factor *= ratio;
    }
    return factor;
}

}  // namespace vae_common

using namespace vae_common;

class AceStepVAEDecoderRuntime::Impl {
public:
    class DecodeGraph {
    public:
        DecodeGraph(
            std::shared_ptr<const AceStepAssets> assets,
            ggml_backend_t backend,
            core::BackendType backend_type,
            int threads,
            std::shared_ptr<const VAEDecoderWeights> weights,
            int64_t latent_frames,
            bool use_col2im_conv_transpose,
            size_t graph_arena_bytes)
            : assets_(std::move(assets)),
              backend_(backend),
              backend_type_(backend_type),
              threads_(threads),
              weights_(std::move(weights)),
              latent_frames_(latent_frames),
              use_col2im_conv_transpose_(use_col2im_conv_transpose) {
            build(graph_arena_bytes);
        }

        ~DecodeGraph() {
            if (backend_ != nullptr && graph_ != nullptr) {
                engine::core::release_backend_graph_resources(backend_, graph_);
            }
            if (gallocr_ != nullptr) {
                ggml_gallocr_free(gallocr_);
            }
        }

        bool can_run(int64_t latent_frames, bool use_col2im_conv_transpose) const noexcept {
            return latent_frames_ >= latent_frames && use_col2im_conv_transpose_ == use_col2im_conv_transpose;
        }

        runtime::AudioBuffer decode(const AceStepLatents & latents) const {
            const auto total_start = Clock::now();
            const auto & config = assets_->config.vae;
            if (latents.frames <= 0 || latents.frames > latent_frames_ ||
                latents.channels != config.decoder_input_channels) {
                throw std::runtime_error("ACE-Step VAE latent shape mismatch");
            }
            const auto input_start = Clock::now();
            bct_input_scratch_.resize(static_cast<size_t>(latent_frames_ * latents.channels));
            std::fill(bct_input_scratch_.begin(), bct_input_scratch_.end(), 0.0F);
            for (int64_t frame = 0; frame < latents.frames; ++frame) {
                for (int64_t channel = 0; channel < latents.channels; ++channel) {
                    bct_input_scratch_[static_cast<size_t>(channel * latent_frames_ + frame)] =
                        latents.values[static_cast<size_t>(frame * latents.channels + channel)];
                }
            }
            core::write_tensor_f32(input_value_, bct_input_scratch_);
            core::set_backend_threads(backend_, threads_);
            engine::debug::timing_log_scalar(
                "ace_step.vae.decode.input_upload_ms",
                engine::debug::elapsed_ms(input_start, Clock::now()));
            const auto compute_start = Clock::now();
            const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
            ggml_backend_synchronize(backend_);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("ACE-Step VAE graph compute failed");
            }
            engine::debug::timing_log_scalar(
                "ace_step.vae.decode.graph.compute_ms",
                engine::debug::elapsed_ms(compute_start, Clock::now()));
            const auto output_start = Clock::now();
            bct_output_scratch_.resize(static_cast<size_t>(audio_frames_ * config.audio_channels));
            core::read_tensor_f32_into(output_value_.tensor, bct_output_scratch_);
            const int64_t request_audio_frames = latents.frames * audio_frames_per_latent_;
            runtime::AudioBuffer out;
            out.sample_rate = config.sample_rate;
            out.channels = config.audio_channels;
            out.samples.resize(static_cast<size_t>(request_audio_frames * config.audio_channels), 0.0F);
            for (int64_t frame = 0; frame < request_audio_frames; ++frame) {
                for (int channel = 0; channel < config.audio_channels; ++channel) {
                    out.samples[static_cast<size_t>(frame * config.audio_channels + channel)] =
                        bct_output_scratch_[static_cast<size_t>((channel * audio_frames_) + frame)];
                }
            }
            engine::debug::timing_log_scalar(
                "ace_step.vae.decode.output_read_ms",
                engine::debug::elapsed_ms(output_start, Clock::now()));
            engine::debug::timing_log_scalar("ace_step.vae.decode.total_ms", engine::debug::elapsed_ms(total_start, Clock::now()));
            return out;
        }

    private:
        void build(size_t graph_arena_bytes) {
            const auto & config = assets_->config.vae;
            ggml_init_params params{graph_arena_bytes, nullptr, true};
            ctx_.reset(ggml_init(params));
            if (ctx_ == nullptr) {
                throw std::runtime_error("ACE-Step VAE ggml context initialization failed");
            }
            core::ModuleBuildContext ctx{ctx_.get(), "ace_step.vae", backend_type_};

            input_value_ = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, config.decoder_input_channels, latent_frames_}));
            ggml_set_input(input_value_.tensor);

            auto hidden = build_conv1d(
                ctx,
                input_value_,
                weights_->conv1,
                config.decoder_input_channels,
                config.decoder_channels * config.channel_multiples.back(),
                true);
            for (size_t i = 0; i < weights_->blocks.size(); ++i) {
                const auto & block = weights_->blocks[i];
                const int64_t in_channels = hidden.shape.dims[1];
                const int64_t out_channels = block.conv_t.conv.weight.shape.dims[1];
                hidden = build_snake1d_exact_bct(ctx, hidden, block.snake, in_channels);
                hidden = use_col2im_conv_transpose_ ?
                    build_conv_transpose1d_col2im(
                        ctx,
                        hidden,
                        block.conv_t,
                        in_channels,
                        out_channels,
                        true) :
                    build_conv_transpose1d(
                        ctx,
                        hidden,
                        block.conv_t,
                        in_channels,
                        out_channels,
                        true);
                hidden = build_residual_unit(ctx, hidden, block.res1, out_channels, 1);
                hidden = build_residual_unit(ctx, hidden, block.res2, out_channels, 3);
                hidden = build_residual_unit(ctx, hidden, block.res3, out_channels, 9);
            }

            hidden = build_snake1d_exact_bct(ctx, hidden, weights_->snake_out, config.decoder_channels);
            hidden = build_conv1d(ctx, hidden, weights_->conv2, config.decoder_channels, config.audio_channels, false);
            output_value_ = hidden;
            audio_frames_ = hidden.shape.dims[2];
            if (audio_frames_ % latent_frames_ != 0) {
                throw std::runtime_error("ACE-Step VAE output length is not divisible by latent frame capacity");
            }
            audio_frames_per_latent_ = audio_frames_ / latent_frames_;
            ggml_set_output(output_value_.tensor);
            graph_ = ggml_new_graph_custom(ctx_.get(), 131072, false);
            ggml_build_forward_expand(graph_, output_value_.tensor);
            gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
            if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
                throw std::runtime_error("ACE-Step VAE backend buffer allocation failed");
            }
        }

        std::shared_ptr<const AceStepAssets> assets_;
        ggml_backend_t backend_ = nullptr;
        core::BackendType backend_type_ = core::BackendType::Cpu;
        int threads_ = 1;
        std::shared_ptr<const VAEDecoderWeights> weights_;
        int64_t latent_frames_ = 0;
        bool use_col2im_conv_transpose_ = false;
        int64_t audio_frames_ = 0;
        int64_t audio_frames_per_latent_ = 0;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
        core::TensorValue input_value_;
        core::TensorValue output_value_;
        ggml_cgraph * graph_ = nullptr;
        mutable std::vector<float> bct_input_scratch_;
        mutable std::vector<float> bct_output_scratch_;
        ggml_gallocr_t gallocr_ = nullptr;
    };

    Impl(
        std::shared_ptr<const AceStepAssets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type)
        : assets_(std::move(assets)),
          backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          graph_arena_bytes_(graph_arena_bytes),
          weights_(std::make_shared<VAEDecoderWeights>(
              load_vae_decoder_weights(*assets_, backend_, backend_type_, weight_context_bytes, weight_storage_type))) {
        if (assets_ == nullptr) {
            throw std::runtime_error("ACE-Step VAE runtime requires assets");
        }
        if (backend_ == nullptr) {
            throw std::runtime_error("ACE-Step VAE backend is not initialized");
        }
        assets_->vae_weights->release_storage();
    }

    static int64_t decode_direct_frame_limit(core::BackendType backend_type) noexcept {
        return core::uses_host_graph_plan(backend_type) ? 256 : 512;
    }

    static int64_t decode_tiled_chunk_frames(core::BackendType backend_type) noexcept {
        return core::uses_host_graph_plan(backend_type) ? 256 : 128;
    }

    static int64_t requested_decode_overlap_frames() noexcept {
        return 64;
    }

    static int64_t effective_decode_overlap(int64_t chunk_frames, int64_t requested_overlap) noexcept {
        constexpr int64_t kMinOverlapFrames = 4;
        int64_t overlap = requested_overlap;
        while (chunk_frames - 2 * overlap <= 0 && overlap > kMinOverlapFrames) {
            overlap /= 2;
        }
        if (overlap < kMinOverlapFrames && requested_overlap >= kMinOverlapFrames) {
            overlap = kMinOverlapFrames;
        }
        return overlap;
    }

    static int64_t decode_chunk_frames(core::BackendType backend_type, int64_t latent_frames) noexcept {
        const int64_t direct_limit = decode_direct_frame_limit(backend_type);
        if (latent_frames <= direct_limit) {
            return direct_limit;
        }
        return decode_tiled_chunk_frames(backend_type);
    }

    static int64_t decode_chunk_overlap_frames(core::BackendType backend_type, int64_t latent_frames) noexcept {
        const int64_t requested_overlap = requested_decode_overlap_frames();
        const int64_t chunk_frames = decode_chunk_frames(backend_type, latent_frames);
        if (latent_frames <= decode_direct_frame_limit(backend_type)) {
            return requested_overlap;
        }
        return effective_decode_overlap(chunk_frames, requested_overlap);
    }

    void ensure_decode_graph(int64_t latent_frames, bool use_col2im_conv_transpose) {
        if (decode_graph_ && decode_graph_->can_run(latent_frames, use_col2im_conv_transpose)) {
            return;
        }
        decode_graph_.reset();
        decode_graph_ = std::make_unique<DecodeGraph>(
            assets_,
            backend_,
            backend_type_,
            threads_,
            weights_,
            latent_frames,
            use_col2im_conv_transpose,
            graph_arena_bytes_);
    }

    runtime::AudioBuffer decode(const AceStepLatents & latents) {
        const auto total_start = Clock::now();
        if (latents.frames <= 0 || latents.channels != assets_->config.vae.decoder_input_channels) {
            throw std::runtime_error("ACE-Step VAE requires valid decoder latents");
        }
        const auto & config = assets_->config.vae;
        const int64_t kChunkFrames = decode_chunk_frames(backend_type_, latents.frames);
        const int64_t kChunkOverlapFrames = decode_chunk_overlap_frames(backend_type_, latents.frames);
        const int64_t frame_factor = audio_frames_per_latent(config);
        const bool use_col2im_conv_transpose =
            !core::uses_host_graph_plan(backend_type_) &&
            latents.frames > decode_direct_frame_limit(backend_type_);
        engine::debug::trace_log_scalar("ace_step.vae.decode.chunk_frames", kChunkFrames);
        engine::debug::trace_log_scalar("ace_step.vae.decode.overlap_frames", kChunkOverlapFrames);
        engine::debug::trace_log_scalar("ace_step.vae.decode.col2im_conv_transpose",
            use_col2im_conv_transpose ? 1 : 0);
        const auto ensure_start = Clock::now();
        ensure_decode_graph(std::min<int64_t>(latents.frames, kChunkFrames), use_col2im_conv_transpose);
        engine::debug::timing_log_scalar(
            "ace_step.vae.decode.graph.ensure_ms",
            engine::debug::elapsed_ms(ensure_start, Clock::now()));
        if (latents.frames > kChunkFrames) {
            const int64_t overlap = std::min<int64_t>(kChunkOverlapFrames, (kChunkFrames - 1) / 2);
            const int64_t stride = kChunkFrames - 2 * overlap;
            if (stride <= 0) {
                throw std::runtime_error("ACE-Step VAE chunked decode stride must be positive");
            }
            runtime::AudioBuffer stitched;
            stitched.sample_rate = config.sample_rate;
            stitched.channels = config.audio_channels;
            stitched.samples.reserve(static_cast<size_t>(latents.frames * frame_factor * config.audio_channels));

            double chunk_decode_ms = 0.0;
            for (int64_t core_start = 0; core_start < latents.frames; core_start += stride) {
                const int64_t core_end = std::min<int64_t>(core_start + stride, latents.frames);
                const int64_t win_start = std::max<int64_t>(0, core_start - overlap);
                const int64_t win_end = std::min<int64_t>(latents.frames, core_end + overlap);
                AceStepLatents window = {};
                window.frames = win_end - win_start;
                window.channels = latents.channels;
                window.values.resize(static_cast<size_t>(window.frames * window.channels), 0.0F);
                const float * src = latents.values.data() + static_cast<size_t>(win_start * latents.channels);
                std::copy_n(src, window.values.size(), window.values.data());
                const auto chunk_start = Clock::now();
                runtime::AudioBuffer decoded = decode_graph_->decode(window);
                chunk_decode_ms += engine::debug::elapsed_ms(chunk_start, Clock::now());
                const int64_t decoded_frames = static_cast<int64_t>(decoded.samples.size()) / decoded.channels;
                const int64_t trim_start = (core_start - win_start) * frame_factor;
                const int64_t trim_end = (win_end - core_end) * frame_factor;
                const int64_t emit_start = trim_start;
                const int64_t emit_end = decoded_frames - trim_end;
                if (emit_start < 0 || emit_end < emit_start || emit_end > decoded_frames) {
                    throw std::runtime_error("ACE-Step VAE chunked decode trim range is invalid");
                }
                const size_t sample_start = static_cast<size_t>(emit_start * decoded.channels);
                const size_t sample_end = static_cast<size_t>(emit_end * decoded.channels);
                stitched.samples.insert(
                    stitched.samples.end(),
                    decoded.samples.begin() + static_cast<std::ptrdiff_t>(sample_start),
                    decoded.samples.begin() + static_cast<std::ptrdiff_t>(sample_end));
            }

            engine::debug::timing_log_scalar("ace_step.vae.decode.chunk_decode_ms", chunk_decode_ms);
            engine::debug::timing_log_scalar(
                "ace_step.vae.decode.runtime_total_ms",
                engine::debug::elapsed_ms(total_start, Clock::now()));
            return stitched;
        }
        runtime::AudioBuffer out = decode_graph_->decode(latents);
        engine::debug::timing_log_scalar(
            "ace_step.vae.decode.runtime_total_ms",
            engine::debug::elapsed_ms(total_start, Clock::now()));
        return out;
    }

    void release_runtime_graphs() {
        decode_graph_.reset();
    }

    std::shared_ptr<const AceStepAssets> assets_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    size_t graph_arena_bytes_ = 0;
    std::shared_ptr<const VAEDecoderWeights> weights_;
    std::unique_ptr<DecodeGraph> decode_graph_;
};

AceStepVAEDecoderRuntime::AceStepVAEDecoderRuntime(
    std::shared_ptr<const AceStepAssets> assets,
    core::ExecutionContext & execution,
    assets::TensorStorageType weight_storage_type,
    size_t graph_arena_bytes,
    size_t weight_context_bytes)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          execution,
          graph_arena_bytes,
          weight_context_bytes,
          weight_storage_type)) {}

AceStepVAEDecoderRuntime::~AceStepVAEDecoderRuntime() = default;

runtime::AudioBuffer AceStepVAEDecoderRuntime::decode(const AceStepLatents & latents) const {
    return impl_->decode(latents);
}

void AceStepVAEDecoderRuntime::release_runtime_graphs() const {
    impl_->release_runtime_graphs();
}

}  // namespace engine::models::ace_step
