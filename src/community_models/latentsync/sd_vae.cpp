#include "engine/community_models/latentsync/sd_vae.h"

#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <stdexcept>

namespace engine::models::latentsync {

SdVaeTowerWeights load_sd_vae_weights(core::BackendWeightStore & store,
    const assets::TensorSource & source, const LatentSyncSdVaeConfig & config,
    bool encoder, assets::TensorStorageType type) {
    namespace binding = modules::binding;
    const std::string root = encoder ? "vae.encoder" : "vae.decoder";
    SdVaeTowerWeights weights;
    const auto load_resnet = [&](const std::string & prefix, int64_t in, int64_t out) {
        SdVaeResnetWeights block;
        block.norm1 = binding::norm_from_source(store, source, prefix + ".norm1", in);
        block.norm2 = binding::norm_from_source(store, source, prefix + ".norm2", out);
        block.conv1 = binding::conv2d_from_source(store, source, prefix + ".conv1", type, out, in, 3, 3, true);
        block.conv2 = binding::conv2d_from_source(store, source, prefix + ".conv2", type, out, out, 3, 3, true);
        if (in != out) {
            block.shortcut = binding::conv2d_from_source(store, source, prefix + ".conv_shortcut", type, out, in, 1, 1, true);
        }
        return block;
    };
    int64_t current = encoder ? config.channels.front() : config.channels.back();
    weights.input = binding::conv2d_from_source(store, source, root + ".conv_in", type,
        current, encoder ? 3 : config.latent_channels, 3, 3, true);
    for (size_t index = 0; index < config.channels.size(); ++index) {
        const int64_t channels = config.channels[encoder ? index : config.channels.size() - index - 1];
        const std::string prefix = root + (encoder ? ".down_blocks." : ".up_blocks.") + std::to_string(index);
        SdVaeStageWeights stage;
        for (int64_t layer = 0; layer < config.layers_per_block + (encoder ? 0 : 1); ++layer) {
            stage.resnets.push_back(load_resnet(prefix + ".resnets." + std::to_string(layer), current, channels));
            current = channels;
        }
        if (index + 1 < config.channels.size()) {
            stage.resample = binding::conv2d_from_source(store, source,
                prefix + (encoder ? ".downsamplers.0.conv" : ".upsamplers.0.conv"),
                type, channels, channels, 3, 3, true);
        }
        weights.stages.push_back(std::move(stage));
    }
    const int64_t mid = config.channels.back();
    for (size_t i = 0; i < weights.mid_resnets.size(); ++i) {
        weights.mid_resnets[i] = load_resnet(root + ".mid_block.resnets." + std::to_string(i), mid, mid);
    }
    const auto attention = root + ".mid_block.attentions.0";
    weights.attention_norm = binding::norm_from_source(store, source, attention + ".group_norm", mid);
    weights.attention.query = binding::linear_from_source(store, source, attention + ".query", type, mid, mid, true);
    weights.attention.key = binding::linear_from_source(store, source, attention + ".key", type, mid, mid, true);
    weights.attention.value = binding::linear_from_source(store, source, attention + ".value", type, mid, mid, true);
    weights.attention.out = binding::linear_from_source(store, source, attention + ".proj_attn", type, mid, mid, true);
    weights.output_norm = binding::norm_from_source(store, source, root + ".conv_norm_out", current);
    const int64_t out = encoder ? config.latent_channels * 2 : 3;
    weights.output = binding::conv2d_from_source(store, source, root + ".conv_out", type, out, current, 3, 3, true);
    const int64_t quant = encoder ? config.latent_channels * 2 : config.latent_channels;
    weights.quant = binding::conv2d_from_source(store, source,
        encoder ? "vae.quant_conv" : "vae.post_quant_conv", type, quant, quant, 1, 1, true);
    return weights;
}

LatentSyncSdVae::LatentSyncSdVae(LatentSyncSdVaeConfig config) : config_(config) {
    if (config_.latent_channels <= 0 || config_.layers_per_block <= 0 || config_.norm_groups <= 0) {
        throw std::runtime_error("Invalid LatentSync SD VAE dimensions");
    }
    for (const int64_t channels : config_.channels) {
        if (channels <= 0 || channels % config_.norm_groups != 0) {
            throw std::runtime_error("Invalid LatentSync SD VAE group normalization channels");
        }
    }
}

core::TensorValue LatentSyncSdVae::build_resnet(core::ModuleBuildContext & ctx,
    const core::TensorValue & input, const SdVaeResnetWeights & weights) const {
    using namespace modules;
    const int64_t in = input.shape.dims[1];
    const int64_t out = weights.conv1.weight.shape.dims[0];
    auto x = GroupNormModule({in, config_.norm_groups, 1e-6F, true, true, config_.cuda_f16_activations}).build(ctx, input, weights.norm1);
    x = SiluModule(config_.cuda_f16_activations).build(ctx, x);
    x = Conv2dModule({in, out, 3, 3, 1, 1, 1, 1}, config_.cuda_cudnn, config_.cuda_f16_activations).build(ctx, x, weights.conv1);
    x = GroupNormModule({out, config_.norm_groups, 1e-6F, true, true, config_.cuda_f16_activations}).build(ctx, x, weights.norm2);
    x = SiluModule(config_.cuda_f16_activations).build(ctx, x);
    x = Conv2dModule({out, out, 3, 3, 1, 1, 1, 1}, config_.cuda_cudnn, config_.cuda_f16_activations).build(ctx, x, weights.conv2);
    auto residual = input;
    if (weights.shortcut) {
        residual = Conv2dModule({in, out, 1, 1}, config_.cuda_cudnn, config_.cuda_f16_activations).build(ctx, input, *weights.shortcut);
    }
    return AddModule(config_.cuda_f16_activations).build(ctx, residual, x);
}

core::TensorValue LatentSyncSdVae::build_mid(core::ModuleBuildContext & ctx,
    const core::TensorValue & input, const SdVaeTowerWeights & weights) const {
    using namespace modules;
    const auto x = build_resnet(ctx, input, weights.mid_resnets[0]);
    const int64_t batch = x.shape.dims[0], channels = x.shape.dims[1];
    const int64_t steps = x.shape.dims[2] * x.shape.dims[3];
    auto attention_input = config_.cuda_f16_activations
        ? core::wrap_tensor(ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_F32), x.shape, GGML_TYPE_F32) : x;
    auto sequence = GroupNormModule({channels, config_.norm_groups, 1e-6F}).build(ctx, attention_input, weights.attention_norm);
    sequence = core::reshape_tensor(ctx, sequence, core::TensorShape::from_dims({batch, channels, steps}));
    sequence = TransposeModule({{0, 2, 1}, 3}).build(ctx, sequence);
    const auto project = [&](const LinearWeights & projection) {
        const auto result = LinearModule({channels, channels, true}).build(ctx, sequence, projection);
        return core::reshape_tensor(ctx, result, core::TensorShape::from_dims({batch, 1, steps, channels}));
    };
    // SD VAE uses one 512-dimensional head, outside the CUDA flash kernel's supported sizes.
    auto attended = ScaledDotProductAttentionModule({channels, ScaledDotProductAttentionLowering::Explicit, GGML_PREC_F32})
        .build(ctx, project(weights.attention.query), project(weights.attention.key), project(weights.attention.value));
    attended = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, attended),
        core::TensorShape::from_dims({batch, steps, channels}));
    attended = LinearModule({channels, channels, true}).build(ctx, attended, weights.attention.out);
    attended = TransposeModule({{0, 2, 1}, 3}).build(ctx, attended);
    attended = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, attended), x.shape);
    if (config_.cuda_f16_activations) {
        attended = core::wrap_tensor(ggml_cast(ctx.ggml, attended.tensor, GGML_TYPE_F16), attended.shape, GGML_TYPE_F16);
    }
    return build_resnet(ctx, AddModule(config_.cuda_f16_activations).build(ctx, x, attended), weights.mid_resnets[1]);
}

core::TensorValue LatentSyncSdVae::build_encoder(core::ModuleBuildContext & ctx,
    const core::TensorValue & image, const SdVaeTowerWeights & weights) const {
    using namespace modules;
    auto x = Conv2dModule({3, config_.channels.front(), 3, 3, 1, 1, 1, 1}, config_.cuda_cudnn, config_.cuda_f16_activations).build(ctx, image, weights.input);
    for (const auto & stage : weights.stages) {
        for (const auto & block : stage.resnets) x = build_resnet(ctx, x, block);
        if (stage.resample) {
            const auto pad_input = config_.cuda_f16_activations
                ? core::wrap_tensor(ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_F32), x.shape, GGML_TYPE_F32) : x;
            x = Pad2dModule({0, 1, 0, 1}).build(ctx, pad_input);
            x = Conv2dModule({x.shape.dims[1], x.shape.dims[1], 3, 3, 2, 2}, config_.cuda_cudnn, config_.cuda_f16_activations).build(ctx, x, *stage.resample);
        }
    }
    x = build_mid(ctx, x, weights);
    x = GroupNormModule({config_.channels.back(), config_.norm_groups, 1e-6F, true, true, config_.cuda_f16_activations}).build(ctx, x, weights.output_norm);
    x = SiluModule(config_.cuda_f16_activations).build(ctx, x);
    x = Conv2dModule({config_.channels.back(), config_.latent_channels * 2, 3, 3, 1, 1, 1, 1}, config_.cuda_cudnn, config_.cuda_f16_activations).build(ctx, x, weights.output);
    x = Conv2dModule({config_.latent_channels * 2, config_.latent_channels * 2, 1, 1}, config_.cuda_cudnn, config_.cuda_f16_activations).build(ctx, x, weights.quant);
    return config_.cuda_f16_activations ? core::wrap_tensor(ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_F32), x.shape, GGML_TYPE_F32) : x;
}

core::TensorValue LatentSyncSdVae::build_decoder(core::ModuleBuildContext & ctx,
    const core::TensorValue & latent, const SdVaeTowerWeights & weights) const {
    using namespace modules;
    auto x = Conv2dModule({config_.latent_channels, config_.latent_channels, 1, 1}, config_.cuda_cudnn, config_.cuda_f16_activations).build(ctx, latent, weights.quant);
    x = Conv2dModule({config_.latent_channels, config_.channels.back(), 3, 3, 1, 1, 1, 1}, config_.cuda_cudnn, config_.cuda_f16_activations).build(ctx, x, weights.input);
    x = build_mid(ctx, x, weights);
    for (const auto & stage : weights.stages) {
        for (const auto & block : stage.resnets) x = build_resnet(ctx, x, block);
        if (stage.resample) {
            const auto upsample_input = config_.cuda_f16_activations
                ? core::wrap_tensor(ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_F32), x.shape, GGML_TYPE_F32) : x;
            x = NearestUpsample2dModule({x.shape.dims[2] * 2, x.shape.dims[3] * 2}).build(ctx, upsample_input);
            x = Conv2dModule({x.shape.dims[1], x.shape.dims[1], 3, 3, 1, 1, 1, 1}, config_.cuda_cudnn, config_.cuda_f16_activations).build(ctx, x, *stage.resample);
        }
    }
    x = GroupNormModule({config_.channels.front(), config_.norm_groups, 1e-6F, true, true, config_.cuda_f16_activations}).build(ctx, x, weights.output_norm);
    x = SiluModule(config_.cuda_f16_activations).build(ctx, x);
    x = Conv2dModule({config_.channels.front(), 3, 3, 3, 1, 1, 1, 1}, config_.cuda_cudnn, config_.cuda_f16_activations).build(ctx, x, weights.output);
    return config_.cuda_f16_activations ? core::wrap_tensor(ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_F32), x.shape, GGML_TYPE_F32) : x;
}

}  // namespace engine::models::latentsync
