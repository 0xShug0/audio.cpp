#include "engine/models/universr/network.h"

#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::models::universr {

namespace binding = modules::binding;

ConvNeXtWeights load_convnext_weights(core::BackendWeightStore & store,
    const assets::TensorSource & source, const std::string & prefix,
    int64_t channels, assets::TensorStorageType storage) {
    ConvNeXtWeights weights;
    weights.depthwise = binding::conv2d_from_source(store, source, prefix + ".dwconv",
        assets::TensorStorageType::F32, channels, 1, 7, 7, true);
    weights.norm = binding::norm_from_source(store, source, prefix + ".norm", channels);
    weights.expansion = binding::linear_from_source(store, source, prefix + ".pwconv1", storage, 4 * channels, channels, true);
    const auto projection = store.load_tensor(source, prefix + ".pwconv2.weight", storage, {channels, 4 * channels});
    const auto projection_storage = assets::tensor_storage_type_for_dtype(ggml_type_name(projection.type));
    const auto projection_f32 = assets::tensor_data_to_f32(prefix + ".pwconv2.weight",
        source.require_tensor(prefix + ".pwconv2.weight", projection_storage, {channels, 4 * channels}));
    const auto beta = source.require_f32(prefix + ".grn.beta", {1, 1, 1, 4 * channels});
    auto bias = source.require_f32(prefix + ".pwconv2.bias", {channels});
    // GRN's constant offset can be applied once through the following linear projection.
    for (int64_t output = 0; output < channels; ++output) {
        double folded = bias[static_cast<size_t>(output)];
        for (int64_t input = 0; input < 4 * channels; ++input) {
            folded += static_cast<double>(projection_f32[static_cast<size_t>(output * 4 * channels + input)]) *
                beta[static_cast<size_t>(input)];
        }
        bias[static_cast<size_t>(output)] = static_cast<float>(folded);
    }
    weights.projection = {
        projection,
        store.make_f32(core::TensorShape::from_dims({channels}), std::move(bias))};
    weights.grn_gamma = store.load_f32_tensor(source, prefix + ".grn.gamma", {1, 1, 1, 4 * channels});
    return weights;
}

core::TensorValue build_convnext_block(core::ModuleBuildContext & ctx,
    const core::TensorValue & input, const ConvNeXtWeights & weights) {
    core::validate_rank_between(input, 4, 4, "UniverSR ConvNeXt input");
    const int64_t channels = input.shape.dims[1];
    auto x = modules::ReflectPad1dModule({3, 3}).build(ctx, input);
    x = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, x);
    x = core::ensure_backend_addressable_layout(ctx, x);
    x = modules::ReflectPad1dModule({3, 3}).build(ctx, x);
    x = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, x);
    x = modules::DepthwiseConv2dModule({channels, 7, 7, 1, 1, 0, 0, 1, 1, true})
        .build(ctx, x, weights.depthwise);
    x = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
    x = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, x);
    x = modules::LayerNormModule({channels, 1e-6f, true, true}).build(ctx, x, weights.norm);
    x = modules::LinearModule({channels, 4 * channels, true}).build(ctx, x, weights.expansion);
    x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);

    // GRN reduces both spatial axes, not channels or independent time chunks.
    auto energy = core::wrap_tensor(ggml_sqr(ctx.ggml, x.tensor), x.shape);
    energy = core::reshape_tensor(ctx, energy,
        core::TensorShape::from_dims({x.shape.dims[0], x.shape.dims[1] * x.shape.dims[2], x.shape.dims[3]}));
    energy = modules::ReduceSumModule({1}).build(ctx, energy);
    energy = core::ensure_backend_addressable_layout(ctx, energy);
    energy = core::reshape_tensor(ctx, energy,
        core::TensorShape::from_dims({x.shape.dims[0], 1, 1, x.shape.dims[3]}));
    auto response = modules::SqrtModule().build(ctx, energy);
    auto mean = modules::ReduceMeanModule({3}).build(ctx, response);
    mean = core::wrap_tensor(ggml_scale_bias(ctx.ggml, mean.tensor, 1.0f, 1e-6f), mean.shape);
    response = core::wrap_tensor(ggml_div(ctx.ggml, response.tensor, mean.tensor), response.shape);
    auto scale = modules::MulModule().build(ctx, response,
        modules::RepeatModule({response.shape}).build(ctx, weights.grn_gamma));
    scale = core::wrap_tensor(ggml_scale_bias(ctx.ggml, scale.tensor, 1.0f, 1.0f), scale.shape);
    auto projection = weights.projection;
    if (x.shape.dims[0] == 1 && projection.weight.type == GGML_TYPE_F32 &&
        x.shape.dims[1] * x.shape.dims[2] > channels) {
        // Scale the smaller projection matrix instead of every spatial activation.
        const auto column_scale = core::reshape_tensor(ctx, scale,
            core::TensorShape::from_dims({1, 4 * channels}));
        projection.weight = modules::MulModule().build(ctx, projection.weight,
            modules::RepeatModule({projection.weight.shape}).build(ctx, column_scale));
    } else {
        x = modules::MulModule().build(ctx, x, modules::RepeatModule({x.shape}).build(ctx, scale));
    }
    x = modules::LinearModule({4 * channels, channels, true}).build(ctx, x, projection);
    x = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, x);
    x = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
    return modules::AddModule().build(ctx, input, x);
}
ConditioningWeights load_conditioning_weights(core::BackendWeightStore & store,
    const assets::TensorSource & source, const UniverSRConfig & config,
    assets::TensorStorageType storage) {
    ConditioningWeights weights;
    const int64_t dim = config.cond_dim;
    weights.frequency_pe = store.load_f32_tensor(source, "freq_pos_enc.pe", {config.total_freq_bins, dim});
    weights.sample_rate_embedding = store.load_f32_tensor(source, "sr_embedder.weight",
        {static_cast<int64_t>(config.sr_to_lr_bins.size()), dim});
    weights.unconditional_embedding = store.load_f32_tensor(source, "uncond_emb", {dim});
    weights.time_frequencies = store.load_f32_tensor(source, "time_embedder.weights", {1, config.time_dim / 2});
    weights.sample_rate_projector = binding::linear_from_source(store, source, "sr_projector", storage, config.time_dim, dim, true);
    weights.low_film = binding::linear_from_source(store, source, "conditioning_encoder.film_generator", storage, 4, dim, true);
    weights.high_film = binding::linear_from_source(store, source, "film_generator", storage, 2 * dim, dim, true);
    // Retain rank four for the store's reshape API, then omit physical singleton axes.
    const auto head = store.load_tensor_as_shape(source, "conditioning_encoder.head.weight", storage,
        {dim, 2, 1, 1}, core::TensorShape::from_dims({1, 1, dim, 2}));
    weights.head = {
        core::wrap_tensor(head.tensor, core::TensorShape::from_dims({dim, 2})),
        store.load_f32_tensor(source, "conditioning_encoder.head.bias", {dim})};
    weights.sample_rate_hidden = binding::linear_from_source(store, source, "conditioning_encoder.sr_adapter.0", storage, dim, dim, true);
    weights.sample_rate_film = binding::linear_from_source(store, source, "conditioning_encoder.sr_adapter.2", storage, 2 * dim, dim, true);
    for (int block = 0; block < config.feature_enc_layers; ++block) {
        weights.blocks.push_back(load_convnext_weights(store, source,
            "conditioning_encoder.blocks." + std::to_string(block), dim, storage));
    }
    return weights;
}

core::TensorValue build_conditioning_encoder(core::ModuleBuildContext & ctx,
    const core::TensorValue & low_spectrum, const core::TensorValue & sample_rate_embedding,
    const ConditioningWeights & weights) {
    core::validate_rank_between(low_spectrum, 4, 4, "UniverSR low spectrum");
    const int64_t bins = low_spectrum.shape.dims[2];
    const int64_t dim = sample_rate_embedding.shape.last_dim();
    auto pe = modules::SliceModule({0, 0, bins}).build(ctx, weights.frequency_pe);
    auto film = modules::LinearModule({dim, 4, true}).build(ctx, pe, weights.low_film);
    auto gamma = modules::SliceModule({1, 0, 2}).build(ctx, film);
    auto beta = modules::SliceModule({1, 2, 2}).build(ctx, film);
    gamma = modules::TransposeModule({{1, 0, 2, 3}, 2}).build(ctx, gamma);
    beta = modules::TransposeModule({{1, 0, 2, 3}, 2}).build(ctx, beta);
    gamma = core::ensure_backend_addressable_layout(ctx, gamma);
    beta = core::ensure_backend_addressable_layout(ctx, beta);
    gamma = core::reshape_tensor(ctx, gamma, core::TensorShape::from_dims({1, 2, bins, 1}));
    beta = core::reshape_tensor(ctx, beta, core::TensorShape::from_dims({1, 2, bins, 1}));
    auto x = modules::MulModule().build(ctx, low_spectrum,
        modules::RepeatModule({low_spectrum.shape}).build(ctx, gamma));
    x = modules::AddModule().build(ctx, x, modules::RepeatModule({x.shape}).build(ctx, beta));
    x = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
    x = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, x);
    x = modules::LinearModule({2, dim, true}).build(ctx, x, weights.head);
    x = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, x);
    x = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
    x = core::ensure_backend_addressable_layout(ctx, x);
    film = modules::LinearModule({dim, dim, true}).build(ctx, sample_rate_embedding, weights.sample_rate_hidden);
    film = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, film);
    film = modules::LinearModule({dim, 2 * dim, true}).build(ctx, film, weights.sample_rate_film);
    gamma = modules::SliceModule({1, 0, dim}).build(ctx, film);
    beta = modules::SliceModule({1, dim, dim}).build(ctx, film);
    gamma = core::ensure_backend_addressable_layout(ctx, gamma);
    beta = core::ensure_backend_addressable_layout(ctx, beta);
    gamma = core::reshape_tensor(ctx, gamma, core::TensorShape::from_dims({1, dim, 1, 1}));
    beta = core::reshape_tensor(ctx, beta, core::TensorShape::from_dims({1, dim, 1, 1}));
    x = modules::MulModule().build(ctx, x, modules::RepeatModule({x.shape}).build(ctx, gamma));
    x = modules::AddModule().build(ctx, x, modules::RepeatModule({x.shape}).build(ctx, beta));
    for (const auto & block : weights.blocks) {
        x = build_convnext_block(ctx, x, block);
    }
    return modules::ReduceMeanModule({2}).build(ctx, x);
}

core::TensorValue build_spatial_conditioning(core::ModuleBuildContext & ctx,
    const core::TensorValue & condition, const UniverSRConfig & config,
    const ConditioningWeights & weights) {
    const int64_t dim = config.cond_dim;
    const int64_t bins = config.hr_freq_bins;
    auto pe = modules::SliceModule({0, config.total_freq_bins - bins, bins}).build(ctx, weights.frequency_pe);
    auto film = modules::LinearModule({dim, 2 * dim, true}).build(ctx, pe, weights.high_film);
    auto gamma = modules::SliceModule({1, 0, dim}).build(ctx, film);
    auto beta = modules::SliceModule({1, dim, dim}).build(ctx, film);
    gamma = modules::TransposeModule({{1, 0, 2, 3}, 2}).build(ctx, gamma);
    beta = modules::TransposeModule({{1, 0, 2, 3}, 2}).build(ctx, beta);
    gamma = core::ensure_backend_addressable_layout(ctx, gamma);
    beta = core::ensure_backend_addressable_layout(ctx, beta);
    gamma = core::reshape_tensor(ctx, gamma, core::TensorShape::from_dims({1, dim, bins, 1}));
    beta = core::reshape_tensor(ctx, beta, core::TensorShape::from_dims({1, dim, bins, 1}));
    auto x = modules::RepeatModule({core::TensorShape::from_dims({1, dim, bins, condition.shape.dims[3]})})
        .build(ctx, condition);
    x = modules::MulModule().build(ctx, x, modules::RepeatModule({x.shape}).build(ctx, gamma));
    return modules::AddModule().build(ctx, x, modules::RepeatModule({x.shape}).build(ctx, beta));
}

core::TensorValue build_projected_conditioning(core::ModuleBuildContext & ctx,
    const core::TensorValue & condition, const UniverSRConfig & config,
    const ConditioningWeights & weights, const core::TensorValue & projection) {
    const int64_t dim = config.cond_dim;
    const int64_t bins = config.hr_freq_bins;
    const int64_t channels = projection.shape.dims[0];
    const int64_t frames = condition.shape.dims[3];
    auto pe = modules::SliceModule({0, config.total_freq_bins - bins, bins}).build(ctx, weights.frequency_pe);
    auto film = modules::LinearModule({dim, 2 * dim, true}).build(ctx, pe, weights.high_film);
    auto gamma = modules::SliceModule({1, 0, dim}).build(ctx, film);
    auto beta = modules::SliceModule({1, dim, dim}).build(ctx, film);
    gamma = core::ensure_backend_addressable_layout(ctx, gamma);
    gamma = core::reshape_tensor(ctx, gamma, core::TensorShape::from_dims({bins, 1, dim}));
    auto kernel = core::ensure_backend_addressable_layout(ctx, projection);
    if (kernel.type != GGML_TYPE_F32) {
        kernel = core::wrap_tensor(ggml_cast(ctx.ggml, kernel.tensor, GGML_TYPE_F32), kernel.shape);
    }
    kernel = core::reshape_tensor(ctx, kernel, core::TensorShape::from_dims({1, channels, dim}));
    kernel = modules::RepeatModule({core::TensorShape::from_dims({bins, channels, dim})}).build(ctx, kernel);
    kernel = modules::MulModule().build(ctx, kernel, modules::RepeatModule({kernel.shape}).build(ctx, gamma));
    auto temporal = modules::TransposeModule({{0, 3, 2, 1}, 4}).build(ctx, condition);
    temporal = core::ensure_backend_addressable_layout(ctx, temporal);
    temporal = core::reshape_tensor(ctx, temporal, core::TensorShape::from_dims({1, frames, dim}));
    // Broadcast one temporal sequence across frequency-specific projection matrices.
    auto output = core::wrap_tensor(ggml_mul_mat(ctx.ggml, temporal.tensor, kernel.tensor),
        core::TensorShape::from_dims({1, bins, channels, frames}));
    ggml_mul_mat_set_prec(output.tensor, GGML_PREC_F32);
    output = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, output);
    output = core::ensure_backend_addressable_layout(ctx, output);
    beta = modules::LinearModule({dim, channels, false}).build(ctx, beta, {projection, std::nullopt});
    beta = core::reshape_tensor(ctx, beta, core::TensorShape::from_dims({1, bins, 1, channels}));
    return modules::AddModule().build(ctx, output, modules::RepeatModule({output.shape}).build(ctx, beta));
}

core::TensorValue build_time_embedding(core::ModuleBuildContext & ctx,
    const core::TensorValue & time, const core::TensorValue & sample_rate_embedding,
    const ConditioningWeights & weights) {
    auto frequency = modules::MulModule().build(ctx, weights.time_frequencies,
        modules::RepeatModule({weights.time_frequencies.shape}).build(ctx, time));
    frequency = core::wrap_tensor(ggml_scale(ctx.ggml, frequency.tensor, 2.0f), frequency.shape);
    frequency = core::wrap_tensor(ggml_scale(ctx.ggml, frequency.tensor, static_cast<float>(std::acos(-1.0))), frequency.shape);
    auto sine = core::wrap_tensor(ggml_sin(ctx.ggml, frequency.tensor), frequency.shape);
    auto cosine = core::wrap_tensor(ggml_cos(ctx.ggml, frequency.tensor), frequency.shape);
    auto x = modules::ConcatModule({1}).build(ctx, sine, cosine);
    x = core::wrap_tensor(ggml_scale(ctx.ggml, x.tensor, std::sqrt(2.0f)), x.shape);
    const auto rate = modules::LinearModule({sample_rate_embedding.shape.last_dim(), x.shape.last_dim(), true})
        .build(ctx, sample_rate_embedding, weights.sample_rate_projector);
    return modules::AddModule().build(ctx, x, rate);
}
namespace {

TimeBlockWeights load_time_block_weights(core::BackendWeightStore & store,
    const assets::TensorSource & source, const std::string & prefix,
    int64_t channels, int64_t time_dim, assets::TensorStorageType storage) {
    TimeBlockWeights weights;
    weights.time_hidden = binding::linear_from_source(store, source, prefix + ".time_adapter.0", storage, time_dim, time_dim, true);
    weights.time_output = binding::linear_from_source(store, source, prefix + ".time_adapter.2", storage, channels, time_dim, true);
    weights.block = load_convnext_weights(store, source, prefix + ".block", channels, storage);
    return weights;
}

}  // namespace

UNetBackboneWeights load_unet_backbone_weights(core::BackendWeightStore & store,
    const assets::TensorSource & source, const UniverSRConfig & config,
    assets::TensorStorageType storage) {
    UNetBackboneWeights weights;
    const int64_t first = config.dims.front();
    const int64_t input_channels = config.cond_dim + 2;
    // The 386-column projection is split after two noise channels, not on a quantization block boundary.
    const auto input_matrix = store.load_tensor_as_shape(source, "init_conv.0.weight", assets::TensorStorageType::F32,
        {first, input_channels, 1, 1}, core::TensorShape::from_dims({1, 1, first, input_channels}));
    weights.input = {
        core::wrap_tensor(input_matrix.tensor, core::TensorShape::from_dims({first, input_channels})),
        store.load_f32_tensor(source, "init_conv.0.bias", {first})};
    weights.input_norm = binding::norm_from_source(store, source, "init_conv.1", first);
    for (size_t stage = 0; stage < config.dims.size(); ++stage) {
        const int64_t in_channels = config.dims[stage];
        const int64_t out_channels = config.dims[std::min(stage + 1, config.dims.size() - 1)];
        const std::string prefix = "encoders." + std::to_string(stage);
        EncoderWeights encoder;
        for (int64_t block = 0; block < config.depths[stage]; ++block) {
            encoder.blocks.push_back(load_time_block_weights(store, source,
                prefix + ".blocks." + std::to_string(block), in_channels, config.time_dim, storage));
        }
        encoder.norm = binding::norm_from_source(store, source, prefix + ".downsampler.0", in_channels);
        encoder.downsample = binding::conv2d_from_source(store, source, prefix + ".downsampler.1",
            assets::TensorStorageType::F32, out_channels, in_channels, 2, 2, true);
        weights.encoders.push_back(std::move(encoder));
    }
    for (int64_t block = 0; block < config.depths.back(); ++block) {
        weights.middle.push_back(load_time_block_weights(store, source,
            "midcoder.blocks." + std::to_string(block), config.dims.back(), config.time_dim, storage));
    }
    for (size_t stage = 0; stage < config.dims.size(); ++stage) {
        const size_t reverse = config.dims.size() - 1 - stage;
        const int64_t in_channels = config.dims[std::min(reverse + 1, config.dims.size() - 1)];
        const int64_t out_channels = config.dims[reverse];
        const std::string prefix = "decoders." + std::to_string(stage);
        DecoderWeights decoder;
        const auto kernel = source.require_f32(prefix + ".upsampler.weight", {in_channels, out_channels, 2, 2});
        std::vector<float> projection(kernel.size());
        for (int64_t input = 0; input < in_channels; ++input) {
            for (int64_t output = 0; output < 4 * out_channels; ++output) {
                projection[static_cast<size_t>(output * in_channels + input)] =
                    kernel[static_cast<size_t>(input * 4 * out_channels + output)];
            }
        }
        decoder.upsample_weight = store.make_f32(core::TensorShape::from_dims({4 * out_channels, in_channels}), std::move(projection));
        decoder.upsample_bias = store.load_f32_tensor(source, prefix + ".upsampler.bias", {out_channels});
        for (int64_t block = 0; block < config.depths[reverse]; ++block) {
            decoder.blocks.push_back(load_time_block_weights(store, source,
                prefix + ".blocks." + std::to_string(block), out_channels, config.time_dim, storage));
        }
        weights.decoders.push_back(std::move(decoder));
    }
    // Retain rank four for the store's reshape API, then omit physical singleton axes.
    const auto output_matrix = store.load_tensor_as_shape(source, "final_conv.weight", storage,
        {2, first, 1, 1}, core::TensorShape::from_dims({1, 1, 2, first}));
    weights.output = {
        core::wrap_tensor(output_matrix.tensor, core::TensorShape::from_dims({2, first})),
        store.load_f32_tensor(source, "final_conv.bias", {2})};
    return weights;
}

core::TensorValue build_time_block(core::ModuleBuildContext & ctx,
    const core::TensorValue & input, const core::TensorValue & time,
    const TimeBlockWeights & weights) {
    const int64_t time_dim = time.shape.last_dim();
    const int64_t channels = input.shape.dims[1];
    auto embedding = modules::LinearModule({time_dim, time_dim, true}).build(ctx, time, weights.time_hidden);
    embedding = modules::SiluModule().build(ctx, embedding);
    embedding = modules::LinearModule({time_dim, channels, true}).build(ctx, embedding, weights.time_output);
    embedding = core::reshape_tensor(ctx, embedding,
        core::TensorShape::from_dims({input.shape.dims[0], channels, 1, 1}));
    auto x = core::ensure_backend_addressable_layout(ctx, input);
    x = modules::AddModule().build(ctx, x, modules::RepeatModule({x.shape}).build(ctx, embedding));
    return build_convnext_block(ctx, x, weights.block);
}

core::TensorValue build_unet_backbone(core::ModuleBuildContext & ctx,
    const core::TensorValue & input, const core::TensorValue & time,
    const UNetBackboneWeights & weights, bool input_projected) {
    core::validate_rank_between(input, 4, 4, "UniverSR U-Net input");
    core::validate_rank_between(time, 2, 2, "UniverSR U-Net time");
    if (weights.encoders.size() != 4 || weights.decoders.size() != 4 ||
        input.shape.dims[2] != 432 || input.shape.dims[3] % 16 != 0) {
        throw std::runtime_error("UniverSR backbone requires four stages and a padded 432-bin spectrum");
    }
    const int64_t channels = weights.input.weight.shape.dims[0];
    auto x = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, input);
    x = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, x);
    if (!input_projected) {
        x = modules::LinearModule({input.shape.dims[1], channels, true}).build(ctx, x, weights.input);
    }
    x = modules::LayerNormModule({channels, 1e-6f, true, true}).build(ctx, x, weights.input_norm);
    x = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, x);
    x = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
    std::vector<core::TensorValue> skips{x};
    for (const auto & encoder : weights.encoders) {
        for (const auto & block : encoder.blocks) {
            x = build_time_block(ctx, x, time, block);
        }
        const int64_t in_channels = x.shape.dims[1];
        const int64_t out_channels = encoder.downsample.weight.shape.dims[0];
        x = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
        x = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, x);
        x = modules::LayerNormModule({in_channels, 1e-6f, true, true}).build(ctx, x, encoder.norm);
        x = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, x);
        x = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
        x = modules::Conv2dModule({in_channels, out_channels, 2, 2, 2, 2, 0, 0, 1, 1, true})
            .build(ctx, x, encoder.downsample);
        skips.push_back(x);
    }
    for (const auto & block : weights.middle) {
        x = build_time_block(ctx, x, time, block);
    }
    for (const auto & decoder : weights.decoders) {
        x = modules::AddModule().build(ctx, x, skips.back());
        skips.pop_back();
        const auto & weight = decoder.upsample_weight;
        const int64_t out_channels = weight.shape.dims[0] / 4;
        const int64_t batch = x.shape.dims[0];
        const int64_t height = x.shape.dims[2];
        const int64_t width = x.shape.dims[3];
        // Kernel size equals stride: each input position produces one non-overlapping 2x2 output patch.
        x = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
        x = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, x);
        x = modules::LinearModule({weight.shape.dims[1], 4 * out_channels, false})
            .build(ctx, x, {weight, std::nullopt});
        x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({batch * height, width, 2 * out_channels, 2}));
        x = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
        x = core::ensure_backend_addressable_layout(ctx, x);
        x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({batch, height, out_channels, 4 * width}));
        x = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
        x = core::ensure_backend_addressable_layout(ctx, x);
        x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({batch, out_channels, 2 * height, 2 * width}));
        const auto bias = core::reshape_tensor(ctx, decoder.upsample_bias,
            core::TensorShape::from_dims({1, out_channels, 1, 1}));
        x = modules::AddModule().build(ctx, x, modules::RepeatModule({x.shape}).build(ctx, bias));
        for (const auto & block : decoder.blocks) {
            x = build_time_block(ctx, x, time, block);
        }
    }
    x = modules::AddModule().build(ctx, x, skips.back());
    x = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
    x = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, x);
    x = modules::LinearModule({channels, 2, true}).build(ctx, x, weights.output);
    x = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, x);
    return modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
}

}  // namespace engine::models::universr
