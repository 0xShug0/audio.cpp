#include "engine/models/latentsync/unet_attention.h"

#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <stdexcept>
#include <algorithm>

namespace engine::models::latentsync {
namespace {

LatentSyncAttentionWeights load_attention(core::BackendWeightStore & store,
    const assets::TensorSource & source, const std::string & prefix,
    int64_t channels, int64_t context_channels, assets::TensorStorageType type) {
    using modules::binding::linear_from_source;
    return {
        linear_from_source(store, source, prefix + ".to_q", type, channels, channels, false),
        linear_from_source(store, source, prefix + ".to_k", type, channels, context_channels, false),
        linear_from_source(store, source, prefix + ".to_v", type, channels, context_channels, false),
        linear_from_source(store, source, prefix + ".to_out.0", type, channels, channels, true),
    };
}

}  // namespace

LatentSyncSpatialTransformerWeights load_spatial_transformer_weights(
    core::BackendWeightStore & store, const assets::TensorSource & source,
    const std::string & prefix, int64_t channels, int64_t audio_channels,
    assets::TensorStorageType type) {
    namespace binding = modules::binding;
    LatentSyncSpatialTransformerWeights out;
    out.input_norm = binding::norm_from_source(store, source, prefix + ".norm", channels);
    out.input = binding::conv2d_from_source(store, source, prefix + ".proj_in", type, channels, channels, 1, 1, true);
    out.output = binding::conv2d_from_source(store, source, prefix + ".proj_out", type, channels, channels, 1, 1, true);
    const auto block = prefix + ".transformer_blocks.0";
    for (size_t i = 0; i < out.norms.size(); ++i) {
        out.norms[i] = binding::norm_from_source(store, source, block + ".norm" + std::to_string(i + 1), channels);
    }
    out.self_attention = load_attention(store, source, block + ".attn1", channels, channels, type);
    out.audio_attention = load_attention(store, source, block + ".attn2", channels, audio_channels, type);
    out.geglu = binding::linear_from_source(store, source, block + ".ff.net.0.proj", type, channels * 8, channels, true);
    out.feed_forward_out = binding::linear_from_source(store, source, block + ".ff.net.2", type, channels, channels * 4, true);
    return out;
}

LatentSyncTemporalTransformerWeights load_temporal_transformer_weights(
    core::BackendWeightStore & store, const assets::TensorSource & source,
    const std::string & prefix, int64_t channels, int64_t max_frames,
    assets::TensorStorageType type) {
    namespace binding = modules::binding;
    LatentSyncTemporalTransformerWeights out;
    out.input_norm = binding::norm_from_source(store, source, prefix + ".norm", channels);
    out.input = binding::linear_from_source(store, source, prefix + ".proj_in", type, channels, channels, true);
    out.output = binding::linear_from_source(store, source, prefix + ".proj_out", type, channels, channels, true);
    const auto block = prefix + ".transformer_blocks.0";
    for (size_t i = 0; i < out.norms.size(); ++i) {
        const auto attention = block + ".attention_blocks." + std::to_string(i);
        out.norms[i] = binding::norm_from_source(store, source, block + ".norms." + std::to_string(i), channels);
        out.attention[i] = load_attention(store, source, attention, channels, channels, type);
        out.position[i] = store.load_tensor(source, attention + ".pos_encoder.pe",
            assets::TensorStorageType::F32, {1, max_frames, channels});
    }
    out.feed_forward_norm = binding::norm_from_source(store, source, block + ".ff_norm", channels);
    out.geglu = binding::linear_from_source(store, source, block + ".ff.net.0.proj", type, channels * 8, channels, true);
    out.feed_forward_out = binding::linear_from_source(store, source, block + ".ff.net.2", type, channels, channels * 4, true);
    return out;
}

core::TensorValue LatentSyncUnetAttention::build_attention(core::ModuleBuildContext & ctx,
    const core::TensorValue & query, const core::TensorValue & conditioning,
    const LatentSyncAttentionWeights & weights) const {
    using namespace modules;
    const int64_t channels = query.shape.dims[2];
    if (heads_ <= 0 || channels % heads_ != 0) {
        throw std::runtime_error("LatentSync attention channels must be divisible by heads");
    }
    const int64_t dim = channels / heads_;
    const auto project = [&](const core::TensorValue & x, const LinearWeights & projection) {
        auto value = LinearModule({x.shape.dims[2], channels, false}).build(ctx, x, projection);
        value = core::reshape_tensor(ctx, value,
            core::TensorShape::from_dims({x.shape.dims[0], x.shape.dims[1], heads_, dim}));
        return TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, value);
    };
    // The 40/80-wide heads cover the large spatial grids. The 160-wide heads are
    // used at lower resolutions and are not supported by the CUDA flash kernel.
    const auto lowering = dim == 40 || dim == 80
        ? ScaledDotProductAttentionLowering::FlashPreserveViews : ScaledDotProductAttentionLowering::Explicit;
    const auto q = project(query, weights.query);
    const auto k = project(conditioning, weights.key);
    const auto v = project(conditioning, weights.value);
    const ScaledDotProductAttentionModule attention({dim, lowering, GGML_PREC_F32});
    core::TensorValue x;
    // CUDA's tiled flash launch packs batch * heads into grid.z (maximum 65535).
    // Temporal attention treats spatial positions as independent batches.
    if (ctx.backend_type == core::BackendType::Cuda &&
        lowering == ScaledDotProductAttentionLowering::FlashPreserveViews && query.shape.dims[0] * heads_ > 65535) {
        const int64_t chunk = std::max<int64_t>(1, 32768 / heads_);
        for (int64_t offset = 0; offset < query.shape.dims[0]; offset += chunk) {
            const int64_t count = std::min(chunk, query.shape.dims[0] - offset);
            const SliceModule slice({0, offset, count});
            const auto part = attention.build(ctx, slice.build(ctx, q), slice.build(ctx, k), slice.build(ctx, v));
            x = offset == 0 ? part : ConcatModule({0}).build(ctx, x, part);
        }
    } else {
        x = attention.build(ctx, q, k, v);
    }
    x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x), query.shape);
    return LinearModule({channels, channels, true}).build(ctx, x, weights.output);
}

core::TensorValue LatentSyncUnetAttention::build_feed_forward(core::ModuleBuildContext & ctx,
    const core::TensorValue & input, const modules::LinearWeights & geglu,
    const modules::LinearWeights & output) const {
    using namespace modules;
    const int64_t channels = input.shape.dims[2], inner = channels * 4;
    const auto packed = LinearModule({channels, inner * 2, true}).build(ctx, input, geglu);
    const auto hidden = SliceModule({2, 0, inner}).build(ctx, packed);
    const auto gate = GeluModule().build(ctx, SliceModule({2, inner, inner}).build(ctx, packed));
    return LinearModule({inner, channels, true}).build(ctx, MulModule().build(ctx, hidden, gate), output);
}

core::TensorValue LatentSyncUnetAttention::build_spatial(core::ModuleBuildContext & ctx,
    const core::TensorValue & image, const core::TensorValue & audio,
    const LatentSyncSpatialTransformerWeights & weights) const {
    using namespace modules;
    const int64_t batch = image.shape.dims[0], channels = image.shape.dims[1];
    const int64_t steps = image.shape.dims[2] * image.shape.dims[3];
    auto x = GroupNormModule({channels, 32, 1e-6F}).build(ctx, image, weights.input_norm);
    x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({batch, channels, steps}));
    x = TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
    // A 1x1 convolution is a per-token linear projection. Stay in the transformer's
    // token layout instead of materializing im2col and transposing its result.
    const auto projection_shape = core::TensorShape::from_dims({channels, channels});
    x = LinearModule({channels, channels, true}).build(ctx, x,
        {core::reshape_tensor(ctx, weights.input.weight, projection_shape), weights.input.bias});
    auto normalized = LayerNormModule({channels}).build(ctx, x, weights.norms[0]);
    x = AddModule().build(ctx, x, build_attention(ctx, normalized, normalized, weights.self_attention));
    normalized = LayerNormModule({channels}).build(ctx, x, weights.norms[1]);
    x = AddModule().build(ctx, x, build_attention(ctx, normalized, audio, weights.audio_attention));
    normalized = LayerNormModule({channels}).build(ctx, x, weights.norms[2]);
    x = AddModule().build(ctx, x, build_feed_forward(ctx, normalized, weights.geglu, weights.feed_forward_out));
    x = LinearModule({channels, channels, true}).build(ctx, x,
        {core::reshape_tensor(ctx, weights.output.weight, projection_shape), weights.output.bias});
    x = TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
    x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x), image.shape);
    return AddModule().build(ctx, image, x);
}

core::TensorValue LatentSyncUnetAttention::build_temporal(core::ModuleBuildContext & ctx,
    const core::TensorValue & image, int64_t frames,
    const LatentSyncTemporalTransformerWeights & weights) const {
    using namespace modules;
    const int64_t batch_frames = image.shape.dims[0], channels = image.shape.dims[1];
    const int64_t steps = image.shape.dims[2] * image.shape.dims[3];
    if (frames <= 0 || batch_frames % frames != 0 || frames > weights.position[0].shape.dims[1]) {
        throw std::runtime_error("Invalid LatentSync temporal attention frame count");
    }
    const int64_t batch = batch_frames / frames;
    auto x = GroupNormModule({channels, 32, 1e-6F}).build(ctx, image, weights.input_norm);
    x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({batch_frames, channels, steps}));
    x = TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
    x = LinearModule({channels, channels, true}).build(ctx, x, weights.input);
    for (size_t i = 0; i < weights.attention.size(); ++i) {
        auto normalized = LayerNormModule({channels}).build(ctx, x, weights.norms[i]);
        normalized = core::reshape_tensor(ctx, normalized, core::TensorShape::from_dims({batch, frames, steps, channels}));
        normalized = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, normalized);
        normalized = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, normalized),
            core::TensorShape::from_dims({batch * steps, frames, channels}));
        const auto position = SliceModule({1, 0, frames}).build(ctx, weights.position[i]);
        normalized = AddModule().build(ctx, normalized, RepeatModule({normalized.shape}).build(ctx, position));
        auto attended = build_attention(ctx, normalized, normalized, weights.attention[i]);
        attended = core::reshape_tensor(ctx, attended, core::TensorShape::from_dims({batch, steps, frames, channels}));
        attended = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, attended);
        attended = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, attended), x.shape);
        x = AddModule().build(ctx, x, attended);
    }
    const auto normalized = LayerNormModule({channels}).build(ctx, x, weights.feed_forward_norm);
    x = AddModule().build(ctx, x, build_feed_forward(ctx, normalized, weights.geglu, weights.feed_forward_out));
    x = LinearModule({channels, channels, true}).build(ctx, x, weights.output);
    x = TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
    x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x), image.shape);
    return AddModule().build(ctx, image, x);
}

}  // namespace engine::models::latentsync
