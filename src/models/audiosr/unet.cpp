#include "engine/models/audiosr/unet.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::audiosr {
namespace {

namespace binding = engine::modules::binding;
namespace core = engine::core;
namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

constexpr int64_t kUnetInputBlocks = 12;
constexpr int64_t kUnetOutputBlocks = 12;
constexpr int64_t kUnetMiddleBlocks = 4;
constexpr int64_t kUnetGroups = 32;
constexpr float kUnetNormEps = 1.0e-6F;
constexpr int64_t kTimeEmbeddingDim = 128;
constexpr int64_t kTimeHiddenDim = 512;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct ResBlockWeights {
    modules::NormWeights in_norm;
    modules::Conv2dWeights in_conv;
    modules::LinearWeights emb;
    modules::NormWeights out_norm;
    modules::Conv2dWeights out_conv;
    std::optional<modules::Conv2dWeights> skip;
};

struct CrossAttentionWeights {
    modules::LinearWeights q;
    modules::LinearWeights k;
    modules::LinearWeights v;
    modules::LinearWeights out;
};

struct TransformerBlockWeights {
    modules::NormWeights norm1;
    CrossAttentionWeights attn1;
    modules::NormWeights norm2;
    CrossAttentionWeights attn2;
    modules::NormWeights norm3;
    modules::LinearWeights ff_in;
    modules::LinearWeights ff_out;
};

struct SpatialTransformerWeights {
    modules::NormWeights norm;
    modules::Conv2dWeights proj_in;
    TransformerBlockWeights block;
    modules::Conv2dWeights proj_out;
};

struct UnetBlockWeights {
    ResBlockWeights res;
    std::vector<SpatialTransformerWeights> transformers;
    std::optional<modules::Conv2dWeights> sample_conv;
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    bool is_downsample = false;
    bool is_upsample = false;
};

struct UnetWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::LinearWeights time0;
    modules::LinearWeights time2;
    modules::Conv2dWeights input_conv;
    UnetBlockWeights input_blocks[kUnetInputBlocks - 1];
    UnetBlockWeights middle_blocks[kUnetMiddleBlocks];
    UnetBlockWeights output_blocks[kUnetOutputBlocks];
    modules::NormWeights out_norm;
    modules::Conv2dWeights out_conv;
};

struct BlockSpec {
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t transformers = 0;
    bool sample = false;
    bool downsample_only = false;
};

modules::Conv2dWeights conv2d(
    core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType type,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel) {
    return binding::conv2d_from_source(store, source, prefix, type, out_channels, in_channels, kernel, kernel, true);
}

modules::LinearWeights linear(
    core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType type,
    int64_t out_features,
    int64_t in_features,
    bool bias) {
    return binding::linear_from_source(store, source, prefix, type, out_features, in_features, bias);
}

modules::NormWeights norm(
    core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels) {
    return binding::norm_from_source(store, source, prefix, channels);
}

ResBlockWeights load_resblock(
    core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType type,
    int64_t in_channels,
    int64_t out_channels) {
    ResBlockWeights out;
    out.in_norm = norm(store, source, prefix + ".in_layers.0", in_channels);
    out.in_conv = conv2d(store, source, prefix + ".in_layers.2", type, out_channels, in_channels, 3);
    out.emb = linear(store, source, prefix + ".emb_layers.1", type, out_channels, kTimeHiddenDim, true);
    out.out_norm = norm(store, source, prefix + ".out_layers.0", out_channels);
    out.out_conv = conv2d(store, source, prefix + ".out_layers.3", type, out_channels, out_channels, 3);
    if (in_channels != out_channels) {
        out.skip = conv2d(store, source, prefix + ".skip_connection", type, out_channels, in_channels, 1);
    }
    return out;
}

CrossAttentionWeights load_cross_attention(
    core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType type,
    int64_t channels) {
    CrossAttentionWeights out;
    out.q = linear(store, source, prefix + ".to_q", type, channels, channels, false);
    out.k = linear(store, source, prefix + ".to_k", type, channels, channels, false);
    out.v = linear(store, source, prefix + ".to_v", type, channels, channels, false);
    out.out = linear(store, source, prefix + ".to_out.0", type, channels, channels, true);
    return out;
}

SpatialTransformerWeights load_spatial_transformer(
    core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType type,
    int64_t channels) {
    SpatialTransformerWeights out;
    out.norm = norm(store, source, prefix + ".norm", channels);
    out.proj_in = conv2d(store, source, prefix + ".proj_in", type, channels, channels, 1);
    const std::string block = prefix + ".transformer_blocks.0";
    out.block.norm1 = norm(store, source, block + ".norm1", channels);
    out.block.attn1 = load_cross_attention(store, source, block + ".attn1", type, channels);
    out.block.norm2 = norm(store, source, block + ".norm2", channels);
    out.block.attn2 = load_cross_attention(store, source, block + ".attn2", type, channels);
    out.block.norm3 = norm(store, source, block + ".norm3", channels);
    out.block.ff_in = linear(store, source, block + ".ff.net.0.proj", type, channels * 8, channels, true);
    out.block.ff_out = linear(store, source, block + ".ff.net.2", type, channels, channels * 4, true);
    out.proj_out = conv2d(store, source, prefix + ".proj_out", type, channels, channels, 1);
    return out;
}

UnetBlockWeights load_block(
    core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType type,
    BlockSpec spec) {
    UnetBlockWeights out;
    out.in_channels = spec.in_channels;
    out.out_channels = spec.out_channels;
    if (spec.downsample_only) {
        out.sample_conv = conv2d(
            store,
            source,
            prefix + ".0.op",
            type,
            spec.out_channels,
            spec.in_channels,
            3);
        out.is_downsample = true;
        return out;
    }
    out.res = load_resblock(store, source, prefix + ".0", type, spec.in_channels, spec.out_channels);
    for (int64_t index = 0; index < spec.transformers; ++index) {
        out.transformers.push_back(load_spatial_transformer(
            store,
            source,
            prefix + "." + std::to_string(index + 1),
            type,
            spec.out_channels));
    }
    if (spec.sample) {
        const int64_t sample_index = 1 + spec.transformers;
        out.sample_conv = conv2d(
            store,
            source,
            prefix + "." + std::to_string(sample_index) + ".conv",
            type,
            spec.out_channels,
            spec.out_channels,
            3);
    }
    return out;
}

UnetWeights load_weights(
    const AudioSRAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    engine::assets::TensorStorageType type) {
    if (assets.weights == nullptr) {
        throw std::runtime_error("AudioSR UNet requires tensor source");
    }
    UnetWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "audiosr.unet.weights",
        4096ull * 1024ull * 1024ull);
    const auto & source = *assets.weights;
    const std::string prefix = "model.diffusion_model";
    weights.time0 = linear(*weights.store, source, prefix + ".time_embed.0", type, kTimeHiddenDim, kTimeEmbeddingDim, true);
    weights.time2 = linear(*weights.store, source, prefix + ".time_embed.2", type, kTimeHiddenDim, kTimeHiddenDim, true);
    weights.input_conv = conv2d(*weights.store, source, prefix + ".input_blocks.0.0", type, 128, 32, 3);
    const std::array<BlockSpec, kUnetInputBlocks - 1> input_specs{{
        {128, 128, 0, false},
        {128, 128, 0, false},
        {128, 128, 0, false, true},
        {128, 256, 2, false},
        {256, 256, 2, false},
        {256, 256, 0, false, true},
        {256, 384, 2, false},
        {384, 384, 2, false},
        {384, 384, 0, false, true},
        {384, 640, 2, false},
        {640, 640, 2, false},
    }};
    for (int64_t index = 1; index < kUnetInputBlocks; ++index) {
        weights.input_blocks[index - 1] = load_block(
            *weights.store,
            source,
            prefix + ".input_blocks." + std::to_string(index),
            type,
            input_specs[static_cast<size_t>(index - 1)]);
        weights.input_blocks[index - 1].is_downsample = input_specs[static_cast<size_t>(index - 1)].downsample_only;
    }
    weights.middle_blocks[0].in_channels = 640;
    weights.middle_blocks[0].out_channels = 640;
    weights.middle_blocks[0].res = load_resblock(*weights.store, source, prefix + ".middle_block.0", type, 640, 640);
    weights.middle_blocks[1].transformers.push_back(load_spatial_transformer(*weights.store, source, prefix + ".middle_block.1", type, 640));
    weights.middle_blocks[2].transformers.push_back(load_spatial_transformer(*weights.store, source, prefix + ".middle_block.2", type, 640));
    weights.middle_blocks[3].in_channels = 640;
    weights.middle_blocks[3].out_channels = 640;
    weights.middle_blocks[3].res = load_resblock(*weights.store, source, prefix + ".middle_block.3", type, 640, 640);
    const std::array<BlockSpec, kUnetOutputBlocks> output_specs{{
        {1280, 640, 2, false},
        {1280, 640, 2, false},
        {1024, 640, 2, true},
        {1024, 384, 2, false},
        {768, 384, 2, false},
        {640, 384, 2, true},
        {640, 256, 2, false},
        {512, 256, 2, false},
        {384, 256, 2, true},
        {384, 128, 0, false},
        {256, 128, 0, false},
        {256, 128, 0, false},
    }};
    for (int64_t index = 0; index < kUnetOutputBlocks; ++index) {
        weights.output_blocks[index] = load_block(
            *weights.store,
            source,
            prefix + ".output_blocks." + std::to_string(index),
            type,
            output_specs[static_cast<size_t>(index)]);
        weights.output_blocks[index].is_upsample = output_specs[static_cast<size_t>(index)].sample;
    }
    weights.out_norm = norm(*weights.store, source, prefix + ".out.0", 128);
    weights.out_conv = conv2d(*weights.store, source, prefix + ".out.2", type, 16, 128, 3);
    weights.store->upload();
    return weights;
}

std::vector<float> timestep_embedding(int64_t timestep) {
    std::vector<float> out(static_cast<size_t>(kTimeEmbeddingDim), 0.0F);
    const int64_t half = kTimeEmbeddingDim / 2;
    const float scale = std::log(10000.0F) / static_cast<float>(half);
    for (int64_t index = 0; index < half; ++index) {
        const float freq = std::exp(-scale * static_cast<float>(index));
        const float value = static_cast<float>(timestep) * freq;
        out[static_cast<size_t>(index)] = std::cos(value);
        out[static_cast<size_t>(half + index)] = std::sin(value);
    }
    return out;
}

core::TensorValue linear_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const CrossAttentionWeights & weights,
    int64_t channels,
    int64_t heads) {
    auto q = modules::LinearModule({channels, channels, false}).build(ctx, input, weights.q);
    auto k = modules::LinearModule({channels, channels, false}).build(ctx, input, weights.k);
    auto v = modules::LinearModule({channels, channels, false}).build(ctx, input, weights.v);
    const int64_t batch = input.shape.dims[0];
    const int64_t tokens = input.shape.dims[1];
    const int64_t head_dim = channels / heads;
    q = modules::ReshapeModule({core::TensorShape::from_dims({batch, tokens, heads, head_dim})}).build(ctx, q);
    k = modules::ReshapeModule({core::TensorShape::from_dims({batch, tokens, heads, head_dim})}).build(ctx, k);
    v = modules::ReshapeModule({core::TensorShape::from_dims({batch, tokens, heads, head_dim})}).build(ctx, v);
    q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
    k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
    v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
    q = core::ensure_backend_addressable_layout(ctx, q);
    k = core::ensure_backend_addressable_layout(ctx, k);
    v = core::ensure_backend_addressable_layout(ctx, v);
    q = modules::ReshapeModule({core::TensorShape::from_dims({batch * heads, tokens, head_dim})}).build(ctx, q);
    k = modules::ReshapeModule({core::TensorShape::from_dims({batch * heads, tokens, head_dim})}).build(ctx, k);
    v = modules::ReshapeModule({core::TensorShape::from_dims({batch * heads, tokens, head_dim})}).build(ctx, v);
    const auto k_t = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, k);
    auto scores = modules::MatMulModule().build(ctx, q, k_t);
    scores = core::wrap_tensor(
        ggml_scale(ctx.ggml, scores.tensor, 1.0F / std::sqrt(static_cast<float>(head_dim))),
        scores.shape,
        GGML_TYPE_F32);
    scores = modules::SoftmaxModule().build(ctx, scores);
    auto attn = modules::MatMulModule().build(ctx, scores, v);
    attn = core::ensure_backend_addressable_layout(ctx, attn);
    attn = modules::ReshapeModule({core::TensorShape::from_dims({batch, heads, tokens, head_dim})}).build(ctx, attn);
    attn = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, attn);
    attn = core::ensure_backend_addressable_layout(ctx, attn);
    attn = modules::ReshapeModule({core::TensorShape::from_dims({batch, tokens, channels})}).build(ctx, attn);
    return modules::LinearModule({channels, channels, true}).build(ctx, attn, weights.out);
}

core::TensorValue geglu(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::LinearWeights & weights,
    int64_t channels) {
    auto projected = modules::LinearModule({channels, channels * 8, true}).build(ctx, input, weights);
    auto x = modules::SliceModule({2, 0, channels * 4}).build(ctx, projected);
    auto gate = modules::SliceModule({2, channels * 4, channels * 4}).build(ctx, projected);
    gate = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, gate);
    return modules::MulModule().build(ctx, x, gate);
}

core::TensorValue group_norm_2d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::NormWeights & weights,
    int64_t channels,
    int64_t groups,
    float eps) {
    if (input.shape.rank != 4 || input.shape.dims[1] != channels) {
        throw std::runtime_error("AudioSR UNet GroupNorm2d input shape mismatch");
    }
    auto x = core::ensure_backend_addressable_layout(ctx, input);
    x = core::wrap_tensor(ggml_group_norm(ctx.ggml, x.tensor, groups, eps), x.shape, GGML_TYPE_F32);
    if (!weights.weight.has_value() || !weights.bias.has_value()) {
        throw std::runtime_error("AudioSR UNet GroupNorm2d requires affine weights");
    }
    auto weight = modules::ReshapeModule({core::TensorShape::from_dims({1, channels, 1, 1})}).build(ctx, *weights.weight);
    weight = modules::RepeatModule({x.shape}).build(ctx, weight);
    x = modules::MulModule().build(ctx, x, weight);
    auto bias = modules::ReshapeModule({core::TensorShape::from_dims({1, channels, 1, 1})}).build(ctx, *weights.bias);
    bias = modules::RepeatModule({x.shape}).build(ctx, bias);
    return modules::AddModule().build(ctx, x, bias);
}

core::TensorValue transformer_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const TransformerBlockWeights & weights,
    int64_t channels) {
    const int64_t heads = channels / 32;
    auto x = input;
    auto h = modules::LayerNormModule({channels, 1.0e-5F, true, true}).build(ctx, x, weights.norm1);
    h = linear_attention(ctx, h, weights.attn1, channels, heads);
    x = modules::AddModule().build(ctx, x, h);
    h = modules::LayerNormModule({channels, 1.0e-5F, true, true}).build(ctx, x, weights.norm2);
    h = linear_attention(ctx, h, weights.attn2, channels, heads);
    x = modules::AddModule().build(ctx, x, h);
    h = modules::LayerNormModule({channels, 1.0e-5F, true, true}).build(ctx, x, weights.norm3);
    h = geglu(ctx, h, weights.ff_in, channels);
    h = modules::LinearModule({channels * 4, channels, true}).build(ctx, h, weights.ff_out);
    return modules::AddModule().build(ctx, x, h);
}

core::TensorValue spatial_transformer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const SpatialTransformerWeights & weights,
    int64_t channels) {
    const int64_t batch = input.shape.dims[0];
    const int64_t height = input.shape.dims[2];
    const int64_t width = input.shape.dims[3];
    auto x = group_norm_2d(ctx, input, weights.norm, channels, kUnetGroups, kUnetNormEps);
    x = modules::Conv2dModule({channels, channels, 1, 1, 1, 1, 0, 0, 1, 1, true}).build(ctx, x, weights.proj_in);
    x = core::ensure_backend_addressable_layout(ctx, x);
    x = modules::ReshapeModule({core::TensorShape::from_dims({batch, channels, height * width})}).build(ctx, x);
    x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    x = core::ensure_backend_addressable_layout(ctx, x);
    x = transformer_block(ctx, x, weights.block, channels);
    x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    x = core::ensure_backend_addressable_layout(ctx, x);
    x = modules::ReshapeModule({core::TensorShape::from_dims({batch, channels, height, width})}).build(ctx, x);
    x = modules::Conv2dModule({channels, channels, 1, 1, 1, 1, 0, 0, 1, 1, true}).build(ctx, x, weights.proj_out);
    return modules::AddModule().build(ctx, input, x);
}

core::TensorValue resblock(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & embedding,
    const ResBlockWeights & weights,
    int64_t in_channels,
    int64_t out_channels) {
    auto h = group_norm_2d(ctx, input, weights.in_norm, in_channels, kUnetGroups, kUnetNormEps);
    h = modules::SiluModule().build(ctx, h);
    h = modules::Conv2dModule({in_channels, out_channels, 3, 3, 1, 1, 1, 1, 1, 1, true}).build(ctx, h, weights.in_conv);
    auto emb = modules::SiluModule().build(ctx, embedding);
    emb = modules::LinearModule({kTimeHiddenDim, out_channels, true}).build(ctx, emb, weights.emb);
    emb = modules::ReshapeModule({core::TensorShape::from_dims({1, out_channels, 1, 1})}).build(ctx, emb);
    emb = modules::RepeatModule({h.shape}).build(ctx, emb);
    h = modules::AddModule().build(ctx, h, emb);
    h = group_norm_2d(ctx, h, weights.out_norm, out_channels, kUnetGroups, kUnetNormEps);
    h = modules::SiluModule().build(ctx, h);
    h = modules::Conv2dModule({out_channels, out_channels, 3, 3, 1, 1, 1, 1, 1, 1, true}).build(ctx, h, weights.out_conv);
    auto residual = input;
    if (weights.skip.has_value()) {
        residual = modules::Conv2dModule({in_channels, out_channels, 1, 1, 1, 1, 0, 0, 1, 1, true}).build(ctx, input, *weights.skip);
    }
    return modules::AddModule().build(ctx, residual, h);
}

core::TensorValue run_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & embedding,
    const UnetBlockWeights & weights) {
    auto x = resblock(ctx, input, embedding, weights.res, weights.in_channels, weights.out_channels);
    for (const auto & transformer : weights.transformers) {
        x = spatial_transformer(ctx, x, transformer, weights.out_channels);
    }
    if (weights.is_upsample) {
        if (!weights.sample_conv.has_value()) {
            throw std::runtime_error("AudioSR UNet upsample block is missing sample convolution");
        }
        x = modules::NearestUpsample2dModule({x.shape.dims[2] * 2, x.shape.dims[3] * 2}).build(ctx, x);
        x = modules::Conv2dModule({weights.out_channels, weights.out_channels, 3, 3, 1, 1, 1, 1, 1, 1, true})
                .build(ctx, x, *weights.sample_conv);
    }
    return x;
}

core::TensorValue run_downsample_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & embedding,
    const UnetBlockWeights & weights) {
    if (!weights.is_downsample) {
        return run_block(ctx, input, embedding, weights);
    }
    (void)embedding;
    return modules::Conv2dModule({weights.in_channels, weights.out_channels, 3, 3, 2, 2, 1, 1, 1, 1, true})
        .build(ctx, input, *weights.sample_conv);
}

core::TensorValue build_unet(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & noisy,
    const core::TensorValue & condition,
    const core::TensorValue & timestep_embedding,
    const UnetWeights & weights,
    float condition_scale) {
    auto emb = modules::LinearModule({kTimeEmbeddingDim, kTimeHiddenDim, true}).build(ctx, timestep_embedding, weights.time0);
    emb = modules::SiluModule().build(ctx, emb);
    emb = modules::LinearModule({kTimeHiddenDim, kTimeHiddenDim, true}).build(ctx, emb, weights.time2);
    auto condition_scaled = core::wrap_tensor(
        ggml_scale(ctx.ggml, condition.tensor, condition_scale),
        condition.shape,
        condition.type);
    auto x = modules::ConcatModule({1}).build(ctx, noisy, condition_scaled);
    x = modules::Conv2dModule({32, 128, 3, 3, 1, 1, 1, 1, 1, 1, true}).build(ctx, x, weights.input_conv);
    std::vector<core::TensorValue> skips;
    skips.reserve(kUnetInputBlocks);
    skips.push_back(x);
    for (int64_t index = 1; index < kUnetInputBlocks; ++index) {
        const auto & block = weights.input_blocks[index - 1];
        x = block.is_downsample
            ? run_downsample_block(ctx, x, emb, block)
            : run_block(ctx, x, emb, block);
        skips.push_back(x);
    }
    x = run_block(ctx, x, emb, weights.middle_blocks[0]);
    x = spatial_transformer(ctx, x, weights.middle_blocks[1].transformers[0], 640);
    x = spatial_transformer(ctx, x, weights.middle_blocks[2].transformers[0], 640);
    x = run_block(ctx, x, emb, weights.middle_blocks[3]);
    for (int64_t index = 0; index < kUnetOutputBlocks; ++index) {
        if (skips.empty()) {
            throw std::runtime_error("AudioSR UNet skip stack underflow");
        }
        auto skip = skips.back();
        skips.pop_back();
        x = modules::ConcatModule({1}).build(ctx, x, skip);
        x = run_block(ctx, x, emb, weights.output_blocks[index]);
    }
    x = group_norm_2d(ctx, x, weights.out_norm, 128, kUnetGroups, kUnetNormEps);
    x = modules::SiluModule().build(ctx, x);
    x = modules::Conv2dModule({128, 16, 3, 3, 1, 1, 1, 1, 1, 1, true}).build(ctx, x, weights.out_conv);
    return x;
}

}  // namespace

struct AudioSRUnetRuntime::Impl {
    Impl(
        std::shared_ptr<const AudioSRAssets> assets,
        core::ExecutionContext & execution,
        engine::assets::TensorStorageType weight_type)
        : assets(std::move(assets)),
          execution(&execution) {
        if (this->assets == nullptr) {
            throw std::runtime_error("AudioSR UNet requires assets");
        }
        weights = load_weights(*this->assets, execution.backend(), execution.backend_type(), weight_type);
    }

    struct Graph {
        Graph(
            core::ExecutionContext & execution,
            std::shared_ptr<const AudioSRAssets> assets_in,
            const UnetWeights & weights,
            int64_t height,
            int64_t width)
            : execution(execution),
              assets(std::move(assets_in)),
              height(height),
              width(width) {
            if (this->assets == nullptr) {
                throw std::runtime_error("AudioSR UNet graph requires assets");
            }
            const auto build_start = Clock::now();
            ggml_init_params params{2048ull * 1024ull * 1024ull, nullptr, true};
            ctx.reset(ggml_init(params));
            ggml_init_params input_params{32ull * 1024ull * 1024ull, nullptr, true};
            input_ctx.reset(ggml_init(input_params));
            if (ctx == nullptr || input_ctx == nullptr) {
                throw std::runtime_error("failed to initialize AudioSR UNet graph context");
            }
            core::ModuleBuildContext graph_ctx{ctx.get(), "audiosr.unet", execution.backend_type()};
            core::ModuleBuildContext input_build_ctx{input_ctx.get(), "audiosr.unet.inputs", execution.backend_type()};
            noisy = core::make_tensor(input_build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, assets->config.latent_channels, height, width})).tensor;
            cond = core::make_tensor(input_build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, assets->config.latent_channels, height, width})).tensor;
            time = core::make_tensor(input_build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, kTimeEmbeddingDim})).tensor;
            ggml_set_input(noisy);
            ggml_set_input(cond);
            ggml_set_input(time);
            auto noisy_value = core::wrap_tensor(noisy, core::TensorShape::from_dims({1, assets->config.latent_channels, height, width}), GGML_TYPE_F32);
            auto cond_value = core::wrap_tensor(cond, core::TensorShape::from_dims({1, assets->config.latent_channels, height, width}), GGML_TYPE_F32);
            auto time_value = core::wrap_tensor(time, core::TensorShape::from_dims({1, kTimeEmbeddingDim}), GGML_TYPE_F32);
            auto x = build_unet(graph_ctx, noisy_value, cond_value, time_value, weights, assets->config.vae_scale_factor);
            output = core::ensure_backend_addressable_layout(graph_ctx, x).tensor;
            ggml_set_output(output);
            graph = ggml_new_graph_custom(ctx.get(), 262144, false);
            ggml_build_forward_expand(graph, output);
            input_buffer = ggml_backend_alloc_ctx_tensors(input_ctx.get(), execution.backend());
            gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
            if (input_buffer == nullptr || gallocr == nullptr ||
                !ggml_gallocr_reserve(gallocr, graph) ||
                !ggml_gallocr_alloc_graph(gallocr, graph)) {
                clear();
                throw std::runtime_error("failed to allocate AudioSR UNet graph");
            }
            engine::debug::timing_log_scalar(
                "audiosr.unet.graph.build_ms",
                engine::debug::elapsed_ms(build_start, Clock::now()));
        }

        ~Graph() {
            clear();
        }

        void clear() {
            if (graph != nullptr) {
                core::release_backend_graph_resources(execution.backend(), graph);
                graph = nullptr;
            }
            if (gallocr != nullptr) {
                ggml_gallocr_free(gallocr);
                gallocr = nullptr;
            }
            if (input_buffer != nullptr) {
                ggml_backend_buffer_free(input_buffer);
                input_buffer = nullptr;
            }
        }

        std::vector<float> run(const AudioSRLatent & latent, int64_t timestep, const AudioSRLatent & condition) {
            const int64_t count = latent.channels * latent.height * latent.width;
            if (latent.channels != assets->config.latent_channels ||
                latent.height != height ||
                latent.width != width ||
                condition.channels != latent.channels ||
                condition.height != height ||
                condition.width != width ||
                static_cast<int64_t>(latent.values.size()) != count ||
                static_cast<int64_t>(condition.values.size()) != count) {
                throw std::runtime_error("AudioSR UNet input shape mismatch");
            }
            auto time_values = timestep_embedding(timestep);
            ggml_backend_tensor_set(noisy, latent.values.data(), 0, latent.values.size() * sizeof(float));
            ggml_backend_tensor_set(cond, condition.values.data(), 0, condition.values.size() * sizeof(float));
            ggml_backend_tensor_set(time, time_values.data(), 0, time_values.size() * sizeof(float));
            core::set_backend_threads(execution.backend(), execution.config().threads);
            const auto start = Clock::now();
            const ggml_status status = core::compute_backend_graph(execution.backend(), graph);
            ggml_backend_synchronize(execution.backend());
            engine::debug::timing_log_scalar(
                "audiosr.unet.graph.compute_ms",
                engine::debug::elapsed_ms(start, Clock::now()));
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("AudioSR UNet graph compute failed");
            }
            std::vector<float> out;
            core::read_tensor_float_into(output, out);
            return out;
        }

        core::ExecutionContext & execution;
        std::shared_ptr<const AudioSRAssets> assets;
        int64_t height = 0;
        int64_t width = 0;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
        std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx;
        ggml_tensor * noisy = nullptr;
        ggml_tensor * cond = nullptr;
        ggml_tensor * time = nullptr;
        ggml_tensor * output = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_gallocr_t gallocr = nullptr;
        ggml_backend_buffer_t input_buffer = nullptr;
    };

    std::shared_ptr<const AudioSRAssets> assets;
    core::ExecutionContext * execution = nullptr;
    UnetWeights weights;
    std::unique_ptr<Graph> graph;
};

AudioSRUnetRuntime::AudioSRUnetRuntime(
    std::shared_ptr<const AudioSRAssets> assets,
    core::ExecutionContext & execution,
    engine::assets::TensorStorageType weight_type)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, weight_type)) {}

AudioSRUnetRuntime::~AudioSRUnetRuntime() = default;

std::vector<float> AudioSRUnetRuntime::predict_v(
    const AudioSRLatent & noisy,
    int64_t timestep,
    const AudioSRLatent & condition) {
    if (impl_->execution == nullptr) {
        throw std::runtime_error("AudioSR UNet execution context is missing");
    }
    if (impl_->graph == nullptr ||
        impl_->graph->height != noisy.height ||
        impl_->graph->width != noisy.width) {
        impl_->graph.reset();
        impl_->graph = std::make_unique<Impl::Graph>(
            *impl_->execution,
            impl_->assets,
            impl_->weights,
            noisy.height,
            noisy.width);
    }
    return impl_->graph->run(noisy, timestep, condition);
}

void AudioSRUnetRuntime::release_runtime_graphs() {
    impl_->graph.reset();
}

}  // namespace engine::models::audiosr
