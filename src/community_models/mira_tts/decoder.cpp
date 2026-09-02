#include "engine/community_models/mira_tts/decoder.h"

#include "engine/framework/audio/flashsr.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::community_models::mira_tts {
namespace {

struct ContextDeleter {
    void operator()(ggml_context * context) const noexcept {
        if (context != nullptr) ggml_free(context);
    }
};

struct SnakeWeights { core::TensorValue alpha; };
struct ConvWeights {
    modules::Conv1dWeights value;
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel = 0;
};
struct UpWeights {
    modules::ConvTranspose1dWeights value;
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel = 0;
};
struct ResidualWeights {
    SnakeWeights snake1;
    ConvWeights conv1;
    SnakeWeights snake2;
    ConvWeights conv2;
};
struct BlockWeights {
    SnakeWeights snake;
    UpWeights up;
    std::vector<ResidualWeights> residuals;
    int stride = 1;
};
struct Weights {
    std::shared_ptr<core::BackendWeightStore> store;
    ConvWeights first;
    std::vector<BlockWeights> blocks;
    SnakeWeights final_snake;
    ConvWeights final_conv;
};

ConvWeights load_conv(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    assets::TensorStorageType storage_type) {
    ConvWeights out;
    out.in_channels = in_channels;
    out.out_channels = out_channels;
    out.kernel = kernel;
    out.value.weight = store.load_tensor(
        source, prefix + ".weight", storage_type,
        {out_channels, in_channels, kernel});
    out.value.bias = store.load_f32_tensor(
        source, prefix + ".bias", {out_channels});
    return out;
}

UpWeights load_up(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    assets::TensorStorageType storage_type) {
    UpWeights out;
    out.in_channels = in_channels;
    out.out_channels = out_channels;
    out.kernel = kernel;
    out.value.weight = store.load_tensor(
        source, prefix + ".weight", storage_type,
        {in_channels, out_channels, kernel});
    out.value.bias = store.load_f32_tensor(
        source, prefix + ".bias", {out_channels});
    return out;
}

SnakeWeights load_snake(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & name,
    int64_t channels) {
    return {store.make_from_f32(
        core::TensorShape::from_dims({channels}),
        assets::TensorStorageType::F32,
        source.require_f32(name, {1, channels, 1}))};
}

ResidualWeights load_residual(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels,
    assets::TensorStorageType storage_type) {
    ResidualWeights out;
    out.snake1 = load_snake(store, source, prefix + ".block.0.alpha", channels);
    out.conv1 = load_conv(store, source, prefix + ".block.1", channels, channels, 7, storage_type);
    out.snake2 = load_snake(store, source, prefix + ".block.2.alpha", channels);
    out.conv2 = load_conv(store, source, prefix + ".block.3", channels, channels, 1, storage_type);
    return out;
}

Weights load_weights(
    const MiraTTSAssets & assets,
    core::ExecutionContext & execution,
    size_t context_bytes,
    assets::TensorStorageType storage_type) {
    Weights out;
    out.store = std::make_shared<core::BackendWeightStore>(
        execution.backend(), execution.backend_type(),
        "mira_tts.decoder.weights", context_bytes);
    const auto & source = *assets.decoder_weights;
    out.first = load_conv(*out.store, source, "model.0", 1536, 1024, 7, storage_type);
    const int strides[] = {8, 5, 4, 2};
    const int kernels[] = {16, 11, 8, 4};
    int64_t channels = 1536;
    for (int stage = 0; stage < 4; ++stage) {
        const int64_t out_channels = channels / 2;
        const std::string prefix = "model." + std::to_string(stage + 1) + ".block";
        BlockWeights block;
        block.stride = strides[stage];
        block.snake = load_snake(*out.store, source, prefix + ".0.alpha", channels);
        block.up = load_up(
            *out.store, source, prefix + ".1", channels, out_channels,
            kernels[stage], storage_type);
        for (int residual = 0; residual < 3; ++residual) {
            block.residuals.push_back(load_residual(
                *out.store, source,
                prefix + "." + std::to_string(residual + 2),
                out_channels, storage_type));
        }
        out.blocks.push_back(std::move(block));
        channels = out_channels;
    }
    out.final_snake = load_snake(*out.store, source, "model.5.alpha", 96);
    out.final_conv = load_conv(*out.store, source, "model.6", 1, 96, 7, storage_type);
    out.store->upload();
    return out;
}

core::TensorValue conv(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ConvWeights & weights,
    int padding,
    int dilation = 1) {
    return modules::Conv1dModule({
        weights.in_channels, weights.out_channels, weights.kernel,
        1, padding, dilation, true}).build(ctx, input, weights.value);
}

core::TensorValue snake(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const SnakeWeights & weights) {
    return modules::Snake1dModule({input.shape.dims[1]}).build(
        ctx, input, {weights.alpha});
}

core::TensorValue residual(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ResidualWeights & weights,
    int dilation) {
    auto x = snake(ctx, input, weights.snake1);
    x = conv(ctx, x, weights.conv1, 3 * dilation, dilation);
    x = snake(ctx, x, weights.snake2);
    x = conv(ctx, x, weights.conv2, 0);
    return modules::AddModule{}.build(ctx, input, x);
}

}  // namespace

struct MiraDecoder::Impl {
    Impl(
        const MiraTTSAssets & assets,
        core::ExecutionContext & execution_in,
        size_t weight_bytes,
        size_t graph_bytes,
        assets::TensorStorageType storage_type)
        : execution(execution_in),
          graph_context_bytes(graph_bytes),
          weights(load_weights(assets, execution_in, weight_bytes, storage_type)),
          upsampler(audio::FlashSrModel::load_from_tensor_source(
              assets.upsampler_weights, execution_in.config())) {}

    runtime::AudioBuffer decode(const std::vector<float> & latents, int64_t frames) {
        if (frames <= 0 || latents.size() != static_cast<size_t>(frames * 1024)) {
            throw std::runtime_error("MiraTTS decoder expects [1024, frames] latents");
        }
        ggml_init_params params{graph_context_bytes, nullptr, true};
        std::unique_ptr<ggml_context, ContextDeleter> ctx(ggml_init(params));
        if (!ctx) throw std::runtime_error("failed to create MiraTTS decoder graph context");
        core::ModuleBuildContext build{ctx.get(), "mira_tts.decoder"};
        auto * input_tensor = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, frames, 1024, 1);
        ggml_set_input(input_tensor);
        auto x = core::wrap_tensor(
            input_tensor, core::TensorShape::from_dims({1, 1024, frames}), GGML_TYPE_F32);
        x = conv(build, x, weights.first, 3);
        for (const auto & block : weights.blocks) {
            x = snake(build, x, block.snake);
            auto upsampled = modules::ConvTranspose1dModule({
                block.up.in_channels, block.up.out_channels, block.up.kernel,
                block.stride, 0, 1, true}).build(build, x, block.up.value);
            const int padding = static_cast<int>(std::ceil(block.stride / 2.0));
            x = modules::SliceModule({2, padding, upsampled.shape.dims[2] - 2 * padding})
                    .build(build, upsampled);
            x = residual(build, x, block.residuals[0], 1);
            x = residual(build, x, block.residuals[1], 3);
            x = residual(build, x, block.residuals[2], 9);
        }
        x = snake(build, x, weights.final_snake);
        x = conv(build, x, weights.final_conv, 3);
        x = modules::TanhModule{}.build(build, x);
        x = core::ensure_backend_addressable_layout(build, x);
        ggml_set_output(x.tensor);
        auto * graph = ggml_new_graph_custom(ctx.get(), 65536, false);
        ggml_build_forward_expand(graph, x.tensor);
        ggml_gallocr_t allocator = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(execution.backend()));
        if (allocator == nullptr || !ggml_gallocr_alloc_graph(allocator, graph)) {
            if (allocator != nullptr) ggml_gallocr_free(allocator);
            throw std::runtime_error("failed to allocate MiraTTS decoder graph");
        }
        ggml_backend_tensor_set(
            input_tensor, latents.data(), 0, latents.size() * sizeof(float));
        core::set_backend_threads(execution.backend(), std::max(1, execution.config().threads));
        const auto status = core::compute_backend_graph(execution.backend(), graph);
        ggml_backend_synchronize(execution.backend());
        if (status != GGML_STATUS_SUCCESS) {
            core::release_backend_graph_resources(execution.backend(), graph);
            ggml_gallocr_free(allocator);
            throw std::runtime_error("MiraTTS decoder graph compute failed");
        }
        std::vector<float> decoded(static_cast<size_t>(x.shape.dims[2]));
        ggml_backend_tensor_get(x.tensor, decoded.data(), 0, decoded.size() * sizeof(float));
        core::release_backend_graph_resources(execution.backend(), graph);
        ggml_gallocr_free(allocator);

        const auto enhanced = upsampler.super_resolve_mono_16k(decoded);
        runtime::AudioBuffer audio;
        audio.samples = enhanced.samples;
        audio.sample_rate = enhanced.sample_rate;
        audio.channels = 1;
        return audio;
    }

    core::ExecutionContext & execution;
    size_t graph_context_bytes;
    Weights weights;
    audio::FlashSrModel upsampler;
};

MiraDecoder::MiraDecoder(
    const MiraTTSAssets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    size_t graph_context_bytes,
    assets::TensorStorageType storage_type)
    : impl_(std::make_unique<Impl>(
          assets, execution, weight_context_bytes, graph_context_bytes, storage_type)) {}

MiraDecoder::~MiraDecoder() = default;

runtime::AudioBuffer MiraDecoder::decode(
    const std::vector<float> & latents,
    int64_t frames) {
    return impl_->decode(latents, frames);
}

}  // namespace engine::community_models::mira_tts
