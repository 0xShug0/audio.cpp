#include "engine/community_models/minimax_music3/vocoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::minimax_music3 {
namespace {

namespace binding = engine::modules::binding;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct GgmlGallocrDeleter {
    void operator()(ggml_gallocr_t alloc) const noexcept {
        if (alloc != nullptr) {
            ggml_gallocr_free(alloc);
        }
    }
};

struct ResidualUnitWeights {
    modules::Snake1dWeights snake1;
    modules::Conv1dWeights conv1;
    modules::Snake1dWeights snake2;
    modules::Conv1dWeights conv2;
    int64_t dilation = 1;
};

struct UpsampleBlockWeights {
    modules::Snake1dWeights snake;
    modules::ConvTranspose1dWeights conv_t;
    std::vector<ResidualUnitWeights> residuals;
};

struct Music3VocoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::Conv1dWeights dec_in_proj;
    modules::Conv1dWeights conv_in;
    std::vector<UpsampleBlockWeights> blocks;
    modules::Snake1dWeights snake_out;
    modules::Conv1dWeights conv_out;
};

core::TensorValue snake_alpha_from_source(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & name,
    int64_t channels) {
    return store.make_f32(
        core::TensorShape::from_dims({channels}),
        source.require_f32(name, {1, channels, 1}));
}

Music3VocoderWeights load_vocoder_weights(
    const MiniMaxMusic3Assets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    const auto & config = assets.config.vocoder;
    const auto & source = *assets.vocoder_weights;
    Music3VocoderWeights out;
    out.store = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        "minimax_music3.vocoder.weights",
        weight_context_bytes);
    out.dec_in_proj = binding::conv1d_from_source(
        *out.store,
        source,
        "dec_in_proj",
        conv_safe_storage_type(source, "dec_in_proj", storage_type),
        config.decoder_input_dim,
        config.latent_channels / 2,
        1,
        true);
    out.conv_in = binding::weight_norm_conv1d_from_source(
        *out.store,
        source,
        "conv_in",
        storage_type,
        config.decoder_hidden_dim,
        config.decoder_input_dim,
        7,
        true);
    int64_t channels = config.decoder_hidden_dim;
    out.blocks.reserve(config.upsample_ratios.size());
    for (size_t block_index = 0; block_index < config.upsample_ratios.size(); ++block_index) {
        const int64_t stride = config.upsample_ratios[block_index];
        const int64_t out_channels = channels / 2;
        const int64_t kernel = stride * 2;
        const std::string prefix = "blocks." + std::to_string(block_index);
        UpsampleBlockWeights block;
        block.snake.alpha = snake_alpha_from_source(*out.store, source, prefix + ".snake1.alpha", channels);
        block.conv_t = binding::weight_norm_conv_transpose1d_from_source(
            *out.store,
            source,
            prefix + ".conv_t1",
            storage_type,
            channels,
            out_channels,
            kernel,
            true);
        const int64_t dilations[3] = {1, 3, 9};
        block.residuals.reserve(3);
        for (int unit = 0; unit < 3; ++unit) {
            const std::string unit_prefix = prefix + ".res_unit" + std::to_string(unit + 1);
            ResidualUnitWeights residual;
            residual.dilation = dilations[unit];
            residual.snake1.alpha = snake_alpha_from_source(*out.store, source, unit_prefix + ".snake1.alpha", out_channels);
            residual.conv1 = binding::weight_norm_conv1d_from_source(
                *out.store,
                source,
                unit_prefix + ".conv1",
                storage_type,
                out_channels,
                out_channels,
                7,
                true);
            residual.snake2.alpha = snake_alpha_from_source(*out.store, source, unit_prefix + ".snake2.alpha", out_channels);
            residual.conv2 = binding::weight_norm_conv1d_from_source(
                *out.store,
                source,
                unit_prefix + ".conv2",
                storage_type,
                out_channels,
                out_channels,
                1,
                true);
            block.residuals.push_back(std::move(residual));
        }
        out.blocks.push_back(std::move(block));
        channels = out_channels;
    }
    out.snake_out.alpha = snake_alpha_from_source(*out.store, source, "snake_out.alpha", channels);
    out.conv_out = binding::weight_norm_conv1d_from_source(
        *out.store,
        source,
        "conv_out",
        storage_type,
        1,
        channels,
        7,
        true);
    out.store->upload();
    assets.vocoder_weights->release_storage();
    return out;
}

core::TensorValue residual_unit(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ResidualUnitWeights & weights,
    int64_t channels) {
    auto x = modules::Snake1dModule({channels}).build(ctx, input, weights.snake1);
    x = modules::Conv1dModule({channels, channels, 7, 1, static_cast<int>(3 * weights.dilation), static_cast<int>(weights.dilation), true})
            .build(ctx, x, weights.conv1);
    x = modules::Snake1dModule({channels}).build(ctx, x, weights.snake2);
    x = modules::Conv1dModule({channels, channels, 1, 1, 0, 1, true}).build(ctx, x, weights.conv2);
    return core::wrap_tensor(ggml_add(ctx.ggml, input.tensor, x.tensor), input.shape, GGML_TYPE_F32);
}

}  // namespace

struct MiniMaxMusic3VocoderRuntime::Impl {
    Impl(
        std::shared_ptr<const MiniMaxMusic3Assets> input_assets,
        core::ExecutionContext & input_execution,
        size_t input_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : assets(std::move(input_assets)),
          execution(input_execution),
          graph_arena_bytes(input_graph_arena_bytes),
          weights(load_vocoder_weights(*assets, execution, weight_context_bytes, storage_type)) {}

    ~Impl() {
        release_runtime_graphs();
    }

    void release_runtime_graphs() {
        if (graph != nullptr) {
            core::release_backend_graph_resources(execution.backend(), graph);
        }
        graph = nullptr;
        input = {};
        output = nullptr;
        gallocr.reset();
        ggml.reset();
        plan.reset();
        latent_frames = 0;
    }

    void ensure_graph(int64_t frames) {
        if (graph != nullptr && latent_frames == frames) {
            return;
        }
        release_runtime_graphs();
        const auto & config = assets->config.vocoder;
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ggml.reset(ggml_init(params));
        if (ggml == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax Music 3 vocoder graph context");
        }
        core::ModuleBuildContext ctx{ggml.get(), "minimax_music3.vocoder", execution.backend_type()};
        input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, config.latent_channels / 2, frames}));
        ggml_set_input(input.tensor);
        auto x = modules::Conv1dModule({config.latent_channels / 2, config.decoder_input_dim, 1, 1, 0, 1, true})
                     .build(ctx, input, weights.dec_in_proj);
        x = modules::Conv1dModule({config.decoder_input_dim, config.decoder_hidden_dim, 7, 1, 3, 1, true})
                .build(ctx, x, weights.conv_in);
        int64_t channels = config.decoder_hidden_dim;
        for (size_t i = 0; i < weights.blocks.size(); ++i) {
            const auto & block = weights.blocks[i];
            const int64_t out_channels = channels / 2;
            const int64_t stride = config.upsample_ratios[i];
            const int64_t padding = stride / 2;
            x = modules::Snake1dModule({channels}).build(ctx, x, block.snake);
            modules::ConvTranspose1dConfig conv_t_config{
                channels,
                out_channels,
                stride * 2,
                static_cast<int>(stride),
                static_cast<int>(padding),
                1,
                true,
            };
            const bool lower_padding_as_crop =
                !modules::is_conv_transpose1d_col2im_fast_path_eligible(ctx, conv_t_config);
            if (lower_padding_as_crop) {
                conv_t_config.padding = 0;
            }
            x = modules::ConvTranspose1dModule(conv_t_config).build(ctx, x, block.conv_t);
            if (lower_padding_as_crop && padding > 0) {
                const int64_t cropped_frames = x.shape.dims[2] - 2 * padding;
                if (cropped_frames <= 0) {
                    throw std::runtime_error("MiniMax Music 3 vocoder ConvTranspose1d crop would produce empty output");
                }
                x = modules::SliceModule({2, padding, cropped_frames}).build(ctx, x);
            }
            for (const auto & residual : block.residuals) {
                x = residual_unit(ctx, x, residual, out_channels);
            }
            channels = out_channels;
        }
        x = modules::Snake1dModule({channels}).build(ctx, x, weights.snake_out);
        x = modules::Conv1dModule({channels, 1, 7, 1, 3, 1, true}).build(ctx, x, weights.conv_out);
        x = modules::TanhModule().build(ctx, x);
        output = x.tensor;
        graph = ggml_new_graph_custom(ggml.get(), 262144, false);
        ggml_build_forward_expand(graph, output);
        gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr.get(), graph) ||
            !ggml_gallocr_alloc_graph(gallocr.get(), graph)) {
            throw std::runtime_error("failed to allocate MiniMax Music 3 vocoder graph");
        }
        core::prepare_host_graph_plan(execution, graph, plan);
        latent_frames = frames;
    }

    runtime::AudioBuffer decode(const std::vector<float> & latents, int64_t frames) {
        const auto & config = assets->config.vocoder;
        if (static_cast<int64_t>(latents.size()) != config.latent_channels * frames) {
            throw std::runtime_error("MiniMax Music 3 vocoder latent shape mismatch");
        }
        ensure_graph(frames);
        core::write_tensor_f32(input, latents);
        if (core::compute_graph(execution, graph, plan, "minimax_music3.vocoder") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiniMax Music 3 vocoder graph compute failed");
        }
        auto decoded = core::read_tensor_f32(output);
        const int64_t samples_per_channel = static_cast<int64_t>(decoded.size()) / 2;
        std::vector<float> interleaved(static_cast<size_t>(samples_per_channel * 2));
        for (int64_t i = 0; i < samples_per_channel; ++i) {
            interleaved[static_cast<size_t>(2 * i)] = decoded[static_cast<size_t>(i)];
            interleaved[static_cast<size_t>(2 * i + 1)] = decoded[static_cast<size_t>(samples_per_channel + i)];
        }
        return runtime::AudioBuffer{config.sample_rate, 2, std::move(interleaved)};
    }

    std::shared_ptr<const MiniMaxMusic3Assets> assets;
    core::ExecutionContext & execution;
    size_t graph_arena_bytes = 0;
    Music3VocoderWeights weights;
    int64_t latent_frames = 0;
    std::unique_ptr<std::remove_pointer_t<ggml_context *>, GgmlContextDeleter> ggml;
    ggml_cgraph * graph = nullptr;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr;
    core::HostGraphPlan plan;
    core::TensorValue input;
    ggml_tensor * output = nullptr;
};

MiniMaxMusic3VocoderRuntime::MiniMaxMusic3VocoderRuntime(
    std::shared_ptr<const MiniMaxMusic3Assets> assets,
    core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          execution,
          graph_arena_bytes,
          weight_context_bytes,
          storage_type)) {}

MiniMaxMusic3VocoderRuntime::~MiniMaxMusic3VocoderRuntime() = default;

runtime::AudioBuffer MiniMaxMusic3VocoderRuntime::decode(const std::vector<float> & latents, int64_t latent_frames) {
    return impl_->decode(latents, latent_frames);
}

void MiniMaxMusic3VocoderRuntime::release_runtime_graphs() {
    if (impl_ != nullptr) {
        impl_->release_runtime_graphs();
    }
}

}  // namespace engine::models::minimax_music3
