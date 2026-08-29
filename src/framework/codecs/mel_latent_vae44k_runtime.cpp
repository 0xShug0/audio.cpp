#include "engine/framework/codecs/mel_latent_vae44k_runtime.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::codecs {
namespace {

namespace binding = engine::modules::binding;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct MelLatentVaeBlockWeights {
    modules::Conv1dWeights conv1;
    modules::Conv1dWeights conv2;
    std::optional<modules::Conv1dWeights> shortcut;
};

struct MelLatentVaeAttentionWeights {
    modules::Conv1dWeights qkv;
    modules::Conv1dWeights proj_out;
};

struct MelLatentVaeLevelWeights {
    std::vector<MelLatentVaeBlockWeights> blocks;
    std::optional<MelLatentVaeAttentionWeights> attention;
    std::optional<modules::Conv1dWeights> upsample;
};

struct MelLatentVaeWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue data_mean;
    core::TensorValue data_std;
    modules::Conv1dWeights conv_in;
    MelLatentVaeBlockWeights mid_block_1;
    MelLatentVaeAttentionWeights mid_attention;
    MelLatentVaeBlockWeights mid_block_2;
    std::vector<MelLatentVaeLevelWeights> levels;
    modules::Conv1dWeights conv_out;
};

std::string join_name(const MelLatentVae44kConfig & config, const std::string & name) {
    if (config.tensor_prefix.empty()) {
        return name;
    }
    return config.tensor_prefix + "." + name;
}

std::vector<float> fold_mpconv_weight(const std::vector<float> & weight, int64_t out_channels, int64_t in_channels, int64_t kernel) {
    if (static_cast<int64_t>(weight.size()) != out_channels * in_channels * kernel) {
        throw std::runtime_error("Mel latent VAE MPConv weight shape mismatch");
    }
    std::vector<float> folded(weight.size(), 0.0F);
    const int64_t inner = in_channels * kernel;
    const float inner_scale = 1.0F / std::sqrt(static_cast<float>(inner));
    const float norm_alpha = std::sqrt(1.0F / static_cast<float>(inner));
    for (int64_t out = 0; out < out_channels; ++out) {
        const size_t base = static_cast<size_t>(out * inner);
        double norm_sq = 0.0;
        for (int64_t index = 0; index < inner; ++index) {
            const float value = weight[base + static_cast<size_t>(index)];
            norm_sq += static_cast<double>(value) * static_cast<double>(value);
        }
        const float denom = 1.0e-4F + std::sqrt(static_cast<float>(norm_sq)) * norm_alpha;
        for (int64_t index = 0; index < inner; ++index) {
            folded[base + static_cast<size_t>(index)] = weight[base + static_cast<size_t>(index)] / denom * inner_scale;
        }
    }
    return folded;
}

modules::Conv1dWeights load_mpconv1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const MelLatentVae44kConfig & config,
    const std::string & name,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    assets::TensorStorageType storage_type,
    float gain = 1.0F) {
    const std::string weight_name = join_name(config, name + ".weight");
    if (config.folded_mpconv_weights) {
        auto weights = binding::conv1d_from_source(
            store,
            source,
            join_name(config, name),
            storage_type,
            out_channels,
            in_channels,
            kernel,
            false);
        if (gain != 1.0F) {
            throw std::runtime_error("Mel latent VAE folded conv_out gain must be baked into the weight source");
        }
        return weights;
    }
    auto values = source.require_f32(weight_name, {out_channels, in_channels, kernel});
    values = fold_mpconv_weight(values, out_channels, in_channels, kernel);
    if (gain != 1.0F) {
        for (float & value : values) {
            value *= gain;
        }
    }
    return {store.make_from_f32(core::TensorShape::from_dims({out_channels, in_channels, kernel}), storage_type, std::move(values)), std::nullopt};
}

MelLatentVaeBlockWeights load_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const MelLatentVae44kConfig & config,
    const std::string & prefix,
    int64_t in_channels,
    int64_t out_channels,
    assets::TensorStorageType storage_type) {
    MelLatentVaeBlockWeights weights;
    weights.conv1 = load_mpconv1d(
        store,
        source,
        config,
        prefix + ".conv1",
        out_channels,
        in_channels,
        config.mp_conv_kernel,
        storage_type);
    weights.conv2 = load_mpconv1d(
        store,
        source,
        config,
        prefix + ".conv2",
        out_channels,
        out_channels,
        config.mp_conv_kernel,
        storage_type);
    if (in_channels != out_channels) {
        weights.shortcut = load_mpconv1d(
            store,
            source,
            config,
            prefix + ".nin_shortcut",
            out_channels,
            in_channels,
            1,
            storage_type);
    }
    return weights;
}

MelLatentVaeAttentionWeights load_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const MelLatentVae44kConfig & config,
    const std::string & prefix,
    int64_t channels,
    assets::TensorStorageType storage_type) {
    return {
        load_mpconv1d(store, source, config, prefix + ".qkv", channels * 3, channels, 1, storage_type),
        load_mpconv1d(store, source, config, prefix + ".proj_out", channels, channels, 1, storage_type),
    };
}

void validate_config(const MelLatentVae44kConfig & config) {
    if (config.latent_channels <= 0 || config.mel_bins <= 0 || config.hidden_dim <= 0 ||
        config.levels != 3 || config.resblocks_per_level != 3 || config.attention_heads != 1) {
        throw std::runtime_error("Mel latent VAE 44k runtime currently expects the ControlFoley VAE_44k decoder shape");
    }
    if (config.mp_conv_kernel != 3 || config.upsample_level < 0 || config.upsample_level >= config.levels) {
        throw std::runtime_error("Mel latent VAE 44k config has invalid decoder convolution or upsample layout");
    }
    if (!(config.mp_norm_eps > 0.0F) || !(config.mp_silu_scale > 0.0F) || !(config.mp_residual_t > 0.0F)) {
        throw std::runtime_error("Mel latent VAE 44k magnitude-preserving config must be positive");
    }
}

MelLatentVaeWeights load_weights(
    const assets::TensorSource & source,
    const MelLatentVae44kConfig & config,
    ggml_backend_t backend,
    core::BackendType backend_type,
    const MelLatentVae44kRuntimeOptions & options) {
    MelLatentVaeWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "framework.mel_latent_vae44k.weights",
        options.weight_context_bytes);
    weights.data_mean = weights.store->load_f32_tensor(source, join_name(config, "data_mean"), {1, config.mel_bins, 1});
    weights.data_std = weights.store->load_f32_tensor(source, join_name(config, "data_std"), {1, config.mel_bins, 1});
    const int64_t top_channels = config.hidden_dim * 4;
    weights.conv_in = load_mpconv1d(
        *weights.store,
        source,
        config,
        "decoder.conv_in",
        top_channels,
        config.latent_channels,
        config.mp_conv_kernel,
        options.weight_storage_type);
    weights.mid_block_1 = load_block(
        *weights.store,
        source,
        config,
        "decoder.mid.block_1",
        top_channels,
        top_channels,
        options.weight_storage_type);
    weights.mid_attention = load_attention(
        *weights.store,
        source,
        config,
        "decoder.mid.attn_1",
        top_channels,
        options.weight_storage_type);
    weights.mid_block_2 = load_block(
        *weights.store,
        source,
        config,
        "decoder.mid.block_2",
        top_channels,
        top_channels,
        options.weight_storage_type);
    weights.levels.resize(static_cast<size_t>(config.levels));
    int64_t block_in = top_channels;
    for (int64_t level = config.levels - 1; level >= 0; --level) {
        const int64_t block_out = config.hidden_dim * (1ll << level);
        auto & level_weights = weights.levels[static_cast<size_t>(level)];
        level_weights.blocks.reserve(static_cast<size_t>(config.resblocks_per_level));
        for (int64_t block = 0; block < config.resblocks_per_level; ++block) {
            level_weights.blocks.push_back(load_block(
                *weights.store,
                source,
                config,
                "decoder.up." + std::to_string(level) + ".block." + std::to_string(block),
                block_in,
                block_out,
                options.weight_storage_type));
            block_in = block_out;
        }
        if (level == config.upsample_level) {
            level_weights.upsample = load_mpconv1d(
                *weights.store,
                source,
                config,
                "decoder.up." + std::to_string(level) + ".upsample.conv",
                block_in,
                block_in,
                config.mp_conv_kernel,
                options.weight_storage_type);
        }
    }
    const std::string gain_name = join_name(config, "decoder.learnable_gain");
    const float gain = source.has_tensor(gain_name) ? source.require_f32(gain_name)[0] + 1.0F : 1.0F;
    weights.conv_out = load_mpconv1d(
        *weights.store,
        source,
        config,
        "decoder.conv_out",
        config.mel_bins,
        config.hidden_dim,
        config.mp_conv_kernel,
        options.weight_storage_type,
        gain);
    weights.store->upload();
    return weights;
}

core::TensorValue mp_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int axis,
    float eps) {
    const auto x = core::ensure_backend_addressable_layout(ctx, input);
    auto squared = core::wrap_tensor(ggml_sqr(ctx.ggml, x.tensor), x.shape, GGML_TYPE_F32);
    auto sum = modules::ReduceSumModule({axis}).build(ctx, squared);
    const int64_t channels = input.shape.dims[static_cast<size_t>(axis)];
    const float norm_alpha = 1.0F / std::sqrt(static_cast<float>(channels));
    auto denom = core::wrap_tensor(
        ggml_scale_bias(ctx.ggml, ggml_sqrt(ctx.ggml, core::ensure_backend_addressable_layout(ctx, sum).tensor), norm_alpha, eps),
        sum.shape,
        GGML_TYPE_F32);
    auto denom_full = modules::RepeatModule({input.shape}).build(ctx, denom);
    return core::wrap_tensor(ggml_div(ctx.ggml, x.tensor, denom_full.tensor), input.shape, GGML_TYPE_F32);
}

core::TensorValue mp_silu(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const MelLatentVae44kConfig & config) {
    auto out = modules::SiluModule().build(ctx, input);
    return core::wrap_tensor(ggml_scale(ctx.ggml, out.tensor, config.mp_silu_scale), out.shape, GGML_TYPE_F32);
}

core::TensorValue mp_sum(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & original,
    const core::TensorValue & residual,
    float t) {
    const float a_scale = 1.0F - t;
    const float b_scale = t;
    const float out_scale = 1.0F / std::sqrt(a_scale * a_scale + b_scale * b_scale);
    auto a = core::wrap_tensor(ggml_scale(ctx.ggml, original.tensor, a_scale), original.shape, GGML_TYPE_F32);
    auto b = core::wrap_tensor(ggml_scale(ctx.ggml, residual.tensor, b_scale), residual.shape, GGML_TYPE_F32);
    auto sum = modules::AddModule().build(ctx, a, b);
    return core::wrap_tensor(ggml_scale(ctx.ggml, sum.tensor, out_scale), sum.shape, GGML_TYPE_F32);
}

core::TensorValue conv1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::Conv1dWeights & weights,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel) {
    return modules::Conv1dModule({in_channels, out_channels, kernel, 1, static_cast<int>(kernel / 2), 1, false})
        .build(ctx, input, weights);
}

core::TensorValue resblock(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const MelLatentVaeBlockWeights & weights,
    const MelLatentVae44kConfig & config,
    int64_t in_channels,
    int64_t out_channels) {
    auto x = mp_norm(ctx, input, 1, config.mp_norm_eps);
    auto h = mp_silu(ctx, x, config);
    h = conv1d(ctx, h, weights.conv1, in_channels, out_channels, config.mp_conv_kernel);
    h = mp_silu(ctx, h, config);
    h = conv1d(ctx, h, weights.conv2, out_channels, out_channels, config.mp_conv_kernel);
    if (in_channels != out_channels) {
        if (!weights.shortcut.has_value()) {
            throw std::runtime_error("Mel latent VAE resblock shortcut is missing");
        }
        x = conv1d(ctx, x, *weights.shortcut, in_channels, out_channels, 1);
    }
    return mp_sum(ctx, x, h, config.mp_residual_t);
}

core::TensorValue attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const MelLatentVaeAttentionWeights & weights,
    const MelLatentVae44kConfig & config,
    const MelLatentVae44kRuntimeOptions & options,
    int64_t channels) {
    auto y = conv1d(ctx, input, weights.qkv, channels, channels * 3, 1);
    y = core::reshape_tensor(ctx, y, core::TensorShape::from_dims({input.shape.dims[0], channels, 3, input.shape.dims[2]}));
    y = mp_norm(ctx, y, 1, config.mp_norm_eps);
    auto q = modules::SliceModule({2, 0, 1}).build(ctx, y);
    auto k = modules::SliceModule({2, 1, 1}).build(ctx, y);
    auto v = modules::SliceModule({2, 2, 1}).build(ctx, y);
    q = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, q),
        core::TensorShape::from_dims({input.shape.dims[0], channels, input.shape.dims[2]}));
    k = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, k),
        core::TensorShape::from_dims({input.shape.dims[0], channels, input.shape.dims[2]}));
    v = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, v),
        core::TensorShape::from_dims({input.shape.dims[0], channels, input.shape.dims[2]}));
    q = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, q);
    k = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, k);
    v = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, v);
    const auto attention_shape = core::TensorShape::from_dims({input.shape.dims[0], 1, input.shape.dims[2], channels});
    q = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, q), attention_shape);
    k = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, k), attention_shape);
    v = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, v), attention_shape);
    auto h = modules::ScaledDotProductAttentionModule({
        channels,
        modules::ScaledDotProductAttentionLowering::Explicit,
        options.attention_precision,
        modules::AttentionCausality::NonCausal,
    }).build(ctx, q, k, v);
    h = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, h),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[2], channels}));
    h = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, h);
    h = core::ensure_backend_addressable_layout(ctx, h);
    h = conv1d(ctx, h, weights.proj_out, channels, channels, 1);
    return mp_sum(ctx, input, h, config.mp_residual_t);
}

core::TensorValue build_decoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & latent,
    const MelLatentVaeWeights & weights,
    const MelLatentVae44kConfig & config,
    const MelLatentVae44kRuntimeOptions & options) {
    const int64_t top_channels = config.hidden_dim * 4;
    auto h = conv1d(ctx, latent, weights.conv_in, config.latent_channels, top_channels, config.mp_conv_kernel);
    h = resblock(ctx, h, weights.mid_block_1, config, top_channels, top_channels);
    h = attention(ctx, h, weights.mid_attention, config, options, top_channels);
    h = resblock(ctx, h, weights.mid_block_2, config, top_channels, top_channels);
    h = core::wrap_tensor(ggml_clamp(ctx.ggml, h.tensor, -config.clip_act, config.clip_act), h.shape, GGML_TYPE_F32);
    int64_t in_channels = top_channels;
    for (int64_t level = config.levels - 1; level >= 0; --level) {
        const int64_t out_channels = config.hidden_dim * (1ll << level);
        const auto & level_weights = weights.levels[static_cast<size_t>(level)];
        for (int64_t block = 0; block < config.resblocks_per_level; ++block) {
            h = resblock(ctx, h, level_weights.blocks[static_cast<size_t>(block)], config, in_channels, out_channels);
            h = core::wrap_tensor(ggml_clamp(ctx.ggml, h.tensor, -config.clip_act, config.clip_act), h.shape, GGML_TYPE_F32);
            in_channels = out_channels;
        }
        if (level == config.upsample_level) {
            if (!level_weights.upsample.has_value()) {
                throw std::runtime_error("Mel latent VAE upsample weights are missing");
            }
            h = modules::Interpolate1dModule({h.shape.dims[2] * 2, modules::Interpolate1dMode::Nearest}).build(ctx, h);
            h = conv1d(ctx, h, *level_weights.upsample, in_channels, in_channels, config.mp_conv_kernel);
        }
    }
    h = mp_silu(ctx, h, config);
    h = conv1d(ctx, h, weights.conv_out, config.hidden_dim, config.mel_bins, config.mp_conv_kernel);
    auto std_full = modules::RepeatModule({h.shape}).build(ctx, weights.data_std);
    auto mean_full = modules::RepeatModule({h.shape}).build(ctx, weights.data_mean);
    h = modules::MulModule().build(ctx, h, std_full);
    return modules::AddModule().build(ctx, h, mean_full);
}

}  // namespace

struct MelLatentVae44kRuntime::Impl {
    Impl(
        std::shared_ptr<const assets::TensorSource> input_source,
        core::ExecutionContext & input_execution,
        MelLatentVae44kConfig input_config,
        MelLatentVae44kRuntimeOptions input_options)
        : source(std::move(input_source)),
          execution(input_execution),
          backend(input_execution.backend()),
          backend_type(input_execution.backend_type()),
          config(std::move(input_config)),
          options(input_options) {
        if (source == nullptr) {
            throw std::runtime_error("Mel latent VAE 44k runtime requires a tensor source");
        }
        if (backend == nullptr) {
            throw std::runtime_error("Mel latent VAE 44k runtime requires an initialized backend");
        }
        validate_config(config);
        weights = std::make_shared<MelLatentVaeWeights>(
            load_weights(*source, config, backend, backend_type, options));
    }

    struct Graph {
        Graph(const Impl & owner, int64_t input_frames)
            : frames(input_frames),
              owner_backend(owner.backend) {
            ggml_init_params params{owner.options.graph_arena_bytes, nullptr, true};
            ctx.reset(ggml_init(params));
            if (ctx == nullptr) {
                throw std::runtime_error("Mel latent VAE 44k failed to create graph context");
            }
            core::ModuleBuildContext build{ctx.get(), "framework.mel_latent_vae44k", owner.backend_type};
            input = core::make_tensor(build, GGML_TYPE_F32, core::TensorShape::from_dims({1, owner.config.latent_channels, frames}));
            output = build_decoder(build, input, *owner.weights, owner.config, owner.options);
            output = core::ensure_backend_addressable_layout(build, output);
            graph = ggml_new_graph_custom(ctx.get(), 262144, false);
            ggml_set_output(output.tensor);
            ggml_build_forward_expand(graph, output.tensor);
            gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(owner.backend));
            if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
                throw std::runtime_error("Mel latent VAE 44k failed to allocate graph");
            }
        }

        ~Graph() {
            if (owner_backend != nullptr && graph != nullptr) {
                core::release_backend_graph_resources(owner_backend, graph);
                graph = nullptr;
            }
            if (gallocr != nullptr) {
                ggml_gallocr_free(gallocr);
                gallocr = nullptr;
            }
        }

        Graph(const Graph &) = delete;
        Graph & operator=(const Graph &) = delete;

        int64_t frames = 0;
        ggml_backend_t owner_backend = nullptr;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
        ggml_cgraph * graph = nullptr;
        ggml_gallocr_t gallocr = nullptr;
        core::TensorValue input;
        core::TensorValue output;
    };

    Graph & graph_for_frames(int64_t frames) {
        if (graph == nullptr || graph->frames != frames) {
            graph = std::make_unique<Graph>(*this, frames);
        }
        return *graph;
    }

    MelLatentVae44kMel decode(const std::vector<float> & latent, int64_t frames) {
        if (frames <= 0 || static_cast<int64_t>(latent.size()) != config.latent_channels * frames) {
            throw std::runtime_error("Mel latent VAE 44k latent shape mismatch");
        }
        auto & active_graph = graph_for_frames(frames);
        core::write_tensor_f32(active_graph.input, latent);
        core::set_backend_threads(execution.backend(), execution.config().threads);
        const ggml_status status = core::compute_backend_graph(execution.backend(), active_graph.graph, nullptr, "framework.mel_latent_vae44k");
        ggml_backend_synchronize(execution.backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Mel latent VAE 44k graph compute failed");
        }
        MelLatentVae44kMel mel;
        mel.bins = config.mel_bins;
        mel.frames = frames * 2;
        core::read_tensor_float_into(active_graph.output.tensor, mel.values);
        if (static_cast<int64_t>(mel.values.size()) != mel.bins * mel.frames) {
            throw std::runtime_error("Mel latent VAE 44k output shape mismatch");
        }
        return mel;
    }

    std::shared_ptr<const assets::TensorSource> source;
    core::ExecutionContext & execution;
    ggml_backend_t backend = nullptr;
    core::BackendType backend_type = core::BackendType::Cpu;
    MelLatentVae44kConfig config;
    MelLatentVae44kRuntimeOptions options;
    std::shared_ptr<MelLatentVaeWeights> weights;
    std::unique_ptr<Graph> graph;
};

MelLatentVae44kRuntime::MelLatentVae44kRuntime(
    std::shared_ptr<const assets::TensorSource> source,
    core::ExecutionContext & execution,
    MelLatentVae44kConfig config,
    MelLatentVae44kRuntimeOptions options)
    : impl_(std::make_unique<Impl>(std::move(source), execution, std::move(config), options)) {}

MelLatentVae44kRuntime::~MelLatentVae44kRuntime() = default;
MelLatentVae44kRuntime::MelLatentVae44kRuntime(MelLatentVae44kRuntime &&) noexcept = default;
MelLatentVae44kRuntime & MelLatentVae44kRuntime::operator=(MelLatentVae44kRuntime &&) noexcept = default;

MelLatentVae44kMel MelLatentVae44kRuntime::decode(const std::vector<float> & latent, int64_t frames) {
    return impl_->decode(latent, frames);
}

void MelLatentVae44kRuntime::release_runtime_graphs() {
    impl_->graph.reset();
}

}  // namespace engine::codecs
