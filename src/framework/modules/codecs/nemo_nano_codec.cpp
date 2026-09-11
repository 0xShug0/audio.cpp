#include "engine/framework/modules/codecs/nemo_nano_codec.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::modules {
namespace {

using Clock = std::chrono::steady_clock;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct CodecResidualBlockWeights {
    Snake1dWeights input_snake;
    Conv1dWeights input_conv;
    Snake1dWeights skip_snake;
    Conv1dWeights skip_conv;
};

struct CodecStageWeights {
    Snake1dWeights up_snake;
    std::vector<ConvTranspose1dWeights> upsample_groups;
    std::vector<CodecResidualBlockWeights> residuals;
};

struct CodecWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    Conv1dWeights pre_conv;
    std::vector<CodecStageWeights> stages;
    Snake1dWeights post_snake;
    Conv1dWeights post_conv;
};

std::vector<float> read_exact_f32_tensor(ggml_tensor * tensor, size_t count, const char * name) {
    auto out = core::read_tensor_f32(tensor);
    if (out.size() != count) {
        throw std::runtime_error(
            std::string(name) + " readback element count mismatch: expected " +
            std::to_string(count) + ", got " + std::to_string(out.size()));
    }
    return out;
}

void validate_config(const NemoNanoCodecConfig & config) {
    if (config.sample_rate <= 0 || config.input_dim <= 0 || config.base_channels <= 0 || config.audio_codebooks <= 0) {
        throw std::runtime_error("NeMo nano codec config requires positive sample_rate, input_dim, base_channels, and audio_codebooks");
    }
    if (config.upsample_rates.empty() || config.resblock_kernel_sizes.empty() || config.resblock_dilation_sizes.empty()) {
        throw std::runtime_error("NeMo nano codec config requires upsample and residual block settings");
    }
    if (config.fsq_num_levels.empty() || config.fsq_num_levels.size() != config.fsq_dim_base_index.size()) {
        throw std::runtime_error("NeMo nano codec config requires matching FSQ levels and base indices");
    }
    if (config.input_dim != config.audio_codebooks * static_cast<int64_t>(config.fsq_num_levels.size())) {
        throw std::runtime_error("NeMo nano codec input_dim must match audio_codebooks times FSQ dimensions per group");
    }
}

std::vector<float> fold_weight_norm(
    const std::vector<float> & g,
    const std::vector<float> & v,
    int64_t outer,
    int64_t inner,
    int64_t kernel) {
    std::vector<float> out(v.size());
    for (int64_t o = 0; o < outer; ++o) {
        double sum = 0.0;
        for (int64_t i = 0; i < inner; ++i) {
            for (int64_t k = 0; k < kernel; ++k) {
                const float value = v[static_cast<size_t>((o * inner + i) * kernel + k)];
                sum += static_cast<double>(value) * static_cast<double>(value);
            }
        }
        const float scale = g[static_cast<size_t>(o)] / static_cast<float>(std::sqrt(sum));
        for (int64_t i = 0; i < inner; ++i) {
            for (int64_t k = 0; k < kernel; ++k) {
                const size_t index = static_cast<size_t>((o * inner + i) * kernel + k);
                out[index] = v[index] * scale;
            }
        }
    }
    return out;
}

std::vector<std::vector<float>> split_grouped_transpose_conv1d_weight(
    const std::vector<float> & weight,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel) {
    if (static_cast<int64_t>(weight.size()) != in_channels * kernel) {
        throw std::runtime_error("NeMo nano codec grouped ConvTranspose1d folded weight shape mismatch");
    }
    const int64_t inputs_per_group = in_channels / out_channels;
    if (inputs_per_group <= 0 || inputs_per_group * out_channels != in_channels) {
        throw std::runtime_error("NeMo nano codec grouped ConvTranspose1d channel ratio is invalid");
    }
    std::vector<std::vector<float>> groups(static_cast<size_t>(out_channels));
    for (auto & group : groups) {
        group.resize(static_cast<size_t>(inputs_per_group * kernel), 0.0F);
    }
    for (int64_t group = 0; group < out_channels; ++group) {
        const int64_t input_start = group * inputs_per_group;
        for (int64_t input_offset = 0; input_offset < inputs_per_group; ++input_offset) {
            const int64_t in_channel = input_start + input_offset;
            for (int64_t tap = 0; tap < kernel; ++tap) {
                groups[static_cast<size_t>(group)][static_cast<size_t>(input_offset * kernel + tap)] =
                    weight[static_cast<size_t>(in_channel * kernel + tap)];
            }
        }
    }
    return groups;
}

std::vector<ConvTranspose1dWeights> load_weight_norm_grouped_convtranspose1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    bool use_bias) {
    const auto g = source.require_f32(prefix + ".parametrizations.weight.original0", {in_channels, 1, 1});
    const auto v = source.require_f32(prefix + ".parametrizations.weight.original1", {in_channels, 1, kernel_size});
    const int64_t inputs_per_group = in_channels / out_channels;
    if (inputs_per_group <= 0 || inputs_per_group * out_channels != in_channels) {
        throw std::runtime_error("NeMo nano codec grouped ConvTranspose1d channel ratio is invalid");
    }
    const auto folded = fold_weight_norm(g, v, in_channels, 1, kernel_size);
    const auto groups = split_grouped_transpose_conv1d_weight(folded, in_channels, out_channels, kernel_size);
    const auto bias = use_bias ? source.require_f32(prefix + ".bias", {out_channels}) : std::vector<float>{};
    std::vector<ConvTranspose1dWeights> weights;
    weights.reserve(static_cast<size_t>(out_channels));
    for (int64_t group = 0; group < out_channels; ++group) {
        ConvTranspose1dWeights item;
        item.weight = store.make_from_f32(
            core::TensorShape::from_dims({inputs_per_group, 1, kernel_size}),
            storage_type,
            groups[static_cast<size_t>(group)]);
        if (use_bias) {
            item.bias = store.make_f32(core::TensorShape::from_dims({1}), {bias[static_cast<size_t>(group)]});
        }
        weights.push_back(std::move(item));
    }
    return weights;
}

Snake1dWeights load_half_snake_alpha(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & name,
    int64_t channels) {
    const int64_t snake_channels = channels / 2;
    const auto values = source.require_f32(name, {1, snake_channels, 1});
    return {store.make_f32(core::TensorShape::from_dims({snake_channels}), values)};
}

CodecWeights load_codec_weights(
    const assets::TensorSource & source,
    const NemoNanoCodecConfig & config,
    ggml_backend_t backend,
    core::BackendType backend_type,
    const NemoNanoCodecRuntimeOptions & options) {
    CodecWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "framework.nemo_nano_codec.weights",
        options.weight_context_bytes);
    weights.pre_conv = binding::weight_norm_conv1d_from_source(
        *weights.store,
        source,
        "audio_decoder.pre_conv.conv",
        options.weight_storage_type,
        config.base_channels,
        config.input_dim,
        7,
        true);
    int64_t in_channels = config.base_channels;
    weights.stages.reserve(config.upsample_rates.size());
    for (size_t stage = 0; stage < config.upsample_rates.size(); ++stage) {
        const int64_t rate = config.upsample_rates[stage];
        const int64_t out_channels = in_channels / 2;
        CodecStageWeights stage_weights;
        stage_weights.up_snake = load_half_snake_alpha(
            *weights.store,
            source,
            "audio_decoder.activations." + std::to_string(stage) + ".activation.snake_act.alpha",
            in_channels);
        stage_weights.upsample_groups = load_weight_norm_grouped_convtranspose1d(
            *weights.store,
            source,
            "audio_decoder.up_sample_conv_layers." + std::to_string(stage) + ".conv",
            options.weight_storage_type,
            in_channels,
            out_channels,
            rate * 2,
            true);
        for (size_t kernel_index = 0; kernel_index < config.resblock_kernel_sizes.size(); ++kernel_index) {
            const int64_t kernel = config.resblock_kernel_sizes[kernel_index];
            for (size_t dilation_index = 0; dilation_index < config.resblock_dilation_sizes.size(); ++dilation_index) {
                const std::string prefix =
                    "audio_decoder.res_layers." + std::to_string(stage) +
                    ".res_blocks." + std::to_string(kernel_index) +
                    ".res_blocks." + std::to_string(dilation_index);
                CodecResidualBlockWeights block;
                block.input_snake = load_half_snake_alpha(
                    *weights.store,
                    source,
                    prefix + ".input_activation.activation.snake_act.alpha",
                    out_channels);
                block.input_conv = binding::weight_norm_conv1d_from_source(
                    *weights.store,
                    source,
                    prefix + ".input_conv.conv",
                    options.weight_storage_type,
                    out_channels,
                    out_channels,
                    kernel,
                    true);
                block.skip_snake = load_half_snake_alpha(
                    *weights.store,
                    source,
                    prefix + ".skip_activation.activation.snake_act.alpha",
                    out_channels);
                block.skip_conv = binding::weight_norm_conv1d_from_source(
                    *weights.store,
                    source,
                    prefix + ".skip_conv.conv",
                    options.weight_storage_type,
                    out_channels,
                    out_channels,
                    kernel,
                    true);
                stage_weights.residuals.push_back(std::move(block));
            }
        }
        weights.stages.push_back(std::move(stage_weights));
        in_channels = out_channels;
    }
    weights.post_snake = load_half_snake_alpha(
        *weights.store,
        source,
        "audio_decoder.post_activation.activation.snake_act.alpha",
        in_channels);
    weights.post_conv = binding::weight_norm_conv1d_from_source(
        *weights.store,
        source,
        "audio_decoder.post_conv.conv",
        options.weight_storage_type,
        1,
        in_channels,
        3,
        true);
    weights.store->upload();
    return weights;
}

core::TensorValue causal_grouped_convtranspose1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const std::vector<ConvTranspose1dWeights> & weights,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    int64_t stride) {
    const int64_t inputs_per_group = in_channels / out_channels;
    if (static_cast<int64_t>(weights.size()) != out_channels) {
        throw std::runtime_error("NeMo nano codec ConvTranspose1d group count mismatch");
    }
    core::TensorValue out;
    for (int64_t group = 0; group < out_channels; ++group) {
        const int64_t input_start = group * inputs_per_group;
        auto input_slice = SliceModule({1, input_start, inputs_per_group}).build(ctx, input);
        const auto & group_weights = weights[static_cast<size_t>(group)];
        auto group_out = ConvTranspose1dModule({
            inputs_per_group,
            1,
            kernel,
            static_cast<int>(stride),
            0,
            1,
            group_weights.bias.has_value(),
        }).build(ctx, input_slice, group_weights);
        out = out.valid() ? ConcatModule({1}).build(ctx, out, group_out) : group_out;
    }
    const int64_t trim_right = kernel - stride;
    if (trim_right <= 0) {
        return out;
    }
    return SliceModule({2, 0, out.shape.dims[2] - trim_right}).build(ctx, out);
}

core::TensorValue half_snake(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const Snake1dWeights & weights) {
    const int64_t snake_channels = input.shape.dims[1] / 2;
    auto left = SliceModule({1, 0, snake_channels}).build(ctx, input);
    auto right = SliceModule({1, snake_channels, input.shape.dims[1] - snake_channels}).build(ctx, input);
    left = Snake1dModule({snake_channels}).build(ctx, left, weights);
    right = LeakyReluModule({0.01F}).build(ctx, right);
    return ConcatModule({1}).build(ctx, left, right);
}

core::TensorValue codec_residual(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const CodecResidualBlockWeights & weights,
    int64_t channels,
    int64_t kernel,
    int64_t dilation) {
    auto x = half_snake(ctx, input, weights.input_snake);
    x = CausalConv1dModule({
        channels,
        channels,
        kernel,
        1,
        static_cast<int>(dilation),
        true,
        CausalConv1dPadMode::Constant,
        CausalConv1dPaddingMode::StrictCausal,
    }).build(ctx, x, weights.input_conv);
    x = half_snake(ctx, x, weights.skip_snake);
    x = CausalConv1dModule({
        channels,
        channels,
        kernel,
        1,
        1,
        true,
        CausalConv1dPadMode::Constant,
        CausalConv1dPaddingMode::StrictCausal,
    }).build(ctx, x, weights.skip_conv);
    return ResidualAddModule().build(ctx, input, x);
}

core::TensorValue codec_residual_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const CodecStageWeights & weights,
    const NemoNanoCodecConfig & config,
    int64_t channels) {
    core::TensorValue summed;
    size_t residual_index = 0;
    for (const int64_t kernel : config.resblock_kernel_sizes) {
        auto branch = input;
        for (const int64_t dilation : config.resblock_dilation_sizes) {
            branch = codec_residual(
                ctx,
                branch,
                weights.residuals[residual_index++],
                channels,
                kernel,
                dilation);
        }
        summed = summed.valid() ? AddModule().build(ctx, summed, branch) : branch;
    }
    return core::wrap_tensor(
        ggml_scale(
            ctx.ggml,
            summed.tensor,
            1.0F / static_cast<float>(config.resblock_kernel_sizes.size())),
        summed.shape,
        summed.type);
}

core::TensorValue build_codec_decoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const NemoNanoCodecConfig & config,
    const CodecWeights & weights) {
    auto x = CausalConv1dModule({
        config.input_dim,
        config.base_channels,
        7,
        1,
        1,
        true,
        CausalConv1dPadMode::Constant,
        CausalConv1dPaddingMode::StrictCausal,
    }).build(ctx, input, weights.pre_conv);
    int64_t channels = config.base_channels;
    for (size_t stage = 0; stage < weights.stages.size(); ++stage) {
        const int64_t rate = config.upsample_rates[stage];
        const int64_t out_channels = channels / 2;
        x = half_snake(ctx, x, weights.stages[stage].up_snake);
        x = causal_grouped_convtranspose1d(ctx, x, weights.stages[stage].upsample_groups, channels, out_channels, rate * 2, rate);
        x = codec_residual_layer(ctx, x, weights.stages[stage], config, out_channels);
        channels = out_channels;
    }
    x = half_snake(ctx, x, weights.post_snake);
    x = CausalConv1dModule({
        channels,
        1,
        3,
        1,
        1,
        true,
        CausalConv1dPadMode::Constant,
        CausalConv1dPaddingMode::StrictCausal,
    }).build(ctx, x, weights.post_conv);
    return core::wrap_tensor(ggml_clamp(ctx.ggml, x.tensor, -1.0F, 1.0F), x.shape, GGML_TYPE_F32);
}

}  // namespace

struct NemoNanoCodecRuntime::Impl {
    struct Graph {
        Graph(
            const Impl & owner,
            int64_t input_frames)
            : frames(input_frames),
              owner_backend(owner.backend) {
            ggml_init_params params{owner.options.graph_arena_bytes, nullptr, true};
            ctx.reset(ggml_init(params));
            if (ctx == nullptr) {
                throw std::runtime_error("NeMo nano codec failed to create graph context");
            }
            core::ModuleBuildContext build{ctx.get(), "framework.nemo_nano_codec", owner.backend_type};
            input = core::make_tensor(build, GGML_TYPE_F32, core::TensorShape::from_dims({1, owner.config.input_dim, frames}));
            output = build_codec_decoder(build, input, owner.config, *owner.weights);
            output = core::ensure_backend_addressable_layout(build, output);
            graph = ggml_new_graph_custom(ctx.get(), 262144, false);
            ggml_set_output(output.tensor);
            ggml_build_forward_expand(graph, output.tensor);
            gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(owner.backend));
            if (gallocr == nullptr) {
                throw std::runtime_error("NeMo nano codec failed to create graph allocator");
            }
            if (!ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
                throw std::runtime_error("NeMo nano codec failed to allocate graph");
            }
        }

        ~Graph() {
            if (owner_backend != nullptr && graph != nullptr) {
                core::release_backend_graph_resources(owner_backend, graph);
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

    Impl(
        std::shared_ptr<const assets::TensorSource> source,
        core::ExecutionContext & execution,
        NemoNanoCodecConfig input_config,
        NemoNanoCodecRuntimeOptions input_options)
        : config(std::move(input_config)),
          backend(execution.backend()),
          backend_type(execution.backend_type()),
          options(input_options) {
        validate_config(config);
        if (source == nullptr) {
            throw std::runtime_error("NeMo nano codec runtime requires tensor source");
        }
        weights = std::make_shared<CodecWeights>(
            load_codec_weights(*source, config, backend, backend_type, options));
    }

    std::vector<float> fsq_decode(const std::vector<int32_t> & codes) const {
        const int64_t frames = static_cast<int64_t>(codes.size()) / config.audio_codebooks;
        if (frames <= 0 || static_cast<int64_t>(codes.size()) != frames * config.audio_codebooks) {
            throw std::runtime_error("NeMo nano codec code shape is invalid");
        }
        const int64_t dims_per_group = static_cast<int64_t>(config.fsq_num_levels.size());
        std::vector<float> out(static_cast<size_t>(config.input_dim * frames), 0.0F);
        for (int64_t frame = 0; frame < frames; ++frame) {
            for (int64_t group = 0; group < config.audio_codebooks; ++group) {
                const int32_t index = codes[static_cast<size_t>(frame * config.audio_codebooks + group)];
                for (int64_t d = 0; d < dims_per_group; ++d) {
                    const int32_t base = config.fsq_dim_base_index[static_cast<size_t>(d)];
                    const int32_t levels = config.fsq_num_levels[static_cast<size_t>(d)];
                    const int32_t nonnegative = (index / base) % levels;
                    const int32_t scale = levels / 2;
                    const float value = static_cast<float>(nonnegative - scale) / static_cast<float>(scale);
                    const int64_t channel = group * dims_per_group + d;
                    out[static_cast<size_t>(channel * frames + frame)] = value;
                }
            }
        }
        return out;
    }

    Graph & graph_for_frames(int64_t frames) {
        if (graph == nullptr || graph->frames != frames) {
            graph = std::make_unique<Graph>(*this, frames);
        }
        return *graph;
    }

    runtime::AudioBuffer decode_codes(const std::vector<int32_t> & codes) {
        const int64_t frames = static_cast<int64_t>(codes.size()) / config.audio_codebooks;
        auto dequantized = fsq_decode(codes);
        auto & graph_ref = graph_for_frames(frames);
        ggml_backend_tensor_set(
            graph_ref.input.tensor,
            dequantized.data(),
            0,
            dequantized.size() * sizeof(float));
        const auto start = Clock::now();
        core::compute_backend_graph(backend, graph_ref.graph);
        debug::timing_log_scalar("nemo_nano_codec.graph.compute_ms", debug::elapsed_ms(start));
        runtime::AudioBuffer audio;
        audio.sample_rate = static_cast<int>(config.sample_rate);
        audio.channels = 1;
        audio.samples = read_exact_f32_tensor(
            graph_ref.output.tensor,
            static_cast<size_t>(graph_ref.output.shape.dims[2]),
            "NeMo nano codec output");
        return audio;
    }

    NemoNanoCodecConfig config;
    ggml_backend_t backend = nullptr;
    core::BackendType backend_type = core::BackendType::Cpu;
    NemoNanoCodecRuntimeOptions options;
    std::shared_ptr<CodecWeights> weights;
    std::unique_ptr<Graph> graph;
};

NemoNanoCodecRuntime::NemoNanoCodecRuntime(
    std::shared_ptr<const assets::TensorSource> source,
    core::ExecutionContext & execution,
    NemoNanoCodecConfig config,
    NemoNanoCodecRuntimeOptions options)
    : impl_(std::make_unique<Impl>(std::move(source), execution, std::move(config), options)) {}

NemoNanoCodecRuntime::~NemoNanoCodecRuntime() = default;

runtime::AudioBuffer NemoNanoCodecRuntime::decode_codes(const std::vector<int32_t> & codes) {
    return impl_->decode_codes(codes);
}

void NemoNanoCodecRuntime::release_runtime_graph() {
    impl_->graph.reset();
}

}  // namespace engine::modules
