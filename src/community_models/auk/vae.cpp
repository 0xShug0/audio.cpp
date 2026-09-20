#include "engine/community_models/auk/vae.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/runtime/graph_optimizer.h"

#include <ggml-alloc.h>
#include <algorithm>
#include <array>

namespace engine::models::auk {
namespace {

using core::TensorShape;
using core::TensorValue;

struct EncoderWeights {
    TensorValue mean;
    TensorValue variance;
    modules::Conv1dWeights pre;
    modules::Conv1dWeights post;
    std::array<modules::Conv1dWeights, 6> down;
    std::array<std::array<modules::Conv1dWeights, 12>, 6> residual;
};

EncoderWeights load_encoder(core::BackendWeightStore & store, const assets::TensorSource & source) {
    const auto conv = [&](const std::string & name, int64_t out, int64_t in, int kernel) {
        const std::string prefix = "audio_encoder.generator." + name;
        const auto weight_v = source.has_tensor(prefix + ".weight_v")
            ? prefix + ".weight_v" : prefix + ".parametrizations.weight.original1";
        const auto storage = assets::tensor_storage_type_for_dtype(source.require_metadata(weight_v).dtype);
        modules::Conv1dWeights result;
        result.weight = modules::binding::weight_norm_tensor_from_source(store, source, prefix,
            storage, TensorShape::from_dims({out, in, kernel}));
        result.bias = store.load_f32_tensor(source, prefix + ".bias", {out});
        return result;
    };
    EncoderWeights weights;
    weights.mean = store.load_f32_tensor(source, "global_mean", {64});
    weights.variance = store.load_f32_tensor(source, "global_log_std", {64});
    weights.pre = conv("0.layer", 12, 1, 3);
    constexpr int rates[] = {2, 2, 2, 3, 4, 5};
    for (int stage = 0; stage < 6; ++stage) {
        const int channels = 24 << stage;
        weights.down[stage] = conv(std::to_string(2 + stage * 3) + ".layer", channels, channels / 2, rates[stage] * 2);
        for (int layer = 0; layer < 6; ++layer) {
            const auto prefix = std::to_string(3 + stage * 3) + ".layers." + std::to_string(layer);
            weights.residual[stage][2 * layer] = conv(prefix + ".1", channels, channels, 3);
            weights.residual[stage][2 * layer + 1] = conv(prefix + ".3", channels, channels, 3);
        }
    }
    weights.post = conv("20.layer", 128, 768, 3);
    return weights;
}

TensorValue build_encoder(core::ModuleBuildContext & ctx, const TensorValue & audio, const EncoderWeights & weights) {
    auto x = modules::Conv1dModule({1, 12, 3, 1, 1, 1, true}).build(ctx, audio, weights.pre);
    x = modules::LeakyReluModule({0.2F}).build(ctx, x);
    constexpr int rates[] = {2, 2, 2, 3, 4, 5};
    for (int stage = 0; stage < 6; ++stage) {
        const int channels = 24 << stage;
        const int rate = rates[stage];
        x = modules::Conv1dModule({channels / 2, channels, rate * 2, rate, rate - 1, 1, true})
            .build(ctx, x, weights.down[stage]);
        for (int layer = 0; layer < 6; ++layer) {
            const int dilation = 1 << layer;
            auto hidden = modules::LeakyReluModule().build(ctx, x);
            hidden = modules::Conv1dModule({channels, channels, 3, 1, dilation, dilation, true})
                .build(ctx, hidden, weights.residual[stage][layer * 2]);
            hidden = modules::LeakyReluModule().build(ctx, hidden);
            hidden = modules::Conv1dModule({channels, channels, 3, 1, 1, 1, true})
                .build(ctx, hidden, weights.residual[stage][layer * 2 + 1]);
            x = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, hidden.tensor), x.shape);
        }
        x = modules::LeakyReluModule({0.2F}).build(ctx, x);
    }
    return modules::Conv1dModule({768, 128, 3, 1, 1, 1, true}).build(ctx, x, weights.post);
}

struct ActivationWeights {
    modules::SnakeBeta1dWeights snake;
    std::array<TensorValue, 2> up_filters;
    TensorValue down_filter;
};

struct ResidualWeights {
    std::array<modules::Conv1dWeights, 3> conv1;
    std::array<modules::Conv1dWeights, 3> conv2;
    std::array<ActivationWeights, 6> activations;
};

struct DecoderWeights {
    TensorValue mean;
    TensorValue variance;
    modules::Conv1dWeights pre;
    modules::Conv1dWeights post;
    std::array<modules::ConvTranspose1dWeights, 6> up;
    std::array<ResidualWeights, 18> residual;
    ActivationWeights activation_post;
};

ActivationWeights load_activation(core::BackendWeightStore & store,
                                  const assets::TensorSource & source,
                                  const std::string & prefix, int64_t channels) {
    ActivationWeights weights;
    weights.snake.alpha = store.load_f32_tensor(source, prefix + ".act.alpha", {channels});
    weights.snake.beta = store.load_f32_tensor(source, prefix + ".act.beta", {channels});
    const auto up = source.require_f32(prefix + ".upsample.filter", {1, 1, 12});
    const auto down = source.require_f32(prefix + ".downsample.lowpass.filter", {1, 1, 12});
    std::array<std::vector<float>, 2> expanded_up{
        std::vector<float>(channels * 6), std::vector<float>(channels * 6)};
    std::vector<float> expanded_down(channels * 12);
    for (int64_t channel = 0; channel < channels; ++channel) {
        for (int tap = 0; tap < 6; ++tap) {
            expanded_up[0][channel * 6 + tap] = up[11 - tap * 2];
            expanded_up[1][channel * 6 + tap] = up[10 - tap * 2];
        }
        std::copy(down.begin(), down.end(), expanded_down.begin() + channel * 12);
    }
    for (int phase = 0; phase < 2; ++phase) {
        weights.up_filters[phase] = store.make_f32(TensorShape::from_dims({channels, 1, 6}), expanded_up[phase]);
    }
    weights.down_filter = store.make_f32(TensorShape::from_dims({channels, 1, 12}), expanded_down);
    return weights;
}

DecoderWeights load_decoder(core::BackendWeightStore & store, const assets::TensorSource & source) {
    using namespace modules::binding;
    const auto conv = [&](const std::string & prefix, int64_t out, int64_t in, int kernel, bool bias = true) {
        const auto weight_v = source.has_tensor(prefix + ".weight_v")
            ? prefix + ".weight_v" : prefix + ".parametrizations.weight.original1";
        const auto storage = assets::tensor_storage_type_for_dtype(source.require_metadata(weight_v).dtype);
        modules::Conv1dWeights result;
        result.weight = weight_norm_tensor_from_source(store, source, prefix,
            storage, TensorShape::from_dims({out, in, kernel}));
        if (bias) {
            result.bias = store.load_f32_tensor(source, prefix + ".bias", {out});
        }
        return result;
    };
    DecoderWeights weights;
    weights.mean = store.load_f32_tensor(source, "global_mean", {64});
    weights.variance = store.load_f32_tensor(source, "global_log_std", {64});
    weights.pre = conv("conv_pre", 1536, 64, 7);
    weights.post = conv("conv_post", 1, 24, 7, false);
    constexpr int rates[] = {5, 4, 3, 2, 2, 2};
    constexpr int kernels[] = {3, 7, 11};
    for (int stage = 0; stage < 6; ++stage) {
        const int channels = 1536 >> (stage + 1);
        const auto prefix = "ups." + std::to_string(stage) + ".0";
        const auto weight_v = source.has_tensor(prefix + ".weight_v")
            ? prefix + ".weight_v" : prefix + ".parametrizations.weight.original1";
        weights.up[stage].weight = weight_norm_tensor_from_source(store, source, prefix,
            assets::tensor_storage_type_for_dtype(source.require_metadata(weight_v).dtype),
            TensorShape::from_dims({channels * 2, channels, rates[stage] * 2}));
        weights.up[stage].bias = store.load_f32_tensor(source, prefix + ".bias", {channels});
        for (int kernel = 0; kernel < 3; ++kernel) {
            auto & residual = weights.residual[stage * 3 + kernel];
            const auto name = "resblocks." + std::to_string(stage * 3 + kernel);
            for (int layer = 0; layer < 3; ++layer) {
                residual.conv1[layer] = conv(name + ".convs1." + std::to_string(layer), channels, channels, kernels[kernel]);
                residual.conv2[layer] = conv(name + ".convs2." + std::to_string(layer), channels, channels, kernels[kernel]);
            }
            for (int layer = 0; layer < 6; ++layer) {
                residual.activations[layer] = load_activation(store, source,
                    name + ".activations." + std::to_string(layer), channels);
            }
        }
    }
    weights.activation_post = load_activation(store, source, "activation_post", 24);
    return weights;
}

TensorValue build_activation(core::ModuleBuildContext & ctx, const TensorValue & input,
                             const ActivationWeights & weights) {
    const auto channels = input.shape.dims[1];
    const auto frames = input.shape.dims[2];
    // AuK pads both ends before upsampling, but only the left before downsampling.
    const auto replicate = [&](ggml_tensor * x, int64_t frame, int64_t count) {
        auto * edge = ggml_view_2d(ctx.ggml, x, 1, channels, x->nb[1], frame * x->nb[0]);
        return ggml_repeat(ctx.ggml, edge, ggml_new_tensor_2d(ctx.ggml, GGML_TYPE_F32, count, channels));
    };
    auto * x = core::ensure_backend_addressable_layout(ctx, input).tensor;
    auto * padded = ggml_concat(ctx.ggml, replicate(x, 0, 3), x, 0);
    padded = ggml_concat(ctx.ggml, padded, replicate(x, frames - 1, 3), 0);
    std::array<ggml_tensor *, 2> phases;
    // Split the stride-two transpose FIR into odd/even taps. The reference's
    // 15-sample crop selects odd taps first, then even taps one frame later.
    for (int phase = 0; phase < 2; ++phase) {
        const auto filtered = modules::DepthwiseConv1dModule({channels, 6, 1, 0, 1, false}).build(ctx,
            core::wrap_tensor(padded, TensorShape::from_dims({1, channels, frames + 6})),
            {weights.up_filters[phase], std::nullopt});
        auto * cropped = ggml_view_3d(ctx.ggml, filtered.tensor, frames, channels, 1,
            filtered.tensor->nb[1], filtered.tensor->nb[2], phase * sizeof(float));
        phases[phase] = ggml_reshape_3d(ctx.ggml, ggml_cont(ctx.ggml, cropped), 1, frames, channels);
    }
    auto * up = ggml_reshape_3d(ctx.ggml, ggml_concat(ctx.ggml, phases[0], phases[1], 0), frames * 2, channels, 1);
    auto activated = modules::SnakeBeta1dModule({channels, true}).build(ctx,
        core::wrap_tensor(ggml_scale(ctx.ggml, up, 2.0F), TensorShape::from_dims({1, channels, frames * 2})), weights.snake);
    auto * causal = ggml_concat(ctx.ggml, replicate(activated.tensor, 0, 11), activated.tensor, 0);
    return modules::DepthwiseConv1dModule({channels, 12, 2, 0, 1, false}).build(ctx,
        core::wrap_tensor(causal, TensorShape::from_dims({1, channels, frames * 2 + 11})), {weights.down_filter, std::nullopt});
}

TensorValue build_decoder(core::ModuleBuildContext & ctx, const TensorValue & latents, const DecoderWeights & weights) {
    auto * scaled = ggml_mul(ctx.ggml, latents.tensor, ggml_sqrt(ctx.ggml, weights.variance.tensor));
    auto * denormalized = ggml_add(ctx.ggml, scaled, weights.mean.tensor);
    const auto frames = latents.shape.dims[1];
    auto x = core::wrap_tensor(ggml_cont(ctx.ggml, ggml_transpose(ctx.ggml, denormalized)),
        TensorShape::from_dims({1, 64, frames}));
    x = modules::Conv1dModule({64, 1536, 7, 1, 3, 1, true}).build(ctx, x, weights.pre);
    constexpr int rates[] = {5, 4, 3, 2, 2, 2};
    constexpr int kernels[] = {3, 7, 11};
    constexpr int dilations[] = {1, 3, 5};
    for (int stage = 0; stage < 6; ++stage) {
        const int channels = 1536 >> (stage + 1);
        const auto length = x.shape.dims[2] * rates[stage];
        x = modules::ConvTranspose1dModule({channels * 2, channels, rates[stage] * 2, rates[stage], 0, 1, true})
            .build(ctx, x, weights.up[stage]);
        x = core::wrap_tensor(ggml_view_3d(ctx.ggml, x.tensor, length, channels, 1,
            x.tensor->nb[1], x.tensor->nb[2], 0), TensorShape::from_dims({1, channels, length}));
        TensorValue sum;
        for (int kernel = 0; kernel < 3; ++kernel) {
            auto residual = x;
            const auto & block = weights.residual[stage * 3 + kernel];
            for (int layer = 0; layer < 3; ++layer) {
                auto hidden = build_activation(ctx, residual, block.activations[layer * 2]);
                modules::CausalConv1dConfig config{channels, channels, kernels[kernel], 1, dilations[layer], true};
                config.padding_mode = modules::StreamingConv1dPaddingMode::StrictCausal;
                hidden = modules::CausalConv1dModule(config).build(ctx, hidden, block.conv1[layer]);
                hidden = build_activation(ctx, hidden, block.activations[layer * 2 + 1]);
                config.dilation = 1;
                hidden = modules::CausalConv1dModule(config).build(ctx, hidden, block.conv2[layer]);
                residual = core::wrap_tensor(ggml_add(ctx.ggml, hidden.tensor, residual.tensor), residual.shape);
            }
            sum = kernel == 0 ? residual : core::wrap_tensor(ggml_add(ctx.ggml, sum.tensor, residual.tensor), residual.shape);
        }
        x = core::wrap_tensor(ggml_scale(ctx.ggml, sum.tensor, 1.0F / 3.0F), sum.shape);
    }
    x = build_activation(ctx, x, weights.activation_post);
    modules::CausalConv1dConfig post{24, 1, 7, 1, 1, false};
    post.padding_mode = modules::StreamingConv1dPaddingMode::StrictCausal;
    x = modules::CausalConv1dModule(post).build(ctx, x, weights.post);
    return core::wrap_tensor(ggml_clamp(ctx.ggml, x.tensor, -1.0F, 1.0F), x.shape);
}

}  // namespace

struct EncoderGraph {
    core::ExecutionContext & execution;
    int64_t samples;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator{nullptr, ggml_gallocr_free};
    ggml_cgraph * graph = nullptr;
    TensorValue input;
    TensorValue noise;
    TensorValue statistics;
    TensorValue output;

    EncoderGraph(core::ExecutionContext & execution_, const EncoderWeights & weights, int64_t samples_)
        : execution(execution_), samples(samples_) {
        constexpr size_t nodes = 8192;
        context.reset(ggml_init({nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true}));
        if (!context) throw std::runtime_error("AuK VAE encoder context allocation failed");
        core::ModuleBuildContext ctx{context.get(), "auk.vae.encoder", execution.backend_type()};
        input = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 1, samples}));
        ggml_set_input(input.tensor);
        statistics = build_encoder(ctx, input, weights);
        ggml_set_output(statistics.tensor);
        const auto frames = statistics.shape.dims[2];
        noise = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 64, frames}));
        ggml_set_input(noise.tensor);
        auto * mean = ggml_view_2d(ctx.ggml, statistics.tensor, frames, 64, statistics.tensor->nb[1], 0);
        auto * log_std = ggml_view_2d(ctx.ggml, statistics.tensor, frames, 64, statistics.tensor->nb[1],
            64 * statistics.tensor->nb[1]);
        auto * sampled = ggml_add(ctx.ggml, mean, ggml_mul(ctx.ggml, noise.tensor, ggml_exp(ctx.ggml, log_std)));
        auto * btc = ggml_cont(ctx.ggml, ggml_transpose(ctx.ggml, sampled));
        auto * centered = ggml_sub(ctx.ggml, btc, weights.mean.tensor);
        output = core::wrap_tensor(ggml_div(ctx.ggml, centered, ggml_sqrt(ctx.ggml, weights.variance.tensor)),
            TensorShape::from_dims({1, frames, 64}));
        ggml_set_output(output.tensor);
        graph = ggml_new_graph_custom(context.get(), nodes, false);
        ggml_build_forward_expand(graph, output.tensor);
        runtime::optimize_graph(*graph, execution.backend_type() == core::BackendType::Cpu
            ? runtime::GraphOptimizationBackend::Cpu : runtime::GraphOptimizationBackend::Gpu);
        allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
            throw std::runtime_error("AuK VAE encoder buffer allocation failed");
        }
    }

    ~EncoderGraph() {
        core::release_backend_graph_resources(execution.backend(), graph, true);
    }
};

struct VaeEncoderRuntime::State {
    core::ExecutionContext & execution;
    core::BackendWeightStore store;
    EncoderWeights weights;
    std::unique_ptr<EncoderGraph> prepared;

    State(core::ExecutionContext & execution_, const assets::TensorSource & source)
        : execution(execution_),
          store(execution.backend(), execution.backend_type(), "auk.vae.encoder", 2 * 1024 * 1024),
          weights(load_encoder(store, source)) {
        store.upload();
    }
};

VaeEncoderRuntime::VaeEncoderRuntime(core::ExecutionContext & execution,
    const assets::TensorSource & source, int64_t samples)
    : state_(std::make_unique<State>(execution, source)) {
    prepare(samples);
}

VaeEncoderRuntime::~VaeEncoderRuntime() = default;

void VaeEncoderRuntime::prepare(int64_t samples) {
    if (samples <= 0) throw std::runtime_error("AuK VAE encoder sample count must be positive");
    auto & state = *state_;
    if (state.prepared && state.prepared->samples == samples) return;
    state.prepared.reset();
    state.prepared = std::make_unique<EncoderGraph>(state.execution, state.weights, samples);
}

int64_t VaeEncoderRuntime::frames() const {
    if (!state_->prepared) throw std::runtime_error("AuK VAE encoder graph is not prepared");
    return state_->prepared->output.shape.dims[1];
}

std::vector<float> VaeEncoderRuntime::encode(const std::vector<float> & audio,
    const std::vector<float> & noise, std::vector<float> * statistics) {
    if (!state_->prepared) throw std::runtime_error("AuK VAE encoder graph is not prepared");
    auto & state = *state_->prepared;
    if (audio.size() != static_cast<size_t>(state.samples)) {
        throw std::runtime_error("AuK VAE audio length differs from prepared graph");
    }
    if (noise.size() != static_cast<size_t>(frames() * 64)) {
        throw std::runtime_error("AuK VAE encoder noise length differs from prepared graph");
    }
    core::write_tensor_f32(state.input, audio);
    core::write_tensor_f32(state.noise, noise);
    if (core::compute_backend_graph(state.execution.backend(), state.graph, nullptr, "auk.vae.encoder") != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("AuK VAE encoder computation failed");
    }
    if (statistics) *statistics = core::read_tensor_f32(state.statistics.tensor);
    return core::read_tensor_f32(state.output.tensor);
}

struct DecoderGraph {
    core::ExecutionContext & execution;
    int64_t frames;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator{nullptr, ggml_gallocr_free};
    ggml_cgraph * graph = nullptr;
    TensorValue input;
    TensorValue output;

    DecoderGraph(core::ExecutionContext & execution_, const DecoderWeights & weights, int64_t frames_)
        : execution(execution_), frames(frames_) {
        if (frames <= 0) {
            throw std::runtime_error("AuK VAE frame count must be positive");
        }
        constexpr size_t nodes = 32768;
        context.reset(ggml_init({nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true}));
        if (!context) {
            throw std::runtime_error("AuK VAE graph context allocation failed");
        }
        core::ModuleBuildContext ctx{context.get(), "auk.vae", execution.backend_type()};
        input = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, frames, 64}));
        ggml_set_input(input.tensor);
        output = build_decoder(ctx, input, weights);
        ggml_set_output(output.tensor);
        graph = ggml_new_graph_custom(context.get(), nodes, false);
        ggml_build_forward_expand(graph, output.tensor);
        const auto optimization = runtime::optimize_graph(*graph,
            execution.backend_type() == core::BackendType::Cpu
                ? runtime::GraphOptimizationBackend::Cpu : runtime::GraphOptimizationBackend::Gpu);
        debug::trace_log_scalar("auk.vae.graph_nodes_before", optimization.nodes_before);
        debug::trace_log_scalar("auk.vae.graph_nodes_after", optimization.nodes_after);
        allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
            throw std::runtime_error("AuK VAE graph buffer allocation failed");
        }
    }

    ~DecoderGraph() {
        core::release_backend_graph_resources(execution.backend(), graph, true);
    }
};

struct VaeDecoderRuntime::State {
    core::ExecutionContext & execution;
    core::BackendWeightStore store;
    DecoderWeights weights;
    std::unique_ptr<DecoderGraph> prepared;

    State(core::ExecutionContext & execution_, const assets::TensorSource & source)
        : execution(execution_),
          store(execution.backend(), execution.backend_type(), "auk.vae", 2 * 1024 * 1024),
          weights(load_decoder(store, source)) {
        store.upload();
    }
};

VaeDecoderRuntime::VaeDecoderRuntime(core::ExecutionContext & execution,
    const assets::TensorSource & source, int64_t frames)
    : state_(std::make_unique<State>(execution, source)) {
    prepare(frames);
}

VaeDecoderRuntime::~VaeDecoderRuntime() = default;

void VaeDecoderRuntime::prepare(int64_t frames) {
    if (frames <= 0) throw std::runtime_error("AuK VAE frame count must be positive");
    auto & state = *state_;
    if (state.prepared && state.prepared->frames == frames) return;
    state.prepared.reset();
    state.prepared = std::make_unique<DecoderGraph>(state.execution, state.weights, frames);
}

std::vector<float> VaeDecoderRuntime::decode(const std::vector<float> & latents) {
    if (!state_->prepared) throw std::runtime_error("AuK VAE graph is not prepared");
    auto & state = *state_->prepared;
    if (latents.size() != static_cast<size_t>(state.frames * 64)) {
        throw std::runtime_error("AuK VAE input length differs from prepared graph");
    }
    core::write_tensor_f32(state.input, latents);
    if (core::compute_backend_graph(state.execution.backend(), state.graph, nullptr, "auk.vae") != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("AuK VAE graph computation failed");
    }
    auto output = core::read_tensor_f32(state.output.tensor);
    debug::trace_log_f32("auk.vae.output", {1, 1, state.frames * 480}, output);
    return output;
}

}  // namespace engine::models::auk
