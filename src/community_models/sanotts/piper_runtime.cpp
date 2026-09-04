#include "engine/community_models/sanotts/piper_runtime.h"

#include "graph_common.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/runtime/cache_slots.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::sanotts {
namespace {

namespace core = engine::core;
namespace modules = engine::modules;
using namespace engine::models::sanotts::graph;   // shared graph helpers

constexpr size_t kIoArenaBytes = 8ULL * 1024ULL * 1024ULL;
constexpr size_t kGraphArenaBytes = 128ULL * 1024ULL * 1024ULL;
constexpr size_t kWeightArenaBytes = 32ULL * 1024ULL * 1024ULL;
constexpr float kLeakyReluSlope = 0.1F;

// PiperResidualBank geometry, fixed by the training code: branch b uses
// kernel kBankKernels[b] with dilation kBankDilations1[b] on its first conv
// and kBankDilations2[b] on its second.
constexpr int64_t kBankKernels[3] = {3, 5, 7};
constexpr int kBankDilations1[3] = {1, 2, 3};
constexpr int kBankDilations2[3] = {2, 6, 12};

// The three ConvTranspose1d stages: stride and PyTorch padding. Kernel sizes
// are read from the weights themselves.
constexpr int kUpStrides[3] = {8, 8, 4};
constexpr int64_t kUpPaddings[3] = {4, 4, 2};

/** Ids outside a component's trained vocab remap to schwa -- the same
 *  fallback the reference runtimes use for the shared-frontend/table
 *  vocab-size mismatch (e.g. duration vocab 127 vs table ids to 156). */
constexpr int32_t kSchwaFallbackId = 59;

size_t expected_piper_tensor_count(const SanoTtsPiperConfig & config) {
    const auto duration = 5 + 5 * config.duration_depth;
    const auto acoustic =
        7 + 5 * (config.acoustic_token_depth + config.acoustic_depth);
    int64_t decoder = 4;   // pre + post
    for (const auto & branches : config.stage_branches) {
        decoder += 2 + 4 * static_cast<int64_t>(branches.size());
    }
    if (config.post_filter_channels > 0) {
        decoder += 4 + 5 * config.post_filter_layers;
    }
    return static_cast<size_t>(duration + acoustic + decoder);
}

std::vector<int32_t> clamp_ids_to_vocab(
    const std::vector<int32_t> & ids,
    int64_t vocab_size) {
    const int32_t fallback =
        kSchwaFallbackId < vocab_size ? kSchwaFallbackId : 0;
    std::vector<int32_t> out(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        const int32_t id = ids[i];
        out[i] = (id < 0 || id >= vocab_size) ? fallback : id;
    }
    return out;
}

int64_t kernel_of(const SanoTtsBackendWeights & weights, const std::string & prefix) {
    return weight(weights, prefix + ".weight").shape.dims[2];
}

core::TensorValue leaky_relu(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    float slope) {
    const auto source = contiguous(ctx, input);
    return core::wrap_tensor(
        ggml_leaky_relu(ctx.ggml, source.tensor, slope, false),
        input.shape,
        GGML_TYPE_F32);
}

core::TensorValue scaled(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value,
    float scale) {
    const auto source = contiguous(ctx, value);
    return core::wrap_tensor(
        ggml_scale(ctx.ggml, source.tensor, scale),
        value.shape,
        GGML_TYPE_F32);
}

/** 'same'-padded conv: pad = dilation * (kernel / 2). */
core::TensorValue conv1d_same(
    core::ModuleBuildContext & ctx,
    const SanoTtsBackendWeights & weights,
    const core::TensorValue & input,
    const std::string & prefix,
    int64_t out_channels,
    int64_t kernel,
    int dilation = 1) {
    return conv1d(
        ctx,
        weights,
        input,
        prefix,
        out_channels,
        kernel,
        dilation * static_cast<int>(kernel / 2),
        dilation);
}

/** ConvTranspose1d producing stride * input_frames samples, PyTorch padding
 *  semantics -- the same construction inflect_v2 uses. */
core::TensorValue conv_transpose1d(
    core::ModuleBuildContext & ctx,
    const SanoTtsBackendWeights & weights,
    const core::TensorValue & input,
    const std::string & prefix,
    int64_t out_channels,
    int64_t kernel,
    int stride,
    int64_t padding) {
    const int64_t input_frames = input.shape.dims[2];
    const int64_t output_frames = input_frames * stride;
    if (ctx.backend_type == core::BackendType::Cpu && padding > 0) {
        const auto source = contiguous(ctx, input);
        auto * input_2d = ggml_reshape_2d(
            ctx.ggml,
            source.tensor,
            input_frames,
            input.shape.dims[1]);
        auto * output = ggml_conv_transpose_1d(
            ctx.ggml,
            weight(weights, prefix + ".weight").tensor,
            input_2d,
            stride,
            0,
            1);
        const int64_t full_frames = (input_frames - 1) * stride + kernel;
        output = ggml_reshape_2d(ctx.ggml, output, full_frames, out_channels);
        output = ggml_view_2d(
            ctx.ggml,
            output,
            output_frames,
            out_channels,
            ggml_row_size(output->type, full_frames),
            ggml_row_size(output->type, padding));
        output = ggml_cont(ctx.ggml, output);
        auto * bias = ggml_reshape_2d(
            ctx.ggml,
            weight(weights, prefix + ".bias").tensor,
            1,
            out_channels);
        output = ggml_add(ctx.ggml, output, bias);
        return core::wrap_tensor(
            ggml_reshape_3d(ctx.ggml, output, output_frames, out_channels, 1),
            core::TensorShape::from_dims({1, out_channels, output_frames}),
            GGML_TYPE_F32);
    }
    auto output = modules::ConvTranspose1dModule({
        input.shape.dims[1],
        out_channels,
        kernel,
        stride,
        0,
        1,
        true,
    }).build(
        ctx,
        input,
        {
            weight(weights, prefix + ".weight"),
            weight(weights, prefix + ".bias"),
        });
    return modules::SliceModule({2, padding, output_frames}).build(ctx, output);
}

/** PiperResidualBank: mean over active branches of
 *  y2 = conv2(lrelu(y1)) + y1, y1 = conv1(lrelu(x)) + x. */
core::TensorValue residual_bank(
    core::ModuleBuildContext & ctx,
    const SanoTtsBackendWeights & weights,
    const core::TensorValue & input,
    const std::string & prefix,
    const std::vector<int64_t> & branches) {
    core::TensorValue sum;
    for (const int64_t branch : branches) {
        const std::string branch_prefix =
            prefix + ".blocks." + std::to_string(branch);
        const int64_t kernel = kBankKernels[branch];
        auto value = leaky_relu(ctx, input, kLeakyReluSlope);
        value = conv1d_same(
            ctx,
            weights,
            value,
            branch_prefix + ".conv1",
            input.shape.dims[1],
            kernel,
            kBankDilations1[branch]);
        const auto y1 = add(ctx, value, input);
        value = leaky_relu(ctx, y1, kLeakyReluSlope);
        value = conv1d_same(
            ctx,
            weights,
            value,
            branch_prefix + ".conv2",
            input.shape.dims[1],
            kernel,
            kBankDilations2[branch]);
        const auto y2 = add(ctx, value, y1);
        sum = sum.valid() ? add(ctx, sum, y2) : y2;
    }
    return scaled(ctx, sum, 1.0F / static_cast<float>(branches.size()));
}

core::TensorValue post_filter(
    core::ModuleBuildContext & ctx,
    const SanoTtsBackendWeights & weights,
    const SanoTtsPiperConfig & config,
    const core::TensorValue & audio) {
    auto value = conv1d_same(
        ctx,
        weights,
        audio,
        "decoder.post_filter.in_conv",
        config.post_filter_channels,
        kernel_of(weights, "decoder.post_filter.in_conv"));
    for (int64_t layer = 0; layer < config.post_filter_layers; ++layer) {
        const std::string prefix =
            "decoder.post_filter.units." + std::to_string(layer);
        auto branch = leaky_relu(ctx, value, kLeakyReluSlope);
        branch = conv1d_same(
            ctx,
            weights,
            branch,
            prefix + ".conv1",
            config.post_filter_channels,
            kernel_of(weights, prefix + ".conv1"),
            static_cast<int>(1 + layer));
        branch = leaky_relu(ctx, branch, kLeakyReluSlope);
        branch = conv1d_same(
            ctx,
            weights,
            branch,
            prefix + ".conv2",
            config.post_filter_channels,
            kernel_of(weights, prefix + ".conv2"));
        const auto branch_source = contiguous(ctx, branch);
        branch = core::wrap_tensor(
            ggml_mul(
                ctx.ggml,
                branch_source.tensor,
                weight(weights, prefix + ".scale").tensor),
            branch.shape,
            GGML_TYPE_F32);
        value = add(ctx, value, branch);
    }
    auto correction = conv1d_same(
        ctx,
        weights,
        value,
        "decoder.post_filter.out_conv",
        1,
        kernel_of(weights, "decoder.post_filter.out_conv"));
    correction = scaled(ctx, correction, static_cast<float>(config.post_filter_scale));
    auto mixed = add(ctx, audio, correction);
    const auto mixed_source = contiguous(ctx, mixed);
    return core::wrap_tensor(
        ggml_tanh(ctx.ggml, mixed_source.tensor),
        mixed.shape,
        GGML_TYPE_F32);
}

// ---- graphs --------------------------------------------------------------

struct DurationGraph : GraphResources {
    int64_t token_count = 0;
    ggml_tensor * tokens = nullptr;
    ggml_tensor * feats = nullptr;
    ggml_tensor * log_duration = nullptr;
};

std::unique_ptr<DurationGraph> build_duration_graph(
    const SanoTtsBackendWeights & weights,
    const SanoTtsPiperConfig & config,
    ggml_backend_t backend,
    core::BackendType backend_type,
    int64_t token_count) {
    auto out = std::make_unique<DurationGraph>();
    out->backend = backend;
    out->token_count = token_count;
    out->io_context.reset(ggml_init({kIoArenaBytes, nullptr, true}));
    out->graph_context.reset(ggml_init({kGraphArenaBytes, nullptr, true}));
    if (out->io_context == nullptr || out->graph_context == nullptr) {
        throw std::runtime_error("sanoTTS failed to create duration graph contexts");
    }
    core::ModuleBuildContext io_ctx{
        out->io_context.get(),
        "sanotts.piper.duration.io",
        backend_type,
    };
    core::ModuleBuildContext ctx{
        out->graph_context.get(),
        "sanotts.piper.duration",
        backend_type,
    };
    auto tokens = core::make_tensor(
        io_ctx,
        GGML_TYPE_I32,
        core::TensorShape::from_dims({1, token_count}));
    auto feats = core::make_tensor(
        io_ctx,
        GGML_TYPE_F32,
        core::TensorShape::from_dims({1, 3, token_count}));
    ggml_set_input(tokens.tensor);
    ggml_set_input(feats.tensor);

    auto hidden = embed_tokens(
        ctx,
        weights,
        tokens,
        "duration.embedding.weight",
        config.duration_vocab,
        config.duration_hidden);
    hidden = modules::ConcatModule({1}).build(ctx, hidden, feats);
    hidden = conv1d(
        ctx,
        weights,
        hidden,
        "duration.input_proj",
        config.duration_hidden,
        1,
        0);
    for (int64_t block = 0; block < config.duration_depth; ++block) {
        hidden = residual_block(
            ctx,
            weights,
            hidden,
            "duration.blocks." + std::to_string(block),
            config.duration_hidden,
            config.duration_kernel);
    }
    auto log_duration = conv1d(ctx, weights, hidden, "duration.output", 1, 1, 0);
    log_duration = contiguous(ctx, log_duration);
    out->tokens = tokens.tensor;
    out->feats = feats.tensor;
    out->log_duration = log_duration.tensor;
    ggml_set_output(out->log_duration);
    out->graph = ggml_new_graph_custom(ctx.ggml, 16384, false);
    ggml_build_forward_expand(out->graph, out->log_duration);
    allocate_graph(*out);
    return out;
}

struct TokenGraph : GraphResources {
    int64_t token_count = 0;
    ggml_tensor * tokens = nullptr;
    ggml_tensor * feats = nullptr;
    ggml_tensor * context = nullptr;
};

std::unique_ptr<TokenGraph> build_token_graph(
    const SanoTtsBackendWeights & weights,
    const SanoTtsPiperConfig & config,
    ggml_backend_t backend,
    core::BackendType backend_type,
    int64_t token_count) {
    auto out = std::make_unique<TokenGraph>();
    out->backend = backend;
    out->token_count = token_count;
    out->io_context.reset(ggml_init({kIoArenaBytes, nullptr, true}));
    out->graph_context.reset(ggml_init({kGraphArenaBytes, nullptr, true}));
    if (out->io_context == nullptr || out->graph_context == nullptr) {
        throw std::runtime_error("sanoTTS failed to create token graph contexts");
    }
    core::ModuleBuildContext io_ctx{
        out->io_context.get(),
        "sanotts.piper.token.io",
        backend_type,
    };
    core::ModuleBuildContext ctx{
        out->graph_context.get(),
        "sanotts.piper.token",
        backend_type,
    };
    auto tokens = core::make_tensor(
        io_ctx,
        GGML_TYPE_I32,
        core::TensorShape::from_dims({1, token_count}));
    auto feats = core::make_tensor(
        io_ctx,
        GGML_TYPE_F32,
        core::TensorShape::from_dims({1, 2, token_count}));
    ggml_set_input(tokens.tensor);
    ggml_set_input(feats.tensor);

    auto hidden = embed_tokens(
        ctx,
        weights,
        tokens,
        "acoustic.embedding.weight",
        config.acoustic_vocab,
        config.acoustic_hidden);
    hidden = modules::ConcatModule({1}).build(ctx, hidden, feats);
    hidden = conv1d(
        ctx,
        weights,
        hidden,
        "acoustic.token_input_proj",
        config.acoustic_hidden,
        1,
        0);
    for (int64_t block = 0; block < config.acoustic_token_depth; ++block) {
        hidden = residual_block(
            ctx,
            weights,
            hidden,
            "acoustic.token_blocks." + std::to_string(block),
            config.acoustic_hidden,
            config.acoustic_kernel);
    }
    hidden = contiguous(ctx, hidden);
    out->tokens = tokens.tensor;
    out->feats = feats.tensor;
    out->context = hidden.tensor;
    ggml_set_output(out->context);
    out->graph = ggml_new_graph_custom(ctx.ggml, 16384, false);
    ggml_build_forward_expand(out->graph, out->context);
    allocate_graph(*out);
    return out;
}

struct DecoderGraph : GraphResources {
    int64_t frames = 0;
    ggml_tensor * context = nullptr;
    ggml_tensor * feats = nullptr;
    ggml_tensor * waveform = nullptr;
};

/** Frame-stage acoustic blocks -> latent -> 3-stage ConvTranspose decoder
 *  with dilated residual banks -> tanh waveform (plus kristin's post
 *  filter when the config carries one). */
std::unique_ptr<DecoderGraph> build_decoder_graph(
    const SanoTtsBackendWeights & weights,
    const SanoTtsPiperConfig & config,
    ggml_backend_t backend,
    core::BackendType backend_type,
    int64_t frames) {
    auto out = std::make_unique<DecoderGraph>();
    out->backend = backend;
    out->frames = frames;
    out->io_context.reset(ggml_init({kIoArenaBytes, nullptr, true}));
    out->graph_context.reset(ggml_init({kGraphArenaBytes, nullptr, true}));
    if (out->io_context == nullptr || out->graph_context == nullptr) {
        throw std::runtime_error("sanoTTS failed to create decoder graph contexts");
    }
    core::ModuleBuildContext io_ctx{
        out->io_context.get(),
        "sanotts.piper.decoder.io",
        backend_type,
    };
    core::ModuleBuildContext ctx{
        out->graph_context.get(),
        "sanotts.piper.decoder",
        backend_type,
    };
    auto context = core::make_tensor(
        io_ctx,
        GGML_TYPE_F32,
        core::TensorShape::from_dims({1, config.acoustic_hidden, frames}));
    auto feats = core::make_tensor(
        io_ctx,
        GGML_TYPE_F32,
        core::TensorShape::from_dims({1, 3, frames}));
    ggml_set_input(context.tensor);
    ggml_set_input(feats.tensor);

    auto hidden = modules::ConcatModule({1}).build(ctx, context, feats);
    hidden = conv1d(
        ctx,
        weights,
        hidden,
        "acoustic.frame_input_proj",
        config.acoustic_hidden,
        1,
        0);
    for (int64_t block = 0; block < config.acoustic_depth; ++block) {
        hidden = residual_block(
            ctx,
            weights,
            hidden,
            "acoustic.frame_blocks." + std::to_string(block),
            config.acoustic_hidden,
            config.acoustic_kernel);
    }
    auto latent = conv1d(
        ctx,
        weights,
        hidden,
        "acoustic.output",
        config.acoustic_out_channels,
        1,
        0);

    auto value = conv1d_same(
        ctx,
        weights,
        latent,
        "decoder.pre",
        config.channels[0],
        kernel_of(weights, "decoder.pre"));
    for (size_t stage = 0; stage < 3; ++stage) {
        const std::string up_name = "decoder.up" + std::to_string(stage);
        value = leaky_relu(ctx, value, kLeakyReluSlope);
        value = conv_transpose1d(
            ctx,
            weights,
            value,
            up_name,
            config.channels[stage + 1],
            kernel_of(weights, up_name),
            kUpStrides[stage],
            kUpPaddings[stage]);
        value = residual_bank(
            ctx,
            weights,
            value,
            "decoder.res" + std::to_string(stage) + ".0",
            config.stage_branches[stage]);
    }
    value = leaky_relu(ctx, value, 0.01F);
    auto audio = conv1d_same(
        ctx,
        weights,
        value,
        "decoder.post",
        1,
        kernel_of(weights, "decoder.post"));
    {
        const auto audio_source = contiguous(ctx, audio);
        audio = core::wrap_tensor(
            ggml_tanh(ctx.ggml, audio_source.tensor),
            audio.shape,
            GGML_TYPE_F32);
    }
    if (config.post_filter_channels > 0) {
        audio = post_filter(ctx, weights, config, audio);
    }
    audio = contiguous(ctx, audio);
    out->context = context.tensor;
    out->feats = feats.tensor;
    out->waveform = audio.tensor;
    ggml_set_output(out->waveform);
    out->graph = ggml_new_graph_custom(ctx.ggml, 16384, false);
    ggml_build_forward_expand(out->graph, out->waveform);
    allocate_graph(*out);
    return out;
}

}  // namespace

struct SanoTtsPiperRuntime::State {
    struct BackendOwner {
        ggml_backend_t value = nullptr;
        ~BackendOwner() {
            if (value != nullptr) {
                ggml_backend_free(value);
            }
        }
    };

    State(
        std::shared_ptr<const SanoTtsAssets> assets_in,
        core::BackendConfig backend_config)
        : assets(std::move(assets_in)),
          threads(std::max(1, backend_config.threads)) {
        if (assets == nullptr) {
            throw std::runtime_error("sanoTTS piper runtime requires assets");
        }
        backend_config.threads = threads;
        backend.value = core::init_backend(backend_config);
        backend_type = core::backend_type(backend.value);
        core::set_backend_threads(backend.value, threads);
        weights = load_weights(
            assets,
            backend.value,
            backend_type,
            expected_piper_tensor_count(assets->piper),
            kWeightArenaBytes);
        if (backend_type == core::BackendType::Cuda) {
            // Durations round to integers and gate the whole frame layout;
            // keep the tiny duration model on the host, as the nano runtime
            // and inflect_v2 do.
            core::BackendConfig duration_config{
                core::BackendType::Cpu,
                0,
                threads,
            };
            duration_backend.value = core::init_backend(duration_config);
            core::set_backend_threads(duration_backend.value, threads);
            duration_weights = load_weights(
                assets,
                duration_backend.value,
                core::BackendType::Cpu,
                expected_piper_tensor_count(assets->piper),
                kWeightArenaBytes);
        }
    }

    DurationGraph & duration_graph(int64_t token_count) {
        if (auto * found = duration_graphs.find(token_count)) {
            engine::debug::trace_log_scalar("sanotts.duration_graph.cache_hit", true);
            return **found;
        }
        engine::debug::trace_log_scalar("sanotts.duration_graph.cache_hit", false);
        const auto backend_value =
            duration_backend.value != nullptr ? duration_backend.value : backend.value;
        const auto graph_backend_type =
            duration_backend.value != nullptr ? core::BackendType::Cpu : backend_type;
        const auto & selected_weights =
            duration_weights != nullptr ? duration_weights : weights;
        duration_graphs.put(
            token_count,
            build_duration_graph(
                *selected_weights,
                assets->piper,
                backend_value,
                graph_backend_type,
                token_count));
        auto * created = duration_graphs.find(token_count);
        if (created == nullptr) {
            throw std::runtime_error("sanoTTS duration graph cache insert failed");
        }
        return **created;
    }

    TokenGraph & token_graph(int64_t token_count) {
        if (auto * found = token_graphs.find(token_count)) {
            engine::debug::trace_log_scalar("sanotts.token_graph.cache_hit", true);
            return **found;
        }
        engine::debug::trace_log_scalar("sanotts.token_graph.cache_hit", false);
        token_graphs.put(
            token_count,
            build_token_graph(
                *weights,
                assets->piper,
                backend.value,
                backend_type,
                token_count));
        auto * created = token_graphs.find(token_count);
        if (created == nullptr) {
            throw std::runtime_error("sanoTTS token graph cache insert failed");
        }
        return **created;
    }

    DecoderGraph & decoder_graph(int64_t frames) {
        if (auto * found = decoder_graphs.find(frames)) {
            engine::debug::trace_log_scalar("sanotts.decoder_graph.cache_hit", true);
            return **found;
        }
        engine::debug::trace_log_scalar("sanotts.decoder_graph.cache_hit", false);
        decoder_graphs.put(
            frames,
            build_decoder_graph(
                *weights,
                assets->piper,
                backend.value,
                backend_type,
                frames));
        auto * created = decoder_graphs.find(frames);
        if (created == nullptr) {
            throw std::runtime_error("sanoTTS decoder graph cache insert failed");
        }
        return **created;
    }

    std::shared_ptr<const SanoTtsAssets> assets;
    int threads = 1;
    core::BackendType backend_type = core::BackendType::Cpu;
    BackendOwner backend;
    std::shared_ptr<const SanoTtsBackendWeights> weights;
    BackendOwner duration_backend;
    std::shared_ptr<const SanoTtsBackendWeights> duration_weights;
    runtime::CacheSlots<int64_t, std::unique_ptr<DurationGraph>> duration_graphs{4};
    runtime::CacheSlots<int64_t, std::unique_ptr<TokenGraph>> token_graphs{4};
    runtime::CacheSlots<int64_t, std::unique_ptr<DecoderGraph>> decoder_graphs{2};
};

SanoTtsPiperRuntime::SanoTtsPiperRuntime(
    std::shared_ptr<const SanoTtsAssets> assets,
    core::BackendConfig backend_config)
    : state_(std::make_unique<State>(std::move(assets), backend_config)) {}

SanoTtsPiperRuntime::~SanoTtsPiperRuntime() = default;

runtime::AudioBuffer SanoTtsPiperRuntime::synthesize(
    const std::vector<int32_t> & token_ids,
    const SanoTtsPiperGenerationOptions & options) {
    const auto & config = state_->assets->piper;
    const auto token_count = static_cast<int64_t>(token_ids.size());
    if (token_count <= 0) {
        throw std::runtime_error("sanoTTS requires at least one phoneme token");
    }
    const auto total_start = std::chrono::steady_clock::now();

    // -- durations ---------------------------------------------------------
    const auto duration_start = std::chrono::steady_clock::now();
    auto & duration = state_->duration_graph(token_count);
    core::write_tensor_i32(
        core::wrap_tensor(
            duration.tokens,
            core::TensorShape::from_dims({1, token_count}),
            GGML_TYPE_I32),
        clamp_ids_to_vocab(token_ids, config.duration_vocab));
    {
        // [positions, length_hint, valid=1] rows; hints computed in double
        // and cast, matching the reference implementation.
        std::vector<float> feats(static_cast<size_t>(3 * token_count));
        linspace01(feats.data(), token_count);
        const auto length_hint = static_cast<float>(
            std::log1p(static_cast<double>(token_count)) /
            std::log1p(static_cast<double>(config.duration_max_tokens)));
        std::fill_n(feats.begin() + token_count, token_count, length_hint);
        std::fill_n(feats.begin() + 2 * token_count, token_count, 1.0F);
        core::write_tensor_f32(
            core::wrap_tensor(
                duration.feats,
                core::TensorShape::from_dims({1, 3, token_count}),
                GGML_TYPE_F32),
            feats);
    }
    compute_graph(duration, "sanoTTS piper duration");
    const auto log_duration = core::read_tensor_f32(duration.log_duration);
    if (static_cast<int64_t>(log_duration.size()) != token_count) {
        throw std::runtime_error("sanoTTS duration graph returned invalid output");
    }
    // exp -> clamp_min(1) -> *scale -> round (ties to even) -> clamp, in
    // double, exactly as the reference computes it.
    const double scale = config.duration_length_scale *
                         static_cast<double>(options.speaking_rate);
    std::vector<int64_t> durations(static_cast<size_t>(token_count));
    int64_t frames = 0;
    for (int64_t token = 0; token < token_count; ++token) {
        double value = std::exp(
            static_cast<double>(log_duration[static_cast<size_t>(token)]));
        if (!std::isfinite(value)) {
            throw std::runtime_error("sanoTTS duration predictor produced a non-finite duration");
        }
        if (value < 1.0) {
            value = 1.0;
        }
        value = std::rint(value * scale);
        if (value < 1.0) {
            value = 1.0;
        }
        if (value > static_cast<double>(config.duration_max_frames)) {
            value = static_cast<double>(config.duration_max_frames);
        }
        durations[static_cast<size_t>(token)] = static_cast<int64_t>(value);
        frames += durations[static_cast<size_t>(token)];
    }
    const int64_t max_frames = config.duration_max_tokens * config.duration_max_frames;
    if (frames < 1 || frames > max_frames) {
        throw std::runtime_error(
            "sanoTTS expanded to " + std::to_string(frames) +
            " frames, outside the supported range");
    }
    engine::debug::timing_log_scalar(
        "sanotts.duration_ms",
        engine::debug::elapsed_ms(duration_start));

    // -- token-stage acoustic context --------------------------------------
    const auto acoustic_start = std::chrono::steady_clock::now();
    auto & token_graph = state_->token_graph(token_count);
    core::write_tensor_i32(
        core::wrap_tensor(
            token_graph.tokens,
            core::TensorShape::from_dims({1, token_count}),
            GGML_TYPE_I32),
        clamp_ids_to_vocab(token_ids, config.acoustic_vocab));
    {
        std::vector<float> feats(static_cast<size_t>(2 * token_count));
        linspace01(feats.data(), token_count);
        double max_duration = 1.0;
        for (int64_t token = 0; token < token_count; ++token) {
            max_duration = std::max(
                max_duration,
                static_cast<double>(durations[static_cast<size_t>(token)]));
        }
        const double log_max_duration = std::log1p(max_duration);
        for (int64_t token = 0; token < token_count; ++token) {
            feats[static_cast<size_t>(token_count + token)] = static_cast<float>(
                std::log1p(static_cast<double>(durations[static_cast<size_t>(token)])) /
                log_max_duration);
        }
        core::write_tensor_f32(
            core::wrap_tensor(
                token_graph.feats,
                core::TensorShape::from_dims({1, 2, token_count}),
                GGML_TYPE_F32),
            feats);
    }
    compute_graph(token_graph, "sanoTTS piper token context");
    const auto token_context = core::read_tensor_f32(token_graph.context);
    if (static_cast<int64_t>(token_context.size()) !=
        config.acoustic_hidden * token_count) {
        throw std::runtime_error("sanoTTS token graph returned invalid output");
    }

    // -- expand to frames --------------------------------------------------
    std::vector<float> expanded(static_cast<size_t>(config.acoustic_hidden * frames));
    for (int64_t channel = 0; channel < config.acoustic_hidden; ++channel) {
        const float * row = token_context.data() + channel * token_count;
        float * out_row = expanded.data() + channel * frames;
        int64_t at = 0;
        for (int64_t token = 0; token < token_count; ++token) {
            const float value = row[token];
            for (int64_t j = 0; j < durations[static_cast<size_t>(token)]; ++j) {
                out_row[at++] = value;
            }
        }
    }
    std::vector<float> frame_feats(static_cast<size_t>(3 * frames));
    linspace01(frame_feats.data(), frames);
    {
        const int64_t denominator = token_count > 1 ? token_count - 1 : 1;
        float * token_pos = frame_feats.data() + frames;
        float * duration_pos = frame_feats.data() + 2 * frames;
        int64_t at = 0;
        for (int64_t token = 0; token < token_count; ++token) {
            const int64_t count = durations[static_cast<size_t>(token)];
            const auto position = static_cast<float>(
                static_cast<double>(token) / static_cast<double>(denominator));
            for (int64_t j = 0; j < count; ++j) {
                token_pos[at] = position;
                duration_pos[at] = count == 1
                    ? 0.0F
                    : static_cast<float>(
                          static_cast<double>(j) / static_cast<double>(count - 1));
                ++at;
            }
        }
    }
    engine::debug::timing_log_scalar(
        "sanotts.acoustic_ms",
        engine::debug::elapsed_ms(acoustic_start));

    // -- frame stage + decoder ---------------------------------------------
    const auto decoder_start = std::chrono::steady_clock::now();
    auto & decoder = state_->decoder_graph(frames);
    core::write_tensor_f32(
        core::wrap_tensor(
            decoder.context,
            core::TensorShape::from_dims({1, config.acoustic_hidden, frames}),
            GGML_TYPE_F32),
        expanded);
    core::write_tensor_f32(
        core::wrap_tensor(
            decoder.feats,
            core::TensorShape::from_dims({1, 3, frames}),
            GGML_TYPE_F32),
        frame_feats);
    compute_graph(decoder, "sanoTTS piper decoder");
    runtime::AudioBuffer out;
    out.sample_rate = static_cast<int>(config.sample_rate);
    out.channels = 1;
    out.samples = core::read_tensor_f32(decoder.waveform);
    const auto expected_samples = static_cast<size_t>(frames) * 256U;
    if (out.samples.size() != expected_samples) {
        throw std::runtime_error("sanoTTS decoder returned an unexpected sample count");
    }
    engine::debug::timing_log_scalar(
        "sanotts.decoder_ms",
        engine::debug::elapsed_ms(decoder_start));
    engine::debug::trace_log_scalar("sanotts.token_count", token_count);
    engine::debug::trace_log_scalar("sanotts.frames", frames);
    engine::debug::trace_log_scalar(
        "sanotts.output_samples",
        static_cast<int64_t>(out.samples.size()));
    engine::debug::timing_log_scalar(
        "session.wall_ms",
        engine::debug::elapsed_ms(total_start));
    return out;
}

}  // namespace engine::models::sanotts
