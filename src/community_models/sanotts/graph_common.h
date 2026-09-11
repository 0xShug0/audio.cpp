#pragma once

// Internal helpers shared by the two sanoTTS runtimes (the nano lineage in
// runtime.cpp and the piperlite lineage in piper_runtime.cpp). Both lineages
// share the same convolutional front-end structure -- residual conv blocks
// over channel-major [1, C, T] values -- and the same graph plumbing.

#include "engine/community_models/sanotts/assets.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/cache_slots.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::sanotts::graph {

struct GgmlContextDeleter {
    void operator()(ggml_context * context) const noexcept {
        if (context != nullptr) {
            ggml_free(context);
        }
    }
};

inline core::TensorValue contiguous(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value) {
    if (core::has_backend_addressable_layout(value.tensor)) {
        return value;
    }
    return core::wrap_tensor(
        ggml_cont(ctx.ggml, value.tensor),
        value.shape,
        value.type);
}

inline core::TensorValue add(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) {
    return modules::AddModule().build(ctx, lhs, rhs);
}

struct SanoTtsBackendWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    std::unordered_map<std::string, core::TensorValue> tensors;
};

inline const core::TensorValue & weight(
    const SanoTtsBackendWeights & weights,
    const std::string & name) {
    const auto found = weights.tensors.find(name);
    if (found == weights.tensors.end()) {
        throw std::runtime_error("sanoTTS missing tensor: " + name);
    }
    return found->second;
}

inline std::shared_ptr<const SanoTtsBackendWeights> load_weights(
    const std::shared_ptr<const SanoTtsAssets> & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t expected_tensors,
    size_t weight_arena_bytes) {
    auto out = std::make_shared<SanoTtsBackendWeights>();
    out->store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "sanotts.weights",
        weight_arena_bytes);
    const auto metadata = assets->weights->tensors();
    if (metadata.size() != expected_tensors) {
        throw std::runtime_error(
            "sanoTTS expects exactly " + std::to_string(expected_tensors) +
            " tensors for this config, found " + std::to_string(metadata.size()));
    }
    out->tensors.reserve(metadata.size());
    for (const auto & tensor : metadata) {
        if (assets::ggml_type_for_tensor_dtype(tensor.dtype) != GGML_TYPE_F32) {
            throw std::runtime_error(
                "sanoTTS supports FP32 weights only: " + tensor.name);
        }
        out->tensors.emplace(
            tensor.name,
            out->store->load_tensor(
                *assets->weights,
                tensor.name,
                assets::TensorStorageType::F32,
                tensor.shape));
    }
    out->store->upload();
    assets->weights->release_storage();
    return out;
}

/** Conv1d over channel-major [1, C, T] with explicit padding and dilation.
 *  A kernel-1 conv lowers to a matmul. */
inline core::TensorValue conv1d(
    core::ModuleBuildContext & ctx,
    const SanoTtsBackendWeights & weights,
    const core::TensorValue & input,
    const std::string & prefix,
    int64_t out_channels,
    int64_t kernel,
    int padding,
    int dilation = 1) {
    const int64_t in_channels = input.shape.dims[1];
    const int64_t input_frames = input.shape.dims[2];
    const int64_t output_frames =
        input_frames + 2 * padding - static_cast<int64_t>(dilation) * (kernel - 1);
    const auto source = contiguous(ctx, input);
    auto * input_2d = ggml_reshape_2d(
        ctx.ggml,
        source.tensor,
        input_frames,
        in_channels);
    auto * kernel_tensor = weight(weights, prefix + ".weight").tensor;
    ggml_tensor * output = nullptr;
    if (kernel == 1 && padding == 0) {
        auto * kernel_2d = ggml_reshape_2d(
            ctx.ggml,
            kernel_tensor,
            in_channels,
            out_channels);
        auto * input_channels_first = ggml_cont(
            ctx.ggml,
            ggml_permute(ctx.ggml, input_2d, 1, 0, 2, 3));
        auto * output_channels_first =
            ggml_mul_mat(ctx.ggml, kernel_2d, input_channels_first);
        output = ggml_reshape_2d(
            ctx.ggml,
            ggml_cont(
                ctx.ggml,
                ggml_permute(ctx.ggml, output_channels_first, 1, 0, 2, 3)),
            output_frames,
            out_channels);
    } else {
        auto * kernel_3d = ggml_reshape_3d(
            ctx.ggml,
            kernel_tensor,
            kernel,
            in_channels,
            out_channels);
        auto * input_3d = ggml_reshape_3d(
            ctx.ggml,
            input_2d,
            input_frames,
            in_channels,
            1);
        auto * output_3d = ggml_conv_1d(
            ctx.ggml,
            kernel_3d,
            input_3d,
            1,
            padding,
            dilation);
        output = ggml_reshape_2d(
            ctx.ggml,
            output_3d,
            output_frames,
            out_channels);
    }
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

/** x + scale * conv2(silu(conv1(x))) -- the front ends' ResidualConvBlock.
 *  `scale` is a learned one-element tensor, broadcast by ggml_mul. */
inline core::TensorValue residual_block(
    core::ModuleBuildContext & ctx,
    const SanoTtsBackendWeights & weights,
    const core::TensorValue & input,
    const std::string & prefix,
    int64_t hidden,
    int64_t kernel) {
    const int padding = static_cast<int>(kernel / 2);
    auto h = conv1d(ctx, weights, input, prefix + ".net.0", hidden, kernel, padding);
    h = modules::SiluModule().build(ctx, h);
    h = conv1d(ctx, weights, h, prefix + ".net.2", hidden, kernel, padding);
    const auto h_source = contiguous(ctx, h);
    const auto scaled_h = core::wrap_tensor(
        ggml_mul(ctx.ggml, h_source.tensor, weight(weights, prefix + ".scale").tensor),
        h.shape,
        GGML_TYPE_F32);
    return add(ctx, input, scaled_h);
}

inline core::TensorValue embed_tokens(
    core::ModuleBuildContext & ctx,
    const SanoTtsBackendWeights & weights,
    const core::TensorValue & tokens,
    const std::string & name,
    int64_t vocab,
    int64_t hidden) {
    auto embedded = modules::EmbeddingModule({vocab, hidden}).build(
        ctx,
        tokens,
        weight(weights, name));
    return modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, embedded);
}

struct GraphResources {
    ~GraphResources() {
        core::free_backend_graph_plan(backend, plan);
        core::release_backend_graph_resources(backend, graph);
        if (allocator != nullptr) {
            ggml_gallocr_free(allocator);
        }
        if (io_buffer != nullptr) {
            ggml_backend_buffer_free(io_buffer);
        }
    }

    std::unique_ptr<ggml_context, GgmlContextDeleter> io_context;
    std::unique_ptr<ggml_context, GgmlContextDeleter> graph_context;
    ggml_backend_buffer_t io_buffer = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_graph_plan_t plan = nullptr;
    ggml_cgraph * graph = nullptr;
};

inline void allocate_graph(GraphResources & resources) {
    resources.io_buffer =
        ggml_backend_alloc_ctx_tensors(resources.io_context.get(), resources.backend);
    if (resources.io_buffer == nullptr) {
        throw std::runtime_error("sanoTTS failed to allocate graph input buffer");
    }
    resources.allocator =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(resources.backend));
    if (resources.allocator == nullptr ||
        !ggml_gallocr_reserve(resources.allocator, resources.graph) ||
        !ggml_gallocr_alloc_graph(resources.allocator, resources.graph)) {
        throw std::runtime_error("sanoTTS failed to allocate backend graph");
    }
    core::validate_backend_graph_supported(
        resources.backend,
        resources.graph,
        "sanoTTS");
    resources.plan =
        core::create_backend_graph_plan_if_host(resources.backend, resources.graph);
}

inline void compute_graph(GraphResources & resources, const char * label) {
    const auto status = core::compute_backend_graph(
        resources.backend,
        resources.graph,
        resources.plan,
        label);
    ggml_backend_synchronize(resources.backend);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(label) + " graph compute failed");
    }
}

/** torch.linspace(0, 1, n) with exact CPU-kernel float semantics: step in
 *  fp32, the first half filled as step*i, the second as fma(-step, n-1-i, 1). */
inline void linspace01(float * dst, int64_t n) {
    if (n <= 0) {
        return;
    }
    if (n == 1) {
        dst[0] = 0.0F;
        return;
    }
    const auto step = 1.0F / static_cast<float>(n - 1);
    const int64_t half = n / 2;
    for (int64_t i = 0; i < half; ++i) {
        dst[i] = step * static_cast<float>(i);
    }
    for (int64_t i = half; i < n; ++i) {
        dst[i] = std::fma(-step, static_cast<float>(n - 1 - i), 1.0F);
    }
}


// ---- shared front-end graphs ---------------------------------------------
//
// Both lineages run the same two token-level stages -- embed + positional
// features + input projection + residual blocks (+ optional 1x1 output conv)
// -- under the same tensor names; only the dimensions differ per package.

struct FrontStageSpec {
    const char * embedding = nullptr;    // "duration.embedding.weight"
    const char * input_proj = nullptr;   // "duration.input_proj"
    const char * block_prefix = nullptr; // "duration.blocks."
    const char * output_conv = nullptr;  // "duration.output", or nullptr
    const char * label = nullptr;        // graph label for contexts/tracing
    int64_t feat_rows = 3;
    int64_t vocab = 0;
    int64_t hidden = 0;
    int64_t depth = 0;
    int64_t kernel = 0;
    int64_t out_channels = 1;            // used when output_conv is set
};

inline FrontStageSpec duration_stage_spec(int64_t vocab, int64_t hidden, int64_t depth, int64_t kernel) {
    return {"duration.embedding.weight", "duration.input_proj", "duration.blocks.",
            "duration.output", "sanotts.duration", 3, vocab, hidden, depth, kernel, 1};
}

inline FrontStageSpec token_stage_spec(int64_t vocab, int64_t hidden, int64_t depth, int64_t kernel) {
    return {"acoustic.embedding.weight", "acoustic.token_input_proj", "acoustic.token_blocks.",
            nullptr, "sanotts.token", 2, vocab, hidden, depth, kernel, 1};
}

struct FrontGraph : GraphResources {
    int64_t token_count = 0;
    ggml_tensor * tokens = nullptr;
    ggml_tensor * feats = nullptr;
    ggml_tensor * output = nullptr;   // log-durations or token context
};

inline std::unique_ptr<FrontGraph> build_front_graph(
    const SanoTtsBackendWeights & weights,
    const FrontStageSpec & spec,
    ggml_backend_t backend,
    core::BackendType backend_type,
    int64_t token_count,
    size_t io_arena_bytes,
    size_t graph_arena_bytes) {
    auto out = std::make_unique<FrontGraph>();
    out->backend = backend;
    out->token_count = token_count;
    out->io_context.reset(ggml_init({io_arena_bytes, nullptr, true}));
    out->graph_context.reset(ggml_init({graph_arena_bytes, nullptr, true}));
    if (out->io_context == nullptr || out->graph_context == nullptr) {
        throw std::runtime_error("sanoTTS failed to create graph contexts");
    }
    core::ModuleBuildContext io_ctx{out->io_context.get(), spec.label, backend_type};
    core::ModuleBuildContext ctx{out->graph_context.get(), spec.label, backend_type};
    auto tokens = core::make_tensor(
        io_ctx,
        GGML_TYPE_I32,
        core::TensorShape::from_dims({1, token_count}));
    auto feats = core::make_tensor(
        io_ctx,
        GGML_TYPE_F32,
        core::TensorShape::from_dims({1, spec.feat_rows, token_count}));
    ggml_set_input(tokens.tensor);
    ggml_set_input(feats.tensor);

    auto hidden = embed_tokens(ctx, weights, tokens, spec.embedding, spec.vocab, spec.hidden);
    hidden = modules::ConcatModule({1}).build(ctx, hidden, feats);
    hidden = conv1d(ctx, weights, hidden, spec.input_proj, spec.hidden, 1, 0);
    for (int64_t block = 0; block < spec.depth; ++block) {
        hidden = residual_block(
            ctx,
            weights,
            hidden,
            spec.block_prefix + std::to_string(block),
            spec.hidden,
            spec.kernel);
    }
    if (spec.output_conv != nullptr) {
        hidden = conv1d(ctx, weights, hidden, spec.output_conv, spec.out_channels, 1, 0);
    }
    hidden = contiguous(ctx, hidden);
    out->tokens = tokens.tensor;
    out->feats = feats.tensor;
    out->output = hidden.tensor;
    ggml_set_output(out->output);
    out->graph = ggml_new_graph_custom(ctx.ggml, 16384, false);
    ggml_build_forward_expand(out->graph, out->output);
    allocate_graph(*out);
    return out;
}

/** The frame-level acoustic stage both decoder graphs open with: expanded
 *  token context + [frame_pos, token_pos, duration_pos] -> residual blocks
 *  -> 1x1 output conv (mel-100 for nano, the 192-ch latent for piperlite). */
inline core::TensorValue acoustic_frame_stage(
    core::ModuleBuildContext & ctx,
    const SanoTtsBackendWeights & weights,
    const core::TensorValue & context,
    const core::TensorValue & feats,
    int64_t hidden,
    int64_t depth,
    int64_t kernel,
    int64_t out_channels) {
    auto value = modules::ConcatModule({1}).build(ctx, context, feats);
    value = conv1d(ctx, weights, value, "acoustic.frame_input_proj", hidden, 1, 0);
    for (int64_t block = 0; block < depth; ++block) {
        value = residual_block(
            ctx,
            weights,
            value,
            "acoustic.frame_blocks." + std::to_string(block),
            hidden,
            kernel);
    }
    return conv1d(ctx, weights, value, "acoustic.output", out_channels, 1, 0);
}

// ---- shared runtime state ------------------------------------------------

struct BackendOwner {
    ggml_backend_t value = nullptr;
    BackendOwner() = default;
    BackendOwner(const BackendOwner &) = delete;
    BackendOwner & operator=(const BackendOwner &) = delete;
    ~BackendOwner() {
        if (value != nullptr) {
            ggml_backend_free(value);
        }
    }
};

/** Backend + uploaded weights, with the tiny duration model mirrored to the
 *  host on CUDA builds: durations round to integers and gate the whole frame
 *  layout, so small TF32 differences must not move them (the inflect_v2
 *  rationale). */
struct BackendState {
    std::shared_ptr<const SanoTtsAssets> assets;
    int threads = 1;
    core::BackendType backend_type = core::BackendType::Cpu;
    BackendOwner backend;
    std::shared_ptr<const SanoTtsBackendWeights> weights;
    BackendOwner duration_backend;
    std::shared_ptr<const SanoTtsBackendWeights> duration_weights;

    BackendState(
        std::shared_ptr<const SanoTtsAssets> assets_in,
        core::BackendConfig backend_config,
        size_t expected_tensors,
        size_t weight_arena_bytes)
        : assets(std::move(assets_in)),
          threads(std::max(1, backend_config.threads)) {
        if (assets == nullptr) {
            throw std::runtime_error("sanoTTS runtime requires assets");
        }
        backend_config.threads = threads;
        backend.value = core::init_backend(backend_config);
        backend_type = core::backend_type(backend.value);
        core::set_backend_threads(backend.value, threads);
        weights = load_weights(
            assets, backend.value, backend_type, expected_tensors, weight_arena_bytes);
        if (backend_type == core::BackendType::Cuda) {
            core::BackendConfig duration_config{core::BackendType::Cpu, 0, threads};
            duration_backend.value = core::init_backend(duration_config);
            core::set_backend_threads(duration_backend.value, threads);
            duration_weights = load_weights(
                assets,
                duration_backend.value,
                core::BackendType::Cpu,
                expected_tensors,
                weight_arena_bytes);
        }
    }

    ggml_backend_t duration_backend_value() const {
        return duration_backend.value != nullptr ? duration_backend.value : backend.value;
    }
    core::BackendType duration_backend_type() const {
        return duration_backend.value != nullptr ? core::BackendType::Cpu : backend_type;
    }
    const SanoTtsBackendWeights & duration_weights_ref() const {
        return duration_weights != nullptr ? *duration_weights : *weights;
    }
};

template <typename Graph, typename Build>
Graph & cached_graph(
    runtime::CacheSlots<int64_t, std::unique_ptr<Graph>> & slots,
    int64_t key,
    const char * trace_name,
    Build && build) {
    if (auto * found = slots.find(key)) {
        engine::debug::trace_log_scalar(trace_name, true);
        return **found;
    }
    engine::debug::trace_log_scalar(trace_name, false);
    slots.put(key, build());
    auto * created = slots.find(key);
    if (created == nullptr) {
        throw std::runtime_error("sanoTTS graph cache insert failed");
    }
    return **created;
}

// ---- shared host-side pieces ---------------------------------------------
//
// Hints are computed in double and cast to fp32, matching the numpy
// references both runtimes are gated against.

/** [positions, length_hint, valid=1] rows for the duration stage. */
inline std::vector<float> duration_features(int64_t token_count, int64_t max_tokens) {
    std::vector<float> feats(static_cast<size_t>(3 * token_count));
    linspace01(feats.data(), token_count);
    const auto length_hint = static_cast<float>(
        std::log1p(static_cast<double>(token_count)) /
        std::log1p(static_cast<double>(max_tokens)));
    std::fill_n(feats.begin() + token_count, token_count, length_hint);
    std::fill_n(feats.begin() + 2 * token_count, token_count, 1.0F);
    return feats;
}

/** [token_pos, duration_hint] rows for the token stage. */
inline std::vector<float> token_features(
    int64_t token_count,
    const std::vector<int64_t> & durations) {
    std::vector<float> feats(static_cast<size_t>(2 * token_count));
    linspace01(feats.data(), token_count);
    double max_duration = 1.0;
    for (const int64_t duration : durations) {
        max_duration = std::max(max_duration, static_cast<double>(duration));
    }
    const double log_max_duration = std::log1p(max_duration);
    for (int64_t token = 0; token < token_count; ++token) {
        feats[static_cast<size_t>(token_count + token)] = static_cast<float>(
            std::log1p(static_cast<double>(durations[static_cast<size_t>(token)])) /
            log_max_duration);
    }
    return feats;
}

/** exp -> clamp_min(1) -> *scale -> round (ties to even) -> clamp, in
 *  double, exactly as the references compute it. Returns per-token frame
 *  counts and validates the total against the trained limits. */
inline std::vector<int64_t> round_durations(
    const std::vector<float> & log_duration,
    double scale,
    int64_t max_frames_per_token,
    int64_t max_total_frames,
    int64_t & total_frames) {
    std::vector<int64_t> durations(log_duration.size());
    total_frames = 0;
    for (size_t token = 0; token < log_duration.size(); ++token) {
        double value = std::exp(static_cast<double>(log_duration[token]));
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
        if (value > static_cast<double>(max_frames_per_token)) {
            value = static_cast<double>(max_frames_per_token);
        }
        durations[token] = static_cast<int64_t>(value);
        total_frames += durations[token];
    }
    if (total_frames < 1 || total_frames > max_total_frames) {
        throw std::runtime_error(
            "sanoTTS expanded to " + std::to_string(total_frames) +
            " frames, outside the supported range");
    }
    return durations;
}

/** Repeat each token's context vector across its own frames. */
inline std::vector<float> expand_context(
    const std::vector<float> & token_context,
    const std::vector<int64_t> & durations,
    int64_t hidden,
    int64_t token_count,
    int64_t frames) {
    std::vector<float> expanded(static_cast<size_t>(hidden * frames));
    for (int64_t channel = 0; channel < hidden; ++channel) {
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
    return expanded;
}

/** [frame_pos, token_pos, duration_pos] rows -- expand_features' documented
 *  float semantics: doubles cast to fp32, a duration of 1 contributing 0. */
inline std::vector<float> frame_features(
    int64_t token_count,
    const std::vector<int64_t> & durations,
    int64_t frames) {
    std::vector<float> feats(static_cast<size_t>(3 * frames));
    linspace01(feats.data(), frames);
    const int64_t denominator = token_count > 1 ? token_count - 1 : 1;
    float * token_pos = feats.data() + frames;
    float * duration_pos = feats.data() + 2 * frames;
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
    return feats;
}

inline void write_i32_input(ggml_tensor * tensor, const core::TensorShape & shape, const std::vector<int32_t> & values) {
    core::write_tensor_i32(core::wrap_tensor(tensor, shape, GGML_TYPE_I32), values);
}

inline void write_f32_input(ggml_tensor * tensor, const core::TensorShape & shape, const std::vector<float> & values) {
    core::write_tensor_f32(core::wrap_tensor(tensor, shape, GGML_TYPE_F32), values);
}

}  // namespace engine::models::sanotts::graph
