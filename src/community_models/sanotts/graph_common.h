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

#include "ggml-alloc.h"

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

}  // namespace engine::models::sanotts::graph
