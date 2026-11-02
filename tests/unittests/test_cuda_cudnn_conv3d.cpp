#include "engine/framework/core/backend.h"
#include "engine/framework/codecs/wan_video_vae_runtime.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/conv_modules.h"

#include <ggml-alloc.h>
#include <algorithm>
#include <limits>
#include <ggml-backend.h>
#include <ggml.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

void run_case(ggml_backend_t backend, engine::modules::Conv3dConfig config,
              int batch, int depth, int height, int width, bool framework, bool f32_input,
              bool expect_unavailable, bool check_cast = false, bool packed_weights = false,
              bool sample_scalar_reference = false) {
    namespace core = engine::core;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({4 * 1024 * 1024, nullptr, true}), ggml_free);
    check(bool(ctx), "context allocation failed");
    core::ModuleBuildContext build{ctx.get(), "test.cudnn_conv3d", core::BackendType::Cuda};
    auto x = core::make_tensor(build, f32_input ? GGML_TYPE_F32 : GGML_TYPE_F16,
        core::TensorShape::from_dims({batch * config.in_channels, depth, height, width}));
    auto w = core::make_tensor(build, GGML_TYPE_F16, core::TensorShape::from_dims(
        {config.out_channels * config.in_channels, config.kernel_depth, config.kernel_height, config.kernel_width}));
    auto bias = core::make_tensor(build, GGML_TYPE_F32, core::TensorShape::from_dims({config.out_channels}));
    auto reference_w = packed_weights ? core::make_tensor(build, GGML_TYPE_F16, w.shape) : w;
    if (packed_weights) ggml_set_input(reference_w.tensor);
    ggml_set_input(x.tensor);
    ggml_set_input(w.tensor);
    ggml_set_input(bias.tensor);
    const ggml_type output_type = config.cuda_f16_output ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ggml_tensor * output;
    if (framework) {
        config.cuda_cudnn = true;
        config.cuda_prepacked_weights = packed_weights;
        output = engine::modules::Conv3dModule(config).build(build, x,
            {w, config.use_bias ? std::optional<core::TensorValue>(bias) : std::nullopt}).tensor;
    } else {
        output = (config.cuda_f16_output ? ggml_conv_3d_direct_f16 : ggml_conv_3d_direct)(ctx.get(), w.tensor, x.tensor,
            config.stride_width, config.stride_height, config.stride_depth,
            config.padding_width, config.padding_height, config.padding_depth,
            config.dilation_width, config.dilation_height, config.dilation_depth,
            int(config.in_channels), batch, int(config.out_channels));
        check(output->type == output_type && output->op_params[12] == int(config.cuda_f16_output),
            "direct Conv3D output type/default marker changed");
        const auto marker = output->op_params[12];
        output->op_params[12] = 0;
        check(!ggml_backend_supports_op(backend, output), "unmarked direct Conv3D advertised on CUDA");
        output->op_params[12] = marker;
        ggml_conv_3d_direct_set_cudnn(output, config.use_bias ? bias.tensor : nullptr, config.cuda_bf16_round, packed_weights, config.cuda_replicate_depth_left);
    }
    check(output->type == output_type && ggml_is_contiguous(output) &&
        output->nb[0] == ggml_type_size(output_type) &&
        ggml_nbytes(output) == size_t(ggml_nelements(output)) * ggml_type_size(output_type),
        "cuDNN Conv3D output dtype/byte size mismatch");
    check(output->op == GGML_OP_CONV_3D, "native output epilogue added a separate graph operation");
    // Independent GPU oracle: identical convolution with standalone F32 bias
    // addition and ROUND_BF16. Keep its output live alongside the fused result.
    ggml_tensor * reference = nullptr;
    if (config.use_bias || config.cuda_bf16_round || config.cuda_f16_output || packed_weights) {
        reference = ggml_conv_3d_direct(ctx.get(), reference_w.tensor, x.tensor,
            config.stride_width, config.stride_height, config.stride_depth,
            config.padding_width, config.padding_height, config.padding_depth,
            config.dilation_width, config.dilation_height, config.dilation_depth,
            int(config.in_channels), batch, int(config.out_channels));
        ggml_conv_3d_direct_set_cudnn(reference, nullptr, false, false, config.cuda_replicate_depth_left);
        if (config.use_bias) {
            reference = ggml_add(ctx.get(), reference,
                ggml_reshape_4d(ctx.get(), bias.tensor, 1, 1, 1, config.out_channels));
        }
        if (config.cuda_bf16_round) reference = ggml_round_bf16(ctx.get(), reference);
        if (config.cuda_f16_output) reference = ggml_cast(ctx.get(), reference, GGML_TYPE_F16);
        ggml_set_output(reference);
    }
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(ctx.get(), 128, false);
    ggml_build_forward_expand(graph, output);
    if (reference) ggml_build_forward_expand(graph, reference);
    ggml_tensor * conv = nullptr;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto * node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_CONV_3D && node == output) conv = node;
    }
    check(conv && conv->op_params[12] == 1, "framework did not select marked direct Conv3D");
    check(conv->src[1] == x.tensor, "marked Conv3D introduced a separate activation cast/copy");
    check(conv->src[2] == (config.use_bias ? bias.tensor : nullptr), "incorrect fused bias dependency");
    check(conv->op_params[13] == int(config.cuda_bf16_round), "incorrect fused BF16 round flag");
    check(conv->op_params[14] == int(packed_weights), "incorrect explicit packed-weight flag");
    conv->op_params[14] = 2;
    check(!ggml_backend_supports_op(backend, conv), "unknown packed-weight flag accepted");
    conv->op_params[14] = int(packed_weights);
    check(conv->op_params[15] == config.cuda_replicate_depth_left + 1, "incorrect replicated depth padding");
    conv->op_params[15] = 2 * config.padding_depth + 2;
    check(!ggml_backend_supports_op(backend, conv), "excess replicated depth padding accepted");
    conv->op_params[15] = config.cuda_replicate_depth_left + 1;
    check(ggml_backend_supports_op(backend, conv) != expect_unavailable,
        "cuDNN Conv3D support does not match the requested build configuration");
    // Reject CPU/Metal scheduling without initializing or executing either backend.
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        const auto dev = ggml_backend_dev_get(i);
        const std::string name = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
        if (name == "CPU" || name == "Metal") {
            check(!ggml_backend_dev_supports_op(dev, conv), "CUDA marker accepted by CPU/Metal");
        }
    }
    conv->op_params[12] = 2;
    check(!ggml_backend_supports_op(backend, conv), "unknown lowering marker accepted");
    conv->op_params[12] = 1;
    conv->op_params[13] = 2;
    check(!ggml_backend_supports_op(backend, conv), "unknown BF16 round flag accepted");
    conv->op_params[13] = int(config.cuda_bf16_round);
    if (config.cuda_f16_output) {
        conv->op_params[13] = 1;
        check(!ggml_backend_supports_op(backend, conv), "F16 output with BF16 rounding advertised");
        conv->op_params[13] = 0;
    }
    auto * saved_bias = conv->src[2];
    conv->src[2] = bias.tensor;
    bias.tensor->type = GGML_TYPE_F16;
    check(!ggml_backend_supports_op(backend, conv), "non-F32 bias advertised");
    bias.tensor->type = GGML_TYPE_F32;
    ++bias.tensor->ne[0];
    check(!ggml_backend_supports_op(backend, conv), "incorrect bias channel count advertised");
    --bias.tensor->ne[0];
    if (config.out_channels > 1) {
        bias.tensor->nb[0] += sizeof(float);
        check(!ggml_backend_supports_op(backend, conv), "noncontiguous bias advertised");
        bias.tensor->nb[0] -= sizeof(float);
    }
    conv->src[2] = saved_bias;
    conv->src[3] = bias.tensor;
    check(!ggml_backend_supports_op(backend, conv), "unexpected extra source advertised");
    conv->src[3] = nullptr;
    const auto saved_type = conv->src[1]->type;
    conv->src[1]->type = GGML_TYPE_BF16;
    check(!ggml_backend_supports_op(backend, conv), "unsupported BF16 input advertised");
    conv->src[1]->type = saved_type;
    for (int axis = 0; axis < 4; ++axis) {
        if (conv->src[0]->ne[axis] == 1) continue;
        const auto saved_stride = conv->src[0]->nb[axis];
        conv->src[0]->nb[axis] += sizeof(ggml_fp16_t);
        check(!ggml_backend_supports_op(backend, conv), "noncontiguous weights advertised");
        conv->src[0]->nb[axis] = saved_stride;
        break;
    }
    if (expect_unavailable) return;
    core::validate_backend_graph_supported(backend, graph, "test.cudnn_conv3d");
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> arena(
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free);
    check(bool(arena) && ggml_gallocr_alloc_graph(arena.get(), graph), "graph allocation failed");
    // Release captured backend resources before arena/context destruction, also on failure.
    const auto release = [backend](ggml_cgraph * g) { core::release_backend_graph_resources(backend, g, true); };
    std::unique_ptr<ggml_cgraph, decltype(release)> graph_resources(graph, release);

    std::vector<float> xf(size_t(ggml_nelements(x.tensor))), wf(size_t(ggml_nelements(w.tensor)));
    std::vector<float> bf(size_t(config.out_channels));
    std::vector<ggml_fp16_t> xh(xf.size()), wh(wf.size());
    // Include non-F16 values and rounding ties. The scalar reference uses exactly
    // the values of the old explicit F32 -> F16 activation cast.
    const float ties[] = {1.00048828125f, 1.00146484375f, -1.00048828125f, -1.00146484375f};
    for (size_t i = 0; i < xf.size(); ++i) {
        xf[i] = float(int(i % 17) - 8) / 16 + float(int(i % 5) - 2) * 0.00013f;
        if (i < 4) xf[i] = ties[i];
        xh[i] = ggml_fp32_to_fp16(xf[i]);
    }
    for (size_t i = 0; i < wf.size(); ++i) {
        wf[i] = float(int(i % 13) - 6) / 32;
        wh[i] = ggml_fp32_to_fp16(wf[i]);
    }
    if (check_cast) {
        check(config.in_channels == 1 && config.out_channels == 1 &&
            config.kernel_depth == 1 && config.kernel_height == 1 && config.kernel_width == 1 &&
            config.stride_depth == 1 && config.stride_height == 1 && config.stride_width == 1 &&
            config.padding_depth == 0 && config.padding_height == 0 && config.padding_width == 0,
            "cast-value check requires identity convolution");
        wf[0] = 1.0f;
        wh[0] = ggml_fp32_to_fp16(1.0f);
    }
    for (size_t i = 0; i < bf.size(); ++i) {
        bf[i] = float(i + 1) / 128;
        if (config.cuda_bf16_round || config.cuda_f16_output || check_cast) bf[i] += float(i + 1) * 0.00013f;
    }
    const int od = int(output->ne[2]), oh = int(output->ne[1]), ow = int(output->ne[0]);
    check(od == (depth + 2 * config.padding_depth - config.dilation_depth * (config.kernel_depth - 1) - 1) / config.stride_depth + 1 &&
          oh == (height + 2 * config.padding_height - config.dilation_height * (config.kernel_height - 1) - 1) / config.stride_height + 1 &&
          ow == (width + 2 * config.padding_width - config.dilation_width * (config.kernel_width - 1) - 1) / config.stride_width + 1 &&
          output->ne[3] == batch * config.out_channels, "incorrect Conv3D output shape");
    std::vector<float> expected(size_t(ggml_nelements(output))), previous;
    if (sample_scalar_reference)
        std::fill(expected.begin(), expected.end(), std::numeric_limits<float>::quiet_NaN());
    std::vector<ggml_fp16_t> previous_half;
    for (int n = 0; n < batch; ++n)
        for (int oc = 0; oc < config.out_channels; ++oc)
            for (int z = 0; z < od; ++z)
                for (int y = 0; y < oh; ++y)
                    for (int col = 0; col < ow; ++col) {
                        // Keep the large workspace-budget case cheap: cover both batches,
                        // channel ends, spatial borders, and the partial final depth tile.
                        if (sample_scalar_reference &&
                            ((oc != 0 && oc != config.out_channels / 2 && oc != config.out_channels - 1) ||
                             (z != 0 && z != od / 2 && z != od - 1) ||
                             (y != 0 && y != oh / 2 && y != oh - 1) ||
                             (col != 0 && col != ow / 2 && col != ow - 1))) continue;
                        float sum = 0;
                        for (int ic = 0; ic < config.in_channels; ++ic)
                            for (int kz = 0; kz < config.kernel_depth; ++kz)
                                for (int ky = 0; ky < config.kernel_height; ++ky)
                                    for (int kx = 0; kx < config.kernel_width; ++kx) {
                                        const int depth_left = config.cuda_replicate_depth_left >= 0
                                            ? config.cuda_replicate_depth_left : config.padding_depth;
                                        int iz = z * config.stride_depth - depth_left + kz * config.dilation_depth;
                                        if (config.cuda_replicate_depth_left >= 0) iz = std::clamp(iz, 0, depth - 1);
                                        const int iy = y * config.stride_height - config.padding_height + ky * config.dilation_height;
                                        const int ix = col * config.stride_width - config.padding_width + kx * config.dilation_width;
                                        if (iz < 0 || iz >= depth || iy < 0 || iy >= height || ix < 0 || ix >= width) continue;
                                        const size_t xi = (((n * config.in_channels + ic) * depth + iz) * height + iy) * width + ix;
                                        const size_t wi = (((oc * config.in_channels + ic) * config.kernel_depth + kz) * config.kernel_height + ky) * config.kernel_width + kx;
                                        sum += ggml_fp16_to_fp32(xh[xi]) * wf[wi];
                                    }
                        const size_t oi = (((n * config.out_channels + oc) * od + z) * oh + y) * ow + col;
                        expected[oi] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(sum)) + (config.use_bias ? bf[oc] : 0);
                        if (config.cuda_bf16_round) {
                            expected[oi] = ggml_bf16_to_fp32(ggml_fp32_to_bf16(expected[oi]));
                        }
                        if (config.cuda_f16_output) {
                            expected[oi] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(expected[oi]));
                        }
                    }
    std::vector<ggml_fp16_t> packed(wh.size());
    if (packed_weights) {
        const size_t kernel = size_t(config.kernel_depth) * config.kernel_height * config.kernel_width;
        for (size_t oc = 0; oc < size_t(config.out_channels); ++oc)
            for (size_t k = 0; k < kernel; ++k)
                for (size_t ic = 0; ic < size_t(config.in_channels); ++ic)
                    packed[(oc * kernel + k) * config.in_channels + ic] =
                        wh[(oc * config.in_channels + ic) * kernel + k];
    }
    for (int repeat = 0; repeat < 3; ++repeat) {
        ggml_backend_tensor_set(x.tensor, f32_input ? static_cast<const void *>(xf.data()) : xh.data(), 0,
            f32_input ? xf.size() * sizeof(float) : xh.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(w.tensor, packed_weights ? packed.data() : wh.data(), 0, wh.size() * sizeof(ggml_fp16_t));
        if (packed_weights) ggml_backend_tensor_set(reference_w.tensor, wh.data(), 0, wh.size() * sizeof(ggml_fp16_t));
        if (config.use_bias) ggml_backend_tensor_set(bias.tensor, bf.data(), 0, bf.size() * sizeof(float));
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "CUDA execution failed");
        ggml_backend_synchronize(backend);
        std::vector<float> actual(expected.size());
        std::vector<ggml_fp16_t> actual_half;
        if (config.cuda_f16_output) {
            actual_half.resize(actual.size());
            ggml_backend_tensor_get(output, actual_half.data(), 0, actual_half.size() * sizeof(ggml_fp16_t));
            for (size_t i = 0; i < actual.size(); ++i) actual[i] = ggml_fp16_to_fp32(actual_half[i]);
        } else {
            ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
        }
        if (reference) {
            if (config.cuda_f16_output) {
                std::vector<ggml_fp16_t> separate(actual_half.size());
                ggml_backend_tensor_get(reference, separate.data(), 0, separate.size() * sizeof(ggml_fp16_t));
                check(std::memcmp(actual_half.data(), separate.data(), actual_half.size() * sizeof(ggml_fp16_t)) == 0,
                    "F16 output differs bitwise from F32 convolution + F32 bias -> F16 GPU oracle");
            } else {
                std::vector<float> separate(actual.size());
                ggml_backend_tensor_get(reference, separate.data(), 0, separate.size() * sizeof(float));
                check(std::memcmp(actual.data(), separate.data(), actual.size() * sizeof(float)) == 0,
                    "fused epilogue differs bitwise from standalone GPU bias/ROUND_BF16");
            }
        }
        for (size_t i = 0; i < actual.size(); ++i) {
            check(std::isfinite(actual[i]), "nonfinite Conv3D output");
            if (sample_scalar_reference && !std::isfinite(expected[i])) continue;
            if (check_cast) check(actual[i] == expected[i], "identity convolution changed cast/bias/round values");
            const float relative_tolerance = config.cuda_bf16_round ? 0.01f : 0.001f;
            check(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) <= 0.002f + relative_tolerance * std::abs(expected[i]),
                "cuDNN Conv3D differs from scalar cross-correlation");
        }
        if (repeat != 0) check(actual == previous, "repeated/captured Conv3D changed output");
        if (repeat != 0 && config.cuda_f16_output) {
            check(actual_half == previous_half, "repeated/captured F16 output changed bits");
        }
        previous_half = std::move(actual_half);
        previous = std::move(actual);
    }
}

// Storage oracle: the existing F32 fused kernel followed by an independent GPU
// cast. Exercise both reduction kernels and their channel/spatial tail handling.
void run_norm_case(ggml_backend_t backend, int channels, bool f32_input,
                   bool coalesced, bool conv_available) {
    namespace core = engine::core;
    namespace codecs = engine::codecs;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({4 * 1024 * 1024, nullptr, true}), ggml_free);
    check(bool(ctx), "norm context allocation failed");
    core::ModuleBuildContext build{ctx.get(), "test.channel_rms_silu", core::BackendType::Cuda};
    auto x = core::make_tensor(build, f32_input ? GGML_TYPE_F32 : GGML_TYPE_F16,
        core::TensorShape::from_dims({channels, 2, 3, 17}));
    auto gamma = core::make_tensor(build, GGML_TYPE_F32, core::TensorShape::from_dims({channels}));
    ggml_set_input(x.tensor);
    ggml_set_input(gamma.tensor);
    codecs::WanVideoRMSNorm3dConfig config;
    config.channels = channels;
    config.eps = 1.e-8f;
    config.cuda_fast_lowering = true;
    config.fuse_silu = true;
    config.cuda_f16_output = true;
    auto fused = codecs::WanVideoRMSNorm3dModule(config).build(build, x, {gamma});
    config.cuda_f16_output = false;
    auto f32 = codecs::WanVideoRMSNorm3dModule(config).build(build, x, {gamma});
    check(fused.type == GGML_TYPE_F16 && f32.type == GGML_TYPE_F32 &&
        fused.tensor->op == GGML_OP_RMS_NORM_CHANNELS_SILU &&
        fused.tensor->src[0] == x.tensor && f32.tensor->src[0] == x.tensor,
        "fused module inserted an input cast or selected the wrong operation/type");
    check(fused.tensor->op_params[1] == GGML_RMS_NORM_CHANNELS_LOWERING_CUDA_COALESCED &&
        f32.tensor->op_params[1] == GGML_RMS_NORM_CHANNELS_LOWERING_CUDA_COALESCED,
        "fused module did not select the established CUDA coalesced lowering");
    if (!coalesced) {
        // Factory default (zero) selects the original block reduction.
        fused.tensor->op_params[1] = 0;
        f32.tensor->op_params[1] = 0;
    }
    auto * oracle = ggml_cast(ctx.get(), f32.tensor, GGML_TYPE_F16);
    const size_t count = size_t(ggml_nelements(x.tensor));
    check(ggml_is_contiguous(fused.tensor) && ggml_nbytes(fused.tensor) == count * sizeof(ggml_fp16_t),
        "F16 norm output byte layout mismatch");
    check(ggml_backend_supports_op(backend, fused.tensor), "CUDA rejected F16 RMSNorm/SiLU");
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto dev = ggml_backend_dev_get(i);
        const std::string name = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
        if (name == "CPU" || name == "Metal" || name == "Vulkan") {
            check(!ggml_backend_dev_supports_op(dev, fused.tensor), "unsupported backend advertised F16 RMSNorm/SiLU");
        }
    }
    // Explicit F16 requests must fail rather than becoming an implicit cast path.
    for (int invalid = 0; invalid < 3; ++invalid) {
        auto bad = config;
        bad.cuda_f16_output = true;
        if (invalid == 0) bad.cuda_fast_lowering = false;
        if (invalid == 1) bad.fuse_silu = false;
        auto non_cuda = build;
        non_cuda.backend_type = core::BackendType::Cpu;
        bool rejected = false;
        try {
            codecs::WanVideoRMSNorm3dModule module(bad);
            if (invalid == 2) module.build(non_cuda, x, {gamma});
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        check(rejected, "invalid F16 norm configuration silently accepted");
    }
    auto non_cuda = build;
    non_cuda.backend_type = core::BackendType::Cpu;
    auto fallback = codecs::WanVideoRMSNorm3dModule(config).build(non_cuda, x, {gamma});
    check(fallback.type == GGML_TYPE_F32 && fallback.tensor->op == GGML_OP_UNARY &&
        ggml_get_unary_op(fallback.tensor) == GGML_UNARY_OP_SILU,
        "generic fused-SiLU request did not use the existing module composition");

    auto * graph = ggml_new_graph_custom(ctx.get(), 128, false);
    ggml_set_output(fused.tensor);
    ggml_set_output(oracle);
    ggml_build_forward_expand(graph, fused.tensor);
    ggml_build_forward_expand(graph, oracle);
    ggml_tensor * conv = nullptr;
    ggml_tensor * conv_oracle = nullptr;
    ggml_tensor * weights = nullptr;
    if (conv_available) {
        weights = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, 1, 1, 1, channels * 3);
        ggml_set_input(weights);
        conv = ggml_conv_3d_direct_f16(ctx.get(), weights, fused.tensor,
            1, 1, 1, 0, 0, 0, 1, 1, 1, channels, 1, 3);
        conv_oracle = ggml_conv_3d_direct_f16(ctx.get(), weights, oracle,
            1, 1, 1, 0, 0, 0, 1, 1, 1, channels, 1, 3);
        ggml_conv_3d_direct_set_cudnn(conv);
        ggml_conv_3d_direct_set_cudnn(conv_oracle);
        check(conv->src[1] == fused.tensor, "norm-to-conv inserted a conversion");
        ggml_set_output(conv);
        ggml_set_output(conv_oracle);
        ggml_build_forward_expand(graph, conv);
        ggml_build_forward_expand(graph, conv_oracle);
    }
    core::validate_backend_graph_supported(backend, graph, "test.channel_rms_silu");
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> arena(
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free);
    check(bool(arena) && ggml_gallocr_alloc_graph(arena.get(), graph), "norm graph allocation failed");
    const auto release = [backend](ggml_cgraph * g) { core::release_backend_graph_resources(backend, g, true); };
    std::unique_ptr<ggml_cgraph, decltype(release)> graph_resources(graph, release);
    std::vector<float> xf(count), gf(channels);
    std::vector<ggml_fp16_t> xh(count), actual(count), expected(count), previous;
    std::vector<ggml_fp16_t> wh(size_t(channels) * 3, ggml_fp32_to_fp16(0.f));
    for (int oc = 0; oc < 3; ++oc) wh[size_t(oc) * channels + oc % channels] = ggml_fp32_to_fp16(1.f);
    if (weights) ggml_backend_tensor_set(weights, wh.data(), 0, wh.size() * sizeof(ggml_fp16_t));
    for (int repeat = 0; repeat < 6; ++repeat) {
        // Change data after capture, then verify repeated replay is stable too.
        const int revision = repeat / 3;
        for (size_t i = 0; i < count; ++i) {
            xf[i] = float(int((i * 7 + revision) % 41) - 20) * 0.07137f;
            if (i % 102 == 0) xf[i] = 0.f;
            xh[i] = ggml_fp32_to_fp16(xf[i]);
        }
        for (int c = 0; c < channels; ++c) gf[c] = float(c % 11 - 5) * 0.1937f + revision * 0.017f;
        ggml_backend_tensor_set(x.tensor, f32_input ? static_cast<const void *>(xf.data()) : xh.data(),
            0, count * (f32_input ? sizeof(float) : sizeof(ggml_fp16_t)));
        ggml_backend_tensor_set(gamma.tensor, gf.data(), 0, gf.size() * sizeof(float));
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "norm CUDA execution failed");
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(fused.tensor, actual.data(), 0, actual.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_get(oracle, expected.data(), 0, expected.size() * sizeof(ggml_fp16_t));
        check(actual == expected, "F16 RMSNorm/SiLU differs bitwise from F32 fused kernel + GPU cast");
        for (auto value : actual) check(std::isfinite(ggml_fp16_to_fp32(value)), "nonfinite norm result");
        if (repeat % 3 != 0) check(actual == previous, "repeated/captured F16 norm changed bits");
        if (repeat == 3) check(actual != previous, "captured norm ignored updated input/gamma");
        previous = actual;
        if (conv) {
            std::vector<ggml_fp16_t> got(size_t(ggml_nelements(conv))), ref(got.size());
            ggml_backend_tensor_get(conv, got.data(), 0, got.size() * sizeof(ggml_fp16_t));
            ggml_backend_tensor_get(conv_oracle, ref.data(), 0, ref.size() * sizeof(ggml_fp16_t));
            check(got == ref, "F16 norm-to-conv differs from independent cast-to-conv oracle");
        }
    }
}
} // namespace

int main(int argc, char ** argv) {
    if ((argc != 3 && argc != 4) || std::string(argv[1]) != "--log" ||
        (argc == 4 && std::string(argv[3]) != "--expect-unavailable")) {
        std::cerr << "Usage: cuda_cudnn_conv3d_test --log PATH [--expect-unavailable]\n";
        return 1;
    }
    engine::debug::configure_logging({true, argv[2]});
    const auto started = std::chrono::steady_clock::now();
    int result = 0;
    try {
        engine::core::ensure_backends_loaded();
        auto reg = ggml_backend_reg_by_name("CUDA");
        if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
            std::cerr << "SKIP: CUDA device unavailable\n";
            result = 77;
        } else {
            std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
                ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr), ggml_backend_free);
            check(bool(backend), "CUDA backend initialization failed");
            engine::modules::Conv3dConfig c;
            c.in_channels = 3; c.out_channels = 5;
            c.kernel_depth = 2; c.kernel_height = 3; c.kernel_width = 2;
            c.stride_depth = 2; c.stride_height = 1; c.stride_width = 2;
            c.padding_depth = 1; c.padding_height = 2; c.padding_width = 1;
            c.dilation_depth = 2; c.dilation_height = 1; c.dilation_width = 2;
            c.use_bias = false;
            const bool unavailable = argc == 4;
            for (int channels : {3, 17, 128, 512, 513}) {
                for (bool f32_input : {false, true}) {
                    for (bool coalesced : {false, true}) {
                        run_norm_case(backend.get(), channels, f32_input, coalesced, !unavailable);
                    }
                }
            }
            run_case(backend.get(), c, 2, 5, 6, 7, false, false, unavailable);
            run_case(backend.get(), c, 2, 5, 6, 7, false, true, unavailable);
            c.use_bias = true;
            run_case(backend.get(), c, 2, 5, 6, 7, true, true, unavailable);
            for (int left : {0, 1, 2}) {
                auto padded = c;
                padded.cuda_replicate_depth_left = left;
                for (bool f32_input : {false, true}) {
                    padded.cuda_f16_output = false;
                    run_case(backend.get(), padded, 2, 5, 6, 7, true, f32_input, unavailable);
                    padded.cuda_f16_output = true;
                    run_case(backend.get(), padded, 2, 5, 6, 7, true, f32_input, unavailable, false, true);
                }
            }
            if (!unavailable) {
                auto tiled = c;
                tiled.in_channels = 256; tiled.out_channels = 2048;
                tiled.kernel_depth = tiled.kernel_height = tiled.kernel_width = 3;
                tiled.stride_depth = tiled.stride_height = tiled.stride_width = 1;
                tiled.dilation_depth = tiled.dilation_height = tiled.dilation_width = 1;
                tiled.padding_depth = tiled.padding_height = tiled.padding_width = 1;
                tiled.cuda_replicate_depth_left = 2;
                tiled.cuda_f16_output = true;
                run_case(backend.get(), tiled, 2, 9, 32, 32, true, true, false, false, true, true);
            }
            for (bool f16_output : {false, true}) {
                auto packed = c;
                packed.cuda_f16_output = f16_output;
                for (bool f32_input : {false, true}) {
                    run_case(backend.get(), packed, 2, 5, 6, 7, false, f32_input, unavailable, false, true);
                }
            }
            auto typed = c;
            typed.cuda_cudnn = true;
            typed.cuda_f16_output = true;
            for (bool cudnn : {false, true}) {
                auto invalid_typed = typed;
                invalid_typed.cuda_cudnn = cudnn;
                invalid_typed.cuda_bf16_round = cudnn;
                bool rejected_typed = false;
                try {
                    engine::modules::Conv3dModule module(invalid_typed);
                } catch (const std::runtime_error &) {
                    rejected_typed = true;
                }
                check(rejected_typed, "invalid F16 output configuration accepted");
            }
            typed.use_bias = false;
            run_case(backend.get(), typed, 2, 5, 6, 7, false, false, unavailable);
            typed.use_bias = true;
            run_case(backend.get(), typed, 2, 5, 6, 7, false, true, unavailable);
            run_case(backend.get(), typed, 2, 5, 6, 7, true, false, unavailable);
            run_case(backend.get(), typed, 2, 5, 6, 7, true, true, unavailable);
            auto invalid = c;
            invalid.cuda_bf16_round = true;
            bool rejected = false;
            try {
                engine::modules::Conv3dModule module(invalid);
            } catch (const std::runtime_error &) {
                rejected = true;
            }
            check(rejected, "BF16 rounding accepted without cuDNN selection");
            c.cuda_bf16_round = true;
            run_case(backend.get(), c, 2, 5, 6, 7, true, true, unavailable);
            c.use_bias = false;
            run_case(backend.get(), c, 2, 5, 6, 7, false, false, unavailable);
            c.cuda_bf16_round = false;
            c.use_bias = true;
            c.kernel_depth = c.kernel_height = c.kernel_width = 1;
            c.stride_depth = c.stride_height = c.stride_width = 1;
            c.padding_depth = c.padding_height = c.padding_width = 0;
            c.dilation_depth = c.dilation_height = c.dilation_width = 1;
            run_case(backend.get(), c, 1, 3, 4, 5, true, false, unavailable);
            c.in_channels = c.out_channels = 1;
            c.use_bias = false;
            run_case(backend.get(), c, 2, 3, 4, 5, true, true, unavailable, true);
            c.use_bias = true;
            run_case(backend.get(), c, 2, 3, 4, 5, true, true, unavailable, true);
            c.cuda_bf16_round = true;
            run_case(backend.get(), c, 2, 3, 4, 5, true, true, unavailable, true);
            run_case(backend.get(), c, 2, 3, 4, 5, false, false, unavailable, true);
            c.use_bias = false;
            run_case(backend.get(), c, 2, 3, 4, 5, true, true, unavailable, true);
            typed = c;
            typed.cuda_cudnn = true;
            typed.cuda_bf16_round = false;
            typed.cuda_f16_output = true;
            run_case(backend.get(), typed, 2, 3, 4, 5, true, true, unavailable, true);
            typed.use_bias = true;
            run_case(backend.get(), typed, 2, 3, 4, 5, true, false, unavailable, true);
            std::cout << "PASS: cuDNN Conv3D contracts" << (unavailable ? " (disabled build)" : " and scalar parity") << '\n';
        }
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        result = 1;
    }
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
    return result;
}
