#include "engine/framework/core/backend.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t kGraphBytes = 512 * 1024 * 1024;
constexpr size_t kGraphNodes = 16384;
constexpr int kWarmupRounds = 1;
constexpr int kMeasureRounds = 5;

struct DiffStats {
    float max_abs = 0.0f;
    double mean_abs = 0.0;
    double cosine = 1.0;
};

struct RunResult {
    bool supported = false;
    std::string error;
    engine::core::TensorShape shape = {};
    std::vector<float> values;
    double avg_ms = 0.0;
};

std::vector<float> make_patterned_f32(size_t count, float phase, float scale) {
    std::vector<float> values(count, 0.0f);
    for (size_t i = 0; i < count; ++i) {
        const float x = static_cast<float>(i);
        values[i] = scale * (
            std::sin(phase + 0.113f * x) +
            0.5f * std::cos(phase * 0.7f + 0.071f * x));
    }
    return values;
}

int64_t conv_out(int64_t input, int64_t kernel, int stride, int padding, int dilation) {
    return (input + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
}

int64_t conv_transpose_out(int64_t input, int64_t kernel, int stride, int padding, int dilation) {
    return (input - 1) * stride - 2 * padding + dilation * (kernel - 1) + 1;
}

const char * backend_name(engine::core::BackendType backend_type) {
    switch (backend_type) {
        case engine::core::BackendType::Cpu: return "cpu";
        case engine::core::BackendType::Cuda: return "cuda";
#ifdef ENGINE_TEST_ENABLE_HIP
        case engine::core::BackendType::Hip: return "hip";
#endif
        case engine::core::BackendType::Vulkan: return "vulkan";
        case engine::core::BackendType::Metal: return "metal";
        default: return "unknown";
    }
}

bool same_shape(const engine::core::TensorShape & lhs, const engine::core::TensorShape & rhs) {
    if (lhs.rank != rhs.rank) {
        return false;
    }
    for (size_t i = 0; i < lhs.rank; ++i) {
        if (lhs.dims[i] != rhs.dims[i]) {
            return false;
        }
    }
    return true;
}

DiffStats diff_values(const std::vector<float> & reference, const std::vector<float> & actual) {
    if (reference.size() != actual.size()) {
        throw std::runtime_error("value count mismatch");
    }
    DiffStats stats;
    double dot = 0.0;
    double ref_norm = 0.0;
    double actual_norm = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const float diff = std::fabs(reference[i] - actual[i]);
        stats.max_abs = std::max(stats.max_abs, diff);
        stats.mean_abs += diff;
        dot += static_cast<double>(reference[i]) * static_cast<double>(actual[i]);
        ref_norm += static_cast<double>(reference[i]) * static_cast<double>(reference[i]);
        actual_norm += static_cast<double>(actual[i]) * static_cast<double>(actual[i]);
    }
    stats.mean_abs /= static_cast<double>(reference.size());
    if (ref_norm > 0.0 && actual_norm > 0.0) {
        stats.cosine = dot / (std::sqrt(ref_norm) * std::sqrt(actual_norm));
    }
    return stats;
}

engine::core::TensorValue add_bias_3d(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & output,
    int64_t channels,
    const std::optional<engine::core::TensorValue> & bias) {
    if (!bias.has_value()) {
        return output;
    }
    auto output_contiguous = engine::core::ensure_backend_addressable_layout(ctx, output);
    auto bias_view = engine::core::reshape_tensor(ctx, *bias, engine::core::TensorShape::from_dims({1, channels, 1}));
    auto repeated = engine::core::wrap_tensor(
        ggml_repeat(ctx.ggml, bias_view.tensor, output_contiguous.tensor),
        output.shape,
        GGML_TYPE_F32);
    return engine::core::wrap_tensor(ggml_add(ctx.ggml, output_contiguous.tensor, repeated.tensor), output.shape, GGML_TYPE_F32);
}

engine::core::TensorValue add_bias_4d(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & output,
    int64_t channels,
    const std::optional<engine::core::TensorValue> & bias) {
    if (!bias.has_value()) {
        return output;
    }
    auto output_contiguous = engine::core::ensure_backend_addressable_layout(ctx, output);
    auto bias_view = engine::core::reshape_tensor(ctx, *bias, engine::core::TensorShape::from_dims({1, channels, 1, 1}));
    auto repeated = engine::core::wrap_tensor(
        ggml_repeat(ctx.ggml, bias_view.tensor, output_contiguous.tensor),
        output.shape,
        GGML_TYPE_F32);
    return engine::core::wrap_tensor(ggml_add(ctx.ggml, output_contiguous.tensor, repeated.tensor), output.shape, GGML_TYPE_F32);
}

engine::core::TensorValue view_batch_matrix(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    int64_t batch_index,
    int64_t channels,
    int64_t frames) {
    auto * view = ggml_view_2d(
        ctx.ggml,
        input.tensor,
        frames,
        channels,
        input.tensor->nb[1],
        static_cast<size_t>(batch_index) * input.tensor->nb[2]);
    return engine::core::wrap_tensor(view, engine::core::TensorShape::from_dims({channels, frames}), input.type);
}

class GraphRunner {
public:
    GraphRunner(const char * name, engine::core::BackendType backend_type) : backend_type_(backend_type) {
        backend_ = engine::core::init_backend({backend_type, 0, 8});
        engine::core::set_backend_threads(backend_, 8);
        ggml_init_params params{};
        params.mem_size = kGraphBytes;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml_ = ggml_init(params);
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize ggml test context");
        }
        ctx_.ggml = ggml_;
        ctx_.module_instance_name = name;
        ctx_.backend_type = backend_type;
    }

    ~GraphRunner() {
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
        if (ggml_ != nullptr) {
            ggml_free(ggml_);
        }
        if (backend_ != nullptr) {
            ggml_backend_free(backend_);
        }
    }

    engine::core::TensorValue make_f32(const engine::core::TensorShape & shape) {
        return engine::core::make_tensor(ctx_, GGML_TYPE_F32, shape);
    }

    engine::core::ModuleBuildContext & ctx() noexcept { return ctx_; }

    RunResult run(
        const engine::core::TensorValue & output,
        const std::vector<std::pair<engine::core::TensorValue, std::vector<float>>> & writes) {
        ggml_cgraph * graph = ggml_new_graph_custom(ggml_, kGraphNodes, false);
        ggml_build_forward_expand(graph, output.tensor);
        engine::core::validate_backend_graph_supported(backend_, graph, "conv_lowering_matrix");
        buffer_ = ggml_backend_alloc_ctx_tensors(ggml_, backend_);
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate backend tensors");
        }
        for (const auto & write : writes) {
            engine::core::write_tensor_f32(write.first, write.second);
        }
        for (int i = 0; i < kWarmupRounds; ++i) {
            if (ggml_backend_graph_compute(backend_, graph) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("warmup graph compute failed");
            }
        }
        double total_ms = 0.0;
        for (int i = 0; i < kMeasureRounds; ++i) {
            const auto start = std::chrono::steady_clock::now();
            if (ggml_backend_graph_compute(backend_, graph) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("graph compute failed");
            }
            const auto end = std::chrono::steady_clock::now();
            total_ms += std::chrono::duration<double, std::milli>(end - start).count();
        }
        RunResult result;
        result.supported = true;
        result.shape = output.shape;
        result.avg_ms = total_ms / static_cast<double>(kMeasureRounds);
        engine::core::read_tensor_f32_into(output.tensor, result.values);
        return result;
    }

private:
    engine::core::BackendType backend_type_;
    ggml_backend_t backend_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_context * ggml_ = nullptr;
    engine::core::ModuleBuildContext ctx_{};
};

template <typename Fn>
RunResult run_guarded(
    const char * label,
    engine::core::BackendType backend_type,
    Fn && fn) {
    try {
        GraphRunner runner(label, backend_type);
        return fn(runner);
    } catch (const std::exception & ex) {
        RunResult result;
        result.supported = false;
        result.error = ex.what();
        return result;
    }
}

struct Conv1dCase {
    const char * name;
    int64_t batch;
    int64_t in_channels;
    int64_t out_channels;
    int64_t frames;
    int64_t kernel;
    int stride;
    int padding;
    int dilation;
    bool bias;
};

RunResult run_conv1d(const Conv1dCase & c, const char * candidate, engine::core::BackendType backend_type) {
    return run_guarded(candidate, backend_type, [&](GraphRunner & runner) {
        const auto input_shape = engine::core::TensorShape::from_dims({c.batch, c.in_channels, c.frames});
        const auto weight_shape = engine::core::TensorShape::from_dims({c.out_channels, c.in_channels, c.kernel});
        const auto bias_shape = engine::core::TensorShape::from_dims({c.out_channels});
        auto input = runner.make_f32(input_shape);
        auto weight = runner.make_f32(weight_shape);
        std::optional<engine::core::TensorValue> bias = c.bias ? std::optional<engine::core::TensorValue>(runner.make_f32(bias_shape)) : std::nullopt;

        engine::core::TensorValue output;
        if (std::string(candidate) == "native") {
            const auto output_shape = engine::core::TensorShape::from_dims(
                {c.batch, c.out_channels, conv_out(c.frames, c.kernel, c.stride, c.padding, c.dilation)});
            if (c.batch == 1) {
                output = engine::core::wrap_tensor(
                    ggml_conv_1d(runner.ctx().ggml, weight.tensor, input.tensor, c.stride, c.padding, c.dilation),
                    output_shape,
                    GGML_TYPE_F32);
            } else {
                for (int64_t batch = 0; batch < c.batch; ++batch) {
                    auto batch_input = view_batch_matrix(runner.ctx(), input, batch, c.in_channels, c.frames);
                    auto batch_output = engine::core::wrap_tensor(
                        ggml_conv_1d(runner.ctx().ggml, weight.tensor, batch_input.tensor, c.stride, c.padding, c.dilation),
                        engine::core::TensorShape::from_dims({1, c.out_channels, output_shape.dims[2]}),
                        GGML_TYPE_F32);
                    output = output.valid() ? engine::modules::ConcatModule({0}).build(runner.ctx(), output, batch_output) : batch_output;
                }
            }
            output = add_bias_3d(runner.ctx(), output, c.out_channels, bias);
        } else if (std::string(candidate) == "conv2d_normal") {
            auto x4 = engine::core::reshape_tensor(runner.ctx(), input, engine::core::TensorShape::from_dims({c.batch, c.in_channels, 1, c.frames}));
            auto w4 = engine::core::reshape_tensor(runner.ctx(), weight, engine::core::TensorShape::from_dims({c.out_channels, c.in_channels, 1, c.kernel}));
            auto y4 = engine::core::wrap_tensor(
                ggml_conv_2d(runner.ctx().ggml, w4.tensor, x4.tensor, c.stride, 1, c.padding, 0, c.dilation, 1),
                engine::core::TensorShape::from_dims({c.batch, c.out_channels, 1, conv_out(c.frames, c.kernel, c.stride, c.padding, c.dilation)}),
                GGML_TYPE_F32);
            y4 = add_bias_4d(runner.ctx(), y4, c.out_channels, bias);
            output = engine::core::reshape_tensor(runner.ctx(), y4, engine::core::TensorShape::from_dims({c.batch, c.out_channels, y4.shape.dims[3]}));
        } else if (std::string(candidate) == "conv2d_direct") {
            auto x4 = engine::core::reshape_tensor(runner.ctx(), input, engine::core::TensorShape::from_dims({c.batch, c.in_channels, 1, c.frames}));
            auto w4 = engine::core::reshape_tensor(runner.ctx(), weight, engine::core::TensorShape::from_dims({c.out_channels, c.in_channels, 1, c.kernel}));
            auto y4 = engine::core::wrap_tensor(
                ggml_conv_2d_direct(runner.ctx().ggml, w4.tensor, x4.tensor, c.stride, 1, c.padding, 0, c.dilation, 1),
                engine::core::TensorShape::from_dims({c.batch, c.out_channels, 1, conv_out(c.frames, c.kernel, c.stride, c.padding, c.dilation)}),
                GGML_TYPE_F32);
            y4 = add_bias_4d(runner.ctx(), y4, c.out_channels, bias);
            output = engine::core::reshape_tensor(runner.ctx(), y4, engine::core::TensorShape::from_dims({c.batch, c.out_channels, y4.shape.dims[3]}));
        } else {
            throw std::runtime_error("unknown conv1d candidate");
        }

        std::vector<std::pair<engine::core::TensorValue, std::vector<float>>> writes;
        writes.push_back({input, make_patterned_f32(static_cast<size_t>(input_shape.num_elements()), 0.19f, 0.031f)});
        writes.push_back({weight, make_patterned_f32(static_cast<size_t>(weight_shape.num_elements()), 0.47f, 0.017f)});
        if (bias) {
            writes.push_back({*bias, make_patterned_f32(static_cast<size_t>(bias_shape.num_elements()), 0.83f, 0.011f)});
        }
        return runner.run(output, writes);
    });
}

struct Conv2dCase {
    const char * name;
    int64_t batch;
    int64_t in_channels;
    int64_t out_channels;
    int64_t height;
    int64_t width;
    int64_t kernel_h;
    int64_t kernel_w;
    int stride_h;
    int stride_w;
    int padding_h;
    int padding_w;
    int dilation_h;
    int dilation_w;
    bool bias;
};

RunResult run_conv2d(const Conv2dCase & c, const char * candidate, engine::core::BackendType backend_type) {
    return run_guarded(candidate, backend_type, [&](GraphRunner & runner) {
        const auto input_shape = engine::core::TensorShape::from_dims({c.batch, c.in_channels, c.height, c.width});
        const auto weight_shape = engine::core::TensorShape::from_dims({c.out_channels, c.in_channels, c.kernel_h, c.kernel_w});
        const auto bias_shape = engine::core::TensorShape::from_dims({c.out_channels});
        auto input = runner.make_f32(input_shape);
        auto weight = runner.make_f32(weight_shape);
        std::optional<engine::core::TensorValue> bias = c.bias ? std::optional<engine::core::TensorValue>(runner.make_f32(bias_shape)) : std::nullopt;

        engine::core::TensorValue output;
        const auto output_shape = engine::core::TensorShape::from_dims({
            c.batch,
            c.out_channels,
            conv_out(c.height, c.kernel_h, c.stride_h, c.padding_h, c.dilation_h),
            conv_out(c.width, c.kernel_w, c.stride_w, c.padding_w, c.dilation_w),
        });
        if (std::string(candidate) == "im2col_matmul") {
            output = engine::core::wrap_tensor(
                ggml_conv_2d(runner.ctx().ggml, weight.tensor, input.tensor, c.stride_w, c.stride_h, c.padding_w, c.padding_h, c.dilation_w, c.dilation_h),
                output_shape,
                GGML_TYPE_F32);
            output = add_bias_4d(runner.ctx(), output, c.out_channels, bias);
        } else if (std::string(candidate) == "direct") {
            output = engine::core::wrap_tensor(
                ggml_conv_2d_direct(runner.ctx().ggml, weight.tensor, input.tensor, c.stride_w, c.stride_h, c.padding_w, c.padding_h, c.dilation_w, c.dilation_h),
                output_shape,
                GGML_TYPE_F32);
            output = add_bias_4d(runner.ctx(), output, c.out_channels, bias);
        } else {
            throw std::runtime_error("unknown conv2d candidate");
        }
        std::vector<std::pair<engine::core::TensorValue, std::vector<float>>> writes;
        writes.push_back({input, make_patterned_f32(static_cast<size_t>(input_shape.num_elements()), 0.21f, 0.021f)});
        writes.push_back({weight, make_patterned_f32(static_cast<size_t>(weight_shape.num_elements()), 0.51f, 0.013f)});
        if (bias) {
            writes.push_back({*bias, make_patterned_f32(static_cast<size_t>(bias_shape.num_elements()), 0.91f, 0.009f)});
        }
        return runner.run(output, writes);
    });
}

struct Depthwise1dCase {
    const char * name;
    int64_t batch;
    int64_t channels;
    int64_t frames;
    int64_t kernel;
    int stride;
    int padding;
    int dilation;
    bool bias;
};

RunResult run_depthwise1d(const Depthwise1dCase & c, const char * candidate, engine::core::BackendType backend_type) {
    return run_guarded(candidate, backend_type, [&](GraphRunner & runner) {
        const auto input_shape = engine::core::TensorShape::from_dims({c.batch, c.channels, c.frames});
        const auto weight_shape = engine::core::TensorShape::from_dims({c.channels, 1, c.kernel});
        const auto bias_shape = engine::core::TensorShape::from_dims({c.channels});
        auto input = runner.make_f32(input_shape);
        auto weight = runner.make_f32(weight_shape);
        std::optional<engine::core::TensorValue> bias = c.bias ? std::optional<engine::core::TensorValue>(runner.make_f32(bias_shape)) : std::nullopt;
        engine::core::TensorValue output;
        if (std::string(candidate) == "dw2d_direct") {
            auto x4 = engine::core::reshape_tensor(runner.ctx(), input, engine::core::TensorShape::from_dims({c.batch, c.channels, 1, c.frames}));
            auto w4 = engine::core::reshape_tensor(runner.ctx(), weight, engine::core::TensorShape::from_dims({c.channels, 1, 1, c.kernel}));
            auto y4 = engine::core::wrap_tensor(
                ggml_conv_2d_dw_direct(runner.ctx().ggml, w4.tensor, x4.tensor, c.stride, 1, c.padding, 0, c.dilation, 1),
                engine::core::TensorShape::from_dims({c.batch, c.channels, 1, conv_out(c.frames, c.kernel, c.stride, c.padding, c.dilation)}),
                GGML_TYPE_F32);
            y4 = add_bias_4d(runner.ctx(), y4, c.channels, bias);
            output = engine::core::reshape_tensor(runner.ctx(), y4, engine::core::TensorShape::from_dims({c.batch, c.channels, y4.shape.dims[3]}));
        } else if (std::string(candidate) == "native_1d_dw") {
            if (c.batch != 1) {
                throw std::runtime_error("native ggml_conv_1d_dw asserts for batched rank-3 input; slice batch first");
            }
            output = engine::core::wrap_tensor(
                ggml_conv_1d_dw(runner.ctx().ggml, weight.tensor, input.tensor, c.stride, c.padding, c.dilation),
                engine::core::TensorShape::from_dims({c.batch, c.channels, conv_out(c.frames, c.kernel, c.stride, c.padding, c.dilation)}),
                GGML_TYPE_F32);
            output = add_bias_3d(runner.ctx(), output, c.channels, bias);
        } else {
            throw std::runtime_error("unknown depthwise1d candidate");
        }
        std::vector<std::pair<engine::core::TensorValue, std::vector<float>>> writes;
        writes.push_back({input, make_patterned_f32(static_cast<size_t>(input_shape.num_elements()), 0.23f, 0.025f)});
        writes.push_back({weight, make_patterned_f32(static_cast<size_t>(weight_shape.num_elements()), 0.53f, 0.015f)});
        if (bias) {
            writes.push_back({*bias, make_patterned_f32(static_cast<size_t>(bias_shape.num_elements()), 0.93f, 0.007f)});
        }
        return runner.run(output, writes);
    });
}

struct Pointwise1dCase {
    const char * name;
    int64_t batch;
    int64_t in_channels;
    int64_t out_channels;
    int64_t frames;
    bool bias;
};

RunResult run_pointwise1d(const Pointwise1dCase & c, const char * candidate, engine::core::BackendType backend_type) {
    return run_guarded(candidate, backend_type, [&](GraphRunner & runner) {
        const auto input_shape = engine::core::TensorShape::from_dims({c.batch, c.in_channels, c.frames});
        const auto weight_shape = engine::core::TensorShape::from_dims({c.out_channels, c.in_channels, 1});
        const auto bias_shape = engine::core::TensorShape::from_dims({c.out_channels});
        auto input = runner.make_f32(input_shape);
        auto weight = runner.make_f32(weight_shape);
        std::optional<engine::core::TensorValue> bias = c.bias ? std::optional<engine::core::TensorValue>(runner.make_f32(bias_shape)) : std::nullopt;

        engine::core::TensorValue output;
        if (std::string(candidate) == "conv1d_kernel1") {
            const auto output_shape = engine::core::TensorShape::from_dims({c.batch, c.out_channels, c.frames});
            if (c.batch == 1) {
                output = engine::core::wrap_tensor(
                    ggml_conv_1d(runner.ctx().ggml, weight.tensor, input.tensor, 1, 0, 1),
                    output_shape,
                    GGML_TYPE_F32);
            } else {
                for (int64_t batch = 0; batch < c.batch; ++batch) {
                    auto batch_input = view_batch_matrix(runner.ctx(), input, batch, c.in_channels, c.frames);
                    auto batch_output = engine::core::wrap_tensor(
                        ggml_conv_1d(runner.ctx().ggml, weight.tensor, batch_input.tensor, 1, 0, 1),
                        engine::core::TensorShape::from_dims({1, c.out_channels, c.frames}),
                        GGML_TYPE_F32);
                    output = output.valid() ? engine::modules::ConcatModule({0}).build(runner.ctx(), output, batch_output) : batch_output;
                }
            }
            output = add_bias_3d(runner.ctx(), output, c.out_channels, bias);
        } else if (std::string(candidate) == "linear_matmul") {
            auto x = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(runner.ctx(), input);
            x = engine::core::ensure_backend_addressable_layout(runner.ctx(), x);
            auto matrix = engine::core::reshape_tensor(runner.ctx(), x, engine::core::TensorShape::from_dims({c.batch * c.frames, c.in_channels}));
            auto w2 = engine::core::reshape_tensor(runner.ctx(), weight, engine::core::TensorShape::from_dims({c.out_channels, c.in_channels}));
            auto projected = engine::core::wrap_tensor(
                ggml_mul_mat(runner.ctx().ggml, w2.tensor, matrix.tensor),
                engine::core::TensorShape::from_dims({c.batch * c.frames, c.out_channels}),
                GGML_TYPE_F32);
            if (bias) {
                projected = engine::core::wrap_tensor(ggml_add(runner.ctx().ggml, projected.tensor, bias->tensor), projected.shape, GGML_TYPE_F32);
            }
            auto y = engine::core::reshape_tensor(runner.ctx(), projected, engine::core::TensorShape::from_dims({c.batch, c.frames, c.out_channels}));
            output = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(runner.ctx(), y);
        } else {
            throw std::runtime_error("unknown pointwise1d candidate");
        }
        std::vector<std::pair<engine::core::TensorValue, std::vector<float>>> writes;
        writes.push_back({input, make_patterned_f32(static_cast<size_t>(input_shape.num_elements()), 0.24f, 0.027f)});
        writes.push_back({weight, make_patterned_f32(static_cast<size_t>(weight_shape.num_elements()), 0.54f, 0.014f)});
        if (bias) {
            writes.push_back({*bias, make_patterned_f32(static_cast<size_t>(bias_shape.num_elements()), 0.94f, 0.007f)});
        }
        return runner.run(output, writes);
    });
}

RunResult run_depthwise2d(const Conv2dCase & c, const char * candidate, engine::core::BackendType backend_type) {
    return run_guarded(candidate, backend_type, [&](GraphRunner & runner) {
        const auto input_shape = engine::core::TensorShape::from_dims({c.batch, c.in_channels, c.height, c.width});
        const auto weight_shape = engine::core::TensorShape::from_dims({c.in_channels, 1, c.kernel_h, c.kernel_w});
        const auto bias_shape = engine::core::TensorShape::from_dims({c.in_channels});
        auto input = runner.make_f32(input_shape);
        auto weight = runner.make_f32(weight_shape);
        std::optional<engine::core::TensorValue> bias = c.bias ? std::optional<engine::core::TensorValue>(runner.make_f32(bias_shape)) : std::nullopt;
        const auto output_shape = engine::core::TensorShape::from_dims({
            c.batch,
            c.in_channels,
            conv_out(c.height, c.kernel_h, c.stride_h, c.padding_h, c.dilation_h),
            conv_out(c.width, c.kernel_w, c.stride_w, c.padding_w, c.dilation_w),
        });
        engine::core::TensorValue output;
        if (std::string(candidate) == "direct") {
            output = engine::core::wrap_tensor(
                ggml_conv_2d_dw_direct(runner.ctx().ggml, weight.tensor, input.tensor, c.stride_w, c.stride_h, c.padding_w, c.padding_h, c.dilation_w, c.dilation_h),
                output_shape,
                GGML_TYPE_F32);
            output = add_bias_4d(runner.ctx(), output, c.in_channels, bias);
        } else if (std::string(candidate) == "im2col_matmul") {
            output = engine::core::wrap_tensor(
                ggml_conv_2d_dw(runner.ctx().ggml, weight.tensor, input.tensor, c.stride_w, c.stride_h, c.padding_w, c.padding_h, c.dilation_w, c.dilation_h),
                output_shape,
                GGML_TYPE_F32);
            output = add_bias_4d(runner.ctx(), output, c.in_channels, bias);
        } else {
            throw std::runtime_error("unknown depthwise2d candidate");
        }
        std::vector<std::pair<engine::core::TensorValue, std::vector<float>>> writes;
        writes.push_back({input, make_patterned_f32(static_cast<size_t>(input_shape.num_elements()), 0.25f, 0.023f)});
        writes.push_back({weight, make_patterned_f32(static_cast<size_t>(weight_shape.num_elements()), 0.55f, 0.012f)});
        if (bias) {
            writes.push_back({*bias, make_patterned_f32(static_cast<size_t>(bias_shape.num_elements()), 0.95f, 0.008f)});
        }
        return runner.run(output, writes);
    });
}

struct ConvTranspose1dCase {
    const char * name;
    int64_t batch;
    int64_t in_channels;
    int64_t out_channels;
    int64_t frames;
    int64_t kernel;
    int stride;
    int padding;
    int dilation;
    bool bias;
};

engine::core::TensorValue build_conv_transpose_native(
    engine::core::ModuleBuildContext & ctx,
    const ConvTranspose1dCase & c,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & weight,
    const std::optional<engine::core::TensorValue> & bias) {
    if (c.padding != 0 || c.dilation != 1) {
        throw std::runtime_error("native ggml_conv_transpose_1d supports only padding=0 and dilation=1");
    }
    engine::core::TensorValue output;
    for (int64_t batch = 0; batch < c.batch; ++batch) {
        auto matrix = view_batch_matrix(ctx, input, batch, c.in_channels, c.frames);
        auto batch_out = engine::core::wrap_tensor(
            ggml_conv_transpose_1d(ctx.ggml, weight.tensor, matrix.tensor, c.stride, c.padding, c.dilation),
            engine::core::TensorShape::from_dims({1, c.out_channels, conv_transpose_out(c.frames, c.kernel, c.stride, c.padding, c.dilation)}),
            GGML_TYPE_F32);
        output = output.valid() ? engine::modules::ConcatModule({0}).build(ctx, output, batch_out) : batch_out;
    }
    return add_bias_3d(ctx, output, c.out_channels, bias);
}

engine::core::TensorValue build_conv_transpose_col2im(
    engine::core::ModuleBuildContext & ctx,
    const ConvTranspose1dCase & c,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & weight,
    const std::optional<engine::core::TensorValue> & bias) {
    if (c.dilation != 1) {
        throw std::runtime_error("col2im lowering currently supports only dilation=1");
    }
    auto * weight_perm = ggml_reshape_2d(
        ctx.ggml,
        ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, weight.tensor, 1, 2, 0, 3)),
        c.in_channels,
        c.kernel * c.out_channels);
    ggml_tensor * bias_matrix = nullptr;
    if (c.bias) {
        if (!bias.has_value()) {
            throw std::runtime_error("missing bias");
        }
        bias_matrix = ggml_reshape_2d(ctx.ggml, bias->tensor, 1, c.out_channels);
    }
    engine::core::TensorValue output;
    for (int64_t batch = 0; batch < c.batch; ++batch) {
        auto * batch_input = ggml_view_2d(
            ctx.ggml,
            input.tensor,
            input.tensor->ne[0],
            input.tensor->ne[1],
            input.tensor->nb[1],
            static_cast<size_t>(batch) * input.tensor->nb[2]);
        auto * transposed_input = ggml_cont(ctx.ggml, ggml_transpose(ctx.ggml, batch_input));
        auto * columns = ggml_mul_mat(ctx.ggml, weight_perm, transposed_input);
        auto * batch_output = ggml_col2im_1d(ctx.ggml, columns, c.stride, static_cast<int>(c.out_channels), c.padding);
        if (bias_matrix != nullptr) {
            batch_output = ggml_add(ctx.ggml, batch_output, bias_matrix);
        }
        auto batch_value = engine::core::wrap_tensor(
            ggml_reshape_3d(ctx.ggml, batch_output, batch_output->ne[0], batch_output->ne[1], 1),
            engine::core::TensorShape::from_dims({1, c.out_channels, batch_output->ne[0]}),
            GGML_TYPE_F32);
        output = output.valid() ? engine::modules::ConcatModule({0}).build(ctx, output, batch_value) : batch_value;
    }
    return output;
}

RunResult run_conv_transpose1d(const ConvTranspose1dCase & c, const char * candidate, engine::core::BackendType backend_type) {
    if (backend_type == engine::core::BackendType::Cpu && std::string(candidate) == "matmul_col2im") {
        RunResult result;
        result.supported = false;
        result.error = "current ggml CPU backend aborts for COL2IM_1D";
        return result;
    }
    return run_guarded(candidate, backend_type, [&](GraphRunner & runner) {
        const auto input_shape = engine::core::TensorShape::from_dims({c.batch, c.in_channels, c.frames});
        const auto weight_shape = engine::core::TensorShape::from_dims({c.in_channels, c.out_channels, c.kernel});
        const auto bias_shape = engine::core::TensorShape::from_dims({c.out_channels});
        auto input = runner.make_f32(input_shape);
        auto weight = runner.make_f32(weight_shape);
        std::optional<engine::core::TensorValue> bias = c.bias ? std::optional<engine::core::TensorValue>(runner.make_f32(bias_shape)) : std::nullopt;
        engine::core::TensorValue output;
        if (std::string(candidate) == "native_direct") {
            output = build_conv_transpose_native(runner.ctx(), c, input, weight, bias);
        } else if (std::string(candidate) == "matmul_col2im") {
            output = build_conv_transpose_col2im(runner.ctx(), c, input, weight, bias);
        } else {
            throw std::runtime_error("unknown conv_transpose1d candidate");
        }
        std::vector<std::pair<engine::core::TensorValue, std::vector<float>>> writes;
        writes.push_back({input, make_patterned_f32(static_cast<size_t>(input_shape.num_elements()), 0.27f, 0.019f)});
        writes.push_back({weight, make_patterned_f32(static_cast<size_t>(weight_shape.num_elements()), 0.57f, 0.011f)});
        if (bias) {
            writes.push_back({*bias, make_patterned_f32(static_cast<size_t>(bias_shape.num_elements()), 0.97f, 0.006f)});
        }
        return runner.run(output, writes);
    });
}

struct MatrixRow {
    std::string module;
    std::string case_name;
    std::string candidate;
    engine::core::BackendType backend;
    RunResult result;
    std::optional<DiffStats> diff;
};

void print_row(const MatrixRow & row) {
    std::cout << "| " << row.module
              << " | " << row.case_name
              << " | " << row.candidate
              << " | " << backend_name(row.backend)
              << " | ";
    if (!row.result.supported) {
        std::string error = row.result.error;
        std::replace(error.begin(), error.end(), '|', '/');
        std::cout << "unsupported | - | - | - | - | " << error << " |\n";
        return;
    }
    std::cout << "ok | " << row.result.shape.to_string()
              << " | " << std::fixed << std::setprecision(4) << row.result.avg_ms
              << " | ";
    if (row.diff.has_value()) {
        std::cout << std::scientific << std::setprecision(3) << row.diff->max_abs
                  << " | " << row.diff->mean_abs
                  << " | " << std::fixed << std::setprecision(9) << row.diff->cosine << " |\n";
    } else {
        std::cout << "- | - | - |\n";
    }
}

void add_result(
    std::vector<MatrixRow> & rows,
    const std::string & module,
    const std::string & case_name,
    const std::string & candidate,
    engine::core::BackendType backend,
    const RunResult & result,
    const RunResult & reference) {
    MatrixRow row{module, case_name, candidate, backend, result, std::nullopt};
    if (result.supported && reference.supported) {
        if (!same_shape(reference.shape, result.shape)) {
            row.result.supported = false;
            row.result.error = "shape mismatch vs reference " + reference.shape.to_string();
        } else {
            row.diff = diff_values(reference.values, result.values);
        }
    }
    rows.push_back(std::move(row));
}

bool backend_available(engine::core::BackendType backend_type) {
    try {
        GraphRunner runner("conv_lowering_matrix.probe", backend_type);
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace

int main() {
    try {
        std::vector<engine::core::BackendType> backends = {engine::core::BackendType::Cpu};
        if (backend_available(engine::core::BackendType::Cuda)) {
            backends.push_back(engine::core::BackendType::Cuda);
        } else {
            std::cout << "[SKIP] cuda backend unavailable\n";
        }
#ifdef ENGINE_TEST_ENABLE_HIP
        if (backend_available(engine::core::BackendType::Hip)) {
            backends.push_back(engine::core::BackendType::Hip);
        } else {
            std::cout << "[SKIP] hip backend unavailable\n";
        }
#endif
        if (backend_available(engine::core::BackendType::Vulkan)) {
            backends.push_back(engine::core::BackendType::Vulkan);
        } else {
            std::cout << "[SKIP] vulkan backend unavailable\n";
        }

        std::vector<MatrixRow> rows;

        const std::vector<Conv1dCase> conv1d_cases = {
            {"citrinet_like_large_regular", 1, 80, 256, 256, 11, 1, 5, 1, true},
            {"bigvgan_like_resblock", 1, 192, 192, 384, 7, 1, 3, 1, true},
            {"batched_stride_regular", 2, 64, 128, 160, 5, 2, 2, 1, true},
            {"dilated_regular", 1, 128, 128, 192, 3, 1, 2, 2, false},
        };
        const std::vector<std::string> conv1d_candidates = {"native", "conv2d_normal", "conv2d_direct"};
        for (const auto & c : conv1d_cases) {
            const auto reference = run_conv1d(c, "native", engine::core::BackendType::Cpu);
            for (const auto backend : backends) {
                for (const auto & candidate : conv1d_candidates) {
                    add_result(rows, "Conv1dModule", c.name, candidate, backend, run_conv1d(c, candidate.c_str(), backend), reference);
                }
            }
        }

        const std::vector<Conv2dCase> conv2d_cases = {
            {"spectrogram_small_kernel", 1, 64, 128, 20, 160, 3, 3, 1, 1, 1, 1, 1, 1, true},
            {"conv1d_lowered_shape", 1, 256, 256, 1, 384, 1, 7, 1, 1, 0, 3, 1, 1, true},
            {"batched_feature_map", 2, 32, 64, 12, 96, 3, 5, 1, 2, 1, 2, 1, 1, false},
        };
        const std::vector<std::string> conv2d_candidates = {"im2col_matmul", "direct"};
        for (const auto & c : conv2d_cases) {
            const auto reference = run_conv2d(c, "im2col_matmul", engine::core::BackendType::Cpu);
            for (const auto backend : backends) {
                for (const auto & candidate : conv2d_candidates) {
                    add_result(rows, "Conv2dModule", c.name, candidate, backend, run_conv2d(c, candidate.c_str(), backend), reference);
                }
            }
        }

        const std::vector<Depthwise1dCase> depthwise1d_cases = {
            {"conformer_like_depthwise", 1, 256, 192, 31, 1, 15, 1, true},
            {"tokenizer_stride_depthwise", 2, 96, 256, 7, 2, 3, 1, true},
            {"dilated_depthwise", 1, 128, 160, 5, 1, 4, 2, false},
        };
        const std::vector<std::string> depthwise1d_candidates = {"dw2d_direct", "native_1d_dw"};
        for (const auto & c : depthwise1d_cases) {
            const auto reference = run_depthwise1d(c, "dw2d_direct", engine::core::BackendType::Cpu);
            for (const auto backend : backends) {
                for (const auto & candidate : depthwise1d_candidates) {
                    add_result(rows, "DepthwiseConv1dModule", c.name, candidate, backend, run_depthwise1d(c, candidate.c_str(), backend), reference);
                }
            }
        }

        const std::vector<Pointwise1dCase> pointwise1d_cases = {
            {"conformer_projection", 1, 256, 512, 192, true},
            {"batched_token_projection", 2, 192, 384, 160, true},
            {"vocoder_channel_mix", 1, 192, 192, 384, false},
        };
        const std::vector<std::string> pointwise1d_candidates = {"conv1d_kernel1", "linear_matmul"};
        for (const auto & c : pointwise1d_cases) {
            const auto reference = run_pointwise1d(c, "conv1d_kernel1", engine::core::BackendType::Cpu);
            for (const auto backend : backends) {
                for (const auto & candidate : pointwise1d_candidates) {
                    add_result(rows, "PointwiseConv1dModule", c.name, candidate, backend, run_pointwise1d(c, candidate.c_str(), backend), reference);
                }
            }
        }

        const std::vector<Conv2dCase> depthwise2d_cases = {
            {"depthwise_1d_lowered_shape", 1, 192, 192, 1, 384, 1, 7, 1, 1, 0, 3, 1, 1, true},
            {"image_depthwise_small", 1, 64, 64, 24, 80, 3, 3, 1, 1, 1, 1, 1, 1, true},
        };
        const std::vector<std::string> depthwise2d_candidates = {"direct", "im2col_matmul"};
        for (const auto & c : depthwise2d_cases) {
            const auto reference = run_depthwise2d(c, "direct", engine::core::BackendType::Cpu);
            for (const auto backend : backends) {
                for (const auto & candidate : depthwise2d_candidates) {
                    add_result(rows, "DepthwiseConv2dModule", c.name, candidate, backend, run_depthwise2d(c, candidate.c_str(), backend), reference);
                }
            }
        }

        const std::vector<ConvTranspose1dCase> conv_transpose_cases = {
            {"qwen3_like_stride5_padding0", 1, 256, 128, 96, 10, 5, 0, 1, true},
            {"vocoder_stride2_padding1", 1, 192, 96, 192, 4, 2, 1, 1, true},
            {"batched_stride2_no_bias", 2, 128, 128, 96, 2, 2, 0, 1, false},
            {"dilated_unsupported_probe", 1, 64, 64, 96, 3, 2, 0, 2, true},
        };
        const std::vector<std::string> conv_transpose_candidates = {"native_direct", "matmul_col2im"};
        for (const auto & c : conv_transpose_cases) {
            const auto reference = run_conv_transpose1d(c, "native_direct", engine::core::BackendType::Cpu);
            for (const auto backend : backends) {
                for (const auto & candidate : conv_transpose_candidates) {
                    add_result(rows, "ConvTranspose1dModule", c.name, candidate, backend, run_conv_transpose1d(c, candidate.c_str(), backend), reference);
                }
            }
        }

        std::cout << "| module | case | candidate | backend | status | shape | avg_ms | max_abs_vs_cpu_ref | mean_abs_vs_cpu_ref | cosine_vs_cpu_ref |\n";
        std::cout << "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |\n";
        for (const auto & row : rows) {
            print_row(row);
        }
    } catch (const std::exception & ex) {
        std::cerr << "[FAIL] " << ex.what() << '\n';
        return 1;
    }
    return 0;
}
