#include "engine/framework/core/backend.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t kGraphBytes = 1024ull * 1024ull * 1024ull;
constexpr size_t kGraphNodes = 65536;
constexpr int64_t kBatch = 2;
constexpr int64_t kChannels = 256;
constexpr int64_t kFrames = 96;
constexpr int64_t kHidden = 512;
constexpr int64_t kKernel = 5;
constexpr int kTimedRuns = 6;
constexpr int kProfileTimedRuns = 3;

struct RunResult {
    engine::core::TensorShape shape;
    std::vector<float> values;
    double warm_ms = 0.0;
    double mean_ms = 0.0;
};

std::vector<float> make_patterned_f32(size_t count, float phase, float scale) {
    std::vector<float> values(count, 0.0f);
    for (size_t i = 0; i < count; ++i) {
        const float x = static_cast<float>(i);
        values[i] = scale * (std::sin(phase + 0.113f * x) + 0.5f * std::cos(phase * 0.7f + 0.071f * x));
    }
    return values;
}

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void write_tensor_f32(const engine::core::TensorValue & tensor, const std::vector<float> & values, const char * name) {
    if (tensor.shape.num_elements() != static_cast<int64_t>(values.size())) {
        std::ostringstream oss;
        oss << name << " value count mismatch: expected " << tensor.shape.num_elements()
            << ", got " << values.size();
        throw std::runtime_error(oss.str());
    }
    ggml_backend_tensor_set(tensor.tensor, values.data(), 0, values.size() * sizeof(float));
}

void require_close(
    const RunResult & original,
    const RunResult & exp,
    float max_allowed,
    double mean_allowed) {
    require(original.shape.rank == exp.shape.rank, "shape rank mismatch");
    for (size_t i = 0; i < original.shape.rank; ++i) {
        require(original.shape.dims[i] == exp.shape.dims[i], "shape dimension mismatch");
    }
    require(original.values.size() == exp.values.size(), "value size mismatch");

    float max_diff = 0.0f;
    size_t max_index = 0;
    double mean_diff = 0.0;
    for (size_t i = 0; i < original.values.size(); ++i) {
        const float diff = std::fabs(original.values[i] - exp.values[i]);
        mean_diff += diff;
        if (diff > max_diff) {
            max_diff = diff;
            max_index = i;
        }
    }
    mean_diff /= static_cast<double>(original.values.size());
    if (max_diff > max_allowed || mean_diff > mean_allowed) {
        std::ostringstream oss;
        oss << "exp graph drift exceeds bounds: max diff=" << max_diff
            << " at " << max_index << " original=" << original.values[max_index]
            << " exp=" << exp.values[max_index] << ", mean diff=" << mean_diff;
        throw std::runtime_error(oss.str());
    }
}

engine::core::TensorValue repeat_to(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & value,
    const engine::core::TensorShape & shape) {
    return engine::modules::RepeatModule({shape}).build(ctx, value);
}

engine::core::TensorValue edge_pad_time(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    int64_t left,
    int64_t right) {
    auto out = input;
    if (left > 0) {
        auto edge = engine::modules::SliceModule({2, 0, 1}).build(ctx, input);
        edge = repeat_to(ctx, edge, engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], left}));
        out = engine::modules::ConcatModule({2}).build(ctx, edge, out);
    }
    if (right > 0) {
        auto edge = engine::modules::SliceModule({2, input.shape.dims[2] - 1, 1}).build(ctx, input);
        edge = repeat_to(ctx, edge, engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], right}));
        out = engine::modules::ConcatModule({2}).build(ctx, out, edge);
    }
    return out;
}

engine::core::TensorValue add(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & lhs,
    const engine::core::TensorValue & rhs) {
    return engine::modules::AddModule().build(ctx, lhs, rhs);
}

engine::core::TensorValue mul(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & lhs,
    const engine::core::TensorValue & rhs) {
    if (lhs.shape.rank == rhs.shape.rank && lhs.shape.dims == rhs.shape.dims) {
        return engine::modules::MulModule().build(ctx, lhs, rhs);
    }
    auto rhs_broadcast = repeat_to(ctx, rhs, lhs.shape);
    return engine::modules::MulModule().build(ctx, lhs, rhs_broadcast);
}

engine::core::TensorValue depthwise_conv1d(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & weight,
    const engine::core::TensorValue & bias,
    int dilation) {
    return engine::modules::DepthwiseConv1dModule({
        input.shape.dims[1],
        weight.shape.dims[2],
        1,
        0,
        dilation,
        true,
    }).build(ctx, input, {weight, bias});
}

engine::core::TensorValue depthwise_conv1d_sliced_batch(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & weight,
    const engine::core::TensorValue & bias,
    int dilation) {
    if (input.shape.dims[0] == 1) {
        return depthwise_conv1d(ctx, input, weight, bias, dilation);
    }
    engine::core::TensorValue output;
    for (int64_t batch = 0; batch < input.shape.dims[0]; ++batch) {
        auto slice = engine::modules::SliceModule({0, batch, 1}).build(ctx, input);
        auto batch_output = depthwise_conv1d(ctx, slice, weight, bias, dilation);
        output = output.valid() ? engine::modules::ConcatModule({0}).build(ctx, output, batch_output) : batch_output;
    }
    return output;
}

engine::core::TensorValue shifted_clamped_time(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    int64_t offset) {
    const int64_t frames = input.shape.dims[2];
    if (offset == 0) {
        return input;
    }
    if (offset < 0) {
        const int64_t prefix_frames = -offset;
        auto edge = engine::modules::SliceModule({2, 0, 1}).build(ctx, input);
        edge = repeat_to(ctx, edge, engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], prefix_frames}));
        auto body = engine::modules::SliceModule({2, 0, frames - prefix_frames}).build(ctx, input);
        return engine::modules::ConcatModule({2}).build(ctx, edge, body);
    }
    auto body = engine::modules::SliceModule({2, offset, frames - offset}).build(ctx, input);
    auto edge = engine::modules::SliceModule({2, frames - 1, 1}).build(ctx, input);
    edge = repeat_to(ctx, edge, engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], offset}));
    return engine::modules::ConcatModule({2}).build(ctx, body, edge);
}

engine::core::TensorValue depthwise_conv1d_shift_sum(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & weight,
    const engine::core::TensorValue & bias,
    int dilation) {
    engine::core::TensorValue output;
    for (int64_t tap = 0; tap < weight.shape.dims[2]; ++tap) {
        const int64_t offset = (tap - weight.shape.dims[2] / 2) * static_cast<int64_t>(dilation);
        auto shifted = shifted_clamped_time(ctx, input, offset);
        auto tap_weight = engine::modules::SliceModule({2, tap, 1}).build(ctx, weight);
        tap_weight = engine::core::ensure_backend_addressable_layout(ctx, tap_weight);
        tap_weight = engine::core::reshape_tensor(ctx, tap_weight, engine::core::TensorShape::from_dims({1, weight.shape.dims[0], 1}));
        auto contribution = mul(ctx, shifted, tap_weight);
        output = output.valid() ? add(ctx, output, contribution) : contribution;
    }
    auto bias_view = engine::core::reshape_tensor(ctx, bias, engine::core::TensorShape::from_dims({1, bias.shape.dims[0], 1}));
    return add(ctx, output, repeat_to(ctx, bias_view, output.shape));
}

engine::core::TensorValue vector_conv1(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & value,
    const engine::core::TensorValue & weight,
    const engine::core::TensorValue & bias) {
    auto value_btc = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, value);
    value_btc = engine::core::ensure_backend_addressable_layout(ctx, value_btc);
    const auto weight_2d = engine::core::reshape_tensor(ctx, weight, engine::core::TensorShape::from_dims({weight.shape.dims[0], weight.shape.dims[1]}));
    auto projected = engine::modules::LinearModule({value.shape.dims[1], weight.shape.dims[0], true}).build(ctx, value_btc, {weight_2d, bias});
    return engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, projected);
}

engine::core::TensorValue vector_linear_btc(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & value_btc,
    const engine::core::TensorValue & weight,
    const engine::core::TensorValue & bias) {
    auto input = engine::core::ensure_backend_addressable_layout(ctx, value_btc);
    const auto weight_2d = engine::core::reshape_tensor(ctx, weight, engine::core::TensorShape::from_dims({weight.shape.dims[0], weight.shape.dims[1]}));
    return engine::modules::LinearModule({value_btc.shape.last_dim(), weight.shape.dims[0], true}).build(ctx, input, {weight_2d, bias});
}

struct ConvNextUnit {
    engine::core::TensorValue dw_weight;
    engine::core::TensorValue dw_bias;
    engine::core::TensorValue norm_weight;
    engine::core::TensorValue norm_bias;
    engine::core::TensorValue pw1_weight;
    engine::core::TensorValue pw1_bias;
    engine::core::TensorValue pw2_weight;
    engine::core::TensorValue pw2_bias;
    engine::core::TensorValue gamma;
};

struct GraphTensors {
    engine::core::TensorValue input;
    engine::core::TensorValue mask;
    std::vector<ConvNextUnit> units;
    engine::core::TensorValue output;
};

enum class DepthwiseMode {
    MergedBatch,
    SlicedBatch,
    ShiftSum,
};

class GraphRunner {
public:
    GraphRunner(const char * name, bool exp, int stop_stage = -1, DepthwiseMode depthwise_mode = DepthwiseMode::MergedBatch)
        : exp_(exp),
          stop_stage_(stop_stage),
          depthwise_mode_(depthwise_mode) {
        config_.type = engine::core::BackendType::Cpu;
        config_.threads = 8;
        backend_ = engine::core::init_backend(config_);
        engine::core::set_backend_threads(backend_, config_.threads);

        ggml_init_params params{};
        params.mem_size = kGraphBytes;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml_ = ggml_init(params);
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize ggml context");
        }
        ctx_.ggml = ggml_;
        ctx_.module_instance_name = name;
        ctx_.backend_type = engine::core::BackendType::Cpu;
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

    GraphTensors build() {
        GraphTensors tensors;
        tensors.input = make_f32({kBatch, kChannels, kFrames});
        tensors.mask = make_f32({kBatch, 1, kFrames});
        tensors.units.reserve(4);
        for (int i = 0; i < 4; ++i) {
            ConvNextUnit unit;
            unit.dw_weight = make_f32({kChannels, 1, kKernel});
            unit.dw_bias = make_f32({kChannels});
            unit.norm_weight = make_f32({kChannels});
            unit.norm_bias = make_f32({kChannels});
            unit.pw1_weight = make_f32({kHidden, kChannels, 1});
            unit.pw1_bias = make_f32({kHidden});
            unit.pw2_weight = make_f32({kChannels, kHidden, 1});
            unit.pw2_bias = make_f32({kChannels});
            unit.gamma = make_f32({1, kChannels, 1});
            tensors.units.push_back(unit);
        }
        tensors.output = exp_ ? build_exp(tensors) : build_original(tensors);
        graph_ = ggml_new_graph_custom(ggml_, kGraphNodes, false);
        ggml_build_forward_expand(graph_, tensors.output.tensor);
        buffer_ = ggml_backend_alloc_ctx_tensors(ggml_, backend_);
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate graph tensors");
        }
        fill(tensors);
        return tensors;
    }

    RunResult run(const engine::core::TensorValue & output, int timed_runs = kTimedRuns) {
        const auto warm_start = std::chrono::steady_clock::now();
        compute();
        const double warm_ms = elapsed_ms(warm_start);

        double total_ms = 0.0;
        for (int i = 0; i < timed_runs; ++i) {
            const auto start = std::chrono::steady_clock::now();
            compute();
            total_ms += elapsed_ms(start);
        }

        RunResult result;
        result.shape = output.shape;
        result.warm_ms = warm_ms;
        result.mean_ms = total_ms / static_cast<double>(timed_runs);
        engine::core::read_tensor_f32_into(output.tensor, result.values);
        return result;
    }

private:
    engine::core::TensorValue depthwise(
        const engine::core::TensorValue & input,
        const engine::core::TensorValue & weight,
        const engine::core::TensorValue & bias,
        int dilation) {
        switch (depthwise_mode_) {
            case DepthwiseMode::MergedBatch:
                return depthwise_conv1d(ctx_, input, weight, bias, dilation);
            case DepthwiseMode::SlicedBatch:
                return depthwise_conv1d_sliced_batch(ctx_, input, weight, bias, dilation);
            case DepthwiseMode::ShiftSum:
                return depthwise_conv1d_shift_sum(ctx_, input, weight, bias, dilation);
        }
        throw std::runtime_error("unknown depthwise mode");
    }

    engine::core::TensorValue maybe_stop(
        engine::core::TensorValue value,
        int & stage,
        bool & stopped) {
        if (stop_stage_ == stage) {
            stopped = true;
            return engine::core::ensure_backend_addressable_layout(ctx_, value);
        }
        ++stage;
        return {};
    }

    engine::core::TensorValue make_f32(std::initializer_list<int64_t> dims) {
        return engine::core::make_tensor(ctx_, GGML_TYPE_F32, engine::core::TensorShape::from_dims(dims));
    }

    engine::core::TensorValue build_original(const GraphTensors & tensors) {
        auto x = tensors.input;
        const int dilations[4] = {1, 2, 4, 8};
        int stage = 0;
        for (int i = 0; i < 4; ++i) {
            const auto & unit = tensors.units[static_cast<size_t>(i)];
            auto residual = mul(ctx_, x, tensors.mask);
            bool stopped = false;
            auto out = maybe_stop(residual, stage, stopped);
            if (stopped) {
                return out;
            }
            auto y = depthwise_mode_ == DepthwiseMode::ShiftSum
                ? residual
                : edge_pad_time(ctx_, residual, static_cast<int64_t>(dilations[i]) * 2, static_cast<int64_t>(dilations[i]) * 2);
            out = maybe_stop(y, stage, stopped);
            if (stopped) {
                return out;
            }
            y = depthwise(y, unit.dw_weight, unit.dw_bias, dilations[i]);
            out = maybe_stop(y, stage, stopped);
            if (stopped) {
                return out;
            }
            y = mul(ctx_, y, tensors.mask);
            out = maybe_stop(y, stage, stopped);
            if (stopped) {
                return out;
            }
            y = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, y);
            out = maybe_stop(y, stage, stopped);
            if (stopped) {
                return out;
            }
            y = engine::modules::LayerNormModule({kChannels, 1.0e-6F, true, true}).build(ctx_, y, {unit.norm_weight, unit.norm_bias});
            out = maybe_stop(y, stage, stopped);
            if (stopped) {
                return out;
            }
            y = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, y);
            out = maybe_stop(y, stage, stopped);
            if (stopped) {
                return out;
            }
            y = vector_conv1(ctx_, y, unit.pw1_weight, unit.pw1_bias);
            out = maybe_stop(y, stage, stopped);
            if (stopped) {
                return out;
            }
            y = engine::modules::GeluModule({engine::modules::GeluApproximation::ExactErf}).build(ctx_, y);
            out = maybe_stop(y, stage, stopped);
            if (stopped) {
                return out;
            }
            y = vector_conv1(ctx_, y, unit.pw2_weight, unit.pw2_bias);
            out = maybe_stop(y, stage, stopped);
            if (stopped) {
                return out;
            }
            y = mul(ctx_, y, unit.gamma);
            out = maybe_stop(y, stage, stopped);
            if (stopped) {
                return out;
            }
            x = add(ctx_, residual, y);
            out = maybe_stop(x, stage, stopped);
            if (stopped) {
                return out;
            }
        }
        return engine::core::ensure_backend_addressable_layout(ctx_, x);
    }

    engine::core::TensorValue build_exp(const GraphTensors & tensors) {
        auto x = tensors.input;
        const int dilations[4] = {1, 2, 4, 8};
        int stage = 0;
        for (int i = 0; i < 4; ++i) {
            const auto & unit = tensors.units[static_cast<size_t>(i)];
            auto residual = mul(ctx_, x, tensors.mask);
            bool stopped = false;
            auto out = maybe_stop(residual, stage, stopped);
            if (stopped) {
                return out;
            }
            auto y = depthwise_mode_ == DepthwiseMode::ShiftSum
                ? residual
                : edge_pad_time(ctx_, residual, static_cast<int64_t>(dilations[i]) * 2, static_cast<int64_t>(dilations[i]) * 2);
            out = maybe_stop(y, stage, stopped);
            if (stopped) {
                return out;
            }
            y = depthwise(y, unit.dw_weight, unit.dw_bias, dilations[i]);
            out = maybe_stop(y, stage, stopped);
            if (stopped) {
                return out;
            }
            y = mul(ctx_, y, tensors.mask);
            out = maybe_stop(y, stage, stopped);
            if (stopped) {
                return out;
            }
            auto y_btc = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, y);
            out = maybe_stop(y_btc, stage, stopped);
            if (stopped) {
                return out;
            }
            y_btc = engine::modules::LayerNormModule({kChannels, 1.0e-6F, true, true}).build(ctx_, y_btc, {unit.norm_weight, unit.norm_bias});
            out = maybe_stop(y_btc, stage, stopped);
            if (stopped) {
                return out;
            }
            y_btc = vector_linear_btc(ctx_, y_btc, unit.pw1_weight, unit.pw1_bias);
            out = maybe_stop(y_btc, stage, stopped);
            if (stopped) {
                return out;
            }
            y_btc = engine::modules::GeluModule({engine::modules::GeluApproximation::ExactErf}).build(ctx_, y_btc);
            out = maybe_stop(y_btc, stage, stopped);
            if (stopped) {
                return out;
            }
            y_btc = vector_linear_btc(ctx_, y_btc, unit.pw2_weight, unit.pw2_bias);
            out = maybe_stop(y_btc, stage, stopped);
            if (stopped) {
                return out;
            }
            y = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, y_btc);
            out = maybe_stop(y, stage, stopped);
            if (stopped) {
                return out;
            }
            y = mul(ctx_, y, unit.gamma);
            out = maybe_stop(y, stage, stopped);
            if (stopped) {
                return out;
            }
            x = add(ctx_, residual, y);
            out = maybe_stop(x, stage, stopped);
            if (stopped) {
                return out;
            }
        }
        return engine::core::ensure_backend_addressable_layout(ctx_, x);
    }

    void fill(const GraphTensors & tensors) {
        write_tensor_f32(tensors.input, make_patterned_f32(static_cast<size_t>(tensors.input.shape.num_elements()), 0.11f, 0.07f), "input");
        std::vector<float> mask(static_cast<size_t>(tensors.mask.shape.num_elements()), 1.0f);
        for (int64_t b = 0; b < kBatch; ++b) {
            for (int64_t t = kFrames - 7; t < kFrames; ++t) {
                mask[static_cast<size_t>(b * kFrames + t)] = 0.0f;
            }
        }
        write_tensor_f32(tensors.mask, mask, "mask");
        for (size_t i = 0; i < tensors.units.size(); ++i) {
            const auto & unit = tensors.units[i];
            const float phase = 0.37f + static_cast<float>(i) * 0.19f;
            write_tensor_f32(unit.dw_weight, make_patterned_f32(static_cast<size_t>(unit.dw_weight.shape.num_elements()), phase, 0.013f), "dw_weight");
            write_tensor_f32(unit.dw_bias, make_patterned_f32(static_cast<size_t>(unit.dw_bias.shape.num_elements()), phase + 0.1f, 0.003f), "dw_bias");
            write_tensor_f32(unit.norm_weight, make_patterned_f32(static_cast<size_t>(unit.norm_weight.shape.num_elements()), phase + 0.2f, 0.01f), "norm_weight");
            write_tensor_f32(unit.norm_bias, make_patterned_f32(static_cast<size_t>(unit.norm_bias.shape.num_elements()), phase + 0.3f, 0.004f), "norm_bias");
            write_tensor_f32(unit.pw1_weight, make_patterned_f32(static_cast<size_t>(unit.pw1_weight.shape.num_elements()), phase + 0.4f, 0.01f), "pw1_weight");
            write_tensor_f32(unit.pw1_bias, make_patterned_f32(static_cast<size_t>(unit.pw1_bias.shape.num_elements()), phase + 0.5f, 0.003f), "pw1_bias");
            write_tensor_f32(unit.pw2_weight, make_patterned_f32(static_cast<size_t>(unit.pw2_weight.shape.num_elements()), phase + 0.6f, 0.01f), "pw2_weight");
            write_tensor_f32(unit.pw2_bias, make_patterned_f32(static_cast<size_t>(unit.pw2_bias.shape.num_elements()), phase + 0.7f, 0.003f), "pw2_bias");
            write_tensor_f32(unit.gamma, make_patterned_f32(static_cast<size_t>(unit.gamma.shape.num_elements()), phase + 0.8f, 0.02f), "gamma");
        }
    }

    void compute() {
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_, nullptr, "supertonic.vector_convnext_exp_test");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("graph compute failed");
        }
        ggml_backend_synchronize(backend_);
    }

    bool exp_ = false;
    int stop_stage_ = -1;
    DepthwiseMode depthwise_mode_ = DepthwiseMode::MergedBatch;
    engine::core::BackendConfig config_{engine::core::BackendType::Cpu, 0, 8};
    ggml_backend_t backend_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_context * ggml_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    engine::core::ModuleBuildContext ctx_{};
};

std::vector<std::string> stage_labels(bool exp) {
    const std::vector<std::string> ops = exp
        ? std::vector<std::string>{
            "residual_mask",
            "edge_pad",
            "depthwise",
            "post_mask",
            "to_btc",
            "layer_norm",
            "pw1_btc",
            "gelu",
            "pw2_btc",
            "to_bct",
            "gamma",
            "residual_add",
        }
        : std::vector<std::string>{
            "residual_mask",
            "edge_pad",
            "depthwise",
            "post_mask",
            "to_btc",
            "layer_norm",
            "to_bct",
            "pw1_bct",
            "gelu",
            "pw2_bct",
            "gamma",
            "residual_add",
        };
    std::vector<std::string> labels;
    labels.reserve(ops.size() * 4);
    for (int unit = 0; unit < 4; ++unit) {
        for (const auto & op : ops) {
            labels.push_back("unit" + std::to_string(unit) + "." + op);
        }
    }
    return labels;
}

void profile_variant(const char * name, bool exp) {
    const auto labels = stage_labels(exp);
    double previous_ms = 0.0;
    std::cout << "[PROFILE] " << name << " cumulative split graph timings\n";
    for (size_t stage = 0; stage < labels.size(); ++stage) {
        GraphRunner runner(name, exp, static_cast<int>(stage));
        const auto tensors = runner.build();
        const RunResult result = runner.run(tensors.output, kProfileTimedRuns);
        std::cout << "[PROFILE] " << name << ' ' << labels[stage]
                  << " cumulative_ms=" << result.mean_ms
                  << " delta_ms=" << (result.mean_ms - previous_ms) << '\n';
        previous_ms = result.mean_ms;
    }
}

}  // namespace

int main() {
    try {
        GraphRunner original_runner("supertonic.vector_convnext.original", false);
        const auto original_tensors = original_runner.build();
        const RunResult original = original_runner.run(original_tensors.output);

        GraphRunner exp_runner("supertonic.vector_convnext.exp", true);
        const auto exp_tensors = exp_runner.build();
        const RunResult exp = exp_runner.run(exp_tensors.output);

        GraphRunner sliced_runner("supertonic.vector_convnext.exp_sliced_depthwise", true, -1, DepthwiseMode::SlicedBatch);
        const auto sliced_tensors = sliced_runner.build();
        const RunResult sliced = sliced_runner.run(sliced_tensors.output);

        GraphRunner shift_sum_runner("supertonic.vector_convnext.exp_shift_sum_depthwise", true, -1, DepthwiseMode::ShiftSum);
        const auto shift_sum_tensors = shift_sum_runner.build();
        const RunResult shift_sum = shift_sum_runner.run(shift_sum_tensors.output);

        require_close(original, exp, 2.5e-5f, 2.0e-6);
        require_close(exp, sliced, 2.5e-5f, 2.0e-6);
        require_close(exp, shift_sum, 2.5e-5f, 2.0e-6);
        std::cout << "[PASS] Supertonic vector ConvNeXt exp parity output " << original.shape.to_string() << '\n';
        std::cout << "[TIMING] original warm_ms=" << original.warm_ms << " mean_ms=" << original.mean_ms << '\n';
        std::cout << "[TIMING] exp warm_ms=" << exp.warm_ms << " mean_ms=" << exp.mean_ms << '\n';
        std::cout << "[TIMING] exp_sliced_depthwise warm_ms=" << sliced.warm_ms << " mean_ms=" << sliced.mean_ms << '\n';
        std::cout << "[TIMING] exp_shift_sum_depthwise warm_ms=" << shift_sum.warm_ms << " mean_ms=" << shift_sum.mean_ms << '\n';
        require(exp.mean_ms < original.mean_ms * 0.95, "exp graph did not improve mean compute time by at least 5%");
        profile_variant("original", false);
        profile_variant("exp", true);
    } catch (const std::exception & ex) {
        std::cerr << "[FAIL] " << ex.what() << '\n';
        return 1;
    }
    return 0;
}
