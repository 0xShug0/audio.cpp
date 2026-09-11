#include "engine/framework/core/backend.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t kTestGraphBytes = 512 * 1024 * 1024;
constexpr size_t kTestGraphNodes = 4096;
constexpr int kRounds = 6;

struct DepthwiseCase {
    const char * name;
    int64_t batch;
    int64_t channels;
    int64_t frames;
    int64_t kernel_size;
    int stride;
    int padding;
    int dilation;
    bool use_bias;
};

struct RunResult {
    std::vector<float> values;
    double compute_ms = 0.0;
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

int64_t output_frames(const DepthwiseCase & test_case) {
    return (test_case.frames + 2 * test_case.padding -
               test_case.dilation * (test_case.kernel_size - 1) - 1) /
        test_case.stride + 1;
}

void require_same_shape(
    const engine::core::TensorShape & lhs,
    const engine::core::TensorShape & rhs,
    const std::string & label) {
    if (lhs.rank != rhs.rank) {
        throw std::runtime_error(label + " rank mismatch");
    }
    for (size_t i = 0; i < lhs.rank; ++i) {
        if (lhs.dims[i] != rhs.dims[i]) {
            throw std::runtime_error(label + " shape mismatch: " + lhs.to_string() + " vs " + rhs.to_string());
        }
    }
}

engine::core::TensorValue add_bias_bct_for_old_lowering(
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
    return engine::core::wrap_tensor(
        ggml_add(ctx.ggml, output_contiguous.tensor, repeated.tensor),
        output.shape,
        GGML_TYPE_F32);
}

engine::core::TensorValue depthwise_conv_single_batch_old(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & weight,
    const DepthwiseCase & test_case) {
    return engine::core::wrap_tensor(
        ggml_conv_1d_dw(
            ctx.ggml,
            weight.tensor,
            input.tensor,
            test_case.stride,
            test_case.padding,
            test_case.dilation),
        engine::core::TensorShape::from_dims({
            input.shape.dims[0],
            test_case.channels,
            output_frames(test_case),
        }),
        GGML_TYPE_F32);
}

engine::core::TensorValue build_batched_depthwise_weight_old(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & weight,
    int64_t batch) {
    auto weight4 = engine::core::reshape_tensor(
        ctx,
        weight,
        engine::core::TensorShape::from_dims({1, weight.shape.dims[0], weight.shape.dims[1], weight.shape.dims[2]}));
    auto repeated = engine::modules::RepeatModule(engine::modules::RepeatConfig{
        engine::core::TensorShape::from_dims({batch, weight.shape.dims[0], weight.shape.dims[1], weight.shape.dims[2]})})
                        .build(ctx, weight4);
    repeated = engine::core::ensure_backend_addressable_layout(ctx, repeated);
    return engine::core::reshape_tensor(
        ctx,
        repeated,
        engine::core::TensorShape::from_dims({batch * weight.shape.dims[0], weight.shape.dims[1], weight.shape.dims[2]}));
}

engine::core::TensorValue build_old_lowering(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & weight,
    const std::optional<engine::core::TensorValue> & bias,
    const DepthwiseCase & test_case) {
    engine::core::TensorValue output;
    if (test_case.batch == 1) {
        output = depthwise_conv_single_batch_old(ctx, input, weight, test_case);
    } else {
        auto merged_input = engine::core::reshape_tensor(
            ctx,
            input,
            engine::core::TensorShape::from_dims({1, test_case.batch * test_case.channels, test_case.frames}));
        auto merged_weight = build_batched_depthwise_weight_old(ctx, weight, test_case.batch);
        auto merged_case = test_case;
        merged_case.batch = 1;
        merged_case.channels = test_case.batch * test_case.channels;
        output = depthwise_conv_single_batch_old(ctx, merged_input, merged_weight, merged_case);
        output = engine::core::reshape_tensor(
            ctx,
            output,
            engine::core::TensorShape::from_dims({test_case.batch, test_case.channels, output.shape.dims[2]}));
    }
    return add_bias_bct_for_old_lowering(ctx, output, test_case.channels, bias);
}

class GraphRunner {
public:
    GraphRunner(const DepthwiseCase & test_case, bool old_lowering)
        : test_case_(test_case) {
        backend_ = engine::core::init_backend({engine::core::BackendType::Cpu, 0, 4});
        ggml_init_params params{};
        params.mem_size = kTestGraphBytes;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml_ = ggml_init(params);
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to init ggml context");
        }
        ctx_.ggml = ggml_;
        ctx_.backend_type = engine::core::BackendType::Cpu;
        ctx_.module_instance_name = old_lowering ? "depthwise_conv1d_old" : "depthwise_conv1d_new";

        const auto input_shape = engine::core::TensorShape::from_dims({test_case.batch, test_case.channels, test_case.frames});
        const auto weight_shape = engine::core::TensorShape::from_dims({test_case.channels, 1, test_case.kernel_size});
        input_ = engine::core::make_tensor(ctx_, GGML_TYPE_F32, input_shape);
        weight_ = engine::core::make_tensor(ctx_, GGML_TYPE_F32, weight_shape);
        if (test_case.use_bias) {
            bias_ = engine::core::make_tensor(ctx_, GGML_TYPE_F32, engine::core::TensorShape::from_dims({test_case.channels}));
        }

        if (old_lowering) {
            output_ = build_old_lowering(ctx_, input_, weight_, bias_, test_case);
        } else {
            output_ = engine::modules::DepthwiseConv1dModule({
                test_case.channels,
                test_case.kernel_size,
                test_case.stride,
                test_case.padding,
                test_case.dilation,
                test_case.use_bias,
            }).build(ctx_, input_, engine::modules::DepthwiseConv1dWeights{weight_, bias_});
        }

        graph_ = ggml_new_graph_custom(ggml_, kTestGraphNodes, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        buffer_ = ggml_backend_alloc_ctx_tensors(ggml_, backend_);
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate backend tensors");
        }
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

    void write_fixed_weights(const std::vector<float> & weight_values, const std::vector<float> & bias_values) {
        engine::core::write_tensor_f32(weight_, weight_values);
        if (bias_.has_value()) {
            engine::core::write_tensor_f32(*bias_, bias_values);
        }
    }

    RunResult run_round(const std::vector<float> & input_values) {
        engine::core::write_tensor_f32(input_, input_values);
        const auto start = std::chrono::steady_clock::now();
        const ggml_status status = ggml_backend_graph_compute(backend_, graph_);
        const auto end = std::chrono::steady_clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            std::ostringstream oss;
            oss << "graph compute failed with status " << static_cast<int>(status);
            throw std::runtime_error(oss.str());
        }
        RunResult result;
        result.compute_ms = std::chrono::duration<double, std::milli>(end - start).count();
        engine::core::read_tensor_f32_into(output_.tensor, result.values);
        return result;
    }

    const engine::core::TensorShape & output_shape() const noexcept {
        return output_.shape;
    }

private:
    DepthwiseCase test_case_;
    ggml_backend_t backend_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_context * ggml_ = nullptr;
    engine::core::ModuleBuildContext ctx_{};
    engine::core::TensorValue input_{};
    engine::core::TensorValue weight_{};
    std::optional<engine::core::TensorValue> bias_;
    engine::core::TensorValue output_{};
    ggml_cgraph * graph_ = nullptr;
};

struct DiffStats {
    float max_diff = 0.0f;
    double mean_diff = 0.0;
};

DiffStats compare_outputs(
    const DepthwiseCase & test_case,
    const std::vector<float> & old_values,
    const std::vector<float> & new_values) {
    if (old_values.size() != new_values.size()) {
        throw std::runtime_error(std::string(test_case.name) + " output size mismatch");
    }
    DiffStats stats;
    for (size_t i = 0; i < old_values.size(); ++i) {
        const float diff = std::fabs(old_values[i] - new_values[i]);
        stats.mean_diff += diff;
        stats.max_diff = std::max(stats.max_diff, diff);
    }
    stats.mean_diff /= static_cast<double>(old_values.size());
    constexpr float kMaxAllowed = 1.0e-4f;
    constexpr double kMeanAllowed = 1.0e-5;
    if (stats.max_diff > kMaxAllowed || stats.mean_diff > kMeanAllowed) {
        std::ostringstream oss;
        oss << test_case.name << " parity drift exceeds bounds: max_diff=" << stats.max_diff
            << " mean_diff=" << stats.mean_diff;
        throw std::runtime_error(oss.str());
    }
    return stats;
}

void run_case(const DepthwiseCase & test_case) {
    GraphRunner old_runner(test_case, true);
    GraphRunner new_runner(test_case, false);
    require_same_shape(old_runner.output_shape(), new_runner.output_shape(), test_case.name);

    const auto input_shape = engine::core::TensorShape::from_dims({test_case.batch, test_case.channels, test_case.frames});
    const auto weight_shape = engine::core::TensorShape::from_dims({test_case.channels, 1, test_case.kernel_size});
    const auto weight_values = make_patterned_f32(static_cast<size_t>(weight_shape.num_elements()), 0.47f, 0.017f);
    const auto bias_values = make_patterned_f32(static_cast<size_t>(test_case.channels), 0.83f, 0.011f);
    old_runner.write_fixed_weights(weight_values, bias_values);
    new_runner.write_fixed_weights(weight_values, bias_values);

    double old_total_ms = 0.0;
    double new_total_ms = 0.0;
    DiffStats worst;
    for (int round = 0; round < kRounds; ++round) {
        const auto input_values = make_patterned_f32(
            static_cast<size_t>(input_shape.num_elements()),
            0.19f + 0.031f * static_cast<float>(round),
            0.031f);
        const auto old_result = old_runner.run_round(input_values);
        const auto new_result = new_runner.run_round(input_values);
        const auto stats = compare_outputs(test_case, old_result.values, new_result.values);
        old_total_ms += old_result.compute_ms;
        new_total_ms += new_result.compute_ms;
        worst.max_diff = std::max(worst.max_diff, stats.max_diff);
        worst.mean_diff = std::max(worst.mean_diff, stats.mean_diff);
    }

    std::cout << "[RESULT] " << test_case.name
              << " rounds=" << kRounds
              << " shape={B=" << test_case.batch
              << ", C=" << test_case.channels
              << ", T=" << test_case.frames
              << ", K=" << test_case.kernel_size
              << "} old_avg_ms=" << old_total_ms / kRounds
              << " new_avg_ms=" << new_total_ms / kRounds
              << " max_diff=" << worst.max_diff
              << " mean_diff=" << worst.mean_diff
              << '\n';
}

}  // namespace

int main() {
    try {
        run_case({"miocodec_depthwise_block", 1, 384, 2048, 7, 1, 3, 1, true});
        run_case({"miocodec_depthwise_batched", 4, 384, 1024, 7, 1, 3, 1, true});
        run_case({"nemotron_encoder_conv", 1, 1024, 1024, 9, 1, 0, 1, false});
    } catch (const std::exception & ex) {
        std::cerr << "[FAIL] " << ex.what() << '\n';
        return 1;
    }
    return 0;
}
