// Numeric and gate coverage for Conv1dModule's Metal per-tap fast path
// (conv1d_pertap_channel_fast / is_conv1d_pertap_fast_path_eligible, added for the
// audio8_tts codec).
//
// The fast path is selected from the build context (backend_type == Metal), so the test
// builds the same module twice -- once with a Metal build context and once with the
// CPU/reference one -- and compares the two graphs. Both are executed on the CPU backend:
// the fast path is built from generic ops (transpose/cont/view/mul_mat/mul_mat_acc/add),
// so running them on the CPU checks the decomposition itself without needing a GPU, and
// the Metal kernels for those ops are covered by their own tests.

#include "engine/framework/core/backend.h"
#include "engine/framework/modules/conv_modules.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t kTestGraphBytes = 256 * 1024 * 1024;
constexpr size_t kTestGraphNodes = 8192;

struct Conv1dCase {
    const char * name;
    int64_t in_channels;
    int64_t out_channels;
    int64_t frames;
    int64_t kernel_size;
    int dilation;
    bool use_bias;
};

struct RunResult {
    engine::core::TensorShape shape;
    std::vector<float> values;
    double compute_ms = 0.0;
};

std::vector<float> make_patterned_f32(size_t count, float phase, float scale) {
    std::vector<float> values(count, 0.0f);
    for (size_t i = 0; i < count; ++i) {
        const float x = static_cast<float>(i);
        values[i] = scale * (std::sin(phase + 0.113f * x) + 0.5f * std::cos(phase * 0.7f + 0.071f * x));
    }
    return values;
}

// The build context decides which conv path is built (that is what these tests vary);
// the compute backend is separate so a case can be timed on the GPU while the reference
// build context stays the default one.
struct GraphRunner {
    engine::core::BackendConfig backend_config{engine::core::BackendType::Cpu, 0, 4};
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_context * ggml = nullptr;
    engine::core::ModuleBuildContext ctx{};

    GraphRunner(
        const char * name,
        engine::core::BackendType build_backend,
        engine::core::BackendType compute_backend = engine::core::BackendType::Cpu) {
        backend_config.type = compute_backend;
        backend = engine::core::init_backend(backend_config);
        if (backend == nullptr) {
            throw std::runtime_error("failed to init test backend");
        }
        ggml_init_params params{};
        params.mem_size = kTestGraphBytes;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml = ggml_init(params);
        if (ggml == nullptr) {
            throw std::runtime_error("failed to init test ggml context");
        }
        ctx.ggml = ggml;
        ctx.module_instance_name = name;
        ctx.backend_type = build_backend;
    }

    ~GraphRunner() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (ggml != nullptr) {
            ggml_free(ggml);
        }
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }

    engine::core::TensorValue make_f32(const engine::core::TensorShape & shape) {
        return engine::core::make_tensor(ctx, GGML_TYPE_F32, shape);
    }

    void allocate_tensors() {
        if (buffer != nullptr) {
            return;
        }
        buffer = ggml_backend_alloc_ctx_tensors(ggml, backend);
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate test backend tensors");
        }
    }

    // Callers allocate first (after the whole graph exists) and then write leaves, which
    // mirrors how the other module tests drive the backend.
    void write(const engine::core::TensorValue & tensor, const std::vector<float> & values, const char * what) {
        if (static_cast<int64_t>(values.size()) != tensor.shape.num_elements()) {
            std::ostringstream oss;
            oss << what << " value count mismatch: expected " << tensor.shape.num_elements()
                << ", got " << values.size();
            throw std::runtime_error(oss.str());
        }
        if (buffer == nullptr) {
            throw std::runtime_error("allocate_tensors() must run before writing tensor data");
        }
        ggml_backend_tensor_set(tensor.tensor, values.data(), 0, values.size() * sizeof(float));
    }

    RunResult run(const engine::core::TensorValue & output) {
        allocate_tensors();
        ggml_cgraph * graph = ggml_new_graph_custom(ggml, kTestGraphNodes, false);
        ggml_build_forward_expand(graph, output.tensor);
        const auto start = std::chrono::steady_clock::now();
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        const auto end = std::chrono::steady_clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            std::ostringstream oss;
            oss << "backend graph compute failed with status " << static_cast<int>(status);
            throw std::runtime_error(oss.str());
        }
        RunResult result{output.shape, {}};
        result.compute_ms = std::chrono::duration<double, std::milli>(end - start).count();
        engine::core::read_tensor_f32_into(output.tensor, result.values);
        return result;
    }

    // Counts nodes of one op without executing anything.
    int64_t count_op(const engine::core::TensorValue & output, ggml_op op) {
        ggml_cgraph * graph = ggml_new_graph_custom(ggml, kTestGraphNodes, false);
        ggml_build_forward_expand(graph, output.tensor);
        int64_t count = 0;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            if (ggml_graph_node(graph, i)->op == op) {
                ++count;
            }
        }
        return count;
    }
};

float max_abs(const std::vector<float> & values) {
    float result = 0.0f;
    for (const float value : values) {
        result = std::max(result, std::fabs(value));
    }
    return result;
}

float max_abs_diff(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error("value vectors differ in size");
    }
    float result = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) {
        result = std::max(result, std::fabs(lhs[i] - rhs[i]));
    }
    return result;
}

engine::modules::Conv1dConfig make_config(const Conv1dCase & test_case, int stride = 1, int padding = 0) {
    engine::modules::Conv1dConfig config;
    config.in_channels = test_case.in_channels;
    config.out_channels = test_case.out_channels;
    config.kernel_size = test_case.kernel_size;
    config.stride = stride;
    config.padding = padding;
    config.dilation = test_case.dilation;
    config.use_bias = test_case.use_bias;
    return config;
}

// Builds the module on the requested build backend and returns the result plus the graph
// shape that identifies the path: the reference path goes through ggml_conv_1d
// (im2col + mul_mat), the fast path decomposes into one matmul per kernel tap and has no
// im2col node at all.
struct ModuleRun {
    RunResult result;
    int64_t mul_mats = 0;
    int64_t im2cols = 0;
};

ModuleRun run_module_case(
    const Conv1dCase & test_case,
    engine::core::BackendType build_backend,
    int stride = 1,
    int padding = 0,
    engine::core::BackendType compute_backend = engine::core::BackendType::Cpu) {
    GraphRunner runner("conv1d_pertap_fast_path_test", build_backend, compute_backend);
    const auto config = make_config(test_case, stride, padding);
    const int64_t frames = test_case.frames;

    const auto input = runner.make_f32(engine::core::TensorShape::from_dims({1, config.in_channels, frames}));
    const auto weight = runner.make_f32(
        engine::core::TensorShape::from_dims({config.out_channels, config.in_channels, config.kernel_size}));
    std::optional<engine::core::TensorValue> bias;
    if (config.use_bias) {
        bias = runner.make_f32(engine::core::TensorShape::from_dims({config.out_channels}));
    }

    const engine::modules::Conv1dModule module(config);
    const auto output = module.build(runner.ctx, input, engine::modules::Conv1dWeights{weight, bias});

    runner.allocate_tensors();
    runner.write(input, make_patterned_f32(static_cast<size_t>(input.shape.num_elements()), 0.71f, 1.0f), "input");
    runner.write(weight, make_patterned_f32(static_cast<size_t>(weight.shape.num_elements()), 0.37f, 0.25f), "weight");
    if (bias.has_value()) {
        runner.write(*bias, make_patterned_f32(static_cast<size_t>(config.out_channels), 1.13f, 0.1f), "bias");
    }
    return ModuleRun{
        runner.run(output),
        runner.count_op(output, GGML_OP_MUL_MAT) + runner.count_op(output, GGML_OP_MUL_MAT_ACC),
        runner.count_op(output, GGML_OP_IM2COL)};
}

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_fast_path_matches_reference() {
    const Conv1dCase cases[] = {
        {"small", 8, 4, 64, 3, 1, true},
        {"no-bias", 16, 12, 33, 5, 1, false},
        {"causal-ish", 32, 32, 17, 7, 1, true},
        {"dilation-2", 24, 8, 40, 5, 2, true},
        {"wide", 96, 96, 25, 3, 1, true},
        // frames == kernel_size: a single output frame, the smallest valid padding=0 case
        {"one-frame-out", 12, 6, 7, 7, 1, true},
    };

    for (const auto & test_case : cases) {
        const auto reference = run_module_case(test_case, engine::core::BackendType::Cpu);
        const auto fast = run_module_case(test_case, engine::core::BackendType::Metal);

        require(
            reference.result.shape.dims == fast.result.shape.dims,
            std::string("shape mismatch in case ") + test_case.name);
        const float scale = std::max(1.0f, max_abs(reference.result.values));
        const float diff = max_abs_diff(reference.result.values, fast.result.values);
        if (diff > 1e-4f * scale) {
            std::ostringstream oss;
            oss << "conv1d pertap fast path differs from the reference in case " << test_case.name
                << ": max_abs_diff=" << diff << " (scale " << scale << ")";
            throw std::runtime_error(oss.str());
        }
        // The gate: the Metal build context must take the per-tap decomposition, the
        // reference build context must stay on the conv op.
        require(
            fast.im2cols == 0 && fast.mul_mats >= test_case.kernel_size,
            std::string("fast path did not decompose into per-tap matmuls in case ") + test_case.name);
        require(
            reference.im2cols >= 1,
            std::string("reference path did not go through ggml_conv_1d in case ") + test_case.name);
    }
    std::cout << "ok: fast path matches the reference for " << (sizeof(cases) / sizeof(cases[0])) << " shapes\n";
}

void test_ineligible_configs_stay_on_the_reference_path() {
    const Conv1dCase base{"stride-2", 16, 8, 32, 3, 1, true};
    const Conv1dCase padded_case{"padded", 16, 8, 32, 3, 1, true};

    // stride != 1 and padding != 0 are not eligible: the Metal build context must fall
    // back to the conv op and still produce the reference numbers.
    const auto stride2 = run_module_case(base, engine::core::BackendType::Metal, 2, 0);
    require(stride2.im2cols >= 1, "stride=2 must not take the per-tap fast path");

    const auto padded = run_module_case(padded_case, engine::core::BackendType::Metal, 1, 1);
    require(padded.im2cols >= 1, "padding=1 must not take the per-tap fast path");

    const auto reference_stride2 = run_module_case(base, engine::core::BackendType::Cpu, 2, 0);
    const float scale = std::max(1.0f, max_abs(reference_stride2.result.values));
    require(
        max_abs_diff(reference_stride2.result.values, stride2.result.values) <= 1e-4f * scale,
        "stride=2 fallback differs from the reference");

    std::cout << "ok: ineligible configs stay on the reference path\n";
}

void test_raw_channel_fast_helper() {
    // The helper the audio8_tts codec chains convs with takes and returns channel-fast
    // tensors; check it against the canonical module path (transposing at the edges).
    const Conv1dCase test_case{"raw-helper", 32, 16, 48, 3, 1, true};
    const auto config = make_config(test_case);
    const auto reference = run_module_case(test_case, engine::core::BackendType::Cpu);

    GraphRunner runner("conv1d_pertap_fast_path_test.raw", engine::core::BackendType::Metal);
    const auto weight = runner.make_f32(
        engine::core::TensorShape::from_dims({config.out_channels, config.in_channels, config.kernel_size}));
    const auto bias = runner.make_f32(engine::core::TensorShape::from_dims({config.out_channels}));

    // channel-fast input [channels, frames]
    const auto input_cf = runner.make_f32(
        engine::core::TensorShape::from_dims({1, config.in_channels, test_case.frames}));
    auto * input_cf_tensor = ggml_cont(runner.ggml, ggml_transpose(runner.ggml, input_cf.tensor));

    auto * raw = engine::modules::conv1d_pertap_channel_fast(
        runner.ctx, engine::modules::Conv1dWeights{weight, bias}, input_cf_tensor, config);
    const auto output_shape = engine::core::TensorShape::from_dims(
        {1, config.out_channels, reference.result.shape.dims[2]});
    const auto output = engine::core::wrap_tensor(
        ggml_cont(runner.ggml, ggml_transpose(runner.ggml, raw)), output_shape, GGML_TYPE_F32);

    runner.allocate_tensors();
    runner.write(
        input_cf, make_patterned_f32(static_cast<size_t>(input_cf.shape.num_elements()), 0.71f, 1.0f), "input");
    runner.write(
        weight, make_patterned_f32(static_cast<size_t>(weight.shape.num_elements()), 0.37f, 0.25f), "weight");
    runner.write(bias, make_patterned_f32(static_cast<size_t>(config.out_channels), 1.13f, 0.1f), "bias");

    const auto result = runner.run(output);
    const float scale = std::max(1.0f, max_abs(reference.result.values));
    const float diff = max_abs_diff(reference.result.values, result.values);
    if (diff > 1e-4f * scale) {
        std::ostringstream oss;
        oss << "channel-fast helper differs from the reference: max_abs_diff=" << diff;
        throw std::runtime_error(oss.str());
    }
    std::cout << "ok: channel-fast helper matches the reference\n";
}

}  // namespace

namespace {

bool backend_available(engine::core::BackendType backend_type) {
    try {
        engine::core::BackendConfig config{backend_type, 0, 4};
        ggml_backend_t backend = engine::core::init_backend(config);
        if (backend == nullptr) {
            return false;
        }
        ggml_backend_free(backend);
        return true;
    } catch (...) {
        return false;
    }
}

// Same-backend timing report (printed, never asserted): the fast path exists because the
// reference im2col conv does strided gathers on Metal. Shapes match the codec chain the
// audio8_tts decoder builds.
void report_backend_timing(engine::core::BackendType compute_backend, const char * label) {
    const Conv1dCase cases[] = {
        {"96ch-k5-4k-frames", 96, 96, 4096, 5, 1, true},
        {"96ch-k5-16k-frames", 96, 96, 16384, 5, 1, true},
    };
    std::cout << "timing on " << label << " (median of 3, per conv):\n";
    for (const auto & test_case : cases) {
        std::vector<double> fast_ms;
        std::vector<double> reference_ms;
        for (int iteration = 0; iteration < 3; ++iteration) {
            const auto fast = run_module_case(test_case, engine::core::BackendType::Metal, 1, 0, compute_backend);
            const auto reference = run_module_case(test_case, engine::core::BackendType::Cpu, 1, 0, compute_backend);
            fast_ms.push_back(fast.result.compute_ms);
            reference_ms.push_back(reference.result.compute_ms);
        }
        std::sort(fast_ms.begin(), fast_ms.end());
        std::sort(reference_ms.begin(), reference_ms.end());
        const double fast = fast_ms[fast_ms.size() / 2];
        const double reference = reference_ms[reference_ms.size() / 2];
        std::cout << "  " << test_case.name << ": reference=" << reference << " ms, per-tap=" << fast
                  << " ms, speedup=" << (fast > 0.0 ? reference / fast : 0.0) << "x\n";
    }
}

}  // namespace

int main() {
    try {
        test_fast_path_matches_reference();
        test_ineligible_configs_stay_on_the_reference_path();
        test_raw_channel_fast_helper();
        if (backend_available(engine::core::BackendType::Metal)) {
            report_backend_timing(engine::core::BackendType::Metal, "Metal");
        } else {
            std::cout << "skip: Metal backend not available\n";
        }
    } catch (const std::exception & error) {
        std::cerr << "conv1d_pertap_fast_path_test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "conv1d_pertap_fast_path_test passed\n";
    return 0;
}
