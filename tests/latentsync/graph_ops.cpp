#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace engine;

void check(bool passed, const char * message) {
    if (!passed) throw std::runtime_error(message);
}

struct Graph {
    ggml_context * context = ggml_init({4 * 1024 * 1024, nullptr, true});
    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_gallocr_t allocator;
    ggml_backend_t backend;
    core::ModuleBuildContext build;

    explicit Graph(core::ExecutionContext & execution)
        : allocator(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()))),
          backend(execution.backend()), build{context, "latentsync.graph_ops.test", execution.backend_type()} {}

    ~Graph() {
        core::release_backend_graph_resources(backend, graph);
        ggml_gallocr_free(allocator);
        ggml_free(context);
    }

    void output(const core::TensorValue & value) {
        ggml_set_output(value.tensor);
        ggml_build_forward_expand(graph, value.tensor);
    }

    void allocate() {
        check(ggml_gallocr_alloc_graph(allocator, graph), "graph allocation failed");
    }
    void compute() {
        check(core::compute_backend_graph(backend, graph) == GGML_STATUS_SUCCESS, "graph execution failed");
    }
};

std::vector<float> fill(const core::TensorValue & tensor, float scale) {
    std::vector<float> values(tensor.shape.num_elements());
    for (size_t i = 0; i < values.size(); ++i) values[i] = scale * std::sin(float(i) * 0.17F);
    ggml_backend_tensor_set(tensor.tensor, values.data(), 0, values.size() * sizeof(float));
    return values;
}

std::vector<float> read(const core::TensorValue & tensor) {
    std::vector<float> values(tensor.shape.num_elements());
    ggml_backend_tensor_get(tensor.tensor, values.data(), 0, values.size() * sizeof(float));
    return values;
}

void group_norm(core::ExecutionContext & execution, bool rank3) {
    constexpr int batch = 2, channels = 64, height = 4, width = 4, groups = 32;
    constexpr int group_size = channels * height * width / groups;
    Graph graph(execution);
    const auto input = core::make_tensor(graph.build, GGML_TYPE_F32,
        rank3 ? core::TensorShape::from_dims({batch, channels, height * width})
              : core::TensorShape::from_dims({batch, channels, height, width}));
    const modules::NormWeights weights{
        core::make_tensor(graph.build, GGML_TYPE_F32, core::TensorShape::from_dims({channels})),
        core::make_tensor(graph.build, GGML_TYPE_F32, core::TensorShape::from_dims({channels}))};
    const auto output = modules::GroupNormModule({channels, groups, 1e-5F, true, true, false, true}).build(
        graph.build, input, weights);
    const auto original = modules::GroupNormModule({channels, groups, 1e-5F}).build(
        graph.build, input, weights);
    for (const auto & value : {input, *weights.weight, *weights.bias}) ggml_set_input(value.tensor);
    graph.output(output);
    graph.output(original);
    int row_norms = 0, group_norms = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph.graph); ++i) {
        const auto op = ggml_graph_node(graph.graph, i)->op;
        row_norms += op == GGML_OP_NORM;
        group_norms += op == GGML_OP_GROUP_NORM;
    }
    check(group_norms == (execution.backend_type() == core::BackendType::Metal ? 1 : 2),
          "default GroupNorm graph changed");
    check(row_norms == (execution.backend_type() == core::BackendType::Metal ? 1 : 0),
          "GroupNorm opt-in selected the wrong backend path");
    graph.allocate();
    auto values = fill(input, 1.0F);
    for (size_t i = 0; i < values.size(); ++i) values[i] += float(i / group_size) * 0.1F;
    ggml_backend_tensor_set(input.tensor, values.data(), 0, values.size() * sizeof(float));
    const auto scale = fill(*weights.weight, 0.7F);
    const auto bias = fill(*weights.bias, 0.2F);
    graph.compute();
    const auto actual = read(output);
    double maximum = 0;
    for (int group = 0; group < batch * groups; ++group) {
        double mean = 0, variance = 0;
        for (int i = 0; i < group_size; ++i) mean += values[group * group_size + i];
        mean /= group_size;
        for (int i = 0; i < group_size; ++i) variance += std::pow(values[group * group_size + i] - mean, 2);
        variance /= group_size;
        for (int i = 0; i < group_size; ++i) {
            const int index = group * group_size + i;
            const int channel = index / (height * width) % channels;
            const double expected = (values[index] - mean) / std::sqrt(variance + 1e-5) * scale[channel] + bias[channel];
            check(std::isfinite(actual[index]), "nonfinite GroupNorm output");
            maximum = std::max(maximum, std::abs(actual[index] - expected));
        }
    }
    std::cout << "batched_group_norm rank=" << input.shape.rank << " max_error=" << maximum << '\n';
    check(maximum < 1e-4, "batched GroupNorm disagrees with scalar reference");
}

void convolution(core::ExecutionContext & execution, int kernel, int stride, int padding, int dilation,
                 int channels = 64) {
    Graph graph(execution);
    const auto input = core::make_tensor(graph.build, GGML_TYPE_F32,
        core::TensorShape::from_dims({2, channels, 23, 25}));
    const modules::Conv2dWeights weights{
        core::make_tensor(graph.build, GGML_TYPE_F32, core::TensorShape::from_dims({96, channels, kernel, kernel})),
        core::make_tensor(graph.build, GGML_TYPE_F32, core::TensorShape::from_dims({96}))};
    modules::Conv2dConfig config{channels, 96, kernel, kernel, stride, stride, padding, padding, dilation, dilation};
    const auto reference = modules::Conv2dModule(config).build(graph.build, input, weights);
    config.metal_per_tap = true;
    const auto rewritten = modules::Conv2dModule(config).build(graph.build, input, weights);
    for (const auto & value : {input, weights.weight, *weights.bias}) ggml_set_input(value.tensor);
    graph.output(reference);
    const int default_nodes = ggml_graph_n_nodes(graph.graph);
    graph.output(rewritten);
    int default_im2col = 0, rewritten_im2col = 0, accumulates = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph.graph); ++i) {
        const auto op = ggml_graph_node(graph.graph, i)->op;
        if (i < default_nodes) default_im2col += op == GGML_OP_IM2COL;
        else {
            rewritten_im2col += op == GGML_OP_IM2COL;
            accumulates += op == GGML_OP_MUL_MAT_ACC;
        }
    }
    const bool per_tap = execution.backend_type() == core::BackendType::Metal && channels >= 64;
    check(default_im2col == 1, "default convolution graph changed");
    check(rewritten_im2col == (per_tap ? 0 : 1), "convolution opt-in selected the wrong path");
    check(accumulates == (per_tap ? kernel * kernel - 1 : 0), "unexpected per-tap accumulation graph");
    graph.allocate();
    fill(input, 0.2F);
    fill(weights.weight, 0.01F);
    fill(*weights.bias, 0.05F);
    graph.compute();
    const auto a = read(reference), b = read(rewritten);
    double squared_error = 0, squared_reference = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        check(std::isfinite(b[i]), "nonfinite convolution output");
        squared_error += std::pow(double(a[i]) - b[i], 2);
        squared_reference += double(a[i]) * a[i];
    }
    const auto error = std::sqrt(squared_error / squared_reference);
    std::cout << "conv kernel=" << kernel << " stride=" << stride << " padding=" << padding
              << " dilation=" << dilation << " relative_rmse=" << error << '\n';
    check(error < 1e-4, "convolution rewrite disagrees with original graph");
}

void geglu(core::ExecutionContext & execution) {
    Graph graph(execution);
    const auto input = core::make_tensor(graph.build, GGML_TYPE_F32,
        core::TensorShape::from_dims({2, 7, 128}));
    ggml_set_input(input.tensor);
    const auto hidden = modules::SliceModule({2, 0, 64}).build(graph.build, input);
    const auto gate = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(graph.build,
        modules::SliceModule({2, 64, 64}).build(graph.build, input));
    const auto reference = modules::MulModule().build(graph.build, hidden, gate);
    const auto rewritten = modules::GegluModule().build(graph.build, input);
    graph.output(reference);
    graph.output(rewritten);
    graph.allocate();
    fill(input, 4.0F);
    graph.compute();
    const auto a = read(reference), b = read(rewritten);
    double maximum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        check(std::isfinite(b[i]), "nonfinite GEGLU output");
        maximum = std::max(maximum, std::abs(double(a[i]) - b[i]));
    }
    std::cout << "geglu max_error=" << maximum << '\n';
    check(maximum < 1e-5, "fused GEGLU disagrees with split exact-erf graph");
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        engine::core::BackendConfig config;
        config.type = argc == 2 && std::string(argv[1]) == "metal"
            ? engine::core::BackendType::Metal : engine::core::BackendType::Cpu;
        engine::core::ExecutionContext execution(config);
        group_norm(execution, false);
        group_norm(execution, true);
        geglu(execution);
        for (int kernel : {1, 3}) for (int stride : {1, 2}) for (int padding : {0, 1}) {
            convolution(execution, kernel, stride, padding, 1);
        }
        convolution(execution, 3, 1, 2, 2);
        convolution(execution, 3, 1, 1, 1, 32);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
