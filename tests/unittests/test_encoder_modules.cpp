#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/attention_modules.h"
#include "engine/framework/modules/conformer_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/runtime/graph_optimizer.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

namespace {

constexpr size_t kTestGraphBytes = 128 * 1024 * 1024;
constexpr size_t kTestGraphNodes = 4096;

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_allclose(
    const std::vector<float> & actual,
    const std::vector<float> & expected,
    float atol,
    const std::string & label) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(label + " size mismatch");
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        if (!std::isfinite(diff) || diff > atol) {
            std::ostringstream oss;
            oss << label << " mismatch at " << i << ": expected " << expected[i] << ", got " << actual[i]
                << ", diff=" << diff;
            throw std::runtime_error(oss.str());
        }
    }
}

void require_max_abs_diff_below(
    const std::vector<float> & actual,
    const std::vector<float> & expected,
    float max_allowed,
    double mean_allowed,
    const std::string & label) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(label + " size mismatch");
    }
    float max_diff = 0.0f;
    size_t max_index = 0;
    double mean_diff = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        mean_diff += diff;
        if (diff > max_diff) {
            max_diff = diff;
            max_index = i;
        }
    }
    mean_diff /= static_cast<double>(actual.size());
    if (max_diff > max_allowed || mean_diff > mean_allowed) {
        std::ostringstream oss;
        oss << label << " drift exceeds bounds: max diff " << max_diff << " (limit " << max_allowed << ")"
            << ", mean diff=" << mean_diff << " (limit " << mean_allowed << ")"
            << " at " << max_index << " (expected " << expected[max_index]
            << ", got " << actual[max_index] << ")";
        throw std::runtime_error(oss.str());
    }
}

std::vector<float> project_sequence(
    const std::vector<float> & input,
    int64_t rows,
    int64_t in_features,
    const std::vector<float> & weight,
    int64_t out_features) {
    std::vector<float> output(static_cast<size_t>(rows * out_features), 0.0f);
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t out = 0; out < out_features; ++out) {
            float sum = 0.0f;
            for (int64_t in = 0; in < in_features; ++in) {
                sum += input[static_cast<size_t>(row * in_features + in)] *
                    weight[static_cast<size_t>(out * in_features + in)];
            }
            output[static_cast<size_t>(row * out_features + out)] = sum;
        }
    }
    return output;
}

struct CpuModuleRunner {
    engine::core::BackendConfig backend_config{engine::core::BackendType::Cpu, 0, 4};
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_context * ggml = nullptr;
    engine::core::ModuleBuildContext ctx{};

    CpuModuleRunner() {
        backend = engine::core::init_backend(backend_config);
        ggml_init_params params{};
        params.mem_size = kTestGraphBytes;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml = ggml_init(params);
        if (ggml == nullptr) {
            throw std::runtime_error("failed to init test ggml context");
        }
        ctx.ggml = ggml;
        ctx.module_instance_name = "encoder_module_test";
    }

    ~CpuModuleRunner() {
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

    engine::core::TensorValue make_i32(const engine::core::TensorShape & shape) {
        return engine::core::make_tensor(ctx, GGML_TYPE_I32, shape);
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

    std::vector<float> run_f32(const engine::core::TensorValue & output) {
        allocate_tensors();
        ggml_cgraph * graph = ggml_new_graph_custom(ggml, kTestGraphNodes, false);
        ggml_build_forward_expand(graph, output.tensor);
        ggml_backend_graph_compute(backend, graph);
        std::vector<float> values;
        engine::core::read_tensor_f32_into(output.tensor, values);
        return values;
    }
};

using ModuleRunner = CpuModuleRunner;

std::vector<float> make_patterned_f32(size_t count, float phase, float scale) {
    std::vector<float> values(count, 0.0f);
    for (size_t i = 0; i < count; ++i) {
        const float x = static_cast<float>(i);
        values[i] = scale * (std::sin(phase + 0.173f * x) + 0.5f * std::cos(phase * 0.7f + 0.097f * x));
    }
    return values;
}

std::vector<float> make_global_attention_bias_for_test(int64_t frames, int64_t valid_frames) {
    std::vector<float> mask(static_cast<size_t>(frames * frames), -std::numeric_limits<float>::infinity());
    for (int64_t q = 0; q < valid_frames; ++q) {
        for (int64_t k = 0; k < valid_frames; ++k) {
            mask[static_cast<size_t>(q * frames + k)] = 0.0f;
        }
    }
    for (int64_t q = valid_frames; q < frames; ++q) {
        mask[static_cast<size_t>(q * frames + q)] = 0.0f;
    }
    return mask;
}

std::vector<int32_t> make_keep_mask_for_test(int64_t frames, int64_t valid_frames) {
    std::vector<int32_t> mask(static_cast<size_t>(frames), 0);
    for (int64_t i = 0; i < valid_frames; ++i) {
        mask[static_cast<size_t>(i)] = 1;
    }
    return mask;
}

bool backend_is_available(engine::core::BackendType type) {
    try {
        ModuleRunner runner;
        runner.backend_config.type = type;
        if (runner.backend != nullptr) {
            ggml_backend_free(runner.backend);
            runner.backend = nullptr;
        }
        runner.backend = engine::core::init_backend(runner.backend_config);
        return true;
    } catch (...) {
        return false;
    }
}

engine::runtime::GraphOptimizationBackend graph_optimizer_backend_for_test(engine::core::BackendType type) {
    switch (type) {
        case engine::core::BackendType::Cpu:
            return engine::runtime::GraphOptimizationBackend::Cpu;
        case engine::core::BackendType::Cuda:
        case engine::core::BackendType::Hip:
            return engine::runtime::GraphOptimizationBackend::Gpu;
        case engine::core::BackendType::Vulkan:
        case engine::core::BackendType::Metal:
        case engine::core::BackendType::BestAvailable:
            return engine::runtime::GraphOptimizationBackend::Other;
    }
    return engine::runtime::GraphOptimizationBackend::Other;
}

void set_runner_backend(ModuleRunner & runner, engine::core::BackendType type) {
    runner.backend_config.type = type;
    if (runner.backend != nullptr) {
        ggml_backend_free(runner.backend);
        runner.backend = nullptr;
    }
    runner.backend = engine::core::init_backend(runner.backend_config);
}

void test_graph_optimizer_elides_metadata_nodes_without_changing_output() {
    CpuModuleRunner runner;
    auto input = runner.make_f32(engine::core::TensorShape::from_dims({2, 2}));
    auto reshaped = engine::core::reshape_tensor(runner.ctx, input, engine::core::TensorShape::from_dims({4}));
    auto output = engine::core::wrap_tensor(
        ggml_add(runner.ctx.ggml, reshaped.tensor, reshaped.tensor),
        reshaped.shape,
        GGML_TYPE_F32);

    ggml_cgraph * graph = ggml_new_graph_custom(runner.ggml, kTestGraphNodes, false);
    ggml_build_forward_expand(graph, output.tensor);

    const auto report = engine::runtime::optimize_graph(*graph);
    require(report.nodes_before > report.nodes_after, "graph optimizer should reduce node count");
    require(report.metadata_only_nodes_elided == 1, "graph optimizer should elide one reshape node");

    runner.allocate_tensors();
    engine::core::write_tensor_f32(input, {1.0f, -2.0f, 3.0f, -4.0f});
    ggml_backend_graph_compute(runner.backend, graph);

    std::vector<float> values;
    engine::core::read_tensor_f32_into(output.tensor, values);
    require_allclose(values, {2.0f, -4.0f, 6.0f, -8.0f}, 1.0e-6f, "graph optimizer output");
}

void test_graph_optimizer_two_sided_broadcast_binary_matches_repeat() {
    auto run_case = [](engine::core::BackendType backend_type, const std::string & backend_label) {
        ModuleRunner runner;
        set_runner_backend(runner, backend_type);

        auto full_shape = engine::core::TensorShape::from_dims({3, 4});
        auto full_like = runner.make_f32(full_shape);
        auto row = runner.make_f32(engine::core::TensorShape::from_dims({1, 4}));
        auto col = runner.make_f32(engine::core::TensorShape::from_dims({3, 1}));
        auto row_repeat = engine::core::wrap_tensor(
            ggml_repeat(runner.ctx.ggml, row.tensor, full_like.tensor),
            full_shape,
            GGML_TYPE_F32);
        auto output = engine::core::wrap_tensor(
            ggml_add(runner.ctx.ggml, row_repeat.tensor, col.tensor),
            full_shape,
            GGML_TYPE_F32);

        ggml_cgraph * graph = ggml_new_graph_custom(runner.ggml, kTestGraphNodes, false);
        ggml_build_forward_expand(graph, output.tensor);
        auto options = engine::runtime::graph_optimization_options_for_backend(
            graph_optimizer_backend_for_test(backend_type),
            true);
        const auto report = engine::runtime::optimize_graph(*graph, options);
        require(report.two_sided_broadcast_repeats_folded == 1,
                "graph optimizer should fold one two-sided broadcast repeat on " + backend_label);
        require(report.candidate_two_sided_broadcast_repeats == 0,
                "two-sided broadcast candidate should be consumed on " + backend_label);

        runner.allocate_tensors();
        engine::core::write_tensor_f32(row, {10.0f, 20.0f, 30.0f, 40.0f});
        engine::core::write_tensor_f32(col, {1.0f, 2.0f, 3.0f});
        ggml_backend_graph_compute(runner.backend, graph);

        std::vector<float> values;
        engine::core::read_tensor_f32_into(output.tensor, values);
        require_allclose(
            values,
            {
                11.0f, 21.0f, 31.0f, 41.0f,
                12.0f, 22.0f, 32.0f, 42.0f,
                13.0f, 23.0f, 33.0f, 43.0f,
            },
            1.0e-6f,
            "two-sided broadcast binary " + backend_label);
    };

    run_case(engine::core::BackendType::Cpu, "cpu");
    if (backend_is_available(engine::core::BackendType::Cuda)) {
        run_case(engine::core::BackendType::Cuda, "cuda");
    } else {
        std::cout << "encoder_module_test: skipping cuda two-sided broadcast\n";
    }
}

void test_graph_optimizer_unary_broadcast_scale_matches_repeat() {
    auto run_case = [](engine::core::BackendType backend_type, const std::string & backend_label) {
        ModuleRunner runner;
        set_runner_backend(runner, backend_type);

        auto full_shape = engine::core::TensorShape::from_dims({3, 4});
        auto full_like = runner.make_f32(full_shape);
        auto scalar = runner.make_f32(engine::core::TensorShape::from_dims({1}));
        auto scalar_repeat = engine::core::wrap_tensor(
            ggml_repeat(runner.ctx.ggml, scalar.tensor, full_like.tensor),
            full_shape,
            GGML_TYPE_F32);
        auto output = engine::core::wrap_tensor(
            ggml_scale(runner.ctx.ggml, scalar_repeat.tensor, 2.5f),
            full_shape,
            GGML_TYPE_F32);

        ggml_cgraph * graph = ggml_new_graph_custom(runner.ggml, kTestGraphNodes, false);
        ggml_build_forward_expand(graph, output.tensor);
        auto options = engine::runtime::graph_optimization_options_for_backend(
            graph_optimizer_backend_for_test(backend_type),
            true);
        const auto report = engine::runtime::optimize_graph(*graph, options);
        require(report.unary_broadcast_repeats_folded == 1,
                "graph optimizer should fold one unary broadcast repeat on " + backend_label);

        runner.allocate_tensors();
        engine::core::write_tensor_f32(scalar, {4.0f});
        ggml_backend_graph_compute(runner.backend, graph);

        std::vector<float> values;
        engine::core::read_tensor_f32_into(output.tensor, values);
        require_allclose(
            values,
            {10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f},
            1.0e-6f,
            "unary broadcast scale " + backend_label);
    };

    run_case(engine::core::BackendType::Cpu, "cpu");
    if (backend_is_available(engine::core::BackendType::Cuda)) {
        run_case(engine::core::BackendType::Cuda, "cuda");
    } else {
        std::cout << "encoder_module_test: skipping cuda unary broadcast scale\n";
    }
}

void test_relative_attention_fused_qkv_matches_split_and_cached_pos() {
    const int64_t batch = 1;
    const int64_t frames = 3;
    const int64_t hidden = 4;
    const int64_t heads = 2;
    const int64_t pos_frames = 2 * frames - 1;

    const std::vector<float> input_values = {
        0.10f, 0.20f, -0.30f, 0.40f,
        0.25f, -0.15f, 0.05f, 0.30f,
        -0.10f, 0.35f, 0.45f, -0.20f,
    };
    const std::vector<float> pos_values = {
        0.05f, 0.10f, -0.05f, 0.15f,
        0.20f, -0.10f, 0.30f, -0.25f,
        -0.15f, 0.05f, 0.25f, 0.10f,
        0.12f, -0.08f, 0.18f, 0.04f,
        -0.07f, 0.16f, -0.11f, 0.09f,
    };

    const std::vector<float> q_weight = {
        0.20f, -0.10f, 0.05f, 0.30f,
        -0.25f, 0.15f, 0.40f, -0.05f,
        0.35f, 0.10f, -0.20f, 0.25f,
        0.05f, 0.30f, 0.15f, -0.10f,
    };
    const std::vector<float> k_weight = {
        -0.10f, 0.25f, 0.15f, 0.05f,
        0.20f, 0.05f, -0.30f, 0.10f,
        0.12f, -0.22f, 0.18f, 0.28f,
        0.08f, 0.14f, -0.12f, 0.32f,
    };
    const std::vector<float> v_weight = {
        0.30f, 0.05f, -0.10f, 0.20f,
        -0.05f, 0.18f, 0.22f, -0.15f,
        0.11f, -0.09f, 0.27f, 0.13f,
        0.07f, 0.26f, -0.04f, 0.19f,
    };
    std::vector<float> qkv_weight;
    qkv_weight.reserve(q_weight.size() + k_weight.size() + v_weight.size());
    qkv_weight.insert(qkv_weight.end(), q_weight.begin(), q_weight.end());
    qkv_weight.insert(qkv_weight.end(), k_weight.begin(), k_weight.end());
    qkv_weight.insert(qkv_weight.end(), v_weight.begin(), v_weight.end());
    const std::vector<float> out_weight = {
        0.15f, 0.05f, -0.20f, 0.25f,
        -0.12f, 0.18f, 0.22f, 0.08f,
        0.30f, -0.05f, 0.12f, 0.14f,
        0.10f, 0.16f, -0.08f, 0.20f,
    };
    const std::vector<float> pos_weight = {
        0.14f, -0.09f, 0.21f, 0.04f,
        0.07f, 0.25f, -0.11f, 0.16f,
        -0.13f, 0.18f, 0.09f, 0.22f,
        0.05f, 0.12f, 0.17f, -0.07f,
    };
    const std::vector<float> pos_bias_u = {
        0.10f, -0.20f,
        0.05f, 0.15f,
    };
    const std::vector<float> pos_bias_v = {
        -0.08f, 0.12f,
        0.07f, -0.03f,
    };

    auto make_relative_weights = [](CpuModuleRunner & runner, int64_t hidden_size, int64_t num_heads, bool fused, bool provide_projected_pos, int64_t batch_size, int64_t pos_frame_count) {
        engine::modules::RelativeAttentionWeights weights{
            {
                runner.make_f32(engine::core::TensorShape::from_dims({hidden_size, hidden_size})),
                std::nullopt,
                runner.make_f32(engine::core::TensorShape::from_dims({hidden_size, hidden_size})),
                std::nullopt,
                runner.make_f32(engine::core::TensorShape::from_dims({hidden_size, hidden_size})),
                std::nullopt,
                fused ? std::optional<engine::core::TensorValue>(runner.make_f32(engine::core::TensorShape::from_dims({hidden_size * 3, hidden_size}))) : std::nullopt,
                std::nullopt,
                runner.make_f32(engine::core::TensorShape::from_dims({hidden_size, hidden_size})),
                std::nullopt,
            },
            runner.make_f32(engine::core::TensorShape::from_dims({hidden_size, hidden_size})),
            runner.make_f32(engine::core::TensorShape::from_dims({num_heads, hidden_size / num_heads})),
            runner.make_f32(engine::core::TensorShape::from_dims({num_heads, hidden_size / num_heads})),
        };
        std::optional<engine::core::TensorValue> projected_pos;
        if (provide_projected_pos) {
            projected_pos = runner.make_f32(engine::core::TensorShape::from_dims({batch_size, pos_frame_count, hidden_size}));
        }
        return std::pair{weights, projected_pos};
    };

    auto write_relative_weights = [&](const engine::modules::RelativeAttentionWeights & weights,
                                      const std::optional<engine::core::TensorValue> & projected_pos) {
        engine::core::write_tensor_f32(weights.attention.q_weight, q_weight);
        engine::core::write_tensor_f32(weights.attention.k_weight, k_weight);
        engine::core::write_tensor_f32(weights.attention.v_weight, v_weight);
        if (weights.attention.qkv_weight.has_value()) {
            engine::core::write_tensor_f32(*weights.attention.qkv_weight, qkv_weight);
        }
        engine::core::write_tensor_f32(weights.attention.out_weight, out_weight);
        engine::core::write_tensor_f32(weights.pos_weight, pos_weight);
        engine::core::write_tensor_f32(weights.pos_bias_u, pos_bias_u);
        engine::core::write_tensor_f32(weights.pos_bias_v, pos_bias_v);
        if (projected_pos.has_value()) {
            engine::core::write_tensor_f32(*projected_pos, project_sequence(pos_values, pos_frames, hidden, pos_weight, hidden));
        }
    };

    CpuModuleRunner fused_runner;
    auto fused_input = fused_runner.make_f32(engine::core::TensorShape::from_dims({batch, frames, hidden}));
    auto fused_pos_emb = fused_runner.make_f32(engine::core::TensorShape::from_dims({batch, pos_frames, hidden}));
    auto [fused_weights, fused_projected_pos] = make_relative_weights(fused_runner, hidden, heads, true, false, batch, pos_frames);
    auto fused_output = engine::modules::RelativeSelfAttentionModule({hidden, heads, false, -1, -1, 0}).build(
        fused_runner.ctx,
        fused_input,
        fused_pos_emb,
        fused_weights,
        std::nullopt,
        std::nullopt,
        fused_projected_pos);
    fused_runner.allocate_tensors();
    engine::core::write_tensor_f32(fused_input, input_values);
    engine::core::write_tensor_f32(fused_pos_emb, pos_values);
    write_relative_weights(fused_weights, fused_projected_pos);
    const auto fused_values = fused_runner.run_f32(fused_output);
    const std::vector<float> expected_fused_values = {
        0.0190383f, 0.00988977f, 0.0295977f, 0.0188725f,
        0.0186866f, 0.00997738f, 0.0296115f, 0.0186434f,
        0.0197160f, 0.00805918f, 0.0315093f, 0.0185213f,
    };
    require_allclose(
        fused_values,
        expected_fused_values,
        1.0e-6f,
        "relative attention fused qkv golden output");

    CpuModuleRunner split_runner;
    auto split_input = split_runner.make_f32(engine::core::TensorShape::from_dims({batch, frames, hidden}));
    auto split_pos_emb = split_runner.make_f32(engine::core::TensorShape::from_dims({batch, pos_frames, hidden}));
    auto [split_weights, split_projected_pos] = make_relative_weights(split_runner, hidden, heads, false, false, batch, pos_frames);
    auto split_output = engine::modules::RelativeSelfAttentionModule({hidden, heads, false, -1, -1, 0}).build(
        split_runner.ctx,
        split_input,
        split_pos_emb,
        split_weights,
        std::nullopt,
        std::nullopt,
        split_projected_pos);
    split_runner.allocate_tensors();
    engine::core::write_tensor_f32(split_input, input_values);
    engine::core::write_tensor_f32(split_pos_emb, pos_values);
    write_relative_weights(split_weights, split_projected_pos);
    const auto split_values = split_runner.run_f32(split_output);
    require_allclose(fused_values, split_values, 1.0e-5f, "relative attention fused qkv vs split");

    CpuModuleRunner cached_runner;
    auto cached_input = cached_runner.make_f32(engine::core::TensorShape::from_dims({batch, frames, hidden}));
    auto cached_pos_emb = cached_runner.make_f32(engine::core::TensorShape::from_dims({batch, pos_frames, hidden}));
    auto [cached_weights, cached_projected_pos] = make_relative_weights(cached_runner, hidden, heads, true, true, batch, pos_frames);
    auto cached_output = engine::modules::RelativeSelfAttentionModule({hidden, heads, false, -1, -1, 0}).build(
        cached_runner.ctx,
        cached_input,
        cached_pos_emb,
        cached_weights,
        std::nullopt,
        std::nullopt,
        cached_projected_pos);
    cached_runner.allocate_tensors();
    engine::core::write_tensor_f32(cached_input, input_values);
    engine::core::write_tensor_f32(cached_pos_emb, pos_values);
    write_relative_weights(cached_weights, cached_projected_pos);
    const auto cached_values = cached_runner.run_f32(cached_output);
    require_allclose(fused_values, cached_values, 1.0e-5f, "relative attention cached projected pos");
}

void test_conformer_conv_pad_mask_ignores_dirty_padded_frames() {
    const int64_t batch = 1;
    const int64_t frames = 5;
    const int64_t hidden = 4;
    const int64_t valid_frames = 3;

    const std::vector<float> clean_input = {
        0.10f, -0.20f, 0.30f, 0.40f,
        0.05f, 0.12f, -0.18f, 0.22f,
        -0.11f, 0.07f, 0.15f, -0.09f,
        0.00f, 0.00f, 0.00f, 0.00f,
        0.00f, 0.00f, 0.00f, 0.00f,
    };
    std::vector<float> dirty_input = clean_input;
    dirty_input[12] = 2.5f; dirty_input[13] = -1.7f; dirty_input[14] = 0.9f; dirty_input[15] = 3.1f;
    dirty_input[16] = -2.2f; dirty_input[17] = 1.4f; dirty_input[18] = -0.6f; dirty_input[19] = 2.8f;
    const auto keep_mask = make_keep_mask_for_test(frames, valid_frames);

    auto make_weights = [](CpuModuleRunner & runner) {
        engine::modules::ConformerConvModuleWeights weights{
            {
                runner.make_f32(engine::core::TensorShape::from_dims({hidden})),
                runner.make_f32(engine::core::TensorShape::from_dims({hidden})),
            },
            {
                runner.make_f32(engine::core::TensorShape::from_dims({hidden * 2, hidden})),
                std::nullopt,
            },
            {
                runner.make_f32(engine::core::TensorShape::from_dims({hidden, 1, 3})),
                std::nullopt,
            },
            {
                runner.make_f32(engine::core::TensorShape::from_dims({hidden})),
                runner.make_f32(engine::core::TensorShape::from_dims({hidden})),
            },
            {
                runner.make_f32(engine::core::TensorShape::from_dims({hidden, hidden})),
                std::nullopt,
            },
        };
        return weights;
    };

    auto write_weights = [](const engine::modules::ConformerConvModuleWeights & weights) {
        engine::core::write_tensor_f32(*weights.norm.weight, {1.0f, 0.9f, 1.1f, 0.95f});
        engine::core::write_tensor_f32(*weights.norm.bias, {0.02f, -0.03f, 0.01f, 0.04f});
        engine::core::write_tensor_f32(weights.pointwise_in.weight, {
            0.20f, -0.10f, 0.05f, 0.30f,
            -0.12f, 0.14f, 0.08f, -0.20f,
            0.09f, 0.18f, -0.15f, 0.07f,
            0.04f, -0.05f, 0.16f, 0.11f,
            -0.03f, 0.17f, 0.06f, 0.12f,
            0.10f, -0.08f, 0.13f, -0.04f,
            0.07f, 0.15f, -0.02f, 0.09f,
            0.05f, 0.01f, 0.14f, -0.06f,
        });
        engine::core::write_tensor_f32(weights.depthwise.weight, {
            0.20f, 0.50f, -0.10f,
            -0.05f, 0.40f, 0.15f,
            0.10f, -0.20f, 0.30f,
            0.25f, 0.05f, -0.12f,
        });
        engine::core::write_tensor_f32(weights.depthwise_norm.scale, {0.95f, 1.05f, 0.90f, 1.10f});
        engine::core::write_tensor_f32(weights.depthwise_norm.bias, {0.01f, -0.02f, 0.03f, 0.00f});
        engine::core::write_tensor_f32(weights.pointwise_out.weight, {
            0.18f, -0.04f, 0.11f, 0.05f,
            -0.09f, 0.16f, 0.07f, 0.12f,
            0.13f, 0.02f, -0.06f, 0.14f,
            0.04f, 0.15f, 0.08f, -0.10f,
        });
    };

    CpuModuleRunner clean_runner;
    auto clean_tensor = clean_runner.make_f32(engine::core::TensorShape::from_dims({batch, frames, hidden}));
    auto pad_tensor_clean = clean_runner.make_i32(engine::core::TensorShape::from_dims({batch, frames}));
    auto clean_weights = make_weights(clean_runner);
    auto clean_output = engine::modules::ConformerConvModule({hidden, 3, false, 1.0e-5f, 0}).build(
        clean_runner.ctx,
        clean_tensor,
        clean_weights,
        pad_tensor_clean);
    clean_runner.allocate_tensors();
    engine::core::write_tensor_f32(clean_tensor, clean_input);
    engine::core::write_tensor_i32(pad_tensor_clean, keep_mask);
    write_weights(clean_weights);
    const auto clean_values = clean_runner.run_f32(clean_output);
    const std::vector<float> expected_clean_values = {
        0.0193402f,  -0.00830123f, 0.00372567f,  -0.00191416f,
        0.0102899f,  -0.00829928f, 0.0123794f,   -0.00320417f,
        -0.00452878f, 0.0123428f,  -0.00722798f,  0.00952792f,
        -0.000882533f, 0.000231944f, -0.00204736f, -0.00295283f,
        0.00297525f, -0.000970502f, -0.000458249f, -6.6002e-05f,
    };
    require_allclose(
        clean_values,
        expected_clean_values,
        1.0e-6f,
        "conformer conv golden output");

    CpuModuleRunner dirty_runner;
    auto dirty_tensor = dirty_runner.make_f32(engine::core::TensorShape::from_dims({batch, frames, hidden}));
    auto pad_tensor_dirty = dirty_runner.make_i32(engine::core::TensorShape::from_dims({batch, frames}));
    auto dirty_weights = make_weights(dirty_runner);
    auto dirty_output = engine::modules::ConformerConvModule({hidden, 3, false, 1.0e-5f, 0}).build(
        dirty_runner.ctx,
        dirty_tensor,
        dirty_weights,
        pad_tensor_dirty);
    dirty_runner.allocate_tensors();
    engine::core::write_tensor_f32(dirty_tensor, dirty_input);
    engine::core::write_tensor_i32(pad_tensor_dirty, keep_mask);
    write_weights(dirty_weights);
    const auto dirty_values = dirty_runner.run_f32(dirty_output);

    require_allclose(clean_values, dirty_values, 1.0e-5f, "conformer conv pad mask invariance");
}

void test_relative_attention_specialized_flash_matches_reference_on_realistic_shapes() {
    const int64_t batch = 1;
    const int64_t frames = 63;
    const int64_t valid_frames = 61;
    const int64_t hidden = 512;
    const int64_t heads = 8;
    const int64_t pos_frames = 2 * frames - 1;

    const auto input_values = make_patterned_f32(static_cast<size_t>(batch * frames * hidden), 0.13f, 0.08f);
    const auto pos_values = make_patterned_f32(static_cast<size_t>(batch * pos_frames * hidden), -0.21f, 0.08f);
    const auto q_weight = make_patterned_f32(static_cast<size_t>(hidden * hidden), 0.07f, 0.020f);
    const auto k_weight = make_patterned_f32(static_cast<size_t>(hidden * hidden), -0.11f, 0.020f);
    const auto v_weight = make_patterned_f32(static_cast<size_t>(hidden * hidden), 0.19f, 0.022f);
    const auto out_weight = make_patterned_f32(static_cast<size_t>(hidden * hidden), -0.17f, 0.018f);
    const auto pos_weight = make_patterned_f32(static_cast<size_t>(hidden * hidden), 0.23f, 0.018f);
    const auto pos_bias_u = make_patterned_f32(static_cast<size_t>(heads * (hidden / heads)), 0.31f, 0.010f);
    const auto pos_bias_v = make_patterned_f32(static_cast<size_t>(heads * (hidden / heads)), -0.29f, 0.010f);
    const auto attention_bias = make_global_attention_bias_for_test(frames, valid_frames);
    const auto keep_mask = make_keep_mask_for_test(frames, valid_frames);
    const auto projected_pos_values = project_sequence(pos_values, pos_frames, hidden, pos_weight, hidden);

    auto run_case = [&](engine::core::BackendType backend_type, const std::string & backend_label) {
        ModuleRunner reference_runner;
        reference_runner.backend_config.type = backend_type;
        if (reference_runner.backend != nullptr) {
            ggml_backend_free(reference_runner.backend);
            reference_runner.backend = nullptr;
        }
        reference_runner.backend = engine::core::init_backend(reference_runner.backend_config);

        auto ref_input = reference_runner.make_f32(engine::core::TensorShape::from_dims({batch, frames, hidden}));
        auto ref_pos = reference_runner.make_f32(engine::core::TensorShape::from_dims({batch, pos_frames, hidden}));
        auto ref_mask = reference_runner.make_f32(engine::core::TensorShape::from_dims({frames, frames}));
        auto ref_keep = reference_runner.make_i32(engine::core::TensorShape::from_dims({batch, frames}));
        engine::modules::RelativeAttentionWeights ref_weights{
            {
                reference_runner.make_f32(engine::core::TensorShape::from_dims({hidden, hidden})),
                std::nullopt,
                reference_runner.make_f32(engine::core::TensorShape::from_dims({hidden, hidden})),
                std::nullopt,
                reference_runner.make_f32(engine::core::TensorShape::from_dims({hidden, hidden})),
                std::nullopt,
                std::nullopt,
                std::nullopt,
                reference_runner.make_f32(engine::core::TensorShape::from_dims({hidden, hidden})),
                std::nullopt,
            },
            reference_runner.make_f32(engine::core::TensorShape::from_dims({hidden, hidden})),
            reference_runner.make_f32(engine::core::TensorShape::from_dims({heads, hidden / heads})),
            reference_runner.make_f32(engine::core::TensorShape::from_dims({heads, hidden / heads})),
        };
        auto ref_output = engine::modules::RelativeSelfAttentionModule({hidden, heads, false, -1, -1, 0, false}).build(
            reference_runner.ctx,
            ref_input,
            ref_pos,
            ref_weights,
            ref_mask,
            ref_keep,
            std::nullopt);
        reference_runner.allocate_tensors();
        engine::core::write_tensor_f32(ref_input, input_values);
        engine::core::write_tensor_f32(ref_pos, pos_values);
        engine::core::write_tensor_f32(ref_mask, attention_bias);
        engine::core::write_tensor_i32(ref_keep, keep_mask);
        engine::core::write_tensor_f32(ref_weights.attention.q_weight, q_weight);
        engine::core::write_tensor_f32(ref_weights.attention.k_weight, k_weight);
        engine::core::write_tensor_f32(ref_weights.attention.v_weight, v_weight);
        engine::core::write_tensor_f32(ref_weights.attention.out_weight, out_weight);
        engine::core::write_tensor_f32(ref_weights.pos_weight, pos_weight);
        engine::core::write_tensor_f32(ref_weights.pos_bias_u, pos_bias_u);
        engine::core::write_tensor_f32(ref_weights.pos_bias_v, pos_bias_v);
        const auto reference_values = reference_runner.run_f32(ref_output);

        ModuleRunner flash_runner;
        flash_runner.backend_config.type = backend_type;
        if (flash_runner.backend != nullptr) {
            ggml_backend_free(flash_runner.backend);
            flash_runner.backend = nullptr;
        }
        flash_runner.backend = engine::core::init_backend(flash_runner.backend_config);

        auto flash_input = flash_runner.make_f32(engine::core::TensorShape::from_dims({batch, frames, hidden}));
        auto flash_pos = flash_runner.make_f32(engine::core::TensorShape::from_dims({batch, pos_frames, hidden}));
        auto flash_mask = flash_runner.make_f32(engine::core::TensorShape::from_dims({frames, frames}));
        auto flash_keep = flash_runner.make_i32(engine::core::TensorShape::from_dims({batch, frames}));
        engine::modules::RelativeAttentionWeights flash_weights{
            {
                flash_runner.make_f32(engine::core::TensorShape::from_dims({hidden, hidden})),
                std::nullopt,
                flash_runner.make_f32(engine::core::TensorShape::from_dims({hidden, hidden})),
                std::nullopt,
                flash_runner.make_f32(engine::core::TensorShape::from_dims({hidden, hidden})),
                std::nullopt,
                std::nullopt,
                std::nullopt,
                flash_runner.make_f32(engine::core::TensorShape::from_dims({hidden, hidden})),
                std::nullopt,
            },
            flash_runner.make_f32(engine::core::TensorShape::from_dims({hidden, hidden})),
            flash_runner.make_f32(engine::core::TensorShape::from_dims({heads, hidden / heads})),
            flash_runner.make_f32(engine::core::TensorShape::from_dims({heads, hidden / heads})),
        };
        auto flash_projected_pos = flash_runner.make_f32(engine::core::TensorShape::from_dims({batch, pos_frames, hidden}));
        auto flash_output = engine::modules::RelativeSelfAttentionModule({hidden, heads, false, -1, -1, 0, true}).build(
            flash_runner.ctx,
            flash_input,
            flash_pos,
            flash_weights,
            flash_mask,
            flash_keep,
            flash_projected_pos);
        flash_runner.allocate_tensors();
        engine::core::write_tensor_f32(flash_input, input_values);
        engine::core::write_tensor_f32(flash_pos, pos_values);
        engine::core::write_tensor_f32(flash_mask, attention_bias);
        engine::core::write_tensor_i32(flash_keep, keep_mask);
        engine::core::write_tensor_f32(flash_projected_pos, projected_pos_values);
        engine::core::write_tensor_f32(flash_weights.attention.q_weight, q_weight);
        engine::core::write_tensor_f32(flash_weights.attention.k_weight, k_weight);
        engine::core::write_tensor_f32(flash_weights.attention.v_weight, v_weight);
        engine::core::write_tensor_f32(flash_weights.attention.out_weight, out_weight);
        engine::core::write_tensor_f32(flash_weights.pos_weight, pos_weight);
        engine::core::write_tensor_f32(flash_weights.pos_bias_u, pos_bias_u);
        engine::core::write_tensor_f32(flash_weights.pos_bias_v, pos_bias_v);
        const auto flash_values = flash_runner.run_f32(flash_output);

        const float max_abs_tol = backend_type == engine::core::BackendType::Cuda ? 4.0e-3f : 4.0e-3f;
        const double mean_abs_tol = backend_type == engine::core::BackendType::Cuda ? 7.5e-4 : 6.0e-4;
        require_max_abs_diff_below(
            flash_values,
            reference_values,
            max_abs_tol,
            mean_abs_tol,
            "relative attention flash parity " + backend_label);
    };

    run_case(engine::core::BackendType::Cpu, "cpu");
    if (backend_is_available(engine::core::BackendType::Cuda)) {
        run_case(engine::core::BackendType::Cuda, "cuda");
    } else {
        std::cout << "encoder_module_test: skipping cuda flash parity\n";
    }
}

void test_glu_contiguous_gate_opt_in() {
    using namespace engine;
    modules::GLUModule legacy = {};
    legacy = []() -> modules::GLUModule { return {}; }();
    const std::vector<float> data{1, -2, 3, 4, 0, 1, -1, 2, -3, 2, 1, -1, -2, 0, 2, 1};
    std::vector<float> expected;
    for (size_t row = 0; row < 2; ++row) {
        for (size_t col = 0; col < 4; ++col) {
            expected.push_back(data[row * 8 + col] / (1.0f + std::exp(-data[row * 8 + col + 4])));
        }
    }
    require(!modules::GLUConfig{}.contiguous_gate, "GLU layout change must be opt-in");
    require(!modules::ConformerBlockConfig{}.contiguous_glu_gate, "Conformer layout change must be opt-in");
    for (const auto type : {core::BackendType::Cpu, core::BackendType::Cuda}) {
        if (type == core::BackendType::Cuda && !backend_is_available(type)) {
            continue;
        }
        for (const bool opt_in : {false, true}) {
            if (type == core::BackendType::Cuda && !opt_in) {
                continue;  // Legacy strided sigmoid is not supported on CUDA.
            }
            ModuleRunner runner;
            set_runner_backend(runner, type);
            runner.ctx.backend_type = type;
            auto input = runner.make_f32(core::TensorShape::from_dims({1, 2, 8}));
            const auto module = opt_in ? modules::GLUModule(modules::GLUConfig{true}) : legacy;
            auto output = module.build(runner.ctx, input);
            auto graph = ggml_new_graph_custom(runner.ggml, kTestGraphNodes, false);
            ggml_build_forward_expand(graph, output.tensor);
            bool found_sigmoid = false;
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                auto node = ggml_graph_node(graph, i);
                if (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SIGMOID) {
                    require(ggml_is_contiguous(node->src[0]) == opt_in, "GLU gate layout opt-in");
                    found_sigmoid = true;
                }
            }
            require(found_sigmoid, "GLU sigmoid node");
            runner.allocate_tensors();
            core::write_tensor_f32(input, data);
            require_allclose(runner.run_f32(output), expected, 1e-5f, "GLU output");
        }
    }
}

void test_depthwise_subsampling_stage_masks() {
    using namespace engine;
    ModuleRunner runner;
    auto input = runner.make_f32(core::TensorShape::from_dims({1, 8, 4}));
    auto first_mask = runner.make_i32(core::TensorShape::from_dims({1, 4}));
    auto second_mask = runner.make_i32(core::TensorShape::from_dims({1, 2}));
    auto kernel = runner.make_f32(core::TensorShape::from_dims({1, 1, 1, 1}));
    auto bias = runner.make_f32(core::TensorShape::from_dims({1}));
    auto depth_bias = runner.make_f32(core::TensorShape::from_dims({1}));
    auto point_kernel = runner.make_f32(core::TensorShape::from_dims({1, 1, 1, 1}));
    auto point_bias = runner.make_f32(core::TensorShape::from_dims({1}));
    auto proj = runner.make_f32(core::TensorShape::from_dims({1, 1}));
    auto proj_bias = runner.make_f32(core::TensorShape::from_dims({1}));
    modules::DepthwiseConvSubsamplingWeights weights{
        {kernel, bias}, {{{kernel, depth_bias}, {point_kernel, point_bias}}}, {proj, proj_bias}};
    const modules::DepthwiseConvSubsamplingModule module({4, 1, 1, 1, 2, 0, true});
    auto masked = module.build(runner.ctx, input, weights, {first_mask, second_mask});
    auto unmasked = module.build(runner.ctx, input, weights);
    require(masked.shape.dims[1] == 2, "two-stage subsampling output length");
    bool rejected = false;
    try {
        module.build(runner.ctx, input, weights, {first_mask});
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, "subsampling rejects missing stage masks");
    runner.allocate_tensors();
    std::vector<float> data(32);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<float>(i + 1);
    core::write_tensor_f32(input, data);
    core::write_tensor_f32(kernel, {1});
    core::write_tensor_f32(bias, {1});
    core::write_tensor_f32(depth_bias, {2});
    core::write_tensor_f32(point_kernel, {3});
    core::write_tensor_f32(point_bias, {5});
    core::write_tensor_f32(proj, {2});
    core::write_tensor_f32(proj_bias, {7});
    core::write_tensor_i32(first_mask, {1, 1, 1, 0});
    core::write_tensor_i32(second_mask, {1, 0});
    require_allclose(runner.run_f32(masked), {41, 7}, 1e-6f, "subsampling tail mask");
    require_allclose(runner.run_f32(unmasked), {41, 137}, 1e-6f, "subsampling unmasked");
    core::write_tensor_i32(first_mask, {1, 1, 0, 0});
    core::write_tensor_i32(second_mask, {1, 1});
    require_allclose(runner.run_f32(masked), {41, 29}, 1e-6f, "subsampling earlier stage mask");
}

class AffineTestSource final : public engine::assets::TensorSource {
public:
    std::unordered_map<std::string, std::vector<float>> values;
    const std::filesystem::path & source_path() const noexcept override { return path_; }
    bool has_tensor(std::string_view name) const noexcept override { return values.count(std::string(name)) != 0; }
    engine::assets::TensorMetadata require_metadata(std::string_view name) const override {
        return {std::string(name), "f32", {static_cast<int64_t>(values.at(std::string(name)).size())}};
    }
    std::vector<engine::assets::TensorMetadata> tensors() const override {
        std::vector<engine::assets::TensorMetadata> result;
        for (const auto & entry : values) result.push_back(require_metadata(entry.first));
        return result;
    }
    engine::assets::RawTensorData require_tensor_data(std::string_view name) const override {
        engine::assets::RawTensorData result;
        result.metadata = require_metadata(name);
        const auto & data = values.at(std::string(name));
        result.bytes.resize(data.size() * sizeof(float));
        std::memcpy(result.bytes.data(), data.data(), result.bytes.size());
        return result;
    }
    std::vector<float> require_f32(std::string_view name,
        const std::optional<std::vector<int64_t>> & shape) const override {
        require(!shape || *shape == require_metadata(name).shape, "affine source shape");
        return values.at(std::string(name));
    }
    std::optional<std::vector<float>> optional_f32(std::string_view name,
        const std::optional<std::vector<int64_t>> & shape) const override {
        return has_tensor(name) ? std::optional<std::vector<float>>(require_f32(name, shape)) : std::nullopt;
    }
    int64_t require_i64_scalar(std::string_view) const override { throw std::runtime_error("not an integer tensor"); }
private:
    std::filesystem::path path_{"affine-test"};
};

void test_batch_norm_eval_binding() {
    using namespace engine;
    ModuleRunner runner;
    core::BackendWeightStore store(runner.backend, core::BackendType::Cpu, "affine-test", 1024 * 1024);
    AffineTestSource source;
    source.values = {{"bn.weight", {2, -3}}, {"bn.bias", {1, 2}},
        {"bn.running_mean", {4, -2}}, {"bn.running_var", {3, 8}}};
    auto weights = modules::binding::batch_norm_eval_from_source(store, source, "bn", 2, 1.0f);
    store.upload();
    std::vector<float> scale, bias;
    core::read_tensor_f32_into(weights.scale.tensor, scale);
    core::read_tensor_f32_into(weights.bias.tensor, bias);
    require_allclose(scale, {1, -1}, 1e-6f, "batch norm eval scale");
    require_allclose(bias, {-3, 0}, 1e-6f, "batch norm eval bias");
}

void test_feed_forward_activation_opt_in() {
    using namespace engine;
    ModuleRunner runner;
    auto input = runner.make_f32(core::TensorShape::from_dims({1, 1, 4}));
    auto identity = runner.make_f32(core::TensorShape::from_dims({4, 4}));
    modules::FeedForwardWeights weights{identity, std::nullopt, identity, std::nullopt};
    modules::FeedForwardConfig config{4, 4, false};
    require(config.activation == modules::FeedForwardActivation::Gelu, "default activation remains GELU");
    auto legacy = modules::FeedForwardModule(config).build(runner.ctx, input, weights);
    config.activation = modules::FeedForwardActivation::Relu;
    auto relu = modules::FeedForwardModule(config).build(runner.ctx, input, weights);
    runner.allocate_tensors();
    const std::vector<float> data{-2, -1, 1, 2};
    core::write_tensor_f32(input, data);
    core::write_tensor_f32(identity, {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1});
    std::vector<float> expected;
    for (float v : data) expected.push_back(0.5f * v * (1.0f + std::erf(v / std::sqrt(2.0f))));
    require_allclose(runner.run_f32(legacy), expected, 1e-6f, "legacy GELU");
    require_allclose(runner.run_f32(relu), {0,0,1,2}, 1e-6f, "opt-in ReLU");
}

void test_cached_decoder_block_matches_composition(bool flash_cross) {
    using namespace engine;
    using core::TensorShape;
    constexpr int64_t hidden = 32;
    ModuleRunner runner;
    std::vector<std::pair<core::TensorValue, std::vector<float>>> initializers;
    auto param = [&](const TensorShape & shape) {
        auto tensor = runner.make_f32(shape);
        initializers.emplace_back(tensor, make_patterned_f32(shape.num_elements(), 0.37f, 0.07f));
        return tensor;
    };
    auto norm = [&]() -> modules::NormWeights {
        auto weight = param(TensorShape::from_dims({hidden}));
        initializers.back().second.assign(hidden, 1.0f);
        return {weight, param(TensorShape::from_dims({hidden}))};
    };
    modules::TransformerDecoderBlockWeights w;
    w.norm1 = norm(); w.norm2 = norm(); w.norm3 = norm();
    w.self_attention.qkv_weight = param(TensorShape::from_dims({3 * hidden, hidden}));
    w.self_attention.qkv_bias = param(TensorShape::from_dims({3 * hidden}));
    w.self_attention.out_weight = param(TensorShape::from_dims({hidden, hidden}));
    w.self_attention.out_bias = param(TensorShape::from_dims({hidden}));
    w.cross_attention.q_weight = param(TensorShape::from_dims({hidden, hidden}));
    w.cross_attention.q_bias = param(TensorShape::from_dims({hidden}));
    w.cross_attention.out_weight = param(TensorShape::from_dims({hidden, hidden}));
    w.cross_attention.out_bias = param(TensorShape::from_dims({hidden}));
    w.feed_forward = {param(TensorShape::from_dims({64, hidden})), param(TensorShape::from_dims({64})),
        param(TensorShape::from_dims({hidden, 64})), param(TensorShape::from_dims({hidden}))};
    auto input = runner.make_f32(TensorShape::from_dims({1, 1, hidden}));
    auto slot = runner.make_i32(TensorShape::from_dims({1}));
    auto mask = core::make_tensor(runner.ctx, GGML_TYPE_F16, TensorShape::from_dims({1, 8}));
    auto memory_mask = runner.make_i32(TensorShape::from_dims({1, 3}));
    auto flash_mask = core::make_tensor(runner.ctx, GGML_TYPE_F16, TensorShape::from_dims({1, 1, 1, 3}));
    modules::CrossAttentionKeyValue memory{param(TensorShape::from_dims({1, 1, 3, hidden})),
        param(TensorShape::from_dims({1, 1, 3, hidden}))};
    std::vector<core::TensorValue> caches;
    for (int i = 0; i < 4; ++i) caches.push_back(core::make_tensor(runner.ctx, GGML_TYPE_F16,
        TensorShape::from_dims({1, 8, 1, hidden})));
    modules::TransformerDecoderBlockConfig config{hidden, 1, 64};
    require(!config.use_flash_cross_attention, "cached cross flash remains opt-in");
    config.use_flash_cross_attention = flash_cross;
    config.activation = modules::FeedForwardActivation::Relu;
    config.use_packed_qkv = true;
    config.use_packed_kv = true;
    auto actual = modules::TransformerDecoderBlockModule(config).build_cached_tail(runner.ctx, input, w,
        caches[0], caches[1], slot, mask, memory, flash_cross ? flash_mask : memory_mask);
    modules::AttentionConfig attention{hidden, 1, true};
    attention.causal = true;
    attention.use_packed_qkv = true;
    auto y = modules::LayerNormModule({hidden}).build(runner.ctx, input, w.norm1);
    y = modules::SelfAttentionModule(attention).build_cached_tail(runner.ctx, y, w.self_attention,
        caches[2], caches[3], slot, mask).output;
    auto expected = modules::AddModule().build(runner.ctx, input, y);
    y = modules::LayerNormModule({hidden}).build(runner.ctx, expected, w.norm2);
    modules::AttentionConfig cross_attention{hidden, 1, true};
    cross_attention.use_packed_kv = true;
    y = modules::CrossAttentionModule(cross_attention).build_cached(runner.ctx, y, memory, w.cross_attention, memory_mask);
    expected = modules::AddModule().build(runner.ctx, expected, y);
    y = modules::LayerNormModule({hidden}).build(runner.ctx, expected, w.norm3);
    y = modules::LinearModule({hidden, 64, true}).build(runner.ctx, y, {w.feed_forward.fc1_weight, w.feed_forward.fc1_bias});
    y = modules::ReluModule().build(runner.ctx, y);
    y = modules::LinearModule({64, hidden, true}).build(runner.ctx, y, {w.feed_forward.fc2_weight, w.feed_forward.fc2_bias});
    expected = modules::AddModule().build(runner.ctx, expected, y);
    runner.allocate_tensors();
    for (const auto & item : initializers) core::write_tensor_f32(item.first, item.second);
    for (const auto & cache : caches) core::write_tensor_f16(cache, std::vector<float>(8 * hidden, 0));
    core::write_tensor_i32(memory_mask, {1, 1, 0});
    core::write_tensor_f16(flash_mask, {0, 0, -std::numeric_limits<float>::infinity()});
    for (int32_t step = 0; step < 3; ++step) {
        core::write_tensor_f32(input, make_patterned_f32(hidden, 0.4f + step, 0.3f));
        core::write_tensor_i32(slot, &step, 1);
        std::vector<float> causal(8, -std::numeric_limits<float>::infinity());
        std::fill_n(causal.begin(), step + 1, 0.0f);
        core::write_tensor_f16(mask, causal);
        require_allclose(runner.run_f32(actual), runner.run_f32(expected), 1e-5f, "cached decoder composition");
    }
}

void test_cached_cross_flash_masks() {
    using namespace engine;
    using core::TensorShape;
    ModuleRunner runner;
    modules::AttentionConfig config{64, 2, false};
    config.use_packed_kv = true;
    const modules::CrossAttentionModule module(config);
    auto query = runner.make_f32(TensorShape::from_dims({2, 2, 64}));
    auto weight = runner.make_f32(TensorShape::from_dims({64, 64}));
    modules::AttentionWeights weights;
    weights.q_weight = weight;
    weights.out_weight = weight;
    modules::CrossAttentionKeyValue kv{
        runner.make_f32(TensorShape::from_dims({2, 2, 3, 32})),
        runner.make_f32(TensorShape::from_dims({2, 2, 3, 32}))};
    auto keep = runner.make_i32(TensorShape::from_dims({2, 3}));
    auto mask = core::make_tensor(runner.ctx, GGML_TYPE_F16, TensorShape::from_dims({2, 1, 2, 3}));
    auto shared_mask = core::make_tensor(runner.ctx, GGML_TYPE_F16, TensorShape::from_dims({1, 1, 2, 3}));
    auto per_head_mask = core::make_tensor(runner.ctx, GGML_TYPE_F16, TensorShape::from_dims({2, 2, 2, 3}));
    auto expected = module.build_cached(runner.ctx, query, kv, weights, keep);
    auto actual = module.build_cached_flash(runner.ctx, query, kv, weights, mask);
    auto shared = module.build_cached_flash(runner.ctx, query, kv, weights, shared_mask);
    auto per_head = module.build_cached_flash(runner.ctx, query, kv, weights, per_head_mask);
    for (const auto & invalid : {keep,
        runner.make_f32(TensorShape::from_dims({2, 1, 2, 3})),
        core::make_tensor(runner.ctx, GGML_TYPE_F16, TensorShape::from_dims({2, 1, 1, 3})),
        core::make_tensor(runner.ctx, GGML_TYPE_F16, TensorShape::from_dims({2, 1, 2, 4}))}) {
        bool rejected = false;
        try { module.build_cached_flash(runner.ctx, query, kv, weights, invalid); }
        catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "cached flash rejects invalid mask contracts");
    }
    runner.allocate_tensors();
    std::vector<float> identity(64 * 64, 0);
    for (size_t i = 0; i < 64; ++i) identity[i * 64 + i] = 1;
    core::write_tensor_f32(weight, identity);
    core::write_tensor_f32(query, make_patterned_f32(256, 0.2f, 0.3f));
    core::write_tensor_f32(kv.key, make_patterned_f32(384, 0.4f, 0.2f));
    core::write_tensor_f32(kv.value, make_patterned_f32(384, 0.7f, 0.1f));
    const float excluded = -std::numeric_limits<float>::infinity();
    core::write_tensor_i32(keep, {1, 1, 0, 1, 0, 1});
    core::write_tensor_f16(mask, {0, 0, excluded, 0, 0, excluded, 0, excluded, 0, 0, excluded, 0});
    require_allclose(runner.run_f32(actual), runner.run_f32(expected), 1e-5f, "batched cached flash padding");
    core::write_tensor_i32(keep, {1, 1, 0, 1, 1, 0});
    core::write_tensor_f16(shared_mask, {0, 0, excluded, 0, 0, excluded});
    std::vector<float> per_head_values(24);
    for (size_t i = 0; i < per_head_values.size(); ++i) per_head_values[i] = i % 3 == 2 ? excluded : 0;
    core::write_tensor_f16(per_head_mask, per_head_values);
    require_allclose(runner.run_f32(shared), runner.run_f32(expected), 1e-5f, "shared cached flash mask");
    require_allclose(runner.run_f32(per_head), runner.run_f32(expected), 1e-5f, "per-head cached flash mask");
}

}  // namespace

int main() {
    try {
        test_glu_contiguous_gate_opt_in();
        test_depthwise_subsampling_stage_masks();
        test_batch_norm_eval_binding();
        test_feed_forward_activation_opt_in();
        test_cached_decoder_block_matches_composition(false);
        test_cached_decoder_block_matches_composition(true);
        test_cached_cross_flash_masks();
        test_graph_optimizer_elides_metadata_nodes_without_changing_output();
        test_graph_optimizer_two_sided_broadcast_binary_matches_repeat();
        test_graph_optimizer_unary_broadcast_scale_matches_repeat();
        test_relative_attention_fused_qkv_matches_split_and_cached_pos();
        test_relative_attention_specialized_flash_matches_reference_on_realistic_shapes();
        test_conformer_conv_pad_mask_ignores_dirty_padded_frames();
        std::cout << "encoder_module_test: ok\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "encoder_module_test: failed: " << ex.what() << "\n";
        return 1;
    }
}
