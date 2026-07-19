#include "engine/framework/core/backend.h"
#include "engine/framework/modules/attention/qwen_causal_decoder.h"
#include "engine/framework/modules/attention/qwen_decoder.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"

#include <cmath>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t kGraphBytes = 16 * 1024 * 1024;
constexpr size_t kGraphNodes = 4096;

std::vector<float> patterned(size_t count, float phase, float scale) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        const float x = static_cast<float>(i);
        values[i] = scale * (std::sin(phase + 0.19f * x) + 0.35f * std::cos(phase + 0.07f * x));
    }
    return values;
}

void require_allclose(
    const std::vector<float> & actual,
    const std::vector<float> & expected,
    float tolerance,
    const std::string & label) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(label + " size mismatch");
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        if (diff > tolerance) {
            std::ostringstream message;
            message << label << " mismatch at " << i << ": expected " << expected[i]
                    << ", got " << actual[i] << ", diff=" << diff;
            throw std::runtime_error(message.str());
        }
    }
}

struct LayerResult {
    std::vector<float> output;
    std::vector<float> key;
    std::vector<float> value;
};

LayerResult run_layer(bool packed) {
    constexpr int64_t batch = 1;
    constexpr int64_t steps = 3;
    constexpr int64_t hidden = 8;
    constexpr int64_t heads = 2;
    constexpr int64_t kv_heads = 1;
    constexpr int64_t head_dim = 4;
    constexpr int64_t intermediate = 12;
    constexpr int64_t q_out = heads * head_dim;
    constexpr int64_t kv_out = kv_heads * head_dim;

    engine::core::BackendConfig backend_config{engine::core::BackendType::Cpu, 0, 4};
    ggml_backend_t backend = engine::core::init_backend(backend_config);
    if (backend == nullptr) {
        throw std::runtime_error("failed to initialize CPU backend");
    }

    ggml_init_params params{kGraphBytes, nullptr, true};
    ggml_context * ggml = ggml_init(params);
    if (ggml == nullptr) {
        ggml_backend_free(backend);
        throw std::runtime_error("failed to initialize GGML context");
    }

    ggml_backend_buffer_t buffer = nullptr;
    try {
        engine::core::ModuleBuildContext ctx{ggml, "qwen_packed_projection_test", engine::core::BackendType::Cpu};
        auto make_f32 = [&](std::initializer_list<int64_t> dims) {
            return engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims(dims));
        };

        auto input = make_f32({batch, steps, hidden});
        auto positions = engine::core::make_tensor(
            ctx,
            GGML_TYPE_I32,
            engine::core::TensorShape::from_dims({steps}));

        engine::modules::QwenDecoderLayerWeights weights;
        weights.input_norm = {make_f32({hidden}), std::nullopt};
        weights.post_norm = {make_f32({hidden}), std::nullopt};
        weights.self_attention.out_weight = make_f32({hidden, hidden});
        weights.mlp.down_proj = {make_f32({hidden, intermediate}), std::nullopt};

        const auto q_values = patterned(static_cast<size_t>(q_out * hidden), 0.1f, 0.12f);
        const auto k_values = patterned(static_cast<size_t>(kv_out * hidden), 0.5f, 0.10f);
        const auto v_values = patterned(static_cast<size_t>(kv_out * hidden), 0.9f, 0.08f);
        const auto gate_values = patterned(static_cast<size_t>(intermediate * hidden), 1.3f, 0.11f);
        const auto up_values = patterned(static_cast<size_t>(intermediate * hidden), 1.7f, 0.09f);

        if (packed) {
            weights.self_attention.qkv_weight = make_f32({q_out + 2 * kv_out, hidden});
            weights.mlp.gate_up_proj = engine::modules::LinearWeights{
                make_f32({intermediate * 2, hidden}),
                std::nullopt,
            };
        } else {
            weights.self_attention.q_weight = make_f32({q_out, hidden});
            weights.self_attention.k_weight = make_f32({kv_out, hidden});
            weights.self_attention.v_weight = make_f32({kv_out, hidden});
            weights.mlp.gate_proj = {make_f32({intermediate, hidden}), std::nullopt};
            weights.mlp.up_proj = {make_f32({intermediate, hidden}), std::nullopt};
        }

        engine::modules::QwenDecoderLayerConfig config;
        config.hidden_size = hidden;
        config.num_attention_heads = heads;
        config.num_key_value_heads = kv_heads;
        config.head_dim = head_dim;
        config.intermediate_size = intermediate;
        config.rms_norm_eps = 1e-5f;
        config.qkv_layout = packed
            ? engine::modules::QwenDecoderQKVLayout::PackedQKV
            : engine::modules::QwenDecoderQKVLayout::Separate;
        config.use_qk_norm = false;
        config.runtime.attention.prefill_mode = engine::modules::QwenDecoderAttentionMode::ManualRepeat;

        const auto outputs = engine::modules::QwenDecoderLayerModule(config).build(
            ctx,
            input,
            positions,
            weights);

        ggml_cgraph * graph = ggml_new_graph_custom(ggml, kGraphNodes, false);
        ggml_build_forward_expand(graph, outputs.output.tensor);
        buffer = ggml_backend_alloc_ctx_tensors(ggml, backend);
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate test tensors");
        }

        engine::core::write_tensor_f32(input, patterned(static_cast<size_t>(batch * steps * hidden), 2.1f, 0.20f));
        engine::core::write_tensor_i32(positions, {0, 1, 2});
        engine::core::write_tensor_f32(*weights.input_norm.weight, patterned(hidden, 0.3f, 0.7f));
        engine::core::write_tensor_f32(*weights.post_norm.weight, patterned(hidden, 0.7f, 0.8f));
        engine::core::write_tensor_f32(
            weights.self_attention.out_weight,
            patterned(static_cast<size_t>(hidden * hidden), 1.1f, 0.10f));
        engine::core::write_tensor_f32(
            weights.mlp.down_proj.weight,
            patterned(static_cast<size_t>(hidden * intermediate), 1.9f, 0.10f));

        if (packed) {
            std::vector<float> qkv_values;
            qkv_values.reserve(q_values.size() + k_values.size() + v_values.size());
            qkv_values.insert(qkv_values.end(), q_values.begin(), q_values.end());
            qkv_values.insert(qkv_values.end(), k_values.begin(), k_values.end());
            qkv_values.insert(qkv_values.end(), v_values.begin(), v_values.end());
            engine::core::write_tensor_f32(*weights.self_attention.qkv_weight, qkv_values);

            std::vector<float> gate_up_values;
            gate_up_values.reserve(gate_values.size() + up_values.size());
            gate_up_values.insert(gate_up_values.end(), gate_values.begin(), gate_values.end());
            gate_up_values.insert(gate_up_values.end(), up_values.begin(), up_values.end());
            engine::core::write_tensor_f32(weights.mlp.gate_up_proj->weight, gate_up_values);
        } else {
            engine::core::write_tensor_f32(weights.self_attention.q_weight, q_values);
            engine::core::write_tensor_f32(weights.self_attention.k_weight, k_values);
            engine::core::write_tensor_f32(weights.self_attention.v_weight, v_values);
            engine::core::write_tensor_f32(weights.mlp.gate_proj.weight, gate_values);
            engine::core::write_tensor_f32(weights.mlp.up_proj.weight, up_values);
        }

        ggml_backend_graph_compute(backend, graph);
        LayerResult result;
        engine::core::read_tensor_f32_into(outputs.output.tensor, result.output);
        engine::core::read_tensor_f32_into(outputs.key.tensor, result.key);
        engine::core::read_tensor_f32_into(outputs.value.tensor, result.value);

        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
        ggml_free(ggml);
        ggml_backend_free(backend);
        return result;
    } catch (...) {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        ggml_free(ggml);
        ggml_backend_free(backend);
        throw;
    }
}

void test_packed_qkv_and_gate_up_match_separate_projections() {
    const auto separate = run_layer(false);
    const auto packed = run_layer(true);
    require_allclose(packed.output, separate.output, 2.0e-5f, "decoder output");
    require_allclose(packed.key, separate.key, 2.0e-5f, "decoder key");
    require_allclose(packed.value, separate.value, 2.0e-5f, "decoder value");
}

void test_suffix_causal_mask() {
    const auto values = engine::modules::qwen_causal_suffix_mask_values(2, 3, 2);
    if (values.size() != 30) {
        throw std::runtime_error("suffix causal mask size mismatch");
    }
    const std::vector<bool> expected{
        true, true, true, false, false,
        true, true, true, true, false,
        true, true, true, true, true,
    };
    for (int batch = 0; batch < 2; ++batch) {
        for (size_t i = 0; i < expected.size(); ++i) {
            const float actual = ggml_fp16_to_fp32(values[static_cast<size_t>(batch) * expected.size() + i]);
            if ((expected[i] && actual != 0.0F) || (!expected[i] && !std::isinf(actual))) {
                throw std::runtime_error("suffix causal mask visibility mismatch");
            }
        }
    }
}

void test_f16_kv_set_rows() {
    engine::core::BackendConfig backend_config{engine::core::BackendType::Cpu, 0, 4};
    ggml_backend_t backend = engine::core::init_backend(backend_config);
    if (backend == nullptr) {
        throw std::runtime_error("failed to initialize CPU backend");
    }

    ggml_init_params params{kGraphBytes, nullptr, true};
    ggml_context * ggml = ggml_init(params);
    if (ggml == nullptr) {
        ggml_backend_free(backend);
        throw std::runtime_error("failed to initialize GGML context");
    }

    ggml_backend_buffer_t buffer = nullptr;
    try {
        engine::core::ModuleBuildContext ctx{ggml, "f16_kv_set_rows_test", engine::core::BackendType::Cpu};
        const auto cache = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F16,
            engine::core::TensorShape::from_dims({1, 3, 1, 2}));
        const auto row = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, 1, 1, 2}));
        const auto row_index = engine::core::make_tensor(
            ctx,
            GGML_TYPE_I64,
            engine::core::TensorShape::from_dims({1}));
        const auto output = engine::modules::FastKVSetRowsModule{}.build(ctx, cache, row, row_index);

        ggml_cgraph * graph = ggml_new_graph_custom(ggml, kGraphNodes, false);
        ggml_build_forward_expand(graph, output.tensor);
        buffer = ggml_backend_alloc_ctx_tensors(ggml, backend);
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate f16 KV test tensors");
        }

        engine::core::write_tensor_f16(cache, std::vector<float>(6, 0.0F));
        engine::core::write_tensor_f32(row, {1.25F, -2.5F});
        const int64_t index = 1;
        ggml_backend_tensor_set(row_index.tensor, &index, 0, sizeof(index));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("f16 KV set-rows graph compute failed");
        }

        const auto values = engine::core::read_tensor_f16(output.tensor);
        require_allclose(values, {0.0F, 0.0F, 1.25F, -2.5F, 0.0F, 0.0F}, 1.0e-3F, "f16 KV cache");

        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
        ggml_free(ggml);
        ggml_backend_free(backend);
    } catch (...) {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        ggml_free(ggml);
        ggml_backend_free(backend);
        throw;
    }
}

void test_f16_kv_set_rows_batched() {
    engine::core::BackendConfig backend_config{engine::core::BackendType::Cpu, 0, 4};
    ggml_backend_t backend = engine::core::init_backend(backend_config);
    if (backend == nullptr) {
        throw std::runtime_error("failed to initialize CPU backend");
    }

    ggml_init_params params{kGraphBytes, nullptr, true};
    ggml_context * ggml = ggml_init(params);
    if (ggml == nullptr) {
        ggml_backend_free(backend);
        throw std::runtime_error("failed to initialize GGML context");
    }

    ggml_backend_buffer_t buffer = nullptr;
    try {
        engine::core::ModuleBuildContext ctx{ggml, "f16_kv_set_rows_batched_test", engine::core::BackendType::Cpu};
        const auto cache = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F16,
            engine::core::TensorShape::from_dims({2, 3, 1, 2}));
        const auto rows = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({2, 1, 1, 2}));
        const auto row_indices = engine::core::make_tensor(
            ctx,
            GGML_TYPE_I64,
            engine::core::TensorShape::from_dims({2}));
        const auto output = engine::modules::FastKVSetRowsModule{}.build(ctx, cache, rows, row_indices);

        ggml_cgraph * graph = ggml_new_graph_custom(ggml, kGraphNodes, false);
        ggml_build_forward_expand(graph, output.tensor);
        buffer = ggml_backend_alloc_ctx_tensors(ggml, backend);
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate batched F16 KV test tensors");
        }

        engine::core::write_tensor_f16(cache, std::vector<float>(12, 0.0F));
        engine::core::write_tensor_f32(rows, {1.25F, -2.5F, 3.5F, -4.5F});
        const std::vector<int64_t> indices{1, 4};
        ggml_backend_tensor_set(row_indices.tensor, indices.data(), 0, indices.size() * sizeof(int64_t));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("batched F16 KV set-rows graph compute failed");
        }

        const auto values = engine::core::read_tensor_f16(output.tensor);
        require_allclose(
            values,
            {0.0F, 0.0F, 1.25F, -2.5F, 0.0F, 0.0F,
             0.0F, 0.0F, 3.5F, -4.5F, 0.0F, 0.0F},
            1.0e-3F,
            "batched f16 KV cache");

        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
        ggml_free(ggml);
        ggml_backend_free(backend);
    } catch (...) {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        ggml_free(ggml);
        ggml_backend_free(backend);
        throw;
    }
}

int count_graph_op(ggml_cgraph * graph, ggml_op op) {
    int count = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        count += node != nullptr && node->op == op ? 1 : 0;
    }
    return count;
}

bool graph_contains_sequence(ggml_cgraph * graph, std::initializer_list<ggml_op> ops) {
    if (ops.size() == 0 || static_cast<int>(ops.size()) > ggml_graph_n_nodes(graph)) {
        return false;
    }
    for (int start = 0; start + static_cast<int>(ops.size()) <= ggml_graph_n_nodes(graph); ++start) {
        bool matches = true;
        int offset = 0;
        for (const ggml_op op : ops) {
            const ggml_tensor * node = ggml_graph_node(graph, start + offset++);
            matches = matches && node != nullptr && node->op == op;
        }
        if (matches) {
            return true;
        }
    }
    return false;
}

std::string graph_ops(ggml_cgraph * graph) {
    std::ostringstream out;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if (i != 0) {
            out << ',';
        }
        out << (node != nullptr ? ggml_op_name(node->op) : "null");
    }
    return out.str();
}

void test_higgs_decode_graph_exposes_cuda_fast_paths() {
    constexpr int64_t hidden = 8;
    constexpr int64_t heads = 2;
    constexpr int64_t kv_heads = 1;
    constexpr int64_t head_dim = 4;
    constexpr int64_t intermediate = 12;
    constexpr int64_t cache_steps = 8;
    constexpr int64_t qkv_out = heads * head_dim + 2 * kv_heads * head_dim;

    ggml_init_params params{kGraphBytes, nullptr, true};
    ggml_context * ggml = ggml_init(params);
    if (ggml == nullptr) {
        throw std::runtime_error("failed to initialize Higgs decode graph test context");
    }

    try {
        engine::core::ModuleBuildContext ctx{ggml, "higgs_decode_fast_path_test", engine::core::BackendType::Cuda};
        auto make_tensor = [&](ggml_type type, std::initializer_list<int64_t> dims) {
            return engine::core::make_tensor(ctx, type, engine::core::TensorShape::from_dims(dims));
        };

        const auto input = make_tensor(GGML_TYPE_F32, {1, 1, hidden});
        const auto positions = make_tensor(GGML_TYPE_I32, {1});
        const auto cache_key = make_tensor(GGML_TYPE_F16, {1, cache_steps, kv_heads, head_dim});
        const auto cache_value = make_tensor(GGML_TYPE_F16, {1, cache_steps, kv_heads, head_dim});
        const auto cache_slot = make_tensor(GGML_TYPE_I64, {1});
        const auto attention_mask = make_tensor(GGML_TYPE_F16, {1, 1, 1, cache_steps});

        engine::modules::QwenDecoderLayerWeights weights;
        weights.input_norm = {make_tensor(GGML_TYPE_F32, {hidden}), std::nullopt};
        weights.q_norm = {make_tensor(GGML_TYPE_F32, {head_dim}), std::nullopt};
        weights.k_norm = {make_tensor(GGML_TYPE_F32, {head_dim}), std::nullopt};
        weights.post_norm = {make_tensor(GGML_TYPE_F32, {hidden}), std::nullopt};
        weights.self_attention.qkv_weight = make_tensor(GGML_TYPE_F32, {qkv_out, hidden});
        weights.self_attention.out_weight = make_tensor(GGML_TYPE_F32, {hidden, hidden});
        weights.mlp.gate_up_proj = engine::modules::LinearWeights{
            make_tensor(GGML_TYPE_F32, {intermediate * 2, hidden}),
            std::nullopt,
        };
        weights.mlp.down_proj = {
            make_tensor(GGML_TYPE_F32, {hidden, intermediate}),
            std::nullopt,
        };

        engine::modules::QwenDecoderLayerConfig config;
        config.hidden_size = hidden;
        config.num_attention_heads = heads;
        config.num_key_value_heads = kv_heads;
        config.head_dim = head_dim;
        config.intermediate_size = intermediate;
        config.qkv_layout = engine::modules::QwenDecoderQKVLayout::PackedQKV;
        config.use_qk_norm = true;
        config.runtime.attention.static_mode =
            engine::modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
        config.runtime.static_cache.update_mode =
            engine::modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;

        ggml_cgraph * graph = ggml_new_graph_custom(ggml, kGraphNodes, false);
        const auto outputs = engine::modules::QwenDecoderLayerModule(config).build_with_static_cache_tail(
            ctx,
            graph,
            input,
            positions,
            weights,
            cache_key,
            cache_value,
            cache_slot,
            attention_mask);
        ggml_build_forward_expand(graph, outputs.output.tensor);

        if (count_graph_op(graph, GGML_OP_FLASH_ATTN_EXT) != 1) {
            throw std::runtime_error("Higgs decode graph must contain one grouped FlashAttention op");
        }
        if (count_graph_op(graph, GGML_OP_SET_ROWS) != 2) {
            throw std::runtime_error("Higgs decode graph must update both F16 KV caches with set-rows");
        }
        if (count_graph_op(graph, GGML_OP_GLU) != 1) {
            throw std::runtime_error("Higgs decode graph must contain one packed SwiGLU op");
        }
        if (count_graph_op(graph, GGML_OP_REPEAT) != 0) {
            throw std::runtime_error("grouped FlashAttention must not materialize repeated KV heads");
        }
        if (!graph_contains_sequence(graph, {GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS})) {
            throw std::runtime_error(
                "Higgs key-cache update must expose CUDA RoPE/view/set-rows fusion; graph=" +
                graph_ops(graph));
        }

        ggml_free(ggml);
    } catch (...) {
        ggml_free(ggml);
        throw;
    }
}

}  // namespace

int main() {
    try {
        test_packed_qkv_and_gate_up_match_separate_projections();
        test_suffix_causal_mask();
        test_f16_kv_set_rows();
        test_f16_kv_set_rows_batched();
        test_higgs_decode_graph_exposes_cuda_fast_paths();
        std::cout << "qwen_decoder_packed_projection_test: ok\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "qwen_decoder_packed_projection_test: failed: " << ex.what() << "\n";
        return 1;
    }
}
