#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/transformers/causal_decoder.h"
#include "engine/framework/modules/transformers/decoder.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/optimizations/fast_projection_modules.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/transformers/causal_decoder_runtime.h"
#include "engine/framework/io/safetensors.h"

#include <cmath>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>
#include <cstdint>

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
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            throw std::runtime_error(label + " contains non-finite values");
        }
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

LayerResult run_layer(bool packed,
    engine::core::BackendType backend_type = engine::core::BackendType::Cpu,
    bool batched_decode = false,
    bool fused_mlp = false) {
    const int64_t batch = batched_decode ? 2 : 1;
    const int64_t steps = batched_decode ? 1 : 3;
    constexpr int64_t cache_steps = 8;
    constexpr int64_t hidden = 8;
    constexpr int64_t heads = 2;
    constexpr int64_t kv_heads = 1;
    constexpr int64_t head_dim = 4;
    constexpr int64_t intermediate = 12;
    constexpr int64_t q_out = heads * head_dim;
    constexpr int64_t kv_out = kv_heads * head_dim;

    engine::core::BackendConfig backend_config{backend_type, 0, 8};
    ggml_backend_t backend = engine::core::init_backend(backend_config);
    if (backend == nullptr) {
        throw std::runtime_error("failed to initialize test backend");
    }

    ggml_init_params params{kGraphBytes, nullptr, true};
    ggml_context * ggml = ggml_init(params);
    if (ggml == nullptr) {
        ggml_backend_free(backend);
        throw std::runtime_error("failed to initialize GGML context");
    }

    ggml_backend_buffer_t buffer = nullptr;
    try {
        engine::core::ModuleBuildContext ctx{ggml, "qwen_packed_projection_test", backend_type};
        ctx.elide_identity_reshapes = fused_mlp;
        auto make_f32 = [&](std::initializer_list<int64_t> dims) {
            return engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims(dims));
        };

        auto input = make_f32({batch, steps, hidden});
        auto positions = engine::core::make_tensor(
            ctx,
            GGML_TYPE_I32,
            engine::core::TensorShape::from_dims({batched_decode ? batch : steps}));

        engine::modules::DecoderLayerWeights weights;
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

        engine::modules::DecoderLayerConfig config;
        config.hidden_size = hidden;
        config.num_attention_heads = heads;
        config.num_key_value_heads = kv_heads;
        config.head_dim = head_dim;
        config.intermediate_size = intermediate;
        config.rms_norm_eps = 1e-5f;
        config.qkv_layout = packed
            ? engine::modules::DecoderQKVLayout::PackedQKV
            : engine::modules::DecoderQKVLayout::Separate;
        config.runtime.mlp.mode = packed
            ? engine::modules::DecoderMLPMode::PackedGateUp
            : (fused_mlp ? engine::modules::DecoderMLPMode::FusedSwiGLU : engine::modules::DecoderMLPMode::Exact);
        config.use_qk_norm = false;
        config.runtime.attention.prefill_mode = engine::modules::DecoderAttentionMode::ManualRepeat;

        ggml_cgraph * graph = ggml_new_graph_custom(ggml, kGraphNodes, false);
        engine::modules::DecoderLayerOutputs outputs;
        engine::core::TensorValue cache_key, cache_value, cache_slot, mask;
        if (batched_decode) {
            cache_key = make_f32({batch, cache_steps, kv_heads, head_dim});
            cache_value = make_f32({batch, cache_steps, kv_heads, head_dim});
            cache_slot = engine::core::make_tensor(ctx, GGML_TYPE_I32,
                engine::core::TensorShape::from_dims({batch}));
            mask = make_f32({batch, 1, 1, cache_steps});
            config.runtime.static_cache.update_mode = engine::modules::DecoderStaticCacheUpdateMode::DirectSetRows;
            config.runtime.attention.static_mode = engine::modules::DecoderAttentionMode::ManualRepeat;
            outputs = engine::modules::DecoderLayerModule(config).build_with_static_cache_tail_batched(
                ctx, graph, input, positions, weights, cache_key, cache_value, cache_slot, mask);
        } else {
            outputs = engine::modules::DecoderLayerModule(config).build(ctx, input, positions, weights);
        }
        ggml_build_forward_expand(graph, outputs.output.tensor);
        if (batched_decode) {
            int rope_nodes = 0;
            int copied_positions = 0;
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                const auto * node = ggml_graph_node(graph, i);
                if (node->op != GGML_OP_ROPE) {
                    continue;
                }
                ++rope_nodes;
                if (node->src[1]->op == GGML_OP_CONT) {
                    ++copied_positions;
                }
                if (backend_type == engine::core::BackendType::Vulkan && node->src[1]->view_offs != 0) {
                    throw std::runtime_error("Vulkan batched RoPE retains an offset position view");
                }
            }
            const int expected_copies = backend_type == engine::core::BackendType::Vulkan ? 2 : 0;
            if (rope_nodes != 4 || copied_positions != expected_copies) {
                throw std::runtime_error("Unexpected batched RoPE position layout");
            }
        }
        buffer = ggml_backend_alloc_ctx_tensors(ggml, backend);
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate test tensors");
        }

        engine::core::write_tensor_f32(input, patterned(static_cast<size_t>(batch * steps * hidden), 2.1f, 0.20f));
        engine::core::write_tensor_i32(positions, batched_decode ? std::vector<int32_t>{2, 5} : std::vector<int32_t>{0, 1, 2});
        if (batched_decode) {
            engine::core::write_tensor_f32(cache_key, patterned(batch * cache_steps * kv_out, 0.4f, 0.1f));
            engine::core::write_tensor_f32(cache_value, patterned(batch * cache_steps * kv_out, 0.8f, 0.1f));
            engine::core::write_tensor_i32(cache_slot, {2, cache_steps + 5});
            std::vector<float> mask_values(batch * cache_steps, -std::numeric_limits<float>::infinity());
            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t t = 0; t <= (b == 0 ? 2 : 5); ++t) {
                    mask_values[b * cache_steps + t] = 0.0f;
                }
            }
            engine::core::write_tensor_f32(mask, mask_values);
        }
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

        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("test graph execution failed");
        }
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
    const auto values = engine::modules::causal_suffix_mask_values(2, 3, 2);
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

void test_f16_kv_set_rows(engine::modules::FastKVSetRowsMode mode) {
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
            GGML_TYPE_I32,
            engine::core::TensorShape::from_dims({1}));
        const auto output = engine::modules::FastKVSetRowsModule({
            mode,
        }).build(ctx, cache, row, row_index);

        ggml_cgraph * graph = ggml_new_graph_custom(ggml, kGraphNodes, false);
        ggml_build_forward_expand(graph, output.tensor);
        buffer = ggml_backend_alloc_ctx_tensors(ggml, backend);
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate f16 KV test tensors");
        }

        engine::core::write_tensor_f16(cache, std::vector<float>(6, 0.0F));
        engine::core::write_tensor_f32(row, {1.25F, -2.5F});
        const int32_t index = 1;
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

void test_f16_kv_set_rows_batched(engine::modules::FastKVSetRowsMode mode) {
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
        const auto output = engine::modules::FastKVSetRowsModule({
            mode,
        }).build(ctx, cache, rows, row_indices);

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

void test_fast_projection_accepts_cuda_or_hip_backends() {
    constexpr int64_t in_features = 8;
    constexpr int64_t out_features = 12;
    for (const auto backend_type : {
             engine::core::BackendType::Cuda,
             engine::core::BackendType::Hip,
         }) {
        ggml_init_params params{kGraphBytes, nullptr, true};
        ggml_context * ggml = ggml_init(params);
        if (ggml == nullptr) {
            throw std::runtime_error("failed to initialize fast projection graph test context");
        }
        try {
            engine::core::ModuleBuildContext ctx{ggml, "fast_projection_cuda_or_hip_test", backend_type};
            const auto input = engine::core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({1, in_features}));
            const auto weight = engine::core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({out_features, in_features}));
            const auto output = engine::modules::FastPackedProjection4Module({in_features, out_features})
                                    .build(ctx, input, {weight, std::nullopt});
            if (output.shape.rank != 2 || output.shape.dims[0] != 1 || output.shape.dims[1] != out_features) {
                throw std::runtime_error("fast projection CUDA/HIP output shape mismatch");
            }
            ggml_free(ggml);
        } catch (...) {
            ggml_free(ggml);
            throw;
        }
    }

    for (const auto rejected_backend : {engine::core::BackendType::Cpu, engine::core::BackendType::Metal}) {
        ggml_init_params params{kGraphBytes, nullptr, true};
        ggml_context * ggml = ggml_init(params);
        if (ggml == nullptr) {
            throw std::runtime_error("failed to initialize fast projection rejection test context");
        }
        bool rejected = false;
        try {
            engine::core::ModuleBuildContext ctx{ggml, "fast_projection_rejection_test", rejected_backend};
            const auto input = engine::core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({1, in_features}));
            const auto weight = engine::core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({out_features, in_features}));
            try {
                (void) engine::modules::FastPackedProjection4Module({in_features, out_features})
                    .build(ctx, input, {weight, std::nullopt});
            } catch (const std::runtime_error &) {
                rejected = true;
            }
            ggml_free(ggml);
        } catch (...) {
            ggml_free(ggml);
            throw;
        }
        if (!rejected) {
            throw std::runtime_error("fast projection must reject non-CUDA/HIP backends");
        }
    }
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

        engine::modules::DecoderLayerWeights weights;
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

        engine::modules::DecoderLayerConfig config;
        config.hidden_size = hidden;
        config.num_attention_heads = heads;
        config.num_key_value_heads = kv_heads;
        config.head_dim = head_dim;
        config.intermediate_size = intermediate;
        config.qkv_layout = engine::modules::DecoderQKVLayout::PackedQKV;
        config.use_qk_norm = true;
        config.runtime.attention.static_mode =
            engine::modules::DecoderAttentionMode::FlashGroupedViewKV;
        config.runtime.static_cache.update_mode =
            engine::modules::DecoderStaticCacheUpdateMode::DirectSetRows;
        config.runtime.static_cache.set_rows_mode =
            engine::modules::DecoderStaticCacheSetRowsMode::BackendViewOptimized;
        config.runtime.mlp.mode = engine::modules::DecoderMLPMode::PackedGateUp;

        ggml_cgraph * graph = ggml_new_graph_custom(ggml, kGraphNodes, false);
        const auto outputs = engine::modules::DecoderLayerModule(config).build_with_static_cache_tail(
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


void run_ar_depth_probe(const std::string & impl, int batch, const std::string & output) {
    using namespace engine;
    core::BackendConfig backend_config{core::BackendType::Hip, 0, 8};
    backend_config.impl = impl;
    core::ExecutionContext execution(backend_config);
    core::BackendWeightStore store(execution.backend(), execution.backend_type(), "ar_probe.weights", 1024 * 1024);
    uint32_t rng = 1234;
    auto matrix = [&](int64_t rows, int64_t cols) {
        std::vector<float> values(static_cast<size_t>(rows * cols));
        for (float & value : values) {
            rng = rng * 1664525U + 1013904223U;
            value = (float(rng >> 16) / 32768.0F - 1.0F) * 0.02F;
        }
        std::vector<std::byte> bytes(ggml_row_size(GGML_TYPE_Q8_0, cols) * rows);
        const size_t written = ggml_quantize_chunk(GGML_TYPE_Q8_0, values.data(), bytes.data(), 0, rows, cols, nullptr);
        if (written != bytes.size()) {
            throw std::runtime_error("AR fixture quantization size mismatch");
        }
        return store.make_tensor(core::TensorShape::from_dims({rows, cols}), GGML_TYPE_Q8_0,
                                 bytes.data(), bytes.size());
    };
    auto norm = [&]() {
        const std::vector<float> values(1024, 1.0F);
        return modules::NormWeights{store.make_tensor(core::TensorShape::from_dims({1024}),
            GGML_TYPE_F32, values.data(), values.size() * sizeof(float)), std::nullopt};
    };
    const std::vector<float> rope_factors = {
        1.000000000F, 1.000000000F, 1.000000000F, 1.000000000F, 1.000000000F, 1.000000000F, 1.000000000F, 1.000000000F,
        1.000000000F, 1.000000000F, 1.000000000F, 1.000000000F, 1.000000000F, 1.000000000F, 1.000000000F, 1.000000000F,
        1.000000000F, 1.000000000F, 1.000000000F, 1.000000000F, 1.000000000F, 1.000000000F, 1.000000000F, 1.000000000F,
        1.000000000F, 1.000000000F, 1.000000000F, 1.000000000F, 1.000000000F, 1.234937034F, 1.651329246F, 2.276659536F,
        3.292262103F, 5.171595583F, 9.666728979F, 32.000000000F, 32.000000000F, 32.000000000F, 32.000000000F, 32.000000000F,
        32.000000000F, 32.000000000F, 32.000000000F, 32.000000000F, 32.000000000F, 32.000000000F, 32.000000000F, 32.000000000F,
        32.000000000F, 32.000000000F, 32.000000000F, 32.000000000F, 32.000000000F, 32.000000000F, 32.000000000F, 32.000000000F,
        32.000000000F, 32.000000000F, 32.000000000F, 32.000000000F, 32.000000000F, 32.000000000F, 32.000000000F, 32.000000000F,
    };
    const auto rope = store.make_tensor(core::TensorShape::from_dims({64}), GGML_TYPE_F32,
        rope_factors.data(), rope_factors.size() * sizeof(float));
    modules::CausalDecoderRuntimeWeights weights;
    for (int i = 0; i < 12; ++i) {
        modules::DecoderLayerWeights layer;
        layer.input_norm = norm();
        layer.post_norm = norm();
        layer.rope_frequency_factors = rope;
        layer.self_attention.qkv_weight = matrix(1536, 1024);
        layer.self_attention.out_weight = matrix(1024, 1024);
        layer.mlp.gate_up_proj = modules::LinearWeights{matrix(16384, 1024), std::nullopt};
        layer.mlp.down_proj = {matrix(1024, 8192), std::nullopt};
        weights.stack.layers.push_back(std::move(layer));
    }
    weights.final_norm = norm();
    store.upload();

    modules::CausalDecoderRuntimeConfig config;
    config.trace_name = "ar_probe.depth";
    config.prefill_graph_arena_bytes = config.decode_graph_arena_bytes = 16 * 1024 * 1024;
    config.output_mode = modules::CausalDecoderOutputMode::Hidden;
    config.return_hidden = true;
    config.decoder.static_cache_type = GGML_TYPE_BF16;
    config.decoder.logits_mode = modules::CausalDecoderLogitsMode::LastStep;
    auto & stack = config.decoder.stack;
    stack.hidden_size = 1024;
    stack.intermediate_size = 8192;
    stack.layers = 12;
    stack.num_attention_heads = 8;
    stack.num_key_value_heads = 2;
    stack.head_dim = 128;
    stack.rms_norm_eps = 1.0e-5F;
    stack.rope_theta = 500000.0F;
    stack.rope_type = GGML_ROPE_TYPE_NEOX;
    stack.use_qk_norm = false;
    stack.qkv_layout = modules::DecoderQKVLayout::PackedQKV;
    stack.runtime.mlp.mode = modules::DecoderMLPMode::PackedGateUp;
    stack.attention_precision = GGML_PREC_F32;
    stack.projection_precision = GGML_PREC_DEFAULT;
    stack.runtime.attention.allow_flash_attention = true;
    stack.runtime.attention.prefill_mode = stack.runtime.attention.static_mode =
        modules::DecoderAttentionMode::FlashGroupedViewKV;
    stack.runtime.static_cache.update_mode = modules::DecoderStaticCacheUpdateMode::DirectSetRows;
    stack.runtime.static_cache.set_rows_mode = modules::DecoderStaticCacheSetRowsMode::BackendViewOptimized;
    auto & cast = stack.activation_cast;
    cast.enabled = true;
    cast.type = GGML_TYPE_BF16;
    cast.fused_round = true;
    cast.after_input_norm = cast.after_qkv_projection = cast.after_qk_norm = true;
    cast.after_rope = cast.after_static_cache_update = cast.after_attention = true;
    cast.after_attention_output = cast.after_residual = cast.after_ffn_norm = true;
    cast.after_mlp_projection = cast.after_mlp_silu = cast.after_mlp_mul = cast.after_output = true;
    modules::CausalDecoderRuntime runtime(execution, config, std::move(weights));
    const auto prefix = patterned(static_cast<size_t>(batch * 2 * 1024), 0.1F, 0.2F);
    std::vector<std::vector<float>> inputs;
    for (int i = 0; i < 14; ++i) {
        inputs.push_back(patterned(static_cast<size_t>(batch * 1024), float(i) * 0.3F, 0.2F));
    }
    std::vector<float> reference;
    for (int repeat = 0; repeat < 21; ++repeat) {
        const auto prefill_start = std::chrono::steady_clock::now();
        auto prefill = runtime.prefill_embeddings_batched(prefix, batch, 2);
        runtime.start_decode_embeddings_batched(prefill.state, 17);
        const double prefill_ms = debug::elapsed_ms(prefill_start);
        std::vector<float> values;
        const auto decode_start = std::chrono::steady_clock::now();
        for (const auto & input : inputs) {
            auto step = runtime.decode_embeddings_batched(input, batch);
            values.insert(values.end(), step.hidden.begin(), step.hidden.end());
        }
        const double decode_ms = debug::elapsed_ms(decode_start);
        if (repeat == 0) {
            reference = std::move(values);
        } else {
            require_allclose(values, reference, 0.0F, "AR probe repeat");
            debug::timing_log_scalar("ar_probe.prefill_ms", prefill_ms);
            debug::timing_log_scalar("ar_probe.decode_ms", decode_ms);
            debug::timing_log_scalar("ar_probe.frame_ms", prefill_ms + decode_ms);
        }
    }
    debug::trace_log_f32("ar_probe.hidden", {14, batch, 1024}, reference);
    io::SafeTensorWriteEntry entry{"hidden", "F32", {14, batch, 1024}, {}};
    entry.data.resize(reference.size() * sizeof(float));
    std::memcpy(entry.data.data(), reference.data(), entry.data.size());
    io::write_safetensors_file(output, {entry});
}

}  // namespace

int main(int argc, char ** argv) {
    const auto started = std::chrono::steady_clock::now();
    try {
        bool vulkan = false;
        bool cuda = false;
        bool ar_probe = false;
        int ar_batch = 1;
        std::string ar_impl = "gfx1151_v1", ar_output;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--ar-probe") {
                ar_probe = true;
            } else if (arg == "--impl" && i + 1 < argc) {
                ar_impl = argv[++i];
            } else if (arg == "--batch" && i + 1 < argc) {
                ar_batch = std::stoi(argv[++i]);
            } else if (arg == "--output" && i + 1 < argc) {
                ar_output = argv[++i];
            } else if (arg == "--vulkan") {
                vulkan = true;
            } else if (arg == "--cuda") {
                cuda = true;
            } else if (arg == "--log") {
                engine::debug::configure_logging({true, ""});
            } else {
                throw std::runtime_error("Unknown argument: " + arg);
            }
        }
        if (ar_probe) {
            if ((ar_batch != 1 && ar_batch != 2) || ar_output.empty()) {
                throw std::runtime_error("AR probe requires batch 1/2 and an output file");
            }
            run_ar_depth_probe(ar_impl, ar_batch, ar_output);
            engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
            return 0;
        }
        test_packed_qkv_and_gate_up_match_separate_projections();
        for (const bool batched : {false, true}) {
            const auto reference = run_layer(false, engine::core::BackendType::Cpu, batched);
            const auto fused = run_layer(false, engine::core::BackendType::Cpu, batched, true);
            require_allclose(fused.output, reference.output, 2.0e-5f, "fused MLP decoder output");
            if (cuda) {
                const auto actual = run_layer(false, engine::core::BackendType::Cuda, batched, true);
                require_allclose(actual.output, reference.output, 2.0e-5f, "CUDA fused MLP decoder output");
            }
            if (vulkan) {
                const auto backend_reference = run_layer(false, engine::core::BackendType::Vulkan, batched);
                const auto actual = run_layer(false, engine::core::BackendType::Vulkan, batched, true);
                require_allclose(actual.output, backend_reference.output, 2.0e-5f, "Vulkan fused MLP decoder output");
            }
        }
        for (bool packed : {false, true}) {
            const auto reference = run_layer(packed, engine::core::BackendType::Cpu, true);
            if (vulkan) {
                const auto actual = run_layer(packed, engine::core::BackendType::Vulkan, true);
                require_allclose(actual.output, reference.output, 2.0e-5f, "Vulkan batched decoder output");
                require_allclose(actual.key, reference.key, 2.0e-5f, "Vulkan batched decoder key");
                require_allclose(actual.value, reference.value, 2.0e-5f, "Vulkan batched decoder value");
            }
        }
        test_suffix_causal_mask();
        for (const auto mode : {engine::modules::FastKVSetRowsMode::Exact,
                                engine::modules::FastKVSetRowsMode::BackendViewOptimized}) {
            test_f16_kv_set_rows(mode);
            test_f16_kv_set_rows_batched(mode);
        }
        test_fast_projection_accepts_cuda_or_hip_backends();
        test_higgs_decode_graph_exposes_cuda_fast_paths();
        std::cout << "decoder_packed_projection_test: ok\n";
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
        return 0;
    } catch (const std::exception & ex) {
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
        std::cerr << "decoder_packed_projection_test: failed: " << ex.what() << "\n";
        return 1;
    }
}
