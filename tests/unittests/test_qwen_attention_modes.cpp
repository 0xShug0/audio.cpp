// Pins the numerical relationship between the QwenDecoderAttentionMode lowerings.
//
// The point of this test is F2.1 / F2.2 of docs/reviews/02-kv-cache-attention-performance.md:
// `static_mode` used to default to FlashGrouped, which copies the whole permuted K/V cache
// with ggml_cont before calling ggml_flash_attn_ext, while FlashGroupedViewKV makes the same
// call on the views. If those two really are the same arithmetic on the same values, the copy
// is pure waste and the default can move. If they are not, the default must stay.
//
// Everything runs on the CPU backend with synthetic weights. No Metal, no model files.
//
//   static_cache_flash_grouped_vs_view_kv   asserts BIT-EXACT equality (max_abs == 0).
//   static_cache_manual_repeat_vs_flash     asserts a stated tolerance, and prints the number.
//   prefill_manual_repeat_vs_flash          asserts a stated tolerance, and prints the number.

#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/transformers/qwen_decoder.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using engine::core::BackendType;
using engine::core::ModuleBuildContext;
using engine::core::TensorShape;
using engine::core::TensorValue;
using engine::modules::QwenDecoderAttentionMode;
using engine::modules::QwenDecoderLayerConfig;
using engine::modules::QwenDecoderLayerModule;
using engine::modules::QwenDecoderLayerWeights;

constexpr size_t kGraphBytes = 64 * 1024 * 1024;
constexpr size_t kGraphNodes = 8192;
constexpr int kRounds = 3;

// Tolerances for the two lowerings that are only mathematically equivalent, not
// bit-identical: an explicit materialised softmax versus a fused online one. These are the
// same thresholds tests/unittests/test_scaled_dot_product_attention.cpp already applies to
// the same class of comparison (explicit SDPA versus ggml_flash_attn_ext) on CUDA. The CPU
// flash kernel accumulates in f32, so the real numbers here should land far below them; the
// test prints what it measured so a regression shows up as a moved number, not just a pass.
constexpr float kFlashMaxAbs = 7.5e-3F;
constexpr double kFlashMeanAbs = 6.0e-4;
constexpr double kFlashMinCosine = 0.99999;

struct LayerDims {
    int64_t hidden_size;
    int64_t heads;
    int64_t kv_heads;
    int64_t head_dim;
    int64_t intermediate_size;
};

struct DiffStats {
    float max_abs = 0.0F;
    double mean_abs = 0.0;
    double cosine = 1.0;
};

std::vector<float> make_patterned_f32(size_t count, float phase, float scale) {
    std::vector<float> values(count, 0.0F);
    for (size_t i = 0; i < count; ++i) {
        const float x = static_cast<float>(i);
        values[i] = scale * (
            std::sin(phase + 0.013F * x) +
            0.5F * std::cos(0.7F * phase + 0.017F * x) +
            0.25F * std::sin(0.11F * phase + 0.031F * x));
    }
    return values;
}

DiffStats diff_stats(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error("diff_stats size mismatch");
    }
    DiffStats stats;
    double abs_sum = 0.0;
    double dot = 0.0;
    double lhs_norm = 0.0;
    double rhs_norm = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        const float diff = std::abs(lhs[i] - rhs[i]);
        stats.max_abs = std::max(stats.max_abs, diff);
        abs_sum += static_cast<double>(diff);
        dot += static_cast<double>(lhs[i]) * static_cast<double>(rhs[i]);
        lhs_norm += static_cast<double>(lhs[i]) * static_cast<double>(lhs[i]);
        rhs_norm += static_cast<double>(rhs[i]) * static_cast<double>(rhs[i]);
    }
    stats.mean_abs = abs_sum / static_cast<double>(std::max<size_t>(size_t{1}, lhs.size()));
    stats.cosine = dot / std::sqrt(std::max(1.0e-30, lhs_norm * rhs_norm));
    return stats;
}

QwenDecoderLayerConfig make_layer_config(const LayerDims & dims) {
    QwenDecoderLayerConfig config;
    config.hidden_size = dims.hidden_size;
    config.num_attention_heads = dims.heads;
    config.num_key_value_heads = dims.kv_heads;
    config.head_dim = dims.head_dim;
    config.intermediate_size = dims.intermediate_size;
    config.rms_norm_eps = 1e-5F;
    config.rope_theta = 10000.0F;
    config.attention_precision = GGML_PREC_F32;
    config.use_qk_norm = true;
    return config;
}

// Every leaf tensor the layer needs, kept alongside its fill parameters so a single pass can
// upload deterministic values for all of them.
struct WeightTensors {
    QwenDecoderLayerWeights layer;
    std::vector<TensorValue> leaves;
    std::vector<float> phases;
    std::vector<float> scales;
    std::vector<float> offsets;

    void add(const TensorValue & tensor, float phase, float scale, float offset) {
        leaves.push_back(tensor);
        phases.push_back(phase);
        scales.push_back(scale);
        offsets.push_back(offset);
    }

    void upload() const {
        for (size_t i = 0; i < leaves.size(); ++i) {
            auto values = make_patterned_f32(
                static_cast<size_t>(leaves[i].shape.num_elements()),
                phases[i],
                scales[i]);
            if (offsets[i] != 0.0F) {
                for (auto & value : values) {
                    value += offsets[i];
                }
            }
            engine::core::write_tensor_f32(leaves[i], values);
        }
    }
};

// Norm gains sit near 1.0 so the block behaves like a trained layer rather than a
// near-degenerate one; projections stay small so activations do not saturate.
WeightTensors make_layer_weights(ModuleBuildContext & ctx, const LayerDims & dims) {
    WeightTensors out;
    const int64_t q_out = dims.heads * dims.head_dim;
    const int64_t kv_out = dims.kv_heads * dims.head_dim;

    auto norm = [&](float phase) {
        auto tensor = engine::core::make_tensor(
            ctx, GGML_TYPE_F32, TensorShape::from_dims({dims.hidden_size}));
        out.add(tensor, phase, 0.05F, 1.0F);
        return tensor;
    };
    auto head_norm = [&](float phase) {
        auto tensor = engine::core::make_tensor(
            ctx, GGML_TYPE_F32, TensorShape::from_dims({dims.head_dim}));
        out.add(tensor, phase, 0.05F, 1.0F);
        return tensor;
    };
    auto matrix = [&](int64_t rows, int64_t cols, float phase) {
        auto tensor = engine::core::make_tensor(
            ctx, GGML_TYPE_F32, TensorShape::from_dims({rows, cols}));
        out.add(tensor, phase, 0.09F, 0.0F);
        return tensor;
    };

    out.layer.input_norm.weight = norm(0.11F);
    out.layer.post_norm.weight = norm(0.23F);
    out.layer.q_norm.weight = head_norm(0.37F);
    out.layer.k_norm.weight = head_norm(0.41F);
    out.layer.self_attention.q_weight = matrix(q_out, dims.hidden_size, 0.53F);
    out.layer.self_attention.k_weight = matrix(kv_out, dims.hidden_size, 0.67F);
    out.layer.self_attention.v_weight = matrix(kv_out, dims.hidden_size, 0.79F);
    out.layer.self_attention.out_weight = matrix(dims.hidden_size, q_out, 0.83F);
    out.layer.mlp.gate_proj.weight = matrix(dims.intermediate_size, dims.hidden_size, 0.97F);
    out.layer.mlp.up_proj.weight = matrix(dims.intermediate_size, dims.hidden_size, 1.09F);
    out.layer.mlp.down_proj.weight = matrix(dims.hidden_size, dims.intermediate_size, 1.13F);
    return out;
}

void compute_or_throw(ggml_backend_t backend, ggml_cgraph * graph, const char * label) {
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    if (status != GGML_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << label << " graph compute failed with status " << static_cast<int>(status);
        throw std::runtime_error(oss.str());
    }
}

void check(const char * label, const DiffStats & stats, bool require_bit_exact) {
    std::cout << "cpu " << label
              << " max_abs=" << stats.max_abs
              << " mean_abs=" << stats.mean_abs
              << " cosine=" << stats.cosine << "\n";
    const bool ok = require_bit_exact
        ? (stats.max_abs == 0.0F && stats.mean_abs == 0.0)
        : (stats.max_abs <= kFlashMaxAbs && stats.mean_abs <= kFlashMeanAbs &&
           stats.cosine >= kFlashMinCosine);
    if (ok) {
        return;
    }
    std::ostringstream oss;
    oss << label << " mismatch: max_abs=" << stats.max_abs
        << " mean_abs=" << stats.mean_abs << " cosine=" << stats.cosine;
    if (require_bit_exact) {
        oss << " -- FlashGrouped and FlashGroupedViewKV are NOT bit-identical on this build."
            << " Revert QwenDecoderAttentionPolicy::static_mode to FlashGrouped in"
            << " include/engine/framework/modules/transformers/qwen_decoder.h";
    }
    throw std::runtime_error(oss.str());
}

// One decoder block, driven through the single-token static-cache path once per attention
// mode. Each mode gets its own K/V cache tensors, filled with identical data, so the modes
// cannot interfere through the ScratchTail write.
void run_static_cache_parity() {
    const LayerDims dims{128, 4, 2, 32, 256};
    const int64_t cache_steps = 48;
    const int64_t visible_steps = 33;
    const int64_t scratch_slot = cache_steps - 1;
    const int64_t step_elems = dims.kv_heads * dims.head_dim;

    ggml_backend_t backend = engine::core::init_backend({BackendType::Cpu, 0, 4});
    ggml_init_params params{kGraphBytes, nullptr, true};
    ggml_context * ggml = ggml_init(params);
    if (ggml == nullptr) {
        ggml_backend_free(backend);
        throw std::runtime_error("failed to initialize qwen attention mode test context");
    }
    ggml_backend_buffer_t buffer = nullptr;

    try {
        ModuleBuildContext ctx{ggml, "qwen.attention_modes.decode", BackendType::Cpu};

        auto weights = make_layer_weights(ctx, dims);
        auto input = engine::core::make_tensor(
            ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 1, dims.hidden_size}));
        auto positions = engine::core::wrap_tensor(
            ggml_new_tensor_1d(ggml, GGML_TYPE_I32, 1), TensorShape::from_dims({1}), GGML_TYPE_I32);
        auto mask = engine::core::wrap_tensor(
            ggml_new_tensor_4d(ggml, GGML_TYPE_F16, cache_steps, 1, 1, 1),
            TensorShape::from_dims({1, 1, 1, cache_steps}),
            GGML_TYPE_F16);

        const std::vector<QwenDecoderAttentionMode> modes = {
            QwenDecoderAttentionMode::FlashGrouped,
            QwenDecoderAttentionMode::FlashGroupedViewKV,
            QwenDecoderAttentionMode::ManualRepeat,
        };

        auto * graph = ggml_new_graph_custom(ggml, kGraphNodes, false);
        std::vector<TensorValue> cache_keys;
        std::vector<TensorValue> cache_values;
        std::vector<TensorValue> outputs;
        for (const auto mode : modes) {
            auto cache_key = engine::core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                TensorShape::from_dims({1, cache_steps, dims.kv_heads, dims.head_dim}));
            auto cache_value = engine::core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                TensorShape::from_dims({1, cache_steps, dims.kv_heads, dims.head_dim}));
            auto config = make_layer_config(dims);
            config.runtime.attention.static_mode = mode;
            auto layer_out = QwenDecoderLayerModule(config).build_with_static_cache_tail(
                ctx,
                graph,
                input,
                positions,
                weights.layer,
                cache_key,
                cache_value,
                std::nullopt,
                mask);
            // Expand each variant before building the next one so the ScratchTail copy for
            // this cache is ordered ahead of the attention that reads it.
            ggml_build_forward_expand(graph, layer_out.output.tensor);
            cache_keys.push_back(cache_key);
            cache_values.push_back(cache_value);
            outputs.push_back(layer_out.output);
        }

        buffer = ggml_backend_alloc_ctx_tensors(ggml, backend);
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate qwen attention mode test tensors");
        }

        weights.upload();

        // Visible for the prefix and for the scratch slot the current token is written into;
        // -inf everywhere else, which is what the shared decode runtime uploads.
        std::vector<float> mask_values(
            static_cast<size_t>(cache_steps), -std::numeric_limits<float>::infinity());
        for (int64_t step = 0; step < visible_steps; ++step) {
            mask_values[static_cast<size_t>(step)] = 0.0F;
        }
        mask_values[static_cast<size_t>(scratch_slot)] = 0.0F;
        engine::core::write_tensor_f16(mask, mask_values);

        const std::vector<int32_t> position_values{static_cast<int32_t>(visible_steps)};
        engine::core::write_tensor_i32(positions, position_values);

        for (int round = 0; round < kRounds; ++round) {
            const auto input_values = make_patterned_f32(
                static_cast<size_t>(dims.hidden_size), 0.31F + static_cast<float>(round), 0.4F);
            engine::core::write_tensor_f32(input, input_values);

            const auto key_values = make_patterned_f32(
                static_cast<size_t>(cache_steps * step_elems),
                1.7F + static_cast<float>(round) * 0.37F,
                0.3F);
            const auto value_values = make_patterned_f32(
                static_cast<size_t>(cache_steps * step_elems),
                2.9F + static_cast<float>(round) * 0.19F,
                0.25F);
            for (size_t variant = 0; variant < cache_keys.size(); ++variant) {
                engine::core::write_tensor_f32(cache_keys[variant], key_values);
                engine::core::write_tensor_f32(cache_values[variant], value_values);
            }

            compute_or_throw(backend, graph, "qwen static-cache attention mode");

            std::vector<std::vector<float>> results(outputs.size());
            for (size_t variant = 0; variant < outputs.size(); ++variant) {
                engine::core::read_tensor_f32_into(outputs[variant].tensor, results[variant]);
            }
            check("static_cache_flash_grouped_vs_view_kv", diff_stats(results[0], results[1]), true);
            check("static_cache_manual_repeat_vs_flash", diff_stats(results[2], results[1]), false);
        }
    } catch (...) {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        ggml_free(ggml);
        ggml_backend_free(backend);
        throw;
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ggml);
    ggml_backend_free(backend);
}

// The prefill path: ManualRepeat (the default) versus the flash lowering that F2.2 proposes
// promoting. These are NOT the same computation -- one materialises [heads, T, T] scores and
// runs ggml_soft_max_ext over them, the other fuses the softmax into ggml_flash_attn_ext --
// so the test states a tolerance rather than asserting equality.
void run_prefill_parity() {
    const LayerDims dims{128, 4, 2, 32, 256};
    const int64_t steps = 24;

    ggml_backend_t backend = engine::core::init_backend({BackendType::Cpu, 0, 4});
    ggml_init_params params{kGraphBytes, nullptr, true};
    ggml_context * ggml = ggml_init(params);
    if (ggml == nullptr) {
        ggml_backend_free(backend);
        throw std::runtime_error("failed to initialize qwen prefill mode test context");
    }
    ggml_backend_buffer_t buffer = nullptr;

    try {
        ModuleBuildContext ctx{ggml, "qwen.attention_modes.prefill", BackendType::Cpu};

        auto weights = make_layer_weights(ctx, dims);
        auto input = engine::core::make_tensor(
            ctx, GGML_TYPE_F32, TensorShape::from_dims({1, steps, dims.hidden_size}));
        auto positions = engine::core::wrap_tensor(
            ggml_new_tensor_1d(ggml, GGML_TYPE_I32, steps),
            TensorShape::from_dims({steps}),
            GGML_TYPE_I32);
        auto mask = engine::core::wrap_tensor(
            ggml_new_tensor_4d(ggml, GGML_TYPE_F16, steps, steps, 1, 1),
            TensorShape::from_dims({1, 1, steps, steps}),
            GGML_TYPE_F16);

        auto * graph = ggml_new_graph_custom(ggml, kGraphNodes, false);
        std::vector<TensorValue> outputs;
        for (const auto mode : {QwenDecoderAttentionMode::ManualRepeat,
                                QwenDecoderAttentionMode::FlashGroupedViewKV}) {
            auto config = make_layer_config(dims);
            config.runtime.attention.prefill_mode = mode;
            auto layer_out = QwenDecoderLayerModule(config).build(
                ctx,
                input,
                positions,
                weights.layer,
                std::nullopt,
                std::nullopt,
                std::optional<TensorValue>(mask));
            ggml_build_forward_expand(graph, layer_out.output.tensor);
            outputs.push_back(layer_out.output);
        }

        buffer = ggml_backend_alloc_ctx_tensors(ggml, backend);
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate qwen prefill mode test tensors");
        }

        weights.upload();

        std::vector<float> mask_values(
            static_cast<size_t>(steps * steps), -std::numeric_limits<float>::infinity());
        for (int64_t query = 0; query < steps; ++query) {
            for (int64_t key = 0; key <= query; ++key) {
                mask_values[static_cast<size_t>(query * steps + key)] = 0.0F;
            }
        }
        engine::core::write_tensor_f16(mask, mask_values);

        std::vector<int32_t> position_values(static_cast<size_t>(steps), 0);
        for (int64_t step = 0; step < steps; ++step) {
            position_values[static_cast<size_t>(step)] = static_cast<int32_t>(step);
        }
        engine::core::write_tensor_i32(positions, position_values);

        for (int round = 0; round < kRounds; ++round) {
            const auto input_values = make_patterned_f32(
                static_cast<size_t>(steps * dims.hidden_size),
                0.47F + static_cast<float>(round),
                0.4F);
            engine::core::write_tensor_f32(input, input_values);

            compute_or_throw(backend, graph, "qwen prefill attention mode");

            std::vector<float> manual_repeat;
            std::vector<float> flash;
            engine::core::read_tensor_f32_into(outputs[0].tensor, manual_repeat);
            engine::core::read_tensor_f32_into(outputs[1].tensor, flash);
            check("prefill_manual_repeat_vs_flash", diff_stats(manual_repeat, flash), false);
        }
    } catch (...) {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        ggml_free(ggml);
        ggml_backend_free(backend);
        throw;
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ggml);
    ggml_backend_free(backend);
}

}  // namespace

int main() try {
    run_static_cache_parity();
    run_prefill_parity();
    std::cout << "qwen_attention_modes passed\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << "qwen_attention_modes_test failed: " << error.what() << "\n";
    return 1;
}
