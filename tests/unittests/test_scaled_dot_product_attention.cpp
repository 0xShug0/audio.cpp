#include "engine/framework/core/backend.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t kGraphBytes = 512 * 1024 * 1024;
constexpr size_t kGraphNodes = 4096;
constexpr int kRounds = 4;

struct SdpaCase {
    const char * name;
    int64_t batch;
    int64_t heads;
    int64_t query_steps;
    int64_t key_steps;
    int64_t head_dim;
    bool causal_mask;
};

struct RunResult {
    std::vector<float> values;
    double compute_ms = 0.0;
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

std::vector<float> make_causal_mask_values(int64_t query_steps, int64_t key_steps, int round) {
    std::vector<float> values(static_cast<size_t>(query_steps * key_steps), -std::numeric_limits<float>::infinity());
    const int64_t prefix = std::max<int64_t>(0, key_steps - query_steps);
    for (int64_t q = 0; q < query_steps; ++q) {
        const int64_t visible = std::min<int64_t>(key_steps - 1, prefix + q + (round % 2));
        for (int64_t k = 0; k <= visible; ++k) {
            values[static_cast<size_t>(q * key_steps + k)] = 0.0F;
        }
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

class SdpaRunner {
public:
    SdpaRunner(const SdpaCase & test_case, engine::modules::ScaledDotProductAttentionLowering lowering)
        : test_case_(test_case) {
        backend_ = engine::core::init_backend({engine::core::BackendType::Cuda, 0, 8});
        ggml_init_params params{};
        params.mem_size = kGraphBytes;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml_ = ggml_init(params);
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize SDPA test context");
        }
        ctx_.ggml = ggml_;
        ctx_.module_instance_name = lowering == engine::modules::ScaledDotProductAttentionLowering::Flash
            ? "sdpa.flash"
            : "sdpa.explicit";
        ctx_.backend_type = engine::core::BackendType::Cuda;

        q_ = engine::core::make_tensor(
            ctx_,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({
                test_case.batch,
                test_case.heads,
                test_case.query_steps,
                test_case.head_dim,
            }));
        k_ = engine::core::make_tensor(
            ctx_,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({
                test_case.batch,
                test_case.heads,
                test_case.key_steps,
                test_case.head_dim,
            }));
        v_ = engine::core::make_tensor(
            ctx_,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({
                test_case.batch,
                test_case.heads,
                test_case.key_steps,
                test_case.head_dim,
            }));
        mask_ = engine::core::make_tensor(
            ctx_,
            GGML_TYPE_F16,
            engine::core::TensorShape::from_dims({
                test_case.batch,
                1,
                test_case.query_steps,
                test_case.key_steps,
            }));

        output_ = engine::modules::ScaledDotProductAttentionModule({
            test_case.head_dim,
            lowering,
            GGML_PREC_F32,
        }).build(ctx_, q_, k_, v_, mask_);

        graph_ = ggml_new_graph_custom(ggml_, kGraphNodes, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        buffer_ = ggml_backend_alloc_ctx_tensors(ggml_, backend_);
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate SDPA test tensors");
        }
    }

    ~SdpaRunner() {
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

    RunResult run(
        const std::vector<float> & q_values,
        const std::vector<float> & k_values,
        const std::vector<float> & v_values,
        const std::vector<float> & mask_values) {
        engine::core::write_tensor_f32(q_, q_values);
        engine::core::write_tensor_f32(k_, k_values);
        engine::core::write_tensor_f32(v_, v_values);
        engine::core::write_tensor_f16(mask_, mask_values);
        const auto start = std::chrono::steady_clock::now();
        const ggml_status status = ggml_backend_graph_compute(backend_, graph_);
        ggml_backend_synchronize(backend_);
        const auto end = std::chrono::steady_clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            std::ostringstream oss;
            oss << "SDPA graph compute failed with status " << static_cast<int>(status);
            throw std::runtime_error(oss.str());
        }
        RunResult result;
        result.compute_ms = std::chrono::duration<double, std::milli>(end - start).count();
        engine::core::read_tensor_f32_into(output_.tensor, result.values);
        return result;
    }

private:
    SdpaCase test_case_;
    ggml_backend_t backend_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_context * ggml_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    engine::core::ModuleBuildContext ctx_{};
    engine::core::TensorValue q_;
    engine::core::TensorValue k_;
    engine::core::TensorValue v_;
    engine::core::TensorValue mask_;
    engine::core::TensorValue output_;
};

void run_case(const SdpaCase & test_case) {
    SdpaRunner explicit_runner(test_case, engine::modules::ScaledDotProductAttentionLowering::Explicit);
    SdpaRunner flash_runner(test_case, engine::modules::ScaledDotProductAttentionLowering::Flash);

    double explicit_ms = 0.0;
    double flash_ms = 0.0;
    for (int round = 0; round < kRounds; ++round) {
        const auto q = make_patterned_f32(
            static_cast<size_t>(test_case.batch * test_case.heads * test_case.query_steps * test_case.head_dim),
            0.31F + static_cast<float>(round),
            0.18F);
        const auto k = make_patterned_f32(
            static_cast<size_t>(test_case.batch * test_case.heads * test_case.key_steps * test_case.head_dim),
            1.17F + static_cast<float>(round) * 0.37F,
            0.16F);
        const auto v = make_patterned_f32(
            static_cast<size_t>(test_case.batch * test_case.heads * test_case.key_steps * test_case.head_dim),
            2.03F + static_cast<float>(round) * 0.19F,
            0.12F);
        const auto mask = make_causal_mask_values(test_case.query_steps, test_case.key_steps, round);

        const auto explicit_result = explicit_runner.run(q, k, v, mask);
        const auto flash_result = flash_runner.run(q, k, v, mask);
        explicit_ms += explicit_result.compute_ms;
        flash_ms += flash_result.compute_ms;
        const auto stats = diff_stats(explicit_result.values, flash_result.values);
        std::cout << test_case.name << " round=" << round
                  << " explicit_ms=" << explicit_result.compute_ms
                  << " flash_ms=" << flash_result.compute_ms
                  << " max_abs=" << stats.max_abs
                  << " mean_abs=" << stats.mean_abs
                  << " cosine=" << stats.cosine << "\n";
        if (stats.max_abs > 7.5e-3F || stats.mean_abs > 6.0e-4 || stats.cosine < 0.99999) {
            std::ostringstream oss;
            oss << test_case.name << " SDPA flash mismatch: max_abs=" << stats.max_abs
                << " mean_abs=" << stats.mean_abs << " cosine=" << stats.cosine;
            throw std::runtime_error(oss.str());
        }
    }
    std::cout << test_case.name << " avg_explicit_ms=" << explicit_ms / kRounds
              << " avg_flash_ms=" << flash_ms / kRounds << "\n";
}

}  // namespace

int main() try {
    const std::vector<SdpaCase> cases = {
        {"moss_qwen_prefill_128", 1, 32, 128, 128, 128, true},
        {"moss_qwen_prefill_96", 1, 32, 96, 96, 128, true},
        {"moss_depth_step_12", 1, 32, 12, 12, 80, true},
    };
    for (const auto & test_case : cases) {
        run_case(test_case);
    }
    return 0;
} catch (const std::exception & error) {
    std::cerr << "scaled_dot_product_attention_test failed: " << error.what() << "\n";
    return 1;
}
