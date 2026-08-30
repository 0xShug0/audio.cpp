#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/transformers/qwen_causal_decode_runtime.h"

#include "test_assert.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using engine::test::require;
using engine::test::require_eq;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

class CpuBackend {
public:
    CpuBackend() {
        backend_ = engine::core::init_backend({engine::core::BackendType::Cpu, 0, 1});
        if (backend_ == nullptr) {
            throw std::runtime_error("failed to initialize the CPU backend");
        }
    }

    ~CpuBackend() {
        if (backend_ != nullptr) {
            ggml_backend_free(backend_);
        }
    }

    CpuBackend(const CpuBackend &) = delete;
    CpuBackend & operator=(const CpuBackend &) = delete;

    ggml_backend_t get() const noexcept {
        return backend_;
    }

private:
    ggml_backend_t backend_ = nullptr;
};

// Distinct, exactly representable float per (row, token) so an equality comparison is
// meaningful: every value below 2^24 is an integer that float holds without rounding.
std::vector<float> synthetic_logits(int64_t rows, int64_t vocab) {
    std::vector<float> out(static_cast<size_t>(rows * vocab), 0.0F);
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t token = 0; token < vocab; ++token) {
            out[static_cast<size_t>(row * vocab + token)] =
                static_cast<float>(row) * 1000000.0F + static_cast<float>(token) - 500.0F;
        }
    }
    return out;
}

struct CompactReadback {
    std::vector<float> values;
    size_t compact_bytes = 0;
    size_t full_bytes = 0;
};

// Builds exactly the graph the decode runtimes build for a compact logits readback,
// computes it on the CPU backend, and returns the gathered row.
CompactReadback run_compact_gather(
    const CpuBackend & backend,
    int64_t rows,
    int64_t vocab,
    const std::vector<float> & values,
    const std::vector<int32_t> & subset) {
    const int64_t compact_size = static_cast<int64_t>(subset.size());
    ggml_init_params params{16ull * 1024ull * 1024ull, nullptr, true};
    std::unique_ptr<ggml_context, GgmlContextDeleter> holder(ggml_init(params));
    if (holder == nullptr) {
        throw std::runtime_error("failed to initialize the test ggml context");
    }

    engine::core::ModuleBuildContext ctx{holder.get(), "test.compact_logits_readback"};
    auto logits = engine::core::make_tensor(
        ctx,
        GGML_TYPE_F32,
        engine::core::TensorShape::from_dims({rows, 1, vocab}));
    ggml_set_input(logits.tensor);
    ggml_tensor * ids = ggml_new_tensor_1d(holder.get(), GGML_TYPE_I32, compact_size);
    ggml_set_input(ids);
    const auto ids_value = engine::core::wrap_tensor(
        ids,
        engine::core::TensorShape::from_dims({compact_size}),
        GGML_TYPE_I32);

    const auto compact =
        engine::modules::build_compact_logits_gather(ctx, logits, ids_value, vocab, compact_size);
    ggml_set_output(compact.tensor);
    ggml_cgraph * graph = ggml_new_graph(holder.get());
    ggml_build_forward_expand(graph, compact.tensor);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(holder.get(), backend.get());
    if (buffer == nullptr) {
        throw std::runtime_error("failed to allocate the compact logits test graph");
    }

    ggml_backend_tensor_set(logits.tensor, values.data(), 0, values.size() * sizeof(float));
    ggml_backend_tensor_set(ids, subset.data(), 0, subset.size() * sizeof(int32_t));
    const ggml_status status = engine::core::compute_backend_graph(backend.get(), graph);
    ggml_backend_synchronize(backend.get());
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer);
        throw std::runtime_error("compact logits test graph compute failed");
    }

    CompactReadback out;
    out.values.assign(static_cast<size_t>(rows) * subset.size(), 0.0F);
    out.compact_bytes = ggml_nbytes(compact.tensor);
    out.full_bytes = ggml_nbytes(logits.tensor);
    ggml_backend_tensor_get(compact.tensor, out.values.data(), 0, out.values.size() * sizeof(float));
    engine::core::release_backend_graph_resources(backend.get(), graph);
    ggml_backend_buffer_free(buffer);
    return out;
}

void require_matches_full_readback(
    int64_t rows,
    int64_t vocab,
    const std::vector<float> & values,
    const std::vector<int32_t> & subset,
    const std::vector<float> & compact,
    const std::string & label) {
    require_eq(compact.size(), static_cast<size_t>(rows) * subset.size(), label + " compact size");
    for (int64_t row = 0; row < rows; ++row) {
        for (size_t index = 0; index < subset.size(); ++index) {
            const float actual = compact[static_cast<size_t>(row) * subset.size() + index];
            const float expected =
                values[static_cast<size_t>(row * vocab + static_cast<int64_t>(subset[index]))];
            if (actual != expected) {
                throw std::runtime_error(
                    label + " row " + std::to_string(row) + " slot " + std::to_string(index) +
                    " (token " + std::to_string(subset[index]) + "): expected " +
                    std::to_string(expected) + ", got " + std::to_string(actual));
            }
        }
    }
}

void test_single_row_subset_matches_full_row(const CpuBackend & backend) {
    constexpr int64_t rows = 1;
    constexpr int64_t vocab = 32;
    const std::vector<int32_t> subset = {7, 0, 31, 7};
    const auto values = synthetic_logits(rows, vocab);
    const auto readback = run_compact_gather(backend, rows, vocab, values, subset);
    require_matches_full_readback(rows, vocab, values, subset, readback.values, "single row");
    require_eq(readback.compact_bytes, size_t{16}, "single row compact byte count");
    require_eq(readback.full_bytes, size_t{128}, "single row full byte count");
}

void test_multi_row_subset_matches_full_row(const CpuBackend & backend) {
    constexpr int64_t rows = 3;
    constexpr int64_t vocab = 17;
    const std::vector<int32_t> subset = {16, 1, 9};
    const auto values = synthetic_logits(rows, vocab);
    const auto readback = run_compact_gather(backend, rows, vocab, values, subset);
    require_matches_full_readback(rows, vocab, values, subset, readback.values, "multi row");
    require_eq(readback.compact_bytes, size_t{3 * 3 * 4}, "multi row compact byte count");
}

// The VibeVoice shape from docs/reviews/03 F3.12: four control tokens out of a 152 064
// entry vocabulary. 608 256 bytes per token become 16.
void test_vibevoice_shape_overfetch_ratio(const CpuBackend & backend) {
    constexpr int64_t rows = 1;
    constexpr int64_t vocab = 152064;
    const std::vector<int32_t> subset = {151643, 151646, 151647, 151645};
    const auto values = synthetic_logits(rows, vocab);
    const auto readback = run_compact_gather(backend, rows, vocab, values, subset);
    require_matches_full_readback(rows, vocab, values, subset, readback.values, "vibevoice shape");
    require_eq(readback.full_bytes, size_t{608256}, "vibevoice full readback bytes");
    require_eq(readback.compact_bytes, size_t{16}, "vibevoice compact readback bytes");
    require_eq(readback.full_bytes / readback.compact_bytes, size_t{38016}, "vibevoice overfetch ratio");
}

void test_rejects_logits_without_vocab_on_last_dim() {
    ggml_init_params params{4ull * 1024ull * 1024ull, nullptr, true};
    std::unique_ptr<ggml_context, GgmlContextDeleter> holder(ggml_init(params));
    require(holder != nullptr, "failed to initialize the validation ggml context");
    engine::core::ModuleBuildContext ctx{holder.get(), "test.compact_logits_validate"};
    auto logits = engine::core::make_tensor(
        ctx,
        GGML_TYPE_F32,
        engine::core::TensorShape::from_dims({1, 1, 16}));
    ggml_tensor * ids = ggml_new_tensor_1d(holder.get(), GGML_TYPE_I32, 2);
    const auto ids_value = engine::core::wrap_tensor(
        ids,
        engine::core::TensorShape::from_dims({2}),
        GGML_TYPE_I32);

    bool threw = false;
    try {
        (void)engine::modules::build_compact_logits_gather(ctx, logits, ids_value, 32, 2);
    } catch (const std::exception &) {
        threw = true;
    }
    require(threw, "compact logits gather accepted a mismatched vocabulary size");

    threw = false;
    try {
        (void)engine::modules::build_compact_logits_gather(ctx, logits, ids_value, 16, 0);
    } catch (const std::exception &) {
        threw = true;
    }
    require(threw, "compact logits gather accepted an empty token subset");
}

}  // namespace

int main() {
    try {
        CpuBackend backend;
        test_single_row_subset_matches_full_row(backend);
        test_multi_row_subset_matches_full_row(backend);
        test_vibevoice_shape_overfetch_ratio(backend);
        test_rejects_logits_without_vocab_on_last_dim();
        std::cout << "compact_logits_readback_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "compact_logits_readback_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
