#include "../../src/models/yue2/attention_layout.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
using engine::core::BackendType;
using engine::models::yue2::detail::prepare_attention_queries;

void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
Context context() {
    Context ctx(ggml_init({256 * 1024 * 1024, nullptr, false}), ggml_free);
    check(ctx != nullptr, "ggml_init failed");
    return ctx;
}

void policy_test() {
    auto ctx = context();
    auto * base = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 128, 4, 33, 2);
    auto * q = ggml_permute(ctx.get(), base, 0, 2, 1, 3);
    check(!ggml_is_contiguous(q), "test must exercise a noncontiguous Q");
    for (auto backend : {BackendType::Cpu, BackendType::Cuda, BackendType::Hip,
                         BackendType::Vulkan, BackendType::Metal,
                         BackendType::Sycl, BackendType::BestAvailable}) {
        for (bool flash : {false, true}) {
            auto * actual = prepare_attention_queries(ctx.get(), q, backend, flash);
            const bool view = backend == BackendType::Sycl && flash;
            check(view ? actual == q : actual->op == GGML_OP_CONT,
                  "packing policy changed for an unintended backend/path");
            check(view || ggml_is_contiguous(actual), "packed Q must be contiguous");
        }
    }
    auto * half = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, 128, 4, 33, 2);
    check(prepare_attention_queries(ctx.get(), half, BackendType::Sycl, true)->op == GGML_OP_CONT,
          "only F32 Q is eligible");
}

void fill(ggml_tensor * tensor, float phase) {
    for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
        const float value = 0.15f * std::sin(phase + float(i % 7919) * 0.013f);
        if (tensor->type == GGML_TYPE_F16) {
            static_cast<ggml_fp16_t *>(tensor->data)[i] = ggml_fp32_to_fp16(value);
        } else {
            static_cast<float *>(tensor->data)[i] = value;
        }
    }
}

int cont_count(ggml_cgraph * graph) {
    int count = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        count += ggml_graph_node(graph, i)->op == GGML_OP_CONT;
    }
    return count;
}

// Builds the SYCL-selected graph but executes the two layouts on CPU. This
// validates ggml values/strides and graph structure, NOT the SYCL device kernel.
int64_t parity_test(int64_t steps, int64_t kv_steps, int64_t heads,
                    int64_t kv_heads, int64_t batches, ggml_type kv_type = GGML_TYPE_F32) {
    auto ctx = context();
    constexpr int64_t dim = 128;
    auto * base_q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, dim, heads, steps, batches);
    auto * base_k = ggml_new_tensor_4d(ctx.get(), kv_type, dim, kv_heads, kv_steps, batches);
    auto * base_v = ggml_new_tensor_4d(ctx.get(), kv_type, dim, kv_heads, kv_steps, batches);
    fill(base_q, 0.2f);
    fill(base_k, 0.9f);
    fill(base_v, 1.7f);
    auto * q = ggml_permute(ctx.get(), base_q, 0, 2, 1, 3);
    auto * k = ggml_permute(ctx.get(), base_k, 0, 2, 1, 3);
    auto * v = ggml_permute(ctx.get(), base_v, 0, 2, 1, 3);
    auto * view = prepare_attention_queries(ctx.get(), q, BackendType::Sycl, true);
    auto * packed = ggml_cont(ctx.get(), q); // original implementation
    auto * reference = ggml_flash_attn_ext(ctx.get(), packed, k, v, nullptr,
                                         1.0f / std::sqrt(float(dim)), 0.0f, 0.0f);
    auto * candidate = ggml_flash_attn_ext(ctx.get(), view, k, v, nullptr,
                                         1.0f / std::sqrt(float(dim)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(reference, GGML_PREC_F32);
    ggml_flash_attn_ext_set_prec(candidate, GGML_PREC_F32);
    auto * ref_graph = ggml_new_graph(ctx.get());
    auto * new_graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(ref_graph, reference);
    ggml_build_forward_expand(new_graph, candidate);
    check(cont_count(ref_graph) == 1 && cont_count(new_graph) == 0,
          "candidate must remove exactly one Q packing operation");
    check(ggml_graph_compute_with_ctx(ctx.get(), ref_graph, 2) == GGML_STATUS_SUCCESS,
          "reference CPU graph failed");
    check(ggml_graph_compute_with_ctx(ctx.get(), new_graph, 2) == GGML_STATUS_SUCCESS,
          "candidate CPU graph failed");

    // Compare all Q coordinates, including head/batch strides and tail rows.
    int64_t checked = 0;
    for (int64_t b = 0; b < batches; ++b)
        for (int64_t h = 0; h < heads; ++h)
            for (int64_t s = 0; s < steps; ++s)
                for (int64_t d = 0; d < dim; ++d) {
                    const auto offset = b * q->nb[3] + h * q->nb[2] + s * q->nb[1] + d * q->nb[0];
                    const auto index = ((b * heads + h) * steps + s) * dim + d;
                    check(std::memcmp(static_cast<char *>(q->data) + offset,
                                      static_cast<float *>(packed->data) + index,
                                      sizeof(float)) == 0, "query coordinate mismatch");
                    ++checked;
                }
    check(ggml_nbytes(reference) == ggml_nbytes(candidate), "output size mismatch");
    const auto * ref = static_cast<const float *>(reference->data);
    const auto * out = static_cast<const float *>(candidate->data);
    for (int64_t i = 0; i < ggml_nelements(candidate); ++i) {
        check(std::isfinite(ref[i]) && std::isfinite(out[i]), "nonfinite attention output");
        check(std::memcmp(ref + i, out + i, sizeof(float)) == 0, "attention output not bit-exact on CPU");
    }
    std::cout << "PASS steps=" << steps << " kv=" << kv_steps << " heads=" << heads
              << "/" << kv_heads << " batches=" << batches << " kv_type=" << ggml_type_name(kv_type) << " values=" << checked
              << " removed_cont=1 output=bit-exact(CPU)\n";
    return checked;
}
} // namespace

int main() {
    try {
        policy_test();
        int64_t total = 0;
        for (int64_t steps : {1, 2, 31, 32, 33, 65}) {
            total += parity_test(steps, steps + 2, 4, 2, 2);
        }
        total += parity_test(2402, 17, 16, 8, 1, GGML_TYPE_F16);
        total += parity_test(6502, 17, 16, 8, 1, GGML_TYPE_F16);
        std::cout << "PASS backend policy + " << total << " Q coordinates and attention outputs\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
