#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-cuda.h"

#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Fault injection exercises a driver rejection without requiring the embedding application that observed it.
static cudaError_t opt_in_error = cudaSuccess;
static std::atomic<int> rejected_requests{0};

extern "C" cudaError_t CUDARTAPI cudaFuncSetAttribute(const void * function, cudaFuncAttribute attribute, int value) {
    static auto real_set_attribute = reinterpret_cast<decltype(&cudaFuncSetAttribute)>(dlsym(RTLD_NEXT, "cudaFuncSetAttribute"));
    if (!real_set_attribute) {
        std::fprintf(stderr, "cannot resolve cudaFuncSetAttribute: %s\n", dlerror());
        std::abort();
    }
    if (attribute == cudaFuncAttributeMaxDynamicSharedMemorySize && opt_in_error != cudaSuccess) {
        ++rejected_requests;
        std::fprintf(stderr, "fault injection: reject shared-memory opt-in of %d bytes\n", value);
        if (opt_in_error == cudaErrorInvalidValue) {
            const auto status = real_set_attribute(function, attribute, -1);
            if (status != cudaErrorInvalidValue) std::abort();
            return status;
        }
        return opt_in_error;
    }
    return real_set_attribute(function, attribute, value);
}

static bool check_softmax(ggml_backend_t backend, int columns, ggml_type mask_type, bool use_mask = true) {
    const int rows = 2;
    const size_t elements = size_t(columns) * rows;
    ggml_init_params params = {ggml_tensor_overhead() * 4 + ggml_graph_overhead_custom(8, false), nullptr, true};
    ggml_context_ptr context(ggml_init(params));
    if (!context) return false;
    auto * input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, columns, rows);
    auto * mask = use_mask ? ggml_new_tensor_2d(context.get(), mask_type, columns, rows) : nullptr;
    auto * output = ggml_soft_max_ext(context.get(), input, mask, 1.0f, 0.0f);
    auto * graph = ggml_new_graph_custom(context.get(), 8, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend));
    if (!buffer) return false;
    std::vector<float> values(elements), mask_values(elements), actual(elements);
    std::vector<ggml_fp16_t> half_mask(elements);
    for (size_t i = 0; i < elements; ++i) {
        values[i] = std::sin(float(i) * 0.17f) * 3.0f;
        mask_values[i] = use_mask ? float(int(i % 7) - 3) * 0.25f : 0.0f;
        half_mask[i] = ggml_fp32_to_fp16(mask_values[i]);
    }
    ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
    if (mask) ggml_backend_tensor_set(mask, mask_type == GGML_TYPE_F32 ? static_cast<void *>(mask_values.data()) : half_mask.data(), 0, ggml_nbytes(mask));
    for (int repeat = 0; repeat < 3; ++repeat) {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
        for (int row = 0; row < rows; ++row) {
            const size_t offset = size_t(row) * columns;
            double maximum = -INFINITY;
            for (int col = 0; col < columns; ++col) maximum = std::max(maximum, double(values[offset + col] + mask_values[offset + col]));
            double sum = 0.0;
            for (int col = 0; col < columns; ++col) sum += std::exp(values[offset + col] + mask_values[offset + col] - maximum);
            for (int col = 0; col < columns; ++col) {
                const double expected = std::exp(values[offset + col] + mask_values[offset + col] - maximum) / sum;
                const double value = actual[offset + col];
                if (!std::isfinite(value) || std::abs(value - expected) > 1e-6 + expected * 1e-4) {
                    std::fprintf(stderr, "columns=%d mask=%s repeat=%d row=%d col=%d actual=%g expected=%g\n",
                                 columns, use_mask ? ggml_type_name(mask_type) : "none", repeat, row, col, value, expected);
                    return false;
                }
            }
        }
    }
    std::printf("PASS columns=%d mask=%s\n", columns, use_mask ? ggml_type_name(mask_type) : "none");
    return true;
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--reject-opt-in") == 0) opt_in_error = cudaErrorInvalidValue;
    else if (argc == 2 && std::strcmp(argv[1], "--unexpected-opt-in-error") == 0) opt_in_error = cudaErrorInvalidResourceHandle;
    else if (argc != 1) return 2;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    int portable = 0, opt_in = 0;
    if (cudaDeviceGetAttribute(&portable, cudaDevAttrMaxSharedMemoryPerBlock, 0) != cudaSuccess ||
        cudaDeviceGetAttribute(&opt_in, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0) != cudaSuccess) return 1;
    ggml_backend_ptr backend(ggml_backend_cuda_init(0));
    if (!backend) return 1;
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        for (int columns : {32, 64, 128, 256, 512, 1024, 2048, 4096, 65, 513, portable / 4 - 32}) {
            if (!check_softmax(backend.get(), columns, type)) return 1;
        }
    }
    if (rejected_requests != 0) {
        std::fprintf(stderr, "portable softmax attempted an unnecessary shared-memory opt-in\n");
        return 1;
    }
    if (opt_in >= portable + 256) {
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
            // Reuse the generic kernel with growing and shrinking inputs after its first opt-in attempt.
            for (int columns : {portable / 4, opt_in / 4 - 32, portable / 4 + 32, 65, opt_in / 4 + 32}) {
                if (!check_softmax(backend.get(), columns, type)) return 1;
            }
        }
        if (!check_softmax(backend.get(), opt_in / 4 - 32, GGML_TYPE_F32, false)) return 1;
        if (opt_in_error != cudaSuccess && rejected_requests != 2) {
            std::fprintf(stderr, "expected one rejected opt-in per mask type, got %d\n", int(rejected_requests));
            return 1;
        }
    }
    if (cudaGetLastError() != cudaSuccess) return 1;
    return 0;
}
