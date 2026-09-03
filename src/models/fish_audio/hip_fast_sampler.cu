#include "hip_fast_sampler.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace engine::models::fish_audio::detail {
namespace {

void check_hip(hipError_t status, const char * label) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(label) + ": " + hipGetErrorString(status));
    }
}

struct Workspace {
    int32_t vocab_size = 0;
    HipFastTopKResult * packed = nullptr;
};

__device__ __forceinline__ bool better(float av, int32_t ai, float bv, int32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

// Exact top-k for Fish Fast-AR (4096 entries for S2 Pro). Keeping only the
// compact result on the host avoids copying and sorting the full logits tensor
// for every acoustic codebook step.
__global__ void exact_topk_kernel(
    const float * logits,
    int32_t count,
    int32_t top_k,
    HipFastTopKResult * result) {
    extern __shared__ unsigned char smem_raw[];
    float * shared_logits = reinterpret_cast<float *>(smem_raw);
    float * best_values = shared_logits + count;
    int32_t * best_indices = reinterpret_cast<int32_t *>(best_values + blockDim.x);
    double * denom_parts = reinterpret_cast<double *>(best_indices + blockDim.x);

    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    for (int32_t i = tid; i < count; i += static_cast<int32_t>(blockDim.x)) {
        const float v = logits[i];
        shared_logits[i] = isfinite(v) ? v : -INFINITY;
    }
    __syncthreads();

    float max_logit = -INFINITY;
    for (int32_t rank = 0; rank < top_k; ++rank) {
        float local_value = -INFINITY;
        int32_t local_index = INT32_MAX;
        for (int32_t i = tid; i < count; i += static_cast<int32_t>(blockDim.x)) {
            const float v = shared_logits[i];
            if (better(v, i, local_value, local_index)) {
                local_value = v;
                local_index = i;
            }
        }
        best_values[tid] = local_value;
        best_indices[tid] = local_index;
        __syncthreads();

        for (int32_t stride = static_cast<int32_t>(blockDim.x) / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float other_value = best_values[tid + stride];
                const int32_t other_index = best_indices[tid + stride];
                if (better(other_value, other_index, best_values[tid], best_indices[tid])) {
                    best_values[tid] = other_value;
                    best_indices[tid] = other_index;
                }
            }
            __syncthreads();
        }

        const float selected_value = best_values[0];
        const int32_t selected_index = best_indices[0];
        if (tid == 0) {
            if (rank == 0) {
                max_logit = selected_value;
                result->max_logit = selected_value;
                result->count = top_k;
            }
            result->logits[rank] = selected_value;
            result->indices[rank] = selected_index;
            shared_logits[selected_index] = -INFINITY;
        }
        __syncthreads();
        if (rank == 0) {
            max_logit = result->max_logit;
        }
    }

    double local_denom = 0.0;
    for (int32_t i = tid; i < count; i += static_cast<int32_t>(blockDim.x)) {
        const float v = logits[i];
        if (isfinite(v)) {
            local_denom += static_cast<double>(expf(v - max_logit));
        }
    }
    denom_parts[tid] = local_denom;
    __syncthreads();
    for (int32_t stride = static_cast<int32_t>(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            denom_parts[tid] += denom_parts[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        result->denom = denom_parts[0];
    }
}

void free_workspace(Workspace * ws) {
    if (ws == nullptr) {
        return;
    }
    if (ws->packed != nullptr) {
        (void) hipFree(ws->packed);
    }
    delete ws;
}

}  // namespace

void * hip_fast_sampler_create(int32_t vocab_size) {
    if (vocab_size <= 0) {
        throw std::invalid_argument("HIP Fish fast sampler requires positive vocab_size");
    }
    auto * ws = new Workspace{};
    ws->vocab_size = vocab_size;
    try {
        check_hip(hipMalloc(&ws->packed, sizeof(HipFastTopKResult)), "hipMalloc Fish sampler result");
    } catch (...) {
        free_workspace(ws);
        throw;
    }
    return ws;
}

void hip_fast_sampler_destroy(void * workspace) {
    free_workspace(static_cast<Workspace *>(workspace));
}

void hip_fast_sampler_topk(
    void * workspace,
    const float * device_logits,
    int32_t top_k,
    HipFastTopKResult * host_result) {
    auto * ws = static_cast<Workspace *>(workspace);
    if (ws == nullptr || device_logits == nullptr || host_result == nullptr) {
        throw std::invalid_argument("HIP Fish fast sampler received a null argument");
    }
    if (top_k <= 0 || top_k > ws->vocab_size || top_k > kHipFastSamplerMaxTopK) {
        throw std::invalid_argument("HIP Fish fast sampler top_k is out of range");
    }

    constexpr int threads = 256;
    const size_t shared_bytes =
        static_cast<size_t>(ws->vocab_size) * sizeof(float) +
        static_cast<size_t>(threads) * (sizeof(float) + sizeof(int32_t) + sizeof(double));
    hipLaunchKernelGGL(
        exact_topk_kernel,
        dim3(1),
        dim3(threads),
        shared_bytes,
        0,
        device_logits,
        ws->vocab_size,
        top_k,
        ws->packed);
    check_hip(hipGetLastError(), "exact_topk_kernel Fish sampler");
    check_hip(
        hipMemcpy(host_result, ws->packed, sizeof(HipFastTopKResult), hipMemcpyDeviceToHost),
        "hipMemcpy Fish sampler result");
}

}  // namespace engine::models::fish_audio::detail
