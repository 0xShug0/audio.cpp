#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cuda_runtime.h>

#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

// Frozen pre-optimization CUDA algorithm. Test the real ggml operator against
// this independent reference, including its original accumulation order.
static __global__ void reference_kernel(int stride, int width, int inputs,
                                       int outputs, int frames, int dst_frames,
                                       const float * weights, const float * input,
                                       float * dst) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= dst_frames * outputs) return;
    const int channel = index / dst_frames;
    const int position = index % dst_frames;
    float accumulator = 0;
    for (int c = 0; c < inputs; ++c) {
        const int kernel_offset = width * outputs * c + channel * width;
        const int input_offset = frames * c;
        for (int k = 0; k < width; ++k) {
            const int shifted = position - k;
            if (shifted < 0 || shifted % stride != 0) continue;
            const int i = shifted / stride;
            if (i >= frames) continue;
            const float weight = weights[kernel_offset + k];
            const float value = input[input_offset + i];
            accumulator += weight * value;
        }
    }
    dst[index] = accumulator;
}

static void checked(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

struct resources {
    ggml_context * context = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    float * expected_device = nullptr;
    ~resources() {
        if (backend) ggml_backend_synchronize(backend);
        if (expected_device) cudaFree(expected_device);
        if (buffer) ggml_backend_buffer_free(buffer);
        if (backend) ggml_backend_free(backend);
        if (context) ggml_free(context);
    }
};

int main() {
    if (ggml_backend_cuda_get_device_count() == 0) {
        std::cout << "SKIP: no CUDA device\n";
        return 77;
    }
    try {
        const int strides[] = {1, 2, 3, 4, 5, 8, 16, 20, 32};
        const int widths[] = {1, 2, 3, 7, 16, 40};
        std::vector<std::tuple<int, int, int, int, int>> cases;
        for (int s : strides) for (int k : widths)
            cases.emplace_back(s, k, 1 + (s+k)%17, 1 + (s+3*k)%11, 1 + (s*k)%19);
        std::mt19937 rng(124578);
        for (int n = 0; n < 40; ++n) {
            // Assign explicitly: function argument evaluation order must not
            // change the seeded shapes across host compilers.
            const int s = strides[rng()%9];
            const int k = 1 + rng()%47;
            const int ci = 1 + rng()%97;
            const int co = 1 + rng()%31;
            const int frames = 1 + rng()%131;
            cases.emplace_back(s, k, ci, co, frames);
        }
        cases.emplace_back(2, 4, 1024, 512, 128);
        cases.emplace_back(4, 8, 512, 256, 256);
        cases.emplace_back(8, 16, 256, 128, 512);
        cases.emplace_back(16, 32, 128, 64, 1024);
        cases.emplace_back(20, 40, 64, 32, 2048);
        std::uniform_real_distribution<float> values(-2.0f, 2.0f);
        for (size_t n = 0; n < cases.size(); ++n) {
            const auto [s, k, ci, co, frames] = cases[n];
            const int dst_frames = (frames-1)*s + k;
            std::vector<float> weights(k*ci*co), input(frames*ci);
            std::vector<float> expected(dst_frames*co), actual(dst_frames*co);
            for (auto & v : weights) v = values(rng);
            for (auto & v : input) v = values(rng);
            if (n < 54 && n%7 == 0) {
                input[0] = -0.0f;
                if (input.size() > 2) input[1] = std::numeric_limits<float>::infinity();
                if (input.size() > 3) input[2] = std::numeric_limits<float>::quiet_NaN();
            }
            resources r;
            r.backend = ggml_backend_cuda_init(0);
            if (!r.backend) throw std::runtime_error("CUDA backend initialization failed");
            ggml_init_params params = {};
            params.mem_size = 8*ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false);
            params.no_alloc = true;
            r.context = ggml_init(params);
            if (!r.context) throw std::runtime_error("ggml context allocation failed");
            auto * w = ggml_new_tensor_3d(r.context, GGML_TYPE_F32, k, co, ci);
            auto * x = ggml_new_tensor_2d(r.context, GGML_TYPE_F32, frames, ci);
            auto * y = ggml_conv_transpose_1d(r.context, w, x, s, 0, 1);
            auto * graph = ggml_new_graph_custom(r.context, 16, false);
            ggml_build_forward_expand(graph, y);
            r.buffer = ggml_backend_alloc_ctx_tensors(r.context, r.backend);
            if (!r.buffer) throw std::runtime_error("CUDA tensor allocation failed");
            ggml_backend_tensor_set(w, weights.data(), 0, weights.size()*sizeof(float));
            ggml_backend_tensor_set(x, input.data(), 0, input.size()*sizeof(float));
            ggml_backend_synchronize(r.backend);
            checked(cudaMalloc(&r.expected_device, expected.size()*sizeof(float)));
            reference_kernel<<<(expected.size()+255)/256, 256>>>(
                s, k, ci, co, frames, dst_frames,
                static_cast<const float *>(w->data), static_cast<const float *>(x->data),
                r.expected_device);
            checked(cudaGetLastError());
            checked(cudaMemcpy(expected.data(), r.expected_device,
                               expected.size()*sizeof(float), cudaMemcpyDeviceToHost));
            // Exercise initial and reused execution of the actual backend op.
            for (int repeat = 0; repeat < 4; ++repeat) {
                if (ggml_backend_graph_compute(r.backend, graph) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("CUDA graph compute failed");
                ggml_backend_tensor_get(y, actual.data(), 0, actual.size()*sizeof(float));
                if (std::memcmp(expected.data(), actual.data(), actual.size()*sizeof(float))) {
                    std::cerr << "FAIL shape=" << n << " repeat=" << repeat
                              << " stride=" << s << " width=" << k << " ci=" << ci
                              << " co=" << co << " frames=" << frames << '\n';
                    return 2;
                }
            }
        }
        std::cout << "PASS: " << cases.size()
                  << " actual ggml CUDA operator shapes, four executions each, bit-exact\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
