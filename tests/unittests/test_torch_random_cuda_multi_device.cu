#include "framework/sampling/torch_random_cuda_runtime.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void checked(cudaError_t error, const char * operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

engine::sampling::TorchCudaSamplingPolicy policy_for(int device) {
    cudaDeviceProp properties{};
    checked(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    engine::sampling::TorchCudaSamplingPolicy policy;
    policy.multiprocessor_count = properties.multiProcessorCount;
    policy.max_threads_per_multiprocessor = properties.maxThreadsPerMultiProcessor;
    policy.cuda_device_index = device;
    return policy;
}

void exercise_device(int device, int64_t hidden_size) {
    constexpr int64_t songs = 1;
    constexpr int64_t vocab = 8;
    const auto policy = policy_for(device);
    checked(cudaSetDevice(device), "cudaSetDevice test");

    std::vector<float> logits(static_cast<size_t>(2 * songs * vocab));
    for (size_t i = 0; i < logits.size(); ++i) {
        logits[i] = static_cast<float>(i) / 10.0F;
    }
    float * device_logits = nullptr;
    checked(cudaMalloc(&device_logits, logits.size() * sizeof(float)), "cudaMalloc logits");
    checked(cudaMemcpy(device_logits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy logits");
    const uint64_t seeds[] = {1234};
    const uint64_t offsets[] = {0};
    int32_t code = -1;
    try {
        engine::sampling::detail::sample_topk_exponential_pairs_cuda(
            device_logits, songs, vocab, 1.0F, vocab, seeds, offsets, 0, policy, &code);
    } catch (...) {
        cudaFree(device_logits);
        throw;
    }
    checked(cudaFree(device_logits), "cudaFree logits");
    if (code < 0 || code >= vocab) {
        throw std::runtime_error("sampler returned an out-of-range code");
    }

    cudaStream_t stream = nullptr;
    checked(cudaStreamCreate(&stream), "cudaStreamCreate");
    try {
        engine::sampling::detail::depth_frame_ensure_cuda(songs, 2, hidden_size, policy);
        engine::sampling::detail::depth_frame_begin_cuda(seeds, offsets, songs, stream);
        checked(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    } catch (...) {
        cudaStreamDestroy(stream);
        throw;
    }
    checked(cudaStreamDestroy(stream), "cudaStreamDestroy");
}

}  // namespace

int main() {
    int count = 0;
    checked(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count < 2) {
        std::cout << "SKIP: two CUDA devices are required\n";
        return 77;
    }
    try {
        // Revisit both devices so the test covers allocation, cross-device
        // release, and allocation again on the same serving thread.
        for (const auto [device, hidden_size] : {
                 std::pair{0, int64_t{4}},
                 std::pair{1, int64_t{4}},
                 std::pair{0, int64_t{8}},
                 std::pair{1, int64_t{8}}}) {
            exercise_device(device, hidden_size);
        }
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "torch_random_cuda_multi_device_test passed\n";
    return 0;
}
