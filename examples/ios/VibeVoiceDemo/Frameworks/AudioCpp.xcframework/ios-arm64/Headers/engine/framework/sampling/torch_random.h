#pragma once

#include "engine/framework/core/backend.h"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace engine::sampling {

struct TorchCudaSamplingPolicy {
    int64_t multiprocessor_count = 1;
    int64_t max_threads_per_multiprocessor = 256;
    bool cuda_fast_path = false;
    int cuda_device_index = 0;
};

enum class TorchCudaSamplingPolicyFailureMode {
    StrictCuda,
    FallbackToDefault,
};

enum class TorchRandnPrecision {
    Float32,
    BFloat16,
};

TorchCudaSamplingPolicy resolve_torch_cuda_sampling_policy(
    engine::core::BackendType backend_type,
    int device_index,
    std::string_view log_category,
    std::string_view model_name,
    TorchCudaSamplingPolicyFailureMode failure_mode = TorchCudaSamplingPolicyFailureMode::StrictCuda);

uint64_t torch_cuda_tensor_iterator_offset_blocks(
    uint64_t total_elements,
    const TorchCudaSamplingPolicy & policy);

void fill_torch_cuda_randn(
    float * output,
    size_t count,
    uint64_t seed,
    TorchRandnPrecision precision = TorchRandnPrecision::Float32,
    uint64_t start_index = 0);

std::vector<float> generate_torch_cuda_randn(
    size_t count,
    uint64_t seed,
    TorchRandnPrecision precision = TorchRandnPrecision::Float32,
    uint64_t start_index = 0);

void fill_torch_cuda_tensor_iterator_randn(
    float * output,
    size_t count,
    uint64_t seed,
    uint64_t offset_blocks,
    const TorchCudaSamplingPolicy & policy,
    TorchRandnPrecision precision = TorchRandnPrecision::Float32);

std::vector<float> generate_torch_cuda_tensor_iterator_randn(
    size_t count,
    uint64_t seed,
    uint64_t offset_blocks,
    const TorchCudaSamplingPolicy & policy,
    TorchRandnPrecision precision = TorchRandnPrecision::Float32);

void fill_torch_cuda_uniform(
    float * output,
    size_t count,
    uint64_t seed,
    uint64_t start_index = 0);

std::vector<float> generate_torch_cuda_uniform(size_t count, uint64_t seed, uint64_t start_index = 0);

float torch_cuda_tensor_iterator_exponential_element(
    uint64_t seed,
    uint64_t total_elements,
    uint64_t element_index,
    uint64_t call_index,
    int64_t multiprocessor_count,
    int64_t max_threads_per_multiprocessor);

}  // namespace engine::sampling
