#include "torch_random_cuda_runtime.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace engine::sampling::detail {
namespace {

constexpr uint32_t kPhiloxM0 = 0xD2511F53U;
constexpr uint32_t kPhiloxM1 = 0xCD9E8D57U;
constexpr uint32_t kPhiloxW0 = 0x9E3779B9U;
constexpr uint32_t kPhiloxW1 = 0xBB67AE85U;
constexpr float kInvTwoPow32 = 2.3283064365386963e-10F;
constexpr float kInvTwoPow32TwoPi = 1.4629180792671596e-09F;

struct Philox4 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t w;
};

void check_cuda(cudaError_t status, const char * label) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(status));
    }
}

uint64_t tensor_iterator_stride(uint64_t total_elements, const TorchCudaSamplingPolicy & policy) {
    constexpr uint64_t block_size = 256;
    uint64_t grid = (total_elements + block_size - 1) / block_size;
    uint64_t blocks_per_sm = static_cast<uint64_t>(policy.max_threads_per_multiprocessor) / block_size;
    if (blocks_per_sm == 0) {
        blocks_per_sm = 1;
    }
    const uint64_t grid_cap = static_cast<uint64_t>(policy.multiprocessor_count) * blocks_per_sm;
    grid = std::max<uint64_t>(1, std::min(grid_cap, grid));
    return block_size * grid;
}

__device__ void mul_hi_lo(uint32_t lhs, uint32_t rhs, uint32_t & hi, uint32_t & lo) {
    const uint64_t product = static_cast<uint64_t>(lhs) * static_cast<uint64_t>(rhs);
    lo = static_cast<uint32_t>(product);
    hi = static_cast<uint32_t>(product >> 32U);
}

__device__ Philox4 philox_round(Philox4 counter, uint32_t key0, uint32_t key1) {
    uint32_t hi0 = 0;
    uint32_t lo0 = 0;
    uint32_t hi1 = 0;
    uint32_t lo1 = 0;
    mul_hi_lo(kPhiloxM0, counter.x, hi0, lo0);
    mul_hi_lo(kPhiloxM1, counter.z, hi1, lo1);
    return Philox4{
        hi1 ^ counter.y ^ key0,
        lo1,
        hi0 ^ counter.w ^ key1,
        lo0,
    };
}

__device__ Philox4 philox_4x32_10(Philox4 counter, uint64_t seed) {
    uint32_t key0 = static_cast<uint32_t>(seed);
    uint32_t key1 = static_cast<uint32_t>(seed >> 32U);
    for (int round = 0; round < 10; ++round) {
        counter = philox_round(counter, key0, key1);
        key0 += kPhiloxW0;
        key1 += kPhiloxW1;
    }
    return counter;
}

__device__ void box_muller(uint32_t uniform0, uint32_t uniform1, float & normal0, float & normal1) {
    const float radius_input =
        static_cast<float>(uniform0) * kInvTwoPow32 + (kInvTwoPow32 * 0.5F);
    const float angle =
        static_cast<float>(uniform1) * kInvTwoPow32TwoPi + (kInvTwoPow32TwoPi * 0.5F);
    const float radius = sqrtf(-2.0F * logf(radius_input));
    sincosf(angle, &normal0, &normal1);
    normal0 *= radius;
    normal1 *= radius;
}

__device__ float round_to_bfloat16(float value) {
    uint32_t bits = __float_as_uint(value);
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    bits &= 0xFFFF0000U;
    return __uint_as_float(bits);
}

__global__ void fill_tensor_iterator_randn_kernel(
    float * output,
    uint64_t total,
    uint64_t seed,
    uint64_t offset_blocks,
    uint64_t stride,
    uint64_t loop_count,
    int precision) {
    constexpr uint64_t unroll_factor = 4;
    const uint64_t sequence = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (sequence >= stride) {
        return;
    }
    for (uint64_t loop_index = 0; loop_index < loop_count; ++loop_index) {
        const uint64_t first_index = loop_index * unroll_factor * stride + sequence;
        if (first_index >= total) {
            continue;
        }
        const Philox4 random = philox_4x32_10(
            Philox4{
                static_cast<uint32_t>(offset_blocks + loop_index),
                static_cast<uint32_t>((offset_blocks + loop_index) >> 32U),
                static_cast<uint32_t>(sequence),
                static_cast<uint32_t>(sequence >> 32U),
            },
            seed);
        float normal0 = 0.0F;
        float normal1 = 0.0F;
        float normal2 = 0.0F;
        float normal3 = 0.0F;
        box_muller(random.x, random.y, normal0, normal1);
        box_muller(random.z, random.w, normal2, normal3);
        if (precision == static_cast<int>(TorchRandnPrecision::BFloat16)) {
            normal0 = round_to_bfloat16(normal0);
            normal1 = round_to_bfloat16(normal1);
            normal2 = round_to_bfloat16(normal2);
            normal3 = round_to_bfloat16(normal3);
        }
        output[static_cast<size_t>(first_index)] = normal0;
        const uint64_t second_index = first_index + stride;
        if (second_index < total) {
            output[static_cast<size_t>(second_index)] = normal1;
        }
        const uint64_t third_index = second_index + stride;
        if (third_index < total) {
            output[static_cast<size_t>(third_index)] = normal2;
        }
        const uint64_t fourth_index = third_index + stride;
        if (fourth_index < total) {
            output[static_cast<size_t>(fourth_index)] = normal3;
        }
    }
}

__device__ float tensor_iterator_exponential_element_device(
    uint64_t seed,
    uint64_t index,
    uint64_t offset_blocks,
    uint64_t stride) {
    constexpr uint64_t unroll_factor = 4;
    const uint64_t chunk = index / stride;
    const int component = static_cast<int>(chunk % unroll_factor);
    const uint64_t loop_index = chunk / unroll_factor;
    const uint64_t sequence = index % stride;
    const Philox4 random = philox_4x32_10(
        Philox4{
            static_cast<uint32_t>(offset_blocks + loop_index),
            static_cast<uint32_t>((offset_blocks + loop_index) >> 32U),
            static_cast<uint32_t>(sequence),
            static_cast<uint32_t>(sequence >> 32U),
        },
        seed);
    uint32_t value = random.x;
    if (component == 1) {
        value = random.y;
    } else if (component == 2) {
        value = random.z;
    } else if (component == 3) {
        value = random.w;
    }
    const float uniform = static_cast<float>(value) * kInvTwoPow32 + (kInvTwoPow32 * 0.5F);
    return -logf(uniform);
}

// One block per song. Rows come in [cond_i; uncond_i] pairs. Reproduces the
// CPU chain bit-for-bit up to logf rounding: bf16-round both branches, mix in
// f32, top-k threshold (value of the k-th largest), then Gumbel-style ranking
// rank = exp(double(score - max)) / double(exponential(token)) with the
// smallest token winning ties, exactly like the sequential CPU scan.
__global__ void sample_topk_exponential_pairs_kernel(
    const float * logits,
    int64_t vocab,
    float guidance,
    int64_t top_k,
    const uint64_t * seeds,
    const uint64_t * offsets,
    uint64_t offset_step_blocks,
    uint64_t stride,
    int32_t * out_codes) {
    extern __shared__ float shared_scores[];  // [vocab original | vocab workspace]
    float * original = shared_scores;
    float * workspace = shared_scores + vocab;
    __shared__ float reduce_val[256];
    __shared__ int reduce_idx[256];
    __shared__ double reduce_rank[256];
    __shared__ float shared_threshold;
    __shared__ float shared_max;

    const int song = blockIdx.x;
    const int tid = static_cast<int>(threadIdx.x);
    const int threads = static_cast<int>(blockDim.x);
    const float * cond = logits + static_cast<size_t>(2 * song) * vocab;
    const float * uncond = logits + static_cast<size_t>(2 * song + 1) * vocab;
    for (int64_t i = tid; i < vocab; i += threads) {
        const float c = round_to_bfloat16(cond[i]);
        const float u = round_to_bfloat16(uncond[i]);
        const float mixed = u + (c - u) * guidance;
        original[i] = mixed;
        workspace[i] = mixed;
    }
    __syncthreads();

    const int64_t keep = top_k < vocab ? (top_k > 0 ? top_k : vocab) : vocab;
    for (int64_t extract = 0; extract < keep; ++extract) {
        float local_best = -INFINITY;
        int local_idx = -1;
        for (int64_t i = tid; i < vocab; i += threads) {
            const float value = workspace[i];
            if (value > local_best) {
                local_best = value;
                local_idx = static_cast<int>(i);
            }
        }
        reduce_val[tid] = local_best;
        reduce_idx[tid] = local_idx;
        __syncthreads();
        for (int step = threads / 2; step > 0; step >>= 1) {
            if (tid < step) {
                const bool take = reduce_val[tid + step] > reduce_val[tid] ||
                    (reduce_val[tid + step] == reduce_val[tid] && reduce_idx[tid + step] >= 0 &&
                     (reduce_idx[tid] < 0 || reduce_idx[tid + step] < reduce_idx[tid]));
                if (take) {
                    reduce_val[tid] = reduce_val[tid + step];
                    reduce_idx[tid] = reduce_idx[tid + step];
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            if (extract == 0) {
                shared_max = reduce_val[0];
            }
            shared_threshold = reduce_val[0];
            if (reduce_idx[0] >= 0) {
                workspace[reduce_idx[0]] = -INFINITY;
            }
        }
        __syncthreads();
    }

    const uint64_t seed = seeds[song];
    const uint64_t offset = offsets[song] + offset_step_blocks;
    double local_rank = -1.0;
    int local_token = -1;
    for (int64_t i = tid; i < vocab; i += threads) {
        const float value = original[i];
        if (isfinite(value) && value >= shared_threshold) {
            const float exponential = tensor_iterator_exponential_element_device(
                seed, static_cast<uint64_t>(i), offset, stride);
            const double weight = exp(static_cast<double>(value - shared_max));
            const double rank = weight / static_cast<double>(exponential);
            if (local_token < 0 || rank > local_rank) {
                local_rank = rank;
                local_token = static_cast<int>(i);
            }
        }
    }
    reduce_rank[tid] = local_rank;
    reduce_idx[tid] = local_token;
    __syncthreads();
    for (int step = threads / 2; step > 0; step >>= 1) {
        if (tid < step) {
            const double other = reduce_rank[tid + step];
            const int other_idx = reduce_idx[tid + step];
            const bool take = other_idx >= 0 &&
                (reduce_idx[tid] < 0 || other > reduce_rank[tid] ||
                 (other == reduce_rank[tid] && other_idx < reduce_idx[tid]));
            if (take) {
                reduce_rank[tid] = other;
                reduce_idx[tid] = other_idx;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_codes[song] = reduce_idx[0];
    }
}

}  // namespace

void sample_topk_exponential_pairs_cuda(
    const void * device_logits_f32,
    int64_t songs,
    int64_t vocab,
    float guidance_scale,
    int64_t top_k,
    const uint64_t * seeds,
    const uint64_t * offset_blocks,
    uint64_t offset_step_blocks,
    const TorchCudaSamplingPolicy & policy,
    int32_t * out_codes) {
    if (device_logits_f32 == nullptr || songs <= 0 || vocab <= 0 || out_codes == nullptr) {
        throw std::invalid_argument("torch CUDA top-k exponential sampler input is invalid");
    }
    constexpr int64_t kMaxVocab = 6144;  // 2 shared copies must fit in 48 KiB
    if (vocab > kMaxVocab) {
        throw std::invalid_argument("torch CUDA top-k exponential sampler vocab exceeds shared memory");
    }
    if (policy.multiprocessor_count <= 0 || policy.max_threads_per_multiprocessor <= 0) {
        throw std::invalid_argument("torch CUDA top-k exponential sampler requires CUDA device properties");
    }
    const uint64_t stride = tensor_iterator_stride(static_cast<uint64_t>(vocab), policy);
    check_cuda(cudaSetDevice(policy.cuda_device_index), "cudaSetDevice");

    constexpr int64_t kMaxSongs = 64;
    if (songs > kMaxSongs) {
        throw std::invalid_argument("torch CUDA top-k exponential sampler song count is too large");
    }
    static thread_local uint64_t * device_seeds = nullptr;
    static thread_local uint64_t * device_offsets = nullptr;
    static thread_local int32_t * device_codes = nullptr;
    if (device_seeds == nullptr) {
        check_cuda(cudaMalloc(&device_seeds, kMaxSongs * sizeof(uint64_t)), "cudaMalloc sampler seeds");
        check_cuda(cudaMalloc(&device_offsets, kMaxSongs * sizeof(uint64_t)), "cudaMalloc sampler offsets");
        check_cuda(cudaMalloc(&device_codes, kMaxSongs * sizeof(int32_t)), "cudaMalloc sampler codes");
    }
    // seeds/offsets are frame constants: callers pass them only on the first
    // call of a frame (offset_step_blocks advances later calls on device).
    if (seeds != nullptr && offset_blocks != nullptr) {
        check_cuda(
            cudaMemcpy(device_seeds, seeds, static_cast<size_t>(songs) * sizeof(uint64_t), cudaMemcpyHostToDevice),
            "cudaMemcpy sampler seeds");
        check_cuda(
            cudaMemcpy(device_offsets, offset_blocks, static_cast<size_t>(songs) * sizeof(uint64_t), cudaMemcpyHostToDevice),
            "cudaMemcpy sampler offsets");
    }
    const size_t shared_bytes = static_cast<size_t>(vocab) * 2 * sizeof(float);
    sample_topk_exponential_pairs_kernel<<<static_cast<int>(songs), 256, shared_bytes>>>(
        static_cast<const float *>(device_logits_f32),
        vocab,
        guidance_scale,
        top_k,
        device_seeds,
        device_offsets,
        offset_step_blocks,
        stride,
        device_codes);
    check_cuda(cudaGetLastError(), "sample_topk_exponential_pairs_kernel");
    check_cuda(
        cudaMemcpy(out_codes, device_codes, static_cast<size_t>(songs) * sizeof(int32_t), cudaMemcpyDeviceToHost),
        "cudaMemcpy sampler codes");
}

namespace {

struct DepthFrameBuffers {
    int32_t * codes = nullptr;      // [levels][songs]
    float * hidden = nullptr;       // [levels][songs][hidden_size]
    uint64_t * seeds = nullptr;     // [songs]
    uint64_t * offsets = nullptr;   // [songs]
    int64_t levels = 0;
    int64_t songs = 0;
    int64_t hidden_size = 0;
};

DepthFrameBuffers & depth_frame_buffers() {
    static thread_local DepthFrameBuffers buffers;
    return buffers;
}

__global__ void depth_frame_residual_kernel(
    int32_t * residual_ids,
    const int32_t * frame_codes,
    int64_t songs,
    int64_t previous_levels,
    int64_t audio_vocab) {
    const int64_t total = 2 * songs * previous_levels;
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    const int64_t row = index / previous_levels;
    const int64_t previous = index % previous_levels;  // 0-based: codebook previous+1
    const int64_t song = row / 2;
    const int32_t code = frame_codes[previous * songs + song];
    residual_ids[index] = code + static_cast<int32_t>(previous) * static_cast<int32_t>(audio_vocab);
}

}  // namespace

void depth_frame_ensure_cuda(int64_t songs, int64_t levels, int64_t hidden_size, const TorchCudaSamplingPolicy & policy) {
    auto & buffers = depth_frame_buffers();
    if (buffers.songs >= songs && buffers.levels >= levels && buffers.hidden_size == hidden_size) {
        return;
    }
    check_cuda(cudaSetDevice(policy.cuda_device_index), "cudaSetDevice");
    if (buffers.codes != nullptr) {
        cudaFree(buffers.codes);
        cudaFree(buffers.hidden);
        cudaFree(buffers.seeds);
        cudaFree(buffers.offsets);
    }
    buffers.levels = levels;
    buffers.songs = songs;
    buffers.hidden_size = hidden_size;
    check_cuda(cudaMalloc(&buffers.codes, static_cast<size_t>(levels * songs) * sizeof(int32_t)), "cudaMalloc frame codes");
    check_cuda(
        cudaMalloc(&buffers.hidden, static_cast<size_t>(levels * songs * hidden_size) * sizeof(float)),
        "cudaMalloc frame hidden");
    check_cuda(cudaMalloc(&buffers.seeds, static_cast<size_t>(songs) * sizeof(uint64_t)), "cudaMalloc frame seeds");
    check_cuda(cudaMalloc(&buffers.offsets, static_cast<size_t>(songs) * sizeof(uint64_t)), "cudaMalloc frame offsets");
}

void depth_frame_begin_cuda(
    const uint64_t * seeds,
    const uint64_t * offset_blocks,
    int64_t songs,
    void * stream) {
    auto & buffers = depth_frame_buffers();
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    check_cuda(
        cudaMemcpyAsync(buffers.seeds, seeds, static_cast<size_t>(songs) * sizeof(uint64_t), cudaMemcpyHostToDevice, cuda_stream),
        "cudaMemcpyAsync frame seeds");
    check_cuda(
        cudaMemcpyAsync(buffers.offsets, offset_blocks, static_cast<size_t>(songs) * sizeof(uint64_t), cudaMemcpyHostToDevice, cuda_stream),
        "cudaMemcpyAsync frame offsets");
}

void depth_frame_sample_cuda(
    const void * device_logits_f32,
    int64_t level_index,
    int64_t songs,
    int64_t vocab,
    float guidance_scale,
    int64_t top_k,
    const TorchCudaSamplingPolicy & policy,
    void * stream) {
    auto & buffers = depth_frame_buffers();
    constexpr int64_t kMaxVocab = 6144;
    if (vocab > kMaxVocab || level_index >= buffers.levels || songs > buffers.songs) {
        throw std::invalid_argument("torch CUDA depth frame sampler shape is invalid");
    }
    const uint64_t stride = tensor_iterator_stride(static_cast<uint64_t>(vocab), policy);
    const uint64_t step_blocks =
        static_cast<uint64_t>(level_index) *
        (((static_cast<uint64_t>(vocab) - 1) / (stride * 4) + 1));
    const size_t shared_bytes = static_cast<size_t>(vocab) * 2 * sizeof(float);
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    sample_topk_exponential_pairs_kernel<<<static_cast<int>(songs), 256, shared_bytes, cuda_stream>>>(
        static_cast<const float *>(device_logits_f32),
        vocab,
        guidance_scale,
        top_k,
        buffers.seeds,
        buffers.offsets,
        step_blocks,
        stride,
        buffers.codes + level_index * buffers.songs);
    check_cuda(cudaGetLastError(), "depth frame sample kernel");
}

void depth_frame_residual_fill_cuda(
    void * residual_ids_i32,
    int64_t previous_levels,
    int64_t songs,
    int64_t audio_vocab,
    void * stream) {
    auto & buffers = depth_frame_buffers();
    const int64_t total = 2 * songs * previous_levels;
    const int threads = 128;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    depth_frame_residual_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<int32_t *>(residual_ids_i32),
        buffers.codes,
        buffers.songs,
        previous_levels,
        audio_vocab);
    check_cuda(cudaGetLastError(), "depth frame residual kernel");
}

void depth_frame_accumulate_hidden_cuda(
    const void * hidden_f32,
    int64_t level_index,
    int64_t songs,
    int64_t hidden_size,
    void * stream) {
    auto & buffers = depth_frame_buffers();
    // cond rows only: source pitch is two rows, destination is packed.
    check_cuda(
        cudaMemcpy2DAsync(
            buffers.hidden + (level_index * buffers.songs) * hidden_size,
            static_cast<size_t>(hidden_size) * sizeof(float),
            hidden_f32,
            static_cast<size_t>(2 * hidden_size) * sizeof(float),
            static_cast<size_t>(hidden_size) * sizeof(float),
            static_cast<size_t>(songs),
            cudaMemcpyDeviceToDevice,
            static_cast<cudaStream_t>(stream)),
        "cudaMemcpy2DAsync frame hidden");
}

void depth_frame_end_cuda(
    int32_t * host_codes,
    float * host_hidden,
    int64_t levels,
    int64_t songs,
    int64_t hidden_size,
    void * stream) {
    auto & buffers = depth_frame_buffers();
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    // The frame code matrix is [levels][buffer_songs]; rows are contiguous per
    // level, so a 2D copy trims to the active song count.
    check_cuda(
        cudaMemcpy2DAsync(
            host_codes,
            static_cast<size_t>(songs) * sizeof(int32_t),
            buffers.codes,
            static_cast<size_t>(buffers.songs) * sizeof(int32_t),
            static_cast<size_t>(songs) * sizeof(int32_t),
            static_cast<size_t>(levels),
            cudaMemcpyDeviceToHost,
            cuda_stream),
        "cudaMemcpy2DAsync frame codes out");
    check_cuda(
        cudaMemcpy2DAsync(
            host_hidden,
            static_cast<size_t>(songs * hidden_size) * sizeof(float),
            buffers.hidden,
            static_cast<size_t>(buffers.songs * hidden_size) * sizeof(float),
            static_cast<size_t>(songs * hidden_size) * sizeof(float),
            static_cast<size_t>(levels),
            cudaMemcpyDeviceToHost,
            cuda_stream),
        "cudaMemcpy2DAsync frame hidden out");
    check_cuda(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize depth frame");
}

void fill_torch_cuda_tensor_iterator_randn_cuda(
    float * output,
    size_t count,
    uint64_t seed,
    uint64_t offset_blocks,
    const TorchCudaSamplingPolicy & policy,
    TorchRandnPrecision precision) {
    if (policy.multiprocessor_count <= 0 || policy.max_threads_per_multiprocessor <= 0) {
        throw std::invalid_argument("torch CUDA TensorIterator randn fast path requires CUDA device properties");
    }
    const uint64_t total = static_cast<uint64_t>(count);
    const uint64_t stride = tensor_iterator_stride(total, policy);
    const uint64_t loop_count = (total + stride * 4 - 1) / (stride * 4);
    constexpr int threads = 256;
    const int blocks = static_cast<int>((stride + threads - 1) / threads);

    check_cuda(cudaSetDevice(policy.cuda_device_index), "cudaSetDevice");
    float * device_output = nullptr;
    check_cuda(cudaMalloc(&device_output, count * sizeof(float)), "cudaMalloc torch random output");
    try {
        fill_tensor_iterator_randn_kernel<<<blocks, threads>>>(
            device_output,
            total,
            seed,
            offset_blocks,
            stride,
            loop_count,
            static_cast<int>(precision));
        check_cuda(cudaGetLastError(), "fill_tensor_iterator_randn_kernel");
        check_cuda(cudaMemcpy(output, device_output, count * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy torch random output");
    } catch (...) {
        cudaFree(device_output);
        throw;
    }
    check_cuda(cudaFree(device_output), "cudaFree torch random output");
}

}  // namespace engine::sampling::detail
