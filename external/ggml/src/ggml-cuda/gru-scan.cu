#include "gru-scan.cuh"
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// A block owns one sequence for its entire recurrence. Double-buffered
// dot-product operands require one block barrier per frame; state stays F32.
template <int lanes>
static __global__ void gru_scan(const float *x, const float *weight,
                                const float *bias, const float *initial,
                                float *out, int h, int batches, int frames) {
  extern __shared__ float storage[];
  float *current = storage;
  float *next = current + h;
  const int d = blockIdx.y, b = blockIdx.x;
  for (int c = threadIdx.x; c < h; c += blockDim.x)
    current[c] = initial[(d * batches + b) * h + c];
  __syncthreads();
  const int lane = threadIdx.x % lanes, group = threadIdx.x / lanes;
  const unsigned mask = ((1u << lanes) - 1u)
                        << ((threadIdx.x % 32) / lanes * lanes);
  for (int step = 0; step < frames; ++step) {
    const int t = d == 1 ? frames - 1 - step : step;
    const float *gates = x + ((int64_t(d) * frames + t) * batches + b) * 3 * h;
    const float *prev =
        step ? out +
                   ((int64_t(d) * frames + (d == 1 ? t + 1 : t - 1)) * batches +
                    b) *
                       h
             : initial + (d * batches + b) * h;
    float *row = out + ((int64_t(d) * frames + t) * batches + b) * h;
    for (int j = group; j < h; j += blockDim.x / lanes) {
      float r = 0, z = 0, n = 0;
      for (int c = lane; c < h; c += lanes) {
        const int64_t i = (int64_t(d) * h + c) * 3 * h + j;
        const float v = float(current[c]);
        r += v * float(weight[i]);
        z += v * float(weight[i + h]);
        n += v * float(weight[i + 2 * h]);
      }
      for (int delta = lanes / 2; delta; delta /= 2) {
        r += __shfl_xor_sync(mask, r, delta);
        z += __shfl_xor_sync(mask, z, delta);
        n += __shfl_xor_sync(mask, n, delta);
      }
      if (lane == 0) {
        r = 1.0f / (1.0f + expf(-(gates[j] + (r + bias[d * 3 * h + j]))));
        z = 1.0f /
            (1.0f + expf(-(gates[j + h] + (z + bias[d * 3 * h + j + h]))));
        n = tanhf(gates[j + 2 * h] + r * (n + bias[d * 3 * h + j + 2 * h]));
        const float value = n + z * (prev[j] - n);
        row[j] = value;
        next[j] = value;
      }
    }
    __syncthreads();
    float *swap = current;
    current = next;
    next = swap;
  }
}

// Small recurrent matrices fit in the block's register file. Load once per
// sequence rather than fetching the same coefficients at every frame.
template <int inputs_per_lane>
static __global__
__launch_bounds__(inputs_per_lane * 16, 1) void gru_scan_registers(
    const float *x, const float *weight, const float *bias,
    const float *initial, float *out, int h, int batches, int frames) {
  extern __shared__ float storage_f32[];
  float *current = storage_f32;
  float *next = current + h;
  const int d = blockIdx.y, b = blockIdx.x, j = threadIdx.x / 4,
            lane = threadIdx.x % 4;
  float wr[inputs_per_lane], wz[inputs_per_lane], wn[inputs_per_lane];
#pragma unroll
  for (int k = 0; k < inputs_per_lane; ++k) {
    const int c = lane + 4 * k;
    const int64_t i = (int64_t(d) * h + c) * 3 * h + j;
    wr[k] = j < h && c < h ? weight[i] : 0.0f;
    wz[k] = j < h && c < h ? weight[i + h] : 0.0f;
    wn[k] = j < h && c < h ? weight[i + 2 * h] : 0.0f;
  }
  for (int c = threadIdx.x; c < h; c += blockDim.x)
    current[c] = initial[(d * batches + b) * h + c];
  __syncthreads();
  const unsigned mask = 0xfu << ((threadIdx.x % 32) / 4 * 4);
  for (int step = 0; step < frames; ++step) {
    const int t = d == 1 ? frames - 1 - step : step;
    float r = 0, z = 0, n = 0;
#pragma unroll
    for (int k = 0; k < inputs_per_lane; ++k) {
      const int c = lane + 4 * k;
      const float v = c < h ? current[c] : 0.0f;
      r += v * wr[k];
      z += v * wz[k];
      n += v * wn[k];
    }
    for (int delta = 2; delta; delta /= 2) {
      r += __shfl_xor_sync(mask, r, delta);
      z += __shfl_xor_sync(mask, z, delta);
      n += __shfl_xor_sync(mask, n, delta);
    }
    if (lane == 0 && j < h) {
      const int64_t offset = ((int64_t(d) * frames + t) * batches + b) * h + j;
      const int64_t input =
          ((int64_t(d) * frames + t) * batches + b) * 3 * h + j;
      r = 1.0f / (1.0f + expf(-(x[input] + (r + bias[d * 3 * h + j]))));
      z = 1.0f / (1.0f + expf(-(x[input + h] + (z + bias[d * 3 * h + j + h]))));
      n = tanhf(x[input + 2 * h] + r * (n + bias[d * 3 * h + j + 2 * h]));
      const float previous = current[j];
      const float value = n + z * (previous - n);
      out[offset] = value;
      next[j] = value;
    }
    __syncthreads();
    float *swap = current;
    current = next;
    next = swap;
  }
}

static void launch_gru(ggml_backend_cuda_context &ctx, ggml_tensor *dst) {
  const int h = dst->ne[0], batches = dst->ne[1], frames = dst->ne[2];
  const dim3 grid(batches, dst->ne[3]);
  const auto *x = static_cast<const float *>(dst->src[0]->data);
  const auto *w = static_cast<const float *>(dst->src[1]->data);
  const auto *bias = static_cast<const float *>(dst->src[2]->data);
  const auto *initial = static_cast<const float *>(dst->src[3]->data);
  auto *out = static_cast<float *>(dst->data);
  const bool parallel_dot =
      batches * dst->ne[3] < ggml_cuda_info().devices[ctx.device].nsm;
  const int threads =
      std::min(256, ((h * (parallel_dot ? 8 : 1) + 31) / 32) * 32);
  if (parallel_dot)
    gru_scan<8><<<grid, threads, 2 * h * sizeof(float), ctx.stream()>>>(
        x, w, bias, initial, out, h, batches, frames);
  else
    gru_scan<1><<<grid, threads, 2 * h * sizeof(float), ctx.stream()>>>(
        x, w, bias, initial, out, h, batches, frames);
}

void ggml_cuda_op_gru_scan(ggml_backend_cuda_context &ctx, ggml_tensor *dst) {
  const int h = dst->ne[0], batches = dst->ne[1], frames = dst->ne[2];
  if (h <= 128) {
    const int threads = ((h * 4 + 31) / 32) * 32;
    const dim3 grid(batches, dst->ne[3]);
    const auto *x = static_cast<const float *>(dst->src[0]->data);
    const auto *w = static_cast<const float *>(dst->src[1]->data);
    const auto *bias = static_cast<const float *>(dst->src[2]->data);
    const auto *initial = static_cast<const float *>(dst->src[3]->data);
    auto *output = static_cast<float *>(dst->data);
    if (h <= 32)
      gru_scan_registers<8>
          <<<grid, threads, 2 * h * sizeof(float), ctx.stream()>>>(
              x, w, bias, initial, output, h, batches, frames);
    else if (h <= 64)
      gru_scan_registers<16>
          <<<grid, threads, 2 * h * sizeof(float), ctx.stream()>>>(
              x, w, bias, initial, output, h, batches, frames);
    else
      gru_scan_registers<32>
          <<<grid, threads, 2 * h * sizeof(float), ctx.stream()>>>(
              x, w, bias, initial, output, h, batches, frames);
  } else
    launch_gru(ctx, dst);
}

static __global__ void gru_cell(const char *x, const char *recurrent,
                                const char *bias, const char *previous,
                                float *out, int64_t h, int64_t batches,
                                int64_t n, size_t x1, size_t x2, size_t q1,
                                size_t q2, size_t bias2, size_t p1, size_t p2) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const int64_t j = i % h, b = i / h % batches, d = i / (h * batches);
  const auto *a = reinterpret_cast<const float *>(x + b * x1 + d * x2) + j;
  const auto *q =
      reinterpret_cast<const float *>(recurrent + b * q1 + d * q2) + j;
  const auto *v = reinterpret_cast<const float *>(bias + d * bias2) + j;
  const auto *p =
      reinterpret_cast<const float *>(previous + b * p1 + d * p2) + j;
  const float r = 1.0f / (1.0f + expf(-(a[0] + (q[0] + v[0]))));
  const float z = 1.0f / (1.0f + expf(-(a[h] + (q[h] + v[h]))));
  const float nvalue = tanhf(a[2 * h] + r * (q[2 * h] + v[2 * h]));
  out[i] = nvalue + z * (*p - nvalue);
}

void ggml_cuda_op_gru_cell(ggml_backend_cuda_context &ctx, ggml_tensor *dst) {
  const auto *x = dst->src[0];
  const auto *q = dst->src[1];
  const auto *bias = dst->src[2];
  const auto *p = dst->src[3];
  const int64_t count = ggml_nelements(p);
  int64_t frame = 0;
  memcpy(&frame, dst->op_params, sizeof(frame));
  gru_cell<<<(count + 255) / 256, 256, 0, ctx.stream()>>>(
      static_cast<const char *>(x->data), static_cast<const char *>(q->data),
      static_cast<const char *>(bias->data), static_cast<const char *>(p->data),
      static_cast<float *>(dst->data) + frame * count, p->ne[0], p->ne[1], count, x->nb[1],
      x->nb[2], q->nb[1], q->nb[2], bias->nb[2], p->nb[1], p->nb[2]);
}
#endif
