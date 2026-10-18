#include <ggml.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

// Internal conversion entry point. Avoid including CUDA device-only helpers
// from convert.cuh in this host C++ test.
using convert_fn = void (*)(const void *, half *, int64_t, int64_t, int64_t, int64_t,
                           int64_t, int64_t, int64_t, cudaStream_t);
convert_fn ggml_get_to_fp16_nc_cuda(ggml_type);

static void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Buffer {
    void * data = nullptr;
    explicit Buffer(size_t bytes) { check(cudaMalloc(&data, bytes)); }
    ~Buffer() { cudaFree(data); }
    Buffer(const Buffer &) = delete;
    Buffer & operator=(const Buffer &) = delete;
};

static void run(int n1, int n2, int n3, int layout) {
    int64_t s1 = 64, s2 = s1*n1, s3 = s2*n2;
    if (layout == 1) { s1 = 192; s2 = s1*n1 + 64; s3 = s2*n2 + 128; }
    if (layout == 2) { s2 = 64; s1 = s2*n2; s3 = s1*n1; }
    if (layout == 3) { s1 = 65; s2 = s1*n1 + 1; s3 = s2*n2 + 1; }
    // The generic 128-wide conversion reads extra values at each row's end;
    // compare only its first 64. Padding makes every such read valid, even
    // when the source rows overlap in that reference view.
    std::vector<float> input((n3-1)*s3 + (n2-1)*s2 + (n1-1)*s1 + 128);
    for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(float(i)*0.11f)*90000.0f;
    const float edge[] = {0.0f, -0.0f, 1.00048828125f, -1.00048828125f,
        std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::denorm_min(),
        0.000000059604644775390625f};
    for (size_t i = 0; i < sizeof(edge)/sizeof(edge[0]); ++i) input[i] = edge[i];
    const size_t rows = size_t(n1)*n2*n3;
    Buffer source(input.size()*sizeof(float)), reference(rows*128*sizeof(half)), packed(rows*64*sizeof(half));
    check(cudaMemcpy(source.data, input.data(), input.size()*sizeof(float), cudaMemcpyHostToDevice));
    auto convert = ggml_get_to_fp16_nc_cuda(GGML_TYPE_F32);
    convert(source.data, static_cast<half *>(reference.data), 128, n1, n2, n3, s1, s2, s3, nullptr);
    convert(source.data, static_cast<half *>(packed.data), 64, n1, n2, n3, s1, s2, s3, nullptr);
    check(cudaDeviceSynchronize());
    std::vector<uint16_t> a(rows*128), b(rows*64);
    check(cudaMemcpy(a.data(), reference.data, a.size()*sizeof(uint16_t), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(b.data(), packed.data, b.size()*sizeof(uint16_t), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < b.size(); ++i) {
        if (a[(i/64)*128 + i%64] != b[i]) throw std::runtime_error("packed conversion differs from generic CUDA conversion");
    }
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    try {
        for (int layout : {0, 1, 2, 3}) {
            run(1, 1, 1, layout);
            run(3, 2, 5, layout);
            run(7, 3, 5, layout);
            run(801, 8, 62, layout);
        }
        std::cout << "PASS: packed conversion bit-matches generic conversion; contiguous, padded and transposed strides, tails, rounding and exceptional values\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
