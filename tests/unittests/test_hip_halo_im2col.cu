#include "im2col.cuh"
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "assets/hip_halo/original_im2col.cuh"

#define HIP_CHECK(x) do { auto rc = (x); if (rc != hipSuccess) throw std::runtime_error(std::string(#x ": ") + hipGetErrorString(rc)); } while (0)

struct Shape {
    std::string name;
    int64_t width, channels, batches, kernel, stride, padding, dilation;
    bool gaps = false, benchmark = false;
    int64_t explicit_channel_stride = 0, explicit_batch_stride = 0;
    ggml_type only_type = GGML_TYPE_COUNT;
};

uint16_t half_reference(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    const uint32_t sign = (bits >> 16) & 0x8000;
    const int exponent = int((bits >> 23) & 255) - 127 + 15;
    const uint32_t mantissa = bits & 0x7fffff;
    if (exponent >= 31) return uint16_t(sign | 0x7c00);
    if (exponent < -10) return uint16_t(sign);
    if (exponent <= 0) {
        const unsigned shift = 14 - exponent;
        const uint32_t significand = mantissa | 0x800000;
        const uint32_t rounded = (significand + ((uint32_t{1} << (shift - 1)) - 1) + ((significand >> shift) & 1)) >> shift;
        return uint16_t(sign | rounded);
    }
    const uint32_t rounded = mantissa + 0xfff + ((mantissa >> 13) & 1);
    return uint16_t(sign | ((uint32_t(exponent) << 10) + (rounded >> 13)));
}

uint16_t bf16_reference(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    return uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

template<typename T>
void original(const Shape & s, int64_t positions, int64_t cs, int64_t bs,
        const float * input, void * output, hipStream_t stream) {
    const int64_t columns = s.channels * s.kernel;
    const dim3 grid(unsigned((columns + 255) / 256), unsigned(std::min<int64_t>(positions, 65535)),
                    unsigned(std::min<int64_t>(s.batches, 65535)));
    original_im2col_kernel<T><<<grid, unsigned(std::min<int64_t>(columns, 256)), 0, stream>>>(
        input, static_cast<T *>(output), s.channels, s.width, 1, 1, positions, s.kernel, 1,
        cs, bs, s.batches, s.kernel, columns, s.stride, 1, s.padding, 0, s.dilation, 1);
}

void run(const Shape & s, ggml_type type, hipStream_t stream) {
    const int64_t positions = (s.width + 2*s.padding - s.dilation*(s.kernel-1) - 1) / s.stride + 1;
    if (positions <= 0) return;
    const int64_t cs = s.explicit_channel_stride ? s.explicit_channel_stride : s.width + (s.gaps ? 7 : 0);
    const int64_t bs = s.explicit_batch_stride ? s.explicit_batch_stride : cs*s.channels + (s.gaps ? 13 : 0);
    const size_t elements = size_t(s.batches) * positions * s.channels * s.kernel;
    const size_t element_size = type == GGML_TYPE_F32 ? 4 : 2;
    const size_t bytes = elements * element_size;
    std::vector<float> host_input(size_t(bs*s.batches) + 32);
    const float edges[] = {0.f, -0.f, 1.f, -1.f, 0.00000001f, 0.00006103515625f, 1.00048828125f, 65504.f, 65520.f};
    for (size_t i = 0; i < host_input.size(); ++i)
        host_input[i] = i % 19 < 9 ? edges[i % 19] : std::ldexp(float(int(i % 4093) - 2046), -7);
    float * input_all = nullptr;
    unsigned char * a = nullptr, * b = nullptr;
    HIP_CHECK(hipMalloc(&input_all, host_input.size()*4));
    HIP_CHECK(hipMalloc(&a, bytes+256));
    HIP_CHECK(hipMalloc(&b, bytes+256));
    auto * input = input_all + 5;
    auto launch = [&](bool halo) {
        if (halo) {
            if (!ggml_hip_halo_im2col_1d(input, b+128, type, s.width, positions, s.kernel, s.channels,
                    s.batches, cs, bs, s.stride, s.padding, s.dilation, stream))
                throw std::runtime_error("Halo rejected test case");
        } else if (type == GGML_TYPE_F32) original<float>(s,positions,cs,bs,input,a+128,stream);
        else if (type == GGML_TYPE_F16) original<__half>(s,positions,cs,bs,input,a+128,stream);
        else original<hip_bfloat16>(s,positions,cs,bs,input,a+128,stream);
        HIP_CHECK(hipGetLastError());
    };
    std::vector<unsigned char> actual(bytes+256), baseline(bytes+256);
    for (int pass = 0; pass < 2; ++pass) {
        if (pass) for (auto & value : host_input) value = -value;
        HIP_CHECK(hipMemcpy(input_all,host_input.data(),host_input.size()*4,hipMemcpyHostToDevice));
        HIP_CHECK(hipMemsetAsync(a, 0xa5+pass, bytes+256, stream));
        HIP_CHECK(hipMemsetAsync(b, 0xa5+pass, bytes+256, stream));
        launch(false); launch(true);
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipMemcpy(actual.data(), b, bytes+256, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(baseline.data(), a, bytes+256, hipMemcpyDeviceToHost));
        if (actual != baseline) throw std::runtime_error(s.name + ": original/Halo byte mismatch");
        for (size_t i = 0; i < 128; ++i) {
            if (actual[i] != 0xa5+pass || actual[bytes+128+i] != 0xa5+pass)
                throw std::runtime_error("output guard overwritten");
        }
        size_t out = 128;
        for (int64_t batch = 0; batch < s.batches; ++batch)
            for (int64_t position = 0; position < positions; ++position)
                for (int64_t channel = 0; channel < s.channels; ++channel)
                    for (int64_t tap = 0; tap < s.kernel; ++tap, out += element_size) {
                        const int64_t ix = position*s.stride + tap*s.dilation - s.padding;
                        const float value = ix >= 0 && ix < s.width ? host_input[5+batch*bs+channel*cs+ix] : 0;
                        uint32_t expected;
                        if (type == GGML_TYPE_F32) std::memcpy(&expected,&value,4);
                        else expected = type == GGML_TYPE_F16 ? half_reference(value) : bf16_reference(value);
                        if (std::memcmp(actual.data()+out,&expected,element_size))
                            throw std::runtime_error(s.name + ": independent scalar reference mismatch at " + std::to_string(out));
                    }
    }
    if (s.benchmark) {
        launch(false); launch(true); HIP_CHECK(hipStreamSynchronize(stream));
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop));
        const int iterations = bytes > 64*1024*1024 ? 4 : bytes > 1024*1024 ? 16 : 64;
        std::vector<float> timings[2];
        for (int round = 0; round < 7; ++round) for (int order = 0; order < 2; ++order) {
            const int mode = (round+order)%2;
            HIP_CHECK(hipEventRecord(start,stream));
            for (int iteration = 0; iteration < iterations; ++iteration) launch(mode);
            HIP_CHECK(hipEventRecord(stop,stream)); HIP_CHECK(hipEventSynchronize(stop));
            float elapsed = 0; HIP_CHECK(hipEventElapsedTime(&elapsed,start,stop));
            timings[mode].push_back(elapsed/iterations);
        }
        for (auto & values : timings) std::sort(values.begin(),values.end());
        std::printf("BENCH,%s,%d,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%.8f,%.8f,%.4f\n",s.name.c_str(),int(type),
            (long long)s.width,(long long)s.channels,(long long)s.batches,(long long)s.kernel,
            (long long)s.stride,(long long)s.padding,(long long)s.dilation,(long long)positions,
            timings[0][3],timings[1][3],timings[0][3]/timings[1][3]);
        HIP_CHECK(hipEventDestroy(start)); HIP_CHECK(hipEventDestroy(stop));
    }
    HIP_CHECK(hipFree(b)); HIP_CHECK(hipFree(a)); HIP_CHECK(hipFree(input_all));
}

int main(int argc, char ** argv) {
    setvbuf(stdout,nullptr,_IONBF,0);
    try {
        int devices = 0;
        if (hipGetDeviceCount(&devices) != hipSuccess || !devices) return 77;
        hipDeviceProp_t device{}; HIP_CHECK(hipGetDeviceProperties(&device,0));
        std::printf("GPU=%s arch=%s\n",device.name,device.gcnArchName);
        if (std::string(device.gcnArchName).substr(0,std::string(device.gcnArchName).find(':')) != "gfx1151") return 77;
        hipStream_t stream; HIP_CHECK(hipStreamCreate(&stream));
        std::vector<Shape> shapes;
        for (int width : {1,31,32,33,65,257}) for (int kernel : {1,3,7,11})
            for (int channels : {1,3,32}) for (int stride : {1,2}) {
                const int dilation = width % 3 == 0 ? 3 : 1;
                shapes.push_back({"boundary",width,channels,3,kernel,stride,(kernel/2)*dilation,dilation,true});
            }
        shapes.push_back({"grid_y_loop",65537,1,1,1,1,0,1});
        shapes.push_back({"grid_z_loop",1,1,65537,1,1,0,1});
        for (int kernel : {32,80,1023,1024}) shapes.push_back({"large_kernel",33,2,3,kernel,1,kernel/2,1,true});
        for (int kernel : {3,7,11}) {
            const int dilation = kernel == 11 ? 5 : 1;
            shapes.push_back({"hift",104281,64,1,kernel,1,kernel/2*dilation,dilation,false,true});
        }
        if (argc == 2) {
            std::ifstream input(argv[1]);
            if (!input) throw std::runtime_error("cannot open production shapes");
            for (std::string line; std::getline(input,line);) {
                int type;
                int64_t width,positions,kernel,channels,batches,cs,bs,stride,padding,dilation,is2d;
                std::istringstream fields(line);
                if (!(fields>>type>>width>>positions>>kernel>>channels>>batches>>cs>>bs>>stride>>padding>>dilation>>is2d))
                    throw std::runtime_error("malformed production shape");
                if (!is2d) shapes.push_back({"production",width,channels,batches,kernel,stride,padding,dilation,
                    false,true,cs,bs,ggml_type(type)});
            }
        }
        int count = 0;
        for (const auto & shape : shapes) for (auto type : {GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16}) {
            if (shape.only_type != GGML_TYPE_COUNT && shape.only_type != type) continue;
            run(shape,type,stream); ++count;
        }
        HIP_CHECK(hipStreamDestroy(stream));
        std::printf("PASS cases=%d; two input updates; original and independent scalar bit parity; guards intact\n",count);
    } catch (const std::exception & error) {
        std::fprintf(stderr,"FAIL %s\n",error.what()); return 1;
    }
}
