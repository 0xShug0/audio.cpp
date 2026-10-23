#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblaslt/hipblaslt.h>

#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#define GGML_CUDA_MAX_DEVICES 16
#define HIPBLASLT_WORKSPACE_SIZE (size_t{32} * 1024 * 1024)
#define GGML_ASSERT(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
#define HIPBLASLT_CHECK(x) do { auto rc = (x); if (rc != HIPBLAS_STATUS_SUCCESS) throw std::runtime_error(#x " status=" + std::to_string(int(rc))); } while (0)
#define HIP_CHECK(x) do { auto rc = (x); if (rc != hipSuccess) throw std::runtime_error(std::string(#x ": ") + hipGetErrorString(rc)); } while (0)
using cudaStream_t = hipStream_t;

// Private handles, workspaces and streams isolate descriptor sharing from other
// same-context concurrency limitations in the full backend.
struct Resources {
    hipblasLtHandle_t handle = nullptr;
    void * workspace = nullptr;
    hipStream_t stream = nullptr;
    Resources() {
        HIP_CHECK(hipSetDevice(0));
        HIPBLASLT_CHECK(hipblasLtCreate(&handle));
        HIP_CHECK(hipMalloc(&workspace, HIPBLASLT_WORKSPACE_SIZE));
        HIP_CHECK(hipStreamCreate(&stream));
    }
    ~Resources() {
        hipStreamDestroy(stream);
        hipFree(workspace);
        hipblasLtDestroy(handle);
    }
};
thread_local Resources * resources = nullptr;
struct ggml_backend_cuda_context {
    int device = 0;
#include "hipblaslt_cache.inc"
    hipblasLtHandle_t hipblaslt_handle() { return resources->handle; }
    void * hipblaslt_workspace(int) { return resources->workspace; }
};

std::mutex observation_mutex;
std::unordered_map<hipblasLtMatmulDesc_t, int> active_descriptors;
std::atomic<int> overlaps{0};
template<class... Args>
hipblasStatus_t observed_matmul(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t descriptor, Args... args) {
    {
        std::lock_guard<std::mutex> lock(observation_mutex);
        if (active_descriptors[descriptor]++ > 0) ++overlaps;
    }
    const auto status = hipblasLtMatmul(handle, descriptor, args...);
    {
        std::lock_guard<std::mutex> lock(observation_mutex);
        if (--active_descriptors[descriptor] == 0) active_descriptors.erase(descriptor);
    }
    return status;
}
#include "hipblaslt_gemm.inc"

struct Barrier {
    std::mutex mutex;
    std::condition_variable changed;
    int count = 0, phase = 0;
    bool stopped = false;
    bool wait() {
        std::unique_lock<std::mutex> lock(mutex);
        const auto previous = phase;
        if (++count == 4) { count = 0; ++phase; changed.notify_all(); }
        changed.wait(lock, [&] { return stopped || phase != previous; });
        return !stopped;
    }
    void cancel() { std::lock_guard<std::mutex> lock(mutex); stopped = true; changed.notify_all(); }
};

bool half_type(int type) { return type == HIP_R_16F || type == 150; }
bool supported_type(int type) { return half_type(type) || type == HIP_R_32F || type == 151; }
size_t element_bytes(int type) { return half_type(type) ? 2 : 4; }
void put(std::vector<unsigned char> & bytes, size_t index, int type, float value) {
    if (half_type(type)) {
        const auto h = __float2half(value);
        std::memcpy(bytes.data() + index * 2, &h, 2);
    } else std::memcpy(bytes.data() + index * 4, &value, 4);
}
float get(const std::vector<unsigned char> & bytes, size_t index, int type) {
    if (half_type(type)) {
        __half h;
        std::memcpy(&h, bytes.data() + index * 2, 2);
        return __half2float(h);
    }
    float value;
    std::memcpy(&value, bytes.data() + index * 4, 4);
    return value;
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc != 3) return 2;
    hipDeviceProp_t device{};
    HIP_CHECK(hipGetDeviceProperties(&device, 0));
    std::printf("GPU=%s arch=%s\n", device.name, device.gcnArchName);
    std::ifstream input(argv[1]);
    int shapes = 0, failed_cases = 0;
    for (std::string line; std::getline(input, line) && shapes < 3;) {
        std::array<int64_t, 13> shape{};
        std::istringstream fields(line);
        bool valid = true;
        for (auto & value : shape) if (!(fields >> value)) valid = false;
        const auto [m,n,k,ta,tb,tc,lda,ldb,ldc,sa,sb,sc,batches] = shape;
        if (!valid || batches != 1 || !supported_type(ta) || !supported_type(tb) || !supported_type(tc)) continue;
        ++shapes;
        std::printf("SHAPE m=%lld n=%lld k=%lld types=%lld,%lld,%lld ld=%lld,%lld,%lld\n",
                    (long long)m,(long long)n,(long long)k,(long long)ta,(long long)tb,(long long)tc,
                    (long long)lda,(long long)ldb,(long long)ldc);
        for (const std::string mode : {"private", "serialized", "shared"}) {
            if (mode != argv[2]) continue;
            ggml_backend_cuda_context shared_context;
            Barrier barrier;
            std::mutex submission_mutex;
            std::atomic<int> failures{0};
            overlaps = 0;
            std::vector<std::thread> workers;
            for (int worker = 0; worker < 4; ++worker) workers.emplace_back([&,worker] {
                try {
                    Resources local_resources;
                    resources = &local_resources;
                    ggml_backend_cuda_context private_context;
                    auto & context = mode == "private" ? private_context : shared_context;
                    std::vector<unsigned char> a(lda*m*element_bytes(ta)), b(ldb*n*element_bytes(tb)), c(ldc*n*element_bytes(tc));
                    std::vector<float> expected(ldc*n, 0);
                    for (int64_t i = 0; i < m; ++i)
                        for (int64_t q = 0; q < k; ++q) put(a, i*lda+q, ta, float(1+(i+worker*3)%7)/64);
                    for (int64_t j = 0; j < n; ++j) {
                        float sum = 0;
                        for (int64_t q = 0; q < k; ++q) {
                            const float value = float(1+(q+j+worker*5)%7)/64;
                            put(b, j*ldb+q, tb, value);
                            sum += value;
                        }
                        for (int64_t i = 0; i < m; ++i) expected[j*ldc+i] = float(1+(i+worker*3)%7)/64 * sum;
                    }
                    void * da = nullptr, * db = nullptr, * dc = nullptr;
                    HIP_CHECK(hipMalloc(&da, a.size()));
                    HIP_CHECK(hipMalloc(&db, b.size()));
                    HIP_CHECK(hipMalloc(&dc, c.size()));
                    HIP_CHECK(hipMemcpy(da, a.data(), a.size(), hipMemcpyHostToDevice));
                    HIP_CHECK(hipMemcpy(db, b.data(), b.size(), hipMemcpyHostToDevice));
                    auto submit = [&] {
                        ggml_hipblaslt_gemm(context, local_resources.stream, m,n,k,
                            da,ta,lda,sa, db,tb,ldb,sb, dc,tc,ldc,sc,batches);
                    };
                    // Warm cache and heuristics serially before the simultaneous calls.
                    { std::lock_guard<std::mutex> lock(submission_mutex); submit(); HIP_CHECK(hipStreamSynchronize(local_resources.stream)); }
                    for (int iteration = 0; iteration < 128; ++iteration) {
                        HIP_CHECK(hipMemsetAsync(dc, 0xff, c.size(), local_resources.stream));
                        HIP_CHECK(hipStreamSynchronize(local_resources.stream));
                        if (!barrier.wait()) break;
                        if (mode == "serialized") { std::lock_guard<std::mutex> lock(submission_mutex); submit(); }
                        else submit();
                        HIP_CHECK(hipStreamSynchronize(local_resources.stream));
                        HIP_CHECK(hipMemcpy(c.data(), dc, c.size(), hipMemcpyDeviceToHost));
                        bool mismatch = false;
                        for (int64_t j = 0; j < n; ++j) for (int64_t i = 0; i < m; ++i) {
                            const auto index = j*ldc+i;
                            const float actual = get(c, index, tc);
                            const float tolerance = half_type(tc) ? 0.001f*std::abs(expected[index])+0.001f : 0.0001f;
                            if (!std::isfinite(actual) || std::abs(actual-expected[index]) > tolerance) mismatch = true;
                        }
                        if (mismatch) {
                            if (failures.fetch_add(1) == 0) std::printf("FIRST_MISMATCH mode=%s worker=%d iteration=%d\n",mode.c_str(),worker,iteration);
                        }
                        if (!barrier.wait()) break;
                    }
                    HIP_CHECK(hipFree(dc)); HIP_CHECK(hipFree(db)); HIP_CHECK(hipFree(da));
                } catch (const std::exception & error) {
                    ++failures;
                    std::fprintf(stderr,"ERROR mode=%s worker=%d %s\n",mode.c_str(),worker,error.what());
                    barrier.cancel();
                }
            });
            for (auto & worker : workers) worker.join();
            std::printf("RESULT shape=%d mode=%s failures=%d overlapping_descriptor_calls=%d\n",shapes,mode.c_str(),failures.load(),overlaps.load());
            if (failures) ++failed_cases;
        }
    }
    std::printf("SUMMARY shapes=%d failed_cases=%d\n",shapes,failed_cases);
    return shapes == 0 ? 2 : failed_cases ? 3 : 0;
}
