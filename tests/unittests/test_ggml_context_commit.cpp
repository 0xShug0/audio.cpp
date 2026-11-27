// ggml context pools on Windows: a large pool is reserved, and committed as objects fill it. A 1 GiB
// no_alloc pool must not commit 1 GiB up front, and headers written across several commit steps
// must read back intact. A context that holds tensor data keeps working too. On other platforms
// the test runs the same calls without the commit checks.

#include "test_assert.h"

#include <ggml.h>

#include <cstddef>
#include <iostream>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#ifdef _MSC_VER
#pragma comment(lib, "psapi.lib")
#endif
#endif

namespace {

constexpr size_t kMiB = size_t(1) << 20;

#ifdef _WIN32
// Memory this process has committed: what Windows charges against the system commit limit.
size_t committed_bytes() {
    PROCESS_MEMORY_COUNTERS_EX counters{};
    GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&counters), sizeof(counters));
    return counters.PrivateUsage;
}
#endif

}  // namespace

int main() try {
    using engine::test::require;
    constexpr size_t kPool = size_t(1) << 30;  // the size of many model graph arenas
#ifdef _WIN32
    const size_t before = committed_bytes();
#endif
    ggml_context * ctx = ggml_init(ggml_init_params{kPool, nullptr, true});
    require(ctx != nullptr, "ggml_init failed");
    require(ggml_get_mem_size(ctx) == kPool, "the pool keeps its full size");
#ifdef _WIN32
    const size_t up_front = committed_bytes() - before;
    require(up_front < 64 * kMiB, "a 1 GiB pool committed " + std::to_string(up_front / kMiB) + " MiB up front");
#endif

    // Enough headers to cross several commit steps (about 11 MiB of them).
    constexpr int kTensors = 30000;
    for (int i = 0; i < kTensors; ++i) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, i % 7 + 1);
        require(t != nullptr, "tensor " + std::to_string(i));
        ggml_format_name(t, "t%d", i);
    }
    int count = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        require(t->ne[0] == 16 && t->ne[1] == count % 7 + 1, "tensor header " + std::to_string(count) + " intact");
        require(std::string(ggml_get_name(t)) == "t" + std::to_string(count), "tensor name " + std::to_string(count));
        ++count;
    }
    require(count == kTensors, "tensor count");
    const size_t used = ggml_used_mem(ctx);
#ifdef _WIN32
    const size_t grown = committed_bytes() - before;
    require(grown < 2 * used + 64 * kMiB,
            "committed " + std::to_string(grown / kMiB) + " MiB for " + std::to_string(used / kMiB) + " MiB of headers");
#endif
    ggml_free(ctx);

    // A context that holds tensor data (no_alloc = false): the data lies inside its objects.
    ggml_context * data_ctx = ggml_init(ggml_init_params{16 * kMiB, nullptr, false});
    require(data_ctx != nullptr, "ggml_init (data) failed");
    ggml_tensor * big = ggml_new_tensor_1d(data_ctx, GGML_TYPE_F32, 2 * 1024 * 1024);  // 8 MiB of data
    require(big != nullptr && big->data != nullptr, "data tensor");
    float * values = static_cast<float *>(big->data);
    values[0] = 1.0f;
    values[2 * 1024 * 1024 - 1] = 2.0f;  // the last element, several commit steps in
    require(values[0] == 1.0f && values[2 * 1024 * 1024 - 1] == 2.0f, "data tensor round trip");
    ggml_free(data_ctx);

    std::cout << "ggml context commit tests passed: " << count << " tensors, " << used / kMiB << " MiB of headers\n";
    return 0;
} catch (const std::exception & e) {
    std::cerr << e.what() << '\n';
    return 1;
}
