#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

// Mirrors the reservation/map/access sequence in ggml_cuda_pool_vmm.
int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const bool prefix = argc == 2 && std::strcmp(argv[1], "prefix") == 0;
    auto check = [](hipError_t error, const char * operation) {
        if (error == hipSuccess) return true;
        std::fprintf(stderr, "%s failed: %s (%d)\n", operation, hipGetErrorString(error), int(error));
        return false;
    };
#define CHECK(call) if (!check((call), #call)) return 2
    CHECK(hipInit(0));
    CHECK(hipSetDevice(0));
    hipDeviceProp_t device{};
    CHECK(hipGetDeviceProperties(&device, 0));
    std::printf("GPU=%s arch=%s\n", device.name, device.gcnArchName);
    int supported = 0;
    CHECK(hipDeviceGetAttribute(&supported, hipDeviceAttributeVirtualMemoryManagementSupported, 0));
    std::printf("VMM supported=%d mode=%s\n", supported, prefix ? "prefix" : "subrange");
    if (!supported) return 77;
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = 0;
    size_t granularity = 0;
    CHECK(hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityRecommended));
    constexpr size_t capacity = size_t{1} << 35;
    void * base = nullptr;
    CHECK(hipMemAddressReserve(&base, capacity, 0, nullptr, 0));
    hipMemAccessDesc access{};
    access.location = prop.location;
    access.flags = hipMemAccessFlagsProtReadWrite;
    size_t mapped = 0;
    std::vector<std::pair<void *, size_t>> mappings;
    for (size_t multiplier : {1, 2, 7, 16, 64}) {
        const size_t bytes = granularity * multiplier;
        auto * start = static_cast<unsigned char *>(base) + mapped;
        hipMemGenericAllocationHandle_t handle;
        CHECK(hipMemCreate(&handle, bytes, &prop, 0));
        CHECK(hipMemMap(start, bytes, 0, handle, 0));
        CHECK(hipMemRelease(handle));
        mappings.emplace_back(start, bytes);
        std::printf("growth=%zu previous=%zu new=%zu access_offset=%zu access_bytes=%zu\n",
                    mappings.size(), mapped, bytes, prefix ? 0 : mapped, prefix ? mapped + bytes : bytes);
        CHECK(hipMemSetAccess(prefix ? base : start, prefix ? mapped + bytes : bytes, &access, 1));
        CHECK(hipMemset(start, 0x5a, bytes));
        CHECK(hipDeviceSynchronize());
        mapped += bytes;
        for (const auto & region : mappings) {
            for (const size_t offset : {size_t{0}, region.second - 1}) {
                unsigned char value = 0;
                CHECK(hipMemcpy(&value, static_cast<unsigned char *>(region.first) + offset, 1, hipMemcpyDeviceToHost));
                if (value != 0x5a) return 3;
            }
        }
    }
    for (const auto & region : mappings) CHECK(hipMemUnmap(region.first, region.second));
    CHECK(hipMemAddressFree(base, capacity));
    std::puts("PASS: five growth events; old and new mappings remain readable");
}
