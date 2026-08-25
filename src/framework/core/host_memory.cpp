#include "engine/framework/core/host_memory.h"

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <cstdio>
#include <cstring>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <unistd.h>
#elif defined(__unix__)
#include <unistd.h>
#endif

namespace engine::core {

#if defined(__linux__)
namespace {

// MemAvailable: what the kernel estimates a new allocation can get without
// swapping, counting reclaimable page cache. sysconf(_SC_AVPHYS_PAGES) reports
// MemFree instead, which sits near zero on any busy machine because spare RAM
// is used for cache -- reading it as memory pressure would be wrong.
size_t mem_available_bytes() {
    std::FILE * meminfo = std::fopen("/proc/meminfo", "r");
    if (meminfo == nullptr) {
        return 0;
    }
    char line[256];
    size_t bytes = 0;
    while (std::fgets(line, sizeof(line), meminfo) != nullptr) {
        unsigned long long kib = 0;
        if (std::sscanf(line, "MemAvailable: %llu kB", &kib) == 1) {
            bytes = static_cast<size_t>(kib) * 1024ull;
            break;
        }
    }
    std::fclose(meminfo);
    return bytes;
}

}  // namespace
#elif defined(__APPLE__)
namespace {

// Mach VM stats: pages that a fresh allocation can claim without swapping.
// free_count is the idle pool; inactive_count is reclaimable file cache /
// anonymous memory; purgeable_count can be dropped on demand. Compressed
// pages are deliberately excluded -- dropping them would force decompression
// and they are already backing live data.
size_t mem_available_bytes() {
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vm_stat{};
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm_stat), &count) != KERN_SUCCESS) {
        return 0;
    }
    const uint64_t page = static_cast<uint64_t>(vm_page_size);
    return static_cast<size_t>(
        (static_cast<uint64_t>(vm_stat.free_count) +
         static_cast<uint64_t>(vm_stat.inactive_count) +
         static_cast<uint64_t>(vm_stat.purgeable_count)) * page);
}

}  // namespace
#endif

size_t available_host_memory_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return static_cast<size_t>(status.ullAvailPhys);
    }
#elif defined(__linux__)
    if (const size_t bytes = mem_available_bytes(); bytes > 0) {
        return bytes;
    }
#elif defined(__APPLE__)
    if (const size_t bytes = mem_available_bytes(); bytes > 0) {
        return bytes;
    }
#elif defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGE_SIZE)
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        return static_cast<size_t>(pages) * static_cast<size_t>(page_size);
    }
#endif
    // A query can fail on any platform: report unknown (0).
    return 0;
}

}  // namespace engine::core
