#pragma once

#include <algorithm>
#include <set>
#include <string>
#include <thread>
#include <utility>

#if defined(__linux__)
#include <fstream>
#include <sched.h>
#endif

namespace engine::core {

// Use the physical CPUs available to this process when Linux exposes topology
// data. This respects affinity/cpuset limits and avoids counting SMT siblings
// as independent cores. Other platforms fall back to the standard C++ CPU
// count; callers can always override this automatic default explicitly.
inline int default_host_thread_count() noexcept {
    static const int count = []() noexcept {
#if defined(__linux__)
        cpu_set_t allowed;
        CPU_ZERO(&allowed);
        if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
            try {
                std::set<std::pair<int, int>> physical_cores;
                for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
                    if (!CPU_ISSET(cpu, &allowed)) {
                        continue;
                    }
                    const std::string root = "/sys/devices/system/cpu/cpu" +
                        std::to_string(cpu) + "/topology/";
                    int package_id = 0;
                    int core_id = cpu;
                    std::ifstream package_file(root + "physical_package_id");
                    std::ifstream core_file(root + "core_id");
                    if (package_file && core_file && package_file >> package_id && core_file >> core_id) {
                        physical_cores.emplace(package_id, core_id);
                    }
                }
                if (!physical_cores.empty()) {
                    return static_cast<int>(physical_cores.size());
                }
            } catch (...) {
                // The hardware concurrency fallback below needs no allocation.
            }
            const int allowed_cpus = CPU_COUNT(&allowed);
            if (allowed_cpus > 0) {
                return allowed_cpus;
            }
        }
#endif
        return static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    }();
    return std::max(1, count);
}

} // namespace engine::core
