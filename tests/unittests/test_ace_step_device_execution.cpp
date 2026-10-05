#include "engine/models/ace_step/device_execution.h"

#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace engine::models::ace_step;
using engine::core::BackendType;
using namespace std::chrono_literals;

namespace engine::models::ace_step {
struct AceStepDeviceExecutionLeaseTestAccess {
    static size_t pending(BackendType backend, int device) {
        auto state = AceStepDeviceExecutionLease::state_for(backend, device);
        std::lock_guard<std::mutex> lock(state->mutex);
        return state->waiting.size();
    }
};
}

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

void wait_pending(size_t count) {
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (AceStepDeviceExecutionLeaseTestAccess::pending(BackendType::Cuda, 0) != count) {
        require(std::chrono::steady_clock::now() < deadline, "waiter did not reach admission queue");
        std::this_thread::yield();
    }
}

int main() {
    int a = 0, b = 0;
    auto first = std::make_unique<AceStepDeviceExecutionLease>(BackendType::Cuda, 0, &a);
    // Same package can overlap on one device.
    auto same = std::async(std::launch::async, [&] {
        AceStepDeviceExecutionLease lease(BackendType::Cuda, 0, &a);
    });
    require(same.wait_for(2s) == std::future_status::ready, "same package was serialized");
    same.get();

    std::promise<void> b_entered, b_release;
    auto release = b_release.get_future().share();
    auto other = std::async(std::launch::async, [&] {
        AceStepDeviceExecutionLease lease(BackendType::Cuda, 0, &b);
        b_entered.set_value();
        release.wait();
    });
    wait_pending(1);
    // Later work from A must not bypass the already queued B package.
    std::promise<void> later_entered;
    auto later_ready = later_entered.get_future();
    auto later = std::async(std::launch::async, [&] {
        AceStepDeviceExecutionLease lease(BackendType::Cuda, 0, &a);
        later_entered.set_value();
    });
    wait_pending(2);
    auto isolated = std::async(std::launch::async, [&] {
        AceStepDeviceExecutionLease cuda_other_device(BackendType::Cuda, 1, &b);
        AceStepDeviceExecutionLease vulkan(BackendType::Vulkan, 0, &b);
        AceStepDeviceExecutionLease cpu(BackendType::Cpu, 0, &b);
    });
    require(isolated.wait_for(2s) == std::future_status::ready, "unrelated backend/device was blocked");
    isolated.get();
    first.reset();
    require(b_entered.get_future().wait_for(2s) == std::future_status::ready, "queued package did not enter");
    require(later_ready.wait_for(0s) != std::future_status::ready, "FIFO package handoff was bypassed");
    b_release.set_value();
    other.get(); later.get();

    try {
        AceStepDeviceExecutionLease lease(BackendType::Cuda, 0, &a);
        throw std::runtime_error("injected operation failure");
    } catch (const std::runtime_error &) {}
    { AceStepDeviceExecutionLease retry(BackendType::Cuda, 0, &b); }

    std::atomic<int> active_a{0}, active_b{0}, errors{0};
    std::vector<std::future<void>> jobs;
    for (int i = 0; i < 8; ++i) jobs.push_back(std::async(std::launch::async, [&, i] {
        for (int j = 0; j < 200; ++j) {
            const bool is_a = (i + j) % 2 == 0;
            AceStepDeviceExecutionLease lease(BackendType::Cuda, 0, is_a ? &a : &b);
            auto & own = is_a ? active_a : active_b;
            auto & foreign = is_a ? active_b : active_a;
            ++own;
            if (foreign.load() != 0) ++errors;
            std::this_thread::yield();
            --own;
        }
    }));
    for (auto & job : jobs) job.get();
    require(errors == 0, "distinct packages overlapped on the same device");
    std::cout << "PASS: package overlap, FIFO handoff, device/backend isolation, failure release, 1600 contended leases\n";
}
