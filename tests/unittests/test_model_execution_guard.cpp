#include "model_execution_guard.h"
#include <future>
#include <iostream>
#include <thread>
#include <vector>

using minitts::server::ModelExecutionGuard;
using minitts::server::ServerBusyError;
using namespace std::chrono_literals;

static void require(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}

static void legacy_path(bool explicit_one) {
    ModelExecutionGuard guard;
    if (explicit_one) { guard.configure(1); }
    require(!guard.parallel(), "one slot must select the original guard");
    {
        auto request = guard.acquire_run(0, "legacy");
        require(request.slot() == 0 && guard.active() == 1, "legacy request owns the direct session");
        require(!guard.try_acquire(), "management cannot enter active legacy inference");
        auto moved = std::move(request);
        require(guard.active() == 1 && moved.slot() == 0, "moving a lease must retain ownership");
        bool expired = false;
        auto waiter = std::async(std::launch::async, [&] {
            try { auto next = guard.acquire_run(40, "legacy"); }
            catch (const ServerBusyError &) { return true; }
            return false;
        });
        expired = waiter.get();
        require(expired, "blocked legacy request must time out");
        require(guard.state().waiting_requests == 0, "timeout must drain legacy request accounting");
    }
    require(guard.active() == 0, "moved legacy lease must release exactly once");
    {
        auto manager = guard.try_acquire();
        require(manager.has_value(), "legacy guard reusable after release");
        bool no_slot = false;
        try { manager->slot(); } catch (const std::logic_error &) { no_slot = true; }
        require(no_slot, "management must not own an inference session");
        auto waiter = std::async(std::launch::async, [&] {
            try { auto next = guard.acquire(40, "legacy"); }
            catch (const ServerBusyError &) { return true; }
            return false;
        });
        require(waiter.get(), "legacy management timeout must be bounded");
        require(guard.state().waiting_management == 0, "timeout must drain management accounting");
    }
    require(guard.try_acquire().has_value(), "legacy management must release ownership");
}

static void parallel_path(int count) {
    ModelExecutionGuard guard;
    guard.configure(count);
    require(guard.parallel(), "explicit multiple slots must select ModelSlots");
    std::vector<ModelExecutionGuard::Lock> leases;
    for (int i = 0; i < count; ++i) {
        leases.push_back(guard.acquire_run(0, "parallel"));
        require(leases.back().slot() == static_cast<size_t>(i), "requests must own distinct slots");
    }
    require(guard.active() == count, "all parallel leases must be visible");
    require(!guard.try_acquire(), "management must wait for all parallel leases");
    auto overflow = std::async(std::launch::async, [&] {
        try { auto lease = guard.acquire_run(40, "parallel"); }
        catch (const ServerBusyError &) { return true; }
        return false;
    });
    require(overflow.get(), "parallel saturation must time out");
    leases.clear();
    require(guard.active() == 0 && guard.state().waiting_requests == 0, "parallel counters must drain");
    auto exclusive = guard.try_acquire();
    require(exclusive.has_value(), "parallel management must acquire after drain");
    guard.configure(count == 2 ? 4 : 2);
    exclusive.reset();
    require(guard.active() == 0, "parallel resizing must keep the guard reusable");
}

static void configuration_identity() {
    for (int count : {1, 2}) {
        ModelExecutionGuard guard;
        guard.configure(count);
        auto manager = guard.acquire(0, "configuration");
        bool rejected = false;
        try { guard.configure(count == 1 ? 2 : 1); }
        catch (const std::runtime_error &) { rejected = true; }
        require(rejected && guard.parallel() == (count > 1), "reconfiguration cannot swap guards under queued callers");
    }
    ModelExecutionGuard legacy;
    for (int invalid : {0, 17}) {
        bool rejected = false;
        try { legacy.configure(invalid); } catch (const std::runtime_error &) { rejected = true; }
        require(rejected && !legacy.parallel(), "invalid count must leave the default intact");
    }
    ModelExecutionGuard other;
    {
        auto first = legacy.acquire_run(0, "first");
        auto second = other.acquire_run(0, "second");
        first = std::move(second);
        require(legacy.active() == 0 && other.active() == 1, "move assignment releases the previous model");
        require(legacy.try_acquire().has_value(), "replaced lease must unlock its original model");
    }
    require(other.active() == 0, "move-assigned lease must release its new model");
}

int main() {
    try {
        legacy_path(false);
        legacy_path(true);
        parallel_path(2);
        parallel_path(4);
        configuration_identity();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "model_execution_guard_test passed\n";
}
