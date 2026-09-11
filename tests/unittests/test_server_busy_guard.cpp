// Behavioral tests for the model busy guard.
//
// The central claim these tests pin down is the one the guard exists to make:
// with the guard disabled (timeout 0) the server behaves exactly as it did before
// -- callers queue behind a wedged inference forever and every worker thread is
// consumed -- and with a positive timeout those same callers fail fast instead.
// Rather than asserting on the code path, each test drives real threads against a
// real BusyGuard whose holder never releases, which is what a wedged CUDA call
// looks like from userspace.

#include "busy_guard.h"

#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono_literals;

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// A run that never finishes, standing in for an inference wedged inside the GPU
// driver. The holder thread parks until `release` is set, so the guard stays taken
// for as long as the test needs.
class WedgedRun {
public:
    explicit WedgedRun(minitts::server::BusyGuard & guard) {
        thread_ = std::thread([this, &guard] {
            auto lock = guard.acquire(0, "wedged");
            held_.store(true);
            while (!release_.load()) {
                std::this_thread::sleep_for(1ms);
            }
        });
        while (!held_.load()) {
            std::this_thread::sleep_for(1ms);
        }
    }

    // Let the wedged run finish. Tests must call this before their std::async futures
    // go out of scope: ~future blocks until its task completes, and locals destruct in
    // reverse declaration order, so an implicit cleanup would join the waiters while
    // this run still holds the guard -- deadlocking the test itself.
    void release() {
        if (thread_.joinable()) {
            release_.store(true);
            thread_.join();
        }
    }

    ~WedgedRun() { release(); }

private:
    std::thread thread_;
    std::atomic<bool> held_{false};
    std::atomic<bool> release_{false};
};

// Is this caller still stuck waiting after `patience`? Returns true when the future
// has not resolved -- i.e. the thread is parked on the lock, exactly the pile-up the
// guard is meant to prevent.
template <typename T>
bool still_blocked(std::future<T> & future, std::chrono::milliseconds patience) {
    return future.wait_for(patience) == std::future_status::timeout;
}

// timeout 0 must reproduce the original std::mutex behavior exactly: a caller
// arriving behind a wedged run waits, and keeps waiting, with no way out.
void test_disabled_guard_queues_forever() {
    minitts::server::BusyGuard guard;
    WedgedRun wedged(guard);

    std::atomic<bool> acquired{false};
    auto waiter = std::async(std::launch::async, [&] {
        auto lock = guard.acquire(0, "model");  // guard disabled
        acquired.store(true);
        return true;
    });

    require(
        still_blocked(waiter, 300ms),
        "with busy_timeout_ms=0 a caller must queue behind the wedged run (original behavior)");
    require(!acquired.load(), "the queued caller must not have entered the model");

    // The waiter is still parked at this point. Releasing the wedged run lets it
    // through so the test can finish -- in the real server nothing performs this
    // release, which is precisely the bug: the thread is gone until the process
    // restarts.
    wedged.release();
    require(waiter.get(), "the waiter proceeds only once the wedged run releases");
    require(acquired.load(), "the queued caller enters the model after the release");
}

// The system-stuck claim: with the guard disabled, every worker that arrives is
// consumed, so a bounded thread pool is fully exhausted and the server stops
// serving -- including requests for other, healthy models.
void test_disabled_guard_exhausts_all_workers() {
    minitts::server::BusyGuard guard;
    WedgedRun wedged(guard);

    constexpr int kWorkers = 8;
    std::atomic<int> entered{0};
    std::vector<std::future<void>> workers;
    workers.reserve(kWorkers);
    for (int i = 0; i < kWorkers; ++i) {
        workers.push_back(std::async(std::launch::async, [&] {
            auto lock = guard.acquire(0, "model");
            entered.fetch_add(1);
        }));
    }

    for (auto & worker : workers) {
        require(
            still_blocked(worker, 100ms),
            "every worker must be stuck behind the wedged run when the guard is disabled");
    }
    require(entered.load() == 0, "no worker may enter the model while it is wedged");

    wedged.release();  // drain the parked workers so the test can exit
    for (auto & worker : workers) {
        worker.get();
    }
    require(entered.load() == kWorkers, "the parked workers drain once the model frees up");
}

// With a positive timeout the same wedged run no longer captures callers: each one
// gives up and reports busy, so the worker returns to the pool.
void test_enabled_guard_fails_fast_instead_of_queuing() {
    minitts::server::BusyGuard guard;
    WedgedRun wedged(guard);

    constexpr int kWorkers = 8;
    std::atomic<int> busy_errors{0};
    std::vector<std::future<void>> workers;
    workers.reserve(kWorkers);
    for (int i = 0; i < kWorkers; ++i) {
        workers.push_back(std::async(std::launch::async, [&] {
            try {
                auto lock = guard.acquire(50, "model");
                require(false, "acquire must not succeed while the model is wedged");
            } catch (const minitts::server::ServerBusyError &) {
                busy_errors.fetch_add(1);
            }
        }));
    }

    // Record the verdict before asserting: if the guard regressed, the workers are
    // parked on the lock and their futures would block on destruction, hanging the
    // test instead of reporting. Releasing the wedged run first frees them either way.
    bool all_returned = true;
    for (auto & worker : workers) {
        if (still_blocked(worker, 2000ms)) {
            all_returned = false;
        }
    }
    wedged.release();
    for (auto & worker : workers) {
        worker.get();
    }

    require(all_returned, "with a positive timeout no worker may stay parked on the lock");
    require(
        busy_errors.load() == kWorkers,
        "every worker must surface ServerBusyError rather than hang");
}

// Once the holder has overrun, later arrivals skip the wait entirely -- they must
// not each burn another full timeout window before reporting busy.
void test_overrun_holder_fails_without_waiting() {
    minitts::server::BusyGuard guard;
    WedgedRun wedged(guard);

    // Let the holder run past a 50 ms bound.
    std::this_thread::sleep_for(120ms);

    const auto started = std::chrono::steady_clock::now();
    bool threw = false;
    try {
        auto lock = guard.acquire(50, "model");
    } catch (const minitts::server::ServerBusyError &) {
        threw = true;
    }
    const auto elapsed = std::chrono::steady_clock::now() - started;

    require(threw, "an already-overrun holder must be reported as busy");
    require(
        elapsed < 40ms,
        "the fail-fast path must not wait out another timeout window");
}

// The guard must not leave a stale timestamp behind: after a normal run completes,
// the next caller acquires immediately rather than inheriting the previous run's
// elapsed time and being wrongly rejected.
void test_guard_is_reusable_after_a_normal_run() {
    minitts::server::BusyGuard guard;
    {
        auto lock = guard.acquire(50, "model");
        std::this_thread::sleep_for(80ms);  // longer than the timeout, but legitimate
    }
    // A completed run -- even one that outlived the timeout -- must not poison the guard.
    auto lock = guard.acquire(50, "model");
}

// The same must hold when a run ends by throwing.
void test_guard_is_reusable_after_a_failed_run() {
    minitts::server::BusyGuard guard;
    try {
        auto lock = guard.acquire(50, "model");
        throw std::runtime_error("inference failed");
    } catch (const std::runtime_error &) {
    }
    auto lock = guard.acquire(50, "model");
}

// An idle model must be entered without delay regardless of the bound.
void test_idle_model_acquires_immediately() {
    minitts::server::BusyGuard guard;
    const auto started = std::chrono::steady_clock::now();
    auto lock = guard.acquire(5000, "model");
    require(
        std::chrono::steady_clock::now() - started < 100ms,
        "an idle model must not wait for the timeout");
}

}  // namespace

int main() {
    try {
        test_disabled_guard_queues_forever();
        test_disabled_guard_exhausts_all_workers();
        test_enabled_guard_fails_fast_instead_of_queuing();
        test_overrun_holder_fails_without_waiting();
        test_guard_is_reusable_after_a_normal_run();
        test_guard_is_reusable_after_a_failed_run();
        test_idle_model_acquires_immediately();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "server_busy_guard_test passed\n";
    return 0;
}
