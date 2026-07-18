#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace minitts::server {

// Thrown when a model is occupied and the effective busy timeout has elapsed.
// Mapped to HTTP 503 so a client can retry or fail over rather than hang.
class ServerBusyError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

inline std::int64_t steady_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// True when a request should fail fast instead of queuing behind the model lock:
// the current holder acquired it at busy_since_ms and has already run longer than
// timeout_ms. busy_since_ms == 0 means the model is idle; timeout_ms <= 0 disables
// the guard. Both times are steady-clock milliseconds.
inline bool model_run_has_overrun(std::int64_t busy_since_ms, std::int64_t now_ms, int timeout_ms) {
    if (timeout_ms <= 0 || busy_since_ms == 0) {
        return false;
    }
    return (now_ms - busy_since_ms) > timeout_ms;
}

// Resolve the timeout for a single run. `ceiling` is server policy (the top-level
// config value, overridden per model); `requested` is the optional per-request
// value. A request may ask for a shorter bound than policy but never a longer one,
// so a client cannot weaken the guard. 0 means "unbounded" and therefore compares
// as +infinity on both sides: a request asking for unbounded is still capped by
// policy, and under an unbounded policy a request's own bound is honored.
inline int resolve_busy_timeout_ms(int ceiling, std::optional<int> requested) {
    if (!requested.has_value()) {
        return ceiling;
    }
    const int value = *requested;
    if (value <= 0) {
        return ceiling;
    }
    if (ceiling <= 0) {
        return value;
    }
    return value < ceiling ? value : ceiling;
}

// Serializes runs on one model and bounds how long a caller waits for its turn.
//
// A single model runs one request at a time. If a running inference wedges the GPU
// -- a CUDA call that never returns cannot be cancelled from userspace -- later
// requests would otherwise block forever and every worker thread would pile up
// behind it. With a positive timeout, a caller instead gives up and throws
// ServerBusyError (-> HTTP 503). A timeout of 0 restores the original unbounded
// std::mutex behavior.
class BusyGuard {
public:
    // Holds the guard for the duration of a run and marks it idle again when it
    // releases (including on exception), so the fail-fast check never sees a stale
    // timestamp. Move-only.
    class Lock {
    public:
        Lock() = default;
        Lock(BusyGuard & guard, std::unique_lock<std::timed_mutex> lock)
            : guard_(&guard), lock_(std::move(lock)) {}
        Lock(Lock &&) = default;
        Lock & operator=(Lock &&) = default;
        Lock(const Lock &) = delete;
        Lock & operator=(const Lock &) = delete;
        ~Lock() {
            if (guard_ != nullptr && lock_.owns_lock()) {
                guard_->busy_since_ms_.store(0, std::memory_order_release);
            }
        }

    private:
        BusyGuard * guard_ = nullptr;
        std::unique_lock<std::timed_mutex> lock_;
    };

    // `label` identifies the model in the error message. Throws ServerBusyError when
    // timeout_ms is positive and either the current holder has already overrun (fail
    // fast, no wait) or the wait itself times out.
    Lock acquire(int timeout_ms, std::string_view label) {
        std::unique_lock<std::timed_mutex> lock(mutex_, std::defer_lock);
        if (timeout_ms <= 0) {
            lock.lock();  // guard disabled: original unbounded wait
        } else {
            // If the current holder has already run past the timeout, treat it as
            // wedged and fail fast rather than queue behind it -- this is what stops
            // requests piling up on a stuck GPU.
            const auto since = busy_since_ms_.load(std::memory_order_acquire);
            const auto now_ms = steady_now_ms();
            if (model_run_has_overrun(since, now_ms, timeout_ms)) {
                throw ServerBusyError(
                    "model '" + std::string(label) + "' is busy: current inference has run " +
                    std::to_string(now_ms - since) + " ms (busy_timeout_ms=" +
                    std::to_string(timeout_ms) +
                    "); the previous request has likely wedged and cannot be cancelled");
            }
            if (!lock.try_lock_for(std::chrono::milliseconds(timeout_ms))) {
                throw ServerBusyError(
                    "model '" + std::string(label) + "' is busy: timed out after " +
                    std::to_string(timeout_ms) + " ms waiting for the inference lock");
            }
        }
        busy_since_ms_.store(steady_now_ms(), std::memory_order_release);
        return Lock(*this, std::move(lock));
    }

private:
    std::timed_mutex mutex_;
    std::atomic<std::int64_t> busy_since_ms_{0};
};

}  // namespace minitts::server
