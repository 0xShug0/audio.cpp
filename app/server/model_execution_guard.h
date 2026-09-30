#pragma once

#include "model_slots.h"
#include <variant>

namespace minitts::server {

// One slot keeps main's timed-mutex admission for inference and management.
// Only an explicit count above one selects the parallel scheduler.
class ModelExecutionGuard {
public:
    class Lock {
    public:
        Lock(const Lock &) = delete;
        Lock & operator=(const Lock &) = delete;
        Lock(Lock && other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), request_(other.request_), lock_(std::move(other.lock_)) {}
        Lock & operator=(Lock && other) noexcept {
            if (this != &other) {
                release();
                owner_ = std::exchange(other.owner_, nullptr);
                request_ = other.request_;
                lock_ = std::move(other.lock_);
            }
            return *this;
        }
        ~Lock() { release(); }
        size_t slot() const {
            if (const auto * parallel = std::get_if<ModelSlots::Lock>(&lock_)) { return parallel->slot(); }
            if (!owner_ || !request_) { throw std::logic_error("management lease has no request slot"); }
            return 0;
        }
    private:
        friend class ModelExecutionGuard;
        Lock(ModelExecutionGuard & owner, BusyGuard::Lock lock, bool request)
            : owner_(&owner), request_(request), lock_(std::move(lock)) {}
        explicit Lock(ModelSlots::Lock lock) : lock_(std::move(lock)) {}
        void release() {
            if (owner_ && request_) { owner_->legacy_active_.store(0); }
            owner_ = nullptr;
            lock_.emplace<std::monostate>();
        }
        ModelExecutionGuard * owner_ = nullptr;
        bool request_ = false;
        std::variant<std::monostate, BusyGuard::Lock, ModelSlots::Lock> lock_;
    };

    void configure(int count) {
        if (count < 1 || count > static_cast<int>(engine::runtime::kMaxParallelSessions)) {
            throw std::runtime_error("slots must be between 1 and 16");
        }
        // Guard identity is fixed at registration. Swapping guards under queued
        // callers would let leases from two schedulers execute simultaneously.
        if (configured_) {
            if (parallel_ != (count > 1)) {
                throw std::runtime_error("switching between one and multiple slots requires a new model ID or server restart");
            }
            if (parallel_) { slots_.configure(count); }
            return;
        }
        if (count > 1) { slots_.configure(count); }
        parallel_ = count > 1;
        configured_ = true;
    }
    bool parallel() const noexcept { return parallel_; }
    Lock acquire_run(int timeout_ms, std::string_view label) {
        if (parallel_) { return Lock(slots_.acquire_run(timeout_ms, label)); }
        return acquire_legacy(timeout_ms, label, true);
    }
    Lock acquire(int timeout_ms, std::string_view label) {
        if (parallel_) { return Lock(slots_.acquire(timeout_ms, label)); }
        return acquire_legacy(timeout_ms, label, false);
    }
    std::optional<Lock> try_acquire() {
        if (parallel_) {
            auto lock = slots_.try_acquire();
            if (!lock) { return std::nullopt; }
            return Lock(std::move(*lock));
        }
        auto lock = legacy_.try_acquire();
        if (!lock) { return std::nullopt; }
        return Lock(*this, std::move(*lock), false);
    }
    ModelSlots::State state() const {
        if (parallel_) { return slots_.state(); }
        return {1, legacy_active_.load(), legacy_waiting_requests_.load(), legacy_waiting_management_.load()};
    }
    int active() const { return state().active; }
private:
    Lock acquire_legacy(int timeout_ms, std::string_view label, bool request) {
        auto & waiting = request ? legacy_waiting_requests_ : legacy_waiting_management_;
        waiting.fetch_add(1);
        try {
            auto lock = legacy_.acquire(timeout_ms, label);
            waiting.fetch_sub(1);
            if (request) { legacy_active_.store(1); }
            return Lock(*this, std::move(lock), request);
        } catch (...) {
            waiting.fetch_sub(1);
            throw;
        }
    }
    BusyGuard legacy_;
    ModelSlots slots_;
    bool configured_ = false;
    bool parallel_ = false;
    std::atomic<int> legacy_active_{0};
    std::atomic<int> legacy_waiting_requests_{0};
    std::atomic<int> legacy_waiting_management_{0};
};
} // namespace minitts::server
