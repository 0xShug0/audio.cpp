#pragma once

#include "busy_guard.h"
#include "engine/framework/runtime/parallel_session.h"
#include <array>
#include <condition_variable>
#include <utility>

namespace minitts::server {

// Requests lease one session; management/eviction lease the entire model.
// Waiting management operations block new requests so unload cannot starve.
class ModelSlots {
public:
    class Lock {
    public:
        Lock(ModelSlots & owner, int slot) : owner_(&owner), slot_(slot) {}
        Lock(const Lock &) = delete;
        Lock & operator=(const Lock &) = delete;
        Lock(Lock && other) noexcept : owner_(std::exchange(other.owner_, nullptr)), slot_(other.slot_) {}
        Lock & operator=(Lock && other) noexcept {
            if (this != &other) { release(); owner_ = std::exchange(other.owner_, nullptr); slot_ = other.slot_; }
            return *this;
        }
        ~Lock() { release(); }
        size_t slot() const {
            if (!owner_ || slot_ < 0) { throw std::logic_error("management lease has no request slot"); }
            return static_cast<size_t>(slot_);
        }
    private:
        void release() {
            if (!owner_) { return; }
            std::lock_guard<std::mutex> lock(owner_->mutex_);
            if (slot_ < 0) { owner_->exclusive_ = false; owner_->exclusive_since_ = 0; }
            else { owner_->since_[slot_] = 0; }
            owner_->cv_.notify_all();
            owner_ = nullptr;
        }
        ModelSlots * owner_;
        int slot_;
    };

    struct State {
        int slots;
        int active;
        int waiting_requests;
        int waiting_management;
    };
    State state() const {
        std::lock_guard<std::mutex> lock(mutex_);
        int count = 0;
        for (auto since : since_) { if (since) { ++count; } }
        return {count_, count, waiting_requests_, waiting_exclusive_};
    }
    int active() const { return state().active; }
    // Called only before publication or under an exclusive lease.
    void configure(int count) {
        if (count < 1 || count > static_cast<int>(engine::runtime::kMaxParallelSessions)) {
            throw std::runtime_error("slots must be between 1 and 16");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto since : since_) {
            if (since) { throw std::logic_error("cannot resize occupied slots"); }
        }
        count_ = count;
    }
    Lock acquire_run(int timeout_ms, std::string_view label) {
        std::unique_lock<std::mutex> lock(mutex_);
        int slot = -1;
        auto ready = [&] {
            if (exclusive_ || waiting_exclusive_) { return false; }
            for (int i = 0; i < count_; ++i) {
                if (since_[i] == 0) { slot = i; return true; }
            }
            return false;
        };
        if (!ready()) {
            if (model_run_has_overrun(exclusive_since_, steady_now_ms(), timeout_ms)) {
                throw ServerBusyError("model '" + std::string(label) + "': exclusive operation exceeded busy_timeout_ms");
            }
            bool all_overrun = !exclusive_ && !waiting_exclusive_;
            for (int i = 0; i < count_; ++i) {
                all_overrun = all_overrun && model_run_has_overrun(since_[i], steady_now_ms(), timeout_ms);
            }
            if (all_overrun) { throw ServerBusyError("model '" + std::string(label) + "': all slots exceeded busy_timeout_ms"); }
            ++waiting_requests_;
            try { wait(lock, timeout_ms, label, ready); }
            catch (...) { --waiting_requests_; throw; }
            --waiting_requests_;
        }
        since_[slot] = steady_now_ms();
        return Lock(*this, slot);
    }
    Lock acquire(int timeout_ms, std::string_view label) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (model_run_has_overrun(exclusive_since_, steady_now_ms(), timeout_ms)) {
            throw ServerBusyError("model '" + std::string(label) + "': exclusive operation exceeded busy_timeout_ms");
        }
        for (auto since : since_) {
            if (model_run_has_overrun(since, steady_now_ms(), timeout_ms)) {
                throw ServerBusyError("model '" + std::string(label) + "': active slot exceeded busy_timeout_ms");
            }
        }
        ++waiting_exclusive_;
        try { wait(lock, timeout_ms, label, [&] { return idle(); }); }
        catch (...) {
            // Requests may be waiting only because management had priority,
            // while some slots are already free. Wake them when it gives up.
            --waiting_exclusive_;
            cv_.notify_all();
            throw;
        }
        --waiting_exclusive_;
        exclusive_ = true;
        exclusive_since_ = steady_now_ms();
        return Lock(*this, -1);
    }
    std::optional<Lock> try_acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!idle() || waiting_exclusive_) { return std::nullopt; }
        exclusive_ = true;
        exclusive_since_ = steady_now_ms();
        return Lock(*this, -1);
    }
private:
    bool idle() const {
        if (exclusive_) { return false; }
        for (auto since : since_) { if (since) { return false; } }
        return true;
    }
    template<class Predicate>
    void wait(std::unique_lock<std::mutex> & lock, int timeout_ms, std::string_view label, Predicate ready) {
        if (timeout_ms <= 0) { cv_.wait(lock, ready); }
        else if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), ready)) {
            throw ServerBusyError("model '" + std::string(label) + "': timed out waiting for a slot");
        }
    }
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::array<int64_t, engine::runtime::kMaxParallelSessions> since_{};
    int count_ = 1;
    int waiting_requests_ = 0;
    int waiting_exclusive_ = 0;
    bool exclusive_ = false;
    int64_t exclusive_since_ = 0;
};
} // namespace minitts::server
