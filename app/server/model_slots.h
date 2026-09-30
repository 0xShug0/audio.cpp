#pragma once

#include "busy_guard.h"
#include "engine/framework/runtime/parallel_session.h"
#include <algorithm>
#include <array>
#include <condition_variable>
#include <deque>
#include <memory>
#include <utility>

namespace minitts::server {

// Requests and management share one admission queue. A manager is a barrier
// for callers behind it, never for older requests. Each waiter owns its wakeup.
class ModelSlots {
    struct Waiter {
        bool management;
        bool granted = false;
        bool expired = false;
        int slot = -1;
        std::optional<std::chrono::steady_clock::time_point> deadline;
        std::condition_variable cv;
    };
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
            owner_->release_slot(slot_);
            owner_ = nullptr;
        }
        ModelSlots * owner_;
        int slot_;
    };

    // Reserve a management barrier now and drain unrelated ready models before
    // waiting for this one. Destruction cancels or releases the reservation.
    class Pending {
    public:
        Pending(const Pending &) = delete;
        Pending & operator=(const Pending &) = delete;
        Pending(Pending && other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), waiter_(std::move(other.waiter_)), label_(std::move(other.label_)) {}
        Pending & operator=(Pending && other) noexcept {
            if (this != &other) {
                cancel(); owner_ = std::exchange(other.owner_, nullptr);
                waiter_ = std::move(other.waiter_); label_ = std::move(other.label_);
            }
            return *this;
        }
        ~Pending() { cancel(); }
        std::optional<Lock> try_acquire() {
            if (!owner_) { throw std::logic_error("admission already consumed"); }
            std::lock_guard<std::mutex> lock(owner_->mutex_);
            if (!waiter_->granted) { return std::nullopt; }
            return take();
        }
        Lock acquire() {
            if (!owner_) { throw std::logic_error("admission already consumed"); }
            std::unique_lock<std::mutex> lock(owner_->mutex_);
            const auto ready = [&] { return waiter_->granted || waiter_->expired; };
            if (waiter_->deadline) { waiter_->cv.wait_until(lock, *waiter_->deadline, ready); }
            else { waiter_->cv.wait(lock, ready); }
            if (!waiter_->granted) {
                owner_->cancel_waiter(waiter_);
                owner_ = nullptr;
                throw ServerBusyError("model '" + label_ + "': timed out waiting for a slot");
            }
            return take();
        }
    private:
        friend class ModelSlots;
        Pending(std::shared_ptr<Waiter> waiter, std::string_view label)
            : waiter_(std::move(waiter)), label_(label) {}
        Lock take() {
            auto * owner = std::exchange(owner_, nullptr);
            return Lock(*owner, waiter_->slot);
        }
        void cancel() {
            if (!owner_) { return; }
            std::lock_guard<std::mutex> lock(owner_->mutex_);
            owner_->cancel_waiter(waiter_);
            owner_ = nullptr;
        }
        ModelSlots * owner_ = nullptr;
        std::shared_ptr<Waiter> waiter_;
        std::string label_;
    };

    struct State { int slots; int active; int waiting_requests; int waiting_management; };
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
        const int slot = free_slot();
        if (!exclusive_ && queue_.empty() && slot >= 0) {
            since_[slot] = steady_now_ms();
            return Lock(*this, slot);
        }
        const auto now = steady_now_ms();
        if (model_run_has_overrun(exclusive_since_, now, timeout_ms)) {
            throw ServerBusyError("model '" + std::string(label) + "': exclusive operation exceeded busy_timeout_ms");
        }
        bool any_active = false, all_overrun = !exclusive_;
        for (int i = 0; i < count_; ++i) {
            if (!since_[i]) { continue; }
            any_active = true;
            all_overrun = all_overrun && model_run_has_overrun(since_[i], now, timeout_ms);
        }
        if (any_active && all_overrun) { throw ServerBusyError("model '" + std::string(label) + "': all active slots exceeded busy_timeout_ms"); }
        auto pending = enqueue(false, timeout_ms, label);
        lock.unlock();
        return pending.acquire();
    }
    Pending queue_management(int timeout_ms, std::string_view label) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto now = steady_now_ms();
        if (model_run_has_overrun(exclusive_since_, now, timeout_ms)) {
            throw ServerBusyError("model '" + std::string(label) + "': exclusive operation exceeded busy_timeout_ms");
        }
        for (auto since : since_) {
            if (model_run_has_overrun(since, now, timeout_ms)) {
                throw ServerBusyError("model '" + std::string(label) + "': active slot exceeded busy_timeout_ms");
            }
        }
        return enqueue(true, timeout_ms, label);
    }
    Lock acquire(int timeout_ms, std::string_view label) { return queue_management(timeout_ms, label).acquire(); }
    std::optional<Lock> try_acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!idle() || !queue_.empty()) { return std::nullopt; }
        exclusive_ = true;
        exclusive_since_ = steady_now_ms();
        return Lock(*this, -1);
    }
private:
    int free_slot() const {
        for (int i = 0; i < count_; ++i) { if (!since_[i]) { return i; } }
        return -1;
    }
    bool idle() const {
        if (exclusive_) { return false; }
        for (auto since : since_) { if (since) { return false; } }
        return true;
    }
    Pending enqueue(bool management, int timeout_ms, std::string_view label) {
        auto waiter = std::make_shared<Waiter>();
        waiter->management = management;
        if (timeout_ms > 0) { waiter->deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms); }
        // Construct labels before queue publication, but only arm cancellation
        // after push_back succeeds: its failure must not lock this mutex again.
        Pending pending(waiter, label);
        queue_.push_back(waiter);
        pending.owner_ = this;
        if (management) { ++waiting_exclusive_; } else { ++waiting_requests_; }
        dispatch();
        return pending;
    }
    void remove_front() {
        if (queue_.front()->management) { --waiting_exclusive_; } else { --waiting_requests_; }
        queue_.pop_front();
    }
    void dispatch() {
        while (!queue_.empty()) {
            auto waiter = queue_.front();
            if (waiter->deadline && std::chrono::steady_clock::now() >= *waiter->deadline) {
                waiter->expired = true; remove_front(); waiter->cv.notify_one(); continue;
            }
            if (exclusive_) { return; }
            if (waiter->management) {
                if (!idle()) { return; }
                exclusive_ = true; exclusive_since_ = steady_now_ms();
            } else {
                waiter->slot = free_slot();
                if (waiter->slot < 0) { return; }
                since_[waiter->slot] = steady_now_ms();
            }
            waiter->granted = true;
            remove_front(); waiter->cv.notify_one();
        }
    }
    void release_slot(int slot) {
        if (slot < 0) { exclusive_ = false; exclusive_since_ = 0; }
        else { since_[slot] = 0; }
        dispatch();
    }
    void cancel_waiter(const std::shared_ptr<Waiter> & waiter) {
        if (waiter->granted) { release_slot(waiter->slot); return; }
        const auto it = std::find(queue_.begin(), queue_.end(), waiter);
        if (it != queue_.end()) {
            if (waiter->management) { --waiting_exclusive_; } else { --waiting_requests_; }
            queue_.erase(it);
        }
        dispatch();
    }
    mutable std::mutex mutex_;
    std::deque<std::shared_ptr<Waiter>> queue_;
    std::array<int64_t, engine::runtime::kMaxParallelSessions> since_{};
    int count_ = 1;
    int waiting_requests_ = 0;
    int waiting_exclusive_ = 0;
    bool exclusive_ = false;
    int64_t exclusive_since_ = 0;
};
} // namespace minitts::server
