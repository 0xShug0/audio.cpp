#pragma once

#include "engine/framework/core/backend.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <utility>

namespace engine::models::ace_step {

// Different CUDA ACE packages must not interleave uploads and GPU execution on the
// same device. Slots of one package may overlap using their existing graph
// guard. FIFO admission prevents a warm package starving another cold package.
class AceStepDeviceExecutionLease {
    friend struct AceStepDeviceExecutionLeaseTestAccess;
    struct Waiter { uint64_t ticket; const void * package; };
    struct State {
        std::mutex mutex;
        std::condition_variable changed;
        std::deque<Waiter> waiting;
        uint64_t next_ticket = 0;
        const void * package = nullptr;
        size_t active = 0;
    };

    static std::shared_ptr<State> state_for(core::BackendType backend, int device) {
        static std::mutex mutex;
        static std::map<std::pair<core::BackendType, int>, std::weak_ptr<State>> states;
        std::lock_guard<std::mutex> lock(mutex);
        auto & entry = states[{backend, device}];
        auto state = entry.lock();
        if (!state) { state = std::make_shared<State>(); entry = state; }
        return state;
    }

public:
    AceStepDeviceExecutionLease(core::BackendType backend, int device, const void * package) {
        if (backend != core::BackendType::Cuda) return;
        state_ = state_for(backend, device);
        std::unique_lock<std::mutex> lock(state_->mutex);
        const auto ticket = state_->next_ticket++;
        state_->waiting.push_back({ticket, package});
        try {
            state_->changed.wait(lock, [&] {
                return state_->waiting.front().ticket == ticket &&
                       (state_->active == 0 || state_->package == package);
            });
        } catch (...) {
            for (auto it = state_->waiting.begin(); it != state_->waiting.end(); ++it) {
                if (it->ticket == ticket) { state_->waiting.erase(it); break; }
            }
            state_->changed.notify_all();
            throw;
        }
        state_->waiting.pop_front();
        state_->package = package;
        ++state_->active;
        state_->changed.notify_all();
    }

    ~AceStepDeviceExecutionLease() {
        if (!state_) return;
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (--state_->active == 0) state_->package = nullptr;
        state_->changed.notify_all();
    }
    AceStepDeviceExecutionLease(const AceStepDeviceExecutionLease &) = delete;
    AceStepDeviceExecutionLease & operator=(const AceStepDeviceExecutionLease &) = delete;

private:
    std::shared_ptr<State> state_;
};

} // namespace engine::models::ace_step
