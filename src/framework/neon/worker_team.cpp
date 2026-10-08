#include "worker_team.h"

#include <stdexcept>

namespace engine::neon::detail {

Barrier::Barrier(int count) : count_(count), left_(count) {
    if (count < 1) {
        throw std::invalid_argument("ARM worker count must be positive");
    }
}

void Barrier::wait() {
    const int generation = generation_.load(std::memory_order_acquire);
    if (left_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        left_.store(count_, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
        return;
    }
    for (int spin = 0; generation_.load(std::memory_order_acquire) == generation; ++spin) {
#if defined(__aarch64__)
        if (spin < 4096) {
            __asm__ __volatile__("yield");
        } else {
            std::this_thread::yield();
        }
#else
        std::this_thread::yield();
#endif
    }
}

WorkerTeam::WorkerTeam(int count) : count_(count), barrier_(count) {
    workers_.reserve(count - 1);
    try {
        for (int index = 1; index < count; ++index) {
            workers_.emplace_back([this, index] {
                size_t seen = 0;
                std::unique_lock<std::mutex> lock(mutex_);
                for (;;) {
                    ready_.wait(lock, [&] { return stop_ || generation_ != seen; });
                    if (stop_) {
                        return;
                    }
                    seen = generation_;
                    const auto *work = work_;
                    lock.unlock();
                    WorkerContext context{index, count_, &barrier_};
                    (*work)(context);
                    lock.lock();
                    if (--pending_ == 0) {
                        finished_.notify_one();
                    }
                }
            });
        }
    } catch (...) {
        stop();
        throw;
    }
}

WorkerTeam::~WorkerTeam() {
    stop();
}

void WorkerTeam::run(const std::function<void(WorkerContext &)> &work) {
    WorkerContext context{0, count_, &barrier_};
    if (count_ == 1) {
        work(context);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        work_ = &work;
        pending_ = count_ - 1;
        ++generation_;
    }
    ready_.notify_all();
    work(context);
    std::unique_lock<std::mutex> lock(mutex_);
    finished_.wait(lock, [&] { return pending_ == 0; });
    work_ = nullptr;
}

void WorkerTeam::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    ready_.notify_all();
    for (auto &worker : workers_) {
        worker.join();
    }
}

} // namespace engine::neon::detail
