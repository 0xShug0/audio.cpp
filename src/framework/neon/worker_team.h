#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace engine::neon::detail {

class Barrier {
public:
    explicit Barrier(int count);
    void wait();

private:
    const int count_;
    std::atomic<int> left_;
    std::atomic<int> generation_{0};
};

struct WorkerContext {
    int index;
    int count;
    Barrier *barrier;
    void synchronize() const {
        if (barrier) {
            barrier->wait();
        }
    }
};

class WorkerTeam {
public:
    explicit WorkerTeam(int count);
    ~WorkerTeam();
    int size() const noexcept {
        return count_;
    }
    void run(const std::function<void(WorkerContext &)> &work);

private:
    void stop();
    int count_;
    int pending_ = 0;
    size_t generation_ = 0;
    bool stop_ = false;
    Barrier barrier_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::condition_variable finished_;
    const std::function<void(WorkerContext &)> *work_ = nullptr;
    std::vector<std::thread> workers_;
};

} // namespace engine::neon::detail
