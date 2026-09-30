#include "model_slots.h"
#include <atomic>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace std::chrono_literals;
using minitts::server::ModelSlots;
using minitts::server::ServerBusyError;
void require(bool value, const char * message) { if (!value) { throw std::runtime_error(message); } }

template<class Predicate>
bool await_state(Predicate ready) {
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (!ready()) {
        if (std::chrono::steady_clock::now() >= deadline) { return false; }
        std::this_thread::yield();
    }
    return true;
}

// A queued unload must not hide an already overdue inference. Include spare
// slots: an older management barrier blocks later work until active work drains.
void test_overdue_with_waiting_management(int count, int occupied, int managers = 1) {
    ModelSlots slots;
    slots.configure(count);
    std::vector<ModelSlots::Lock> held;
    for (int i = 0; i < occupied; ++i) { held.push_back(slots.acquire_run(0, "overdue")); }
    std::vector<std::future<void>> management;
    for (int i = 0; i < managers; ++i) {
        management.push_back(std::async(std::launch::async, [&] { auto exclusive = slots.acquire(0, "unload"); }));
    }
    const bool queued = await_state([&] { return slots.state().waiting_management == managers; });
    std::this_thread::sleep_for(30ms);
    const auto started = std::chrono::steady_clock::now();
    std::string error;
    try { auto lease = slots.acquire_run(15, "overdue"); }
    catch (const ServerBusyError & e) { error = e.what(); }
    const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    const auto state = slots.state();
    // Drain every thread before asserting, including on the regression path.
    held.clear();
    for (auto & future : management) { future.get(); }
    require(queued, "management did not queue behind inference");
    require(error.find("slots exceeded busy_timeout_ms") != std::string::npos,
            "waiting management suppressed overdue rejection");
    require(state.active == occupied && state.waiting_requests == 0 && state.waiting_management == managers,
            "overdue rejection changed lease ownership or leaked a waiter");
    require(slots.try_acquire().has_value(), "overdue rejection stranded management");
    std::cout << "PASS overdue with management: slots=" << count << " occupied=" << occupied
              << " managers=" << managers << " rejection_ms=" << elapsed << '\n';
}

// One overdue request must not cause an early rejection while another blocker
// is healthy. Timeout zero must also keep waiting, even if all work is overdue.
void test_management_priority(bool disable_timeout) {
    ModelSlots slots;
    slots.configure(2);
    std::vector<ModelSlots::Lock> held;
    held.push_back(slots.acquire_run(0, "old"));
    std::this_thread::sleep_for(225ms);
    if (!disable_timeout) { held.push_back(slots.acquire_run(0, "healthy")); }
    auto management = std::async(std::launch::async, [&] { return slots.acquire(0, "unload"); });
    const bool manager_queued = await_state([&] { return slots.state().waiting_management == 1; });
    auto request = std::async(std::launch::async, [&] {
        try { auto lease = slots.acquire_run(disable_timeout ? 0 : 200, "waiting"); return true; }
        catch (const ServerBusyError &) { return false; }
    });
    const bool request_queued = await_state([&] {
        return slots.state().waiting_requests == 1 || request.wait_for(0ms) == std::future_status::ready;
    }) && slots.state().waiting_requests == 1;
    held.clear();
    bool kept_priority;
    {
        auto exclusive = management.get();
        kept_priority = request.wait_for(20ms) == std::future_status::timeout && slots.state().active == 0;
    }
    const bool completed = request.get();
    require(manager_queued && request_queued, "healthy or unbounded request rejected instead of queuing");
    require(kept_priority, "new request overtook a waiting management operation");
    require(completed && slots.state().waiting_requests == 0, "management release failed to drain request queue");
}

// Admission order is observable under a held slot, independent of wake timing.
// Older inference must drain before management; later work cannot barge ahead.
void test_fifo(int count) {
    ModelSlots slots; slots.configure(count);
    std::vector<ModelSlots::Lock> held;
    for (int i = 0; i < count; ++i) { held.push_back(slots.acquire_run(0, "held")); }
    std::mutex order_mutex; std::vector<int> order;
    std::vector<std::thread> work;
    auto submit = [&](int id, bool management) {
        work.emplace_back([&, id, management] {
            auto lease = management ? slots.acquire(0, "manager") : slots.acquire_run(0, "request");
            std::lock_guard<std::mutex> lock(order_mutex); order.push_back(id);
        });
    };
    submit(1, false);
    bool observed = await_state([&] { return slots.state().waiting_requests == 1; });
    submit(2, true);
    observed = observed && await_state([&] { return slots.state().waiting_management == 1; });
    for (int i = 0; i < 16; ++i) { submit(3 + i, false); }
    observed = observed && await_state([&] { return slots.state().waiting_requests == 17; });
    if (!observed) { const auto s = slots.state(); std::cerr << "FIFO setup slots=" << count << " active=" << s.active << " requests=" << s.waiting_requests << " management=" << s.waiting_management << "\n"; }
    held.clear();
    for (auto & future : work) { future.join(); }
    require(observed, "failed to establish staggered FIFO arrivals");
    require(order.size() == 18 && order[0] == 1 && order[1] == 2,
            "management or a later request overtook the oldest request");
    const auto state = slots.state();
    require(state.active == 0 && state.waiting_requests == 0 && state.waiting_management == 0,
            "FIFO admission leaked ownership/accounting");
}

void test_management_reservations() {
    ModelSlots slots; slots.configure(2);
    std::optional<ModelSlots::Lock> held(slots.acquire_run(0, "held"));
    {
        auto barrier = slots.queue_management(0, "bulk");
        require(!barrier.try_acquire(), "bulk barrier ignored running inference");
        auto later = std::async(std::launch::async, [&] { return slots.acquire_run(0, "later"); });
        const bool queued = await_state([&] { return slots.state().waiting_requests == 1; });
        // Destroying a pending reservation cancels it and wakes usable slots.
        barrier = slots.queue_management(0, "replacement");
        const bool completed = later.wait_for(500ms) == std::future_status::ready;
        held.reset();
        { auto lease = later.get(); }
        require(queued && completed, "cancelled management stranded a runnable request");
    }
    require(slots.try_acquire().has_value(), "reserved management leaked an exclusive lease");
    {
        auto ready = slots.queue_management(0, "ready");
        auto lease = ready.try_acquire();
        require(lease.has_value() && !slots.try_acquire(), "ready bulk reservation did not own management");
    }
    require(slots.try_acquire().has_value(), "consumed management reservation did not release");
    {
        auto held_exclusive = slots.acquire(0, "holder");
        auto expired = slots.queue_management(10, "expiring");
        auto next = slots.queue_management(0, "next");
        std::this_thread::sleep_for(20ms);
        // Expiry is also enforced by dispatch, before assigning another lease.
    }
    const auto state = slots.state();
    require(state.active == 0 && state.waiting_requests == 0 && state.waiting_management == 0,
            "expired/cancelled bulk barriers leaked queue entries");
}

// No manager is needed to enforce fairness. Only one slot becomes free: its
// oldest waiter must get it before any of the sixteen staggered later callers.
void test_oldest_request(int count) {
    ModelSlots slots; slots.configure(count);
    std::vector<std::optional<ModelSlots::Lock>> held;
    for (int i = 0; i < count; ++i) { held.emplace_back(slots.acquire_run(0, "held")); }
    std::packaged_task<ModelSlots::Lock()> oldest([&] { return slots.acquire_run(2000, "oldest"); });
    auto result = oldest.get_future();
    std::thread first(std::move(oldest));
    bool established = await_state([&] { return slots.state().waiting_requests == 1; });
    std::atomic<int> admitted{0};
    std::vector<std::thread> later;
    for (int i = 0; i < 16; ++i) {
        later.emplace_back([&] { auto lease = slots.acquire_run(0, "later"); ++admitted; });
        established = established && await_state([&] { return slots.state().waiting_requests == i + 2; });
    }
    const auto released = held[0]->slot(); held[0].reset();
    const bool ready = result.wait_for(500ms) == std::future_status::ready;
    first.join();
    std::optional<ModelSlots::Lock> oldest_lease;
    try { oldest_lease.emplace(result.get()); } catch (const ServerBusyError &) {}
    const bool ordered = oldest_lease && oldest_lease->slot() == released && admitted == 0;
    held.clear(); oldest_lease.reset();
    for (auto & thread : later) { thread.join(); }
    require(established && ready && ordered, "later requests delayed the oldest waiter");
    require(admitted == 16 && slots.state().waiting_requests == 0, "fairness test stranded queued work");
}

int main() {
    try {
        test_fifo(1);
        test_fifo(2);
        test_fifo(4);
        test_oldest_request(1);
        test_oldest_request(2);
        test_oldest_request(4);
        test_management_reservations();
        test_overdue_with_waiting_management(1, 1);
        test_overdue_with_waiting_management(2, 2);
        test_overdue_with_waiting_management(4, 4);
        test_overdue_with_waiting_management(2, 1);
        test_overdue_with_waiting_management(4, 2);
        test_overdue_with_waiting_management(2, 2, 2);
        test_management_priority(false);
        test_management_priority(true);
        ModelSlots slots;
        slots.configure(2);
        std::optional<ModelSlots::Lock> a(slots.acquire_run(100, "test"));
        std::optional<ModelSlots::Lock> b(slots.acquire_run(100, "test"));
        require(a->slot() != b->slot(), "simultaneous requests shared a session");
        require(!slots.try_acquire(), "evicted a running model");
        bool timed_out = false;
        try { auto c = slots.acquire_run(20, "test"); }
        catch (const ServerBusyError &) { timed_out = true; }
        require(timed_out, "exhausted slots did not time out");
        const auto released_slot = a->slot();
        a.reset();
        { auto c = slots.acquire_run(100, "test"); require(c.slot() == released_slot, "free slot not reused"); }
        auto admin = std::async(std::launch::async, [&] { return slots.acquire(1000, "test"); });
        require(admin.wait_for(20ms) == std::future_status::timeout, "unload did not wait for last request");
        b.reset();
        { auto exclusive = admin.get(); require(!slots.try_acquire(), "two management leases");
          auto request = std::async(std::launch::async, [&] {
              try { auto c = slots.acquire_run(20, "test"); return false; }
              catch (const ServerBusyError &) { return true; }
          });
          require(request.get(), "request ran during unload"); }
        // Concurrent queue draining: no slot may ever be shared by two callers.
        std::array<std::atomic<int>, 2> active{};
        std::atomic<int> completed{0}, errors{0};
        std::vector<std::thread> threads;
        for (int i = 0; i < 12; ++i) {
            threads.emplace_back([&] {
                for (int j = 0; j < 20; ++j) {
                    auto lease = slots.acquire_run(0, "test");
                    const auto index = lease.slot();
                    if (active[index].fetch_add(1) != 0) { ++errors; }
                    std::this_thread::yield();
                    --active[index]; ++completed;
                }
            });
        }
        for (auto & t : threads) { t.join(); }
        require(completed == 240 && errors == 0, "queued requests collided or were lost");
        try { auto lease = slots.acquire_run(0, "test"); throw std::runtime_error("inference failed"); }
        catch (const std::runtime_error &) {}
        require(slots.try_acquire().has_value(), "exception leaked a slot");
        // A waiting manager has priority over new work, but a timed-out manager
        // must wake requests that can use a free slot even if another stays busy.
        {
            std::optional<ModelSlots::Lock> held(slots.acquire_run(0, "test"));
            bool rejected_resize = false;
            try { slots.configure(1); } catch (const std::logic_error &) { rejected_resize = true; }
            require(rejected_resize, "resized occupied slots");
            auto management = std::async(std::launch::async, [&] {
                try { auto lock = slots.acquire(150, "test"); return false; }
                catch (const ServerBusyError &) { return true; }
            });
            const auto deadline = std::chrono::steady_clock::now() + 1s;
            while (!slots.state().waiting_management && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::yield();
            }
            auto waiter = std::async(std::launch::async, [&] { auto lock = slots.acquire_run(500, "test"); return lock.slot(); });
            const bool timed_out = management.get();
            const bool woke = waiter.wait_for(200ms) == std::future_status::ready;
            held.reset(); // Always let threads drain, including on test failure.
            waiter.get();
            require(timed_out && woke, "management timeout stranded a runnable request");
            require(slots.state().waiting_requests == 0, "queued request count leaked");
        }
        // A stuck slot must not prevent the remaining slots from serving work.
        {
            auto held = slots.acquire_run(0, "test");
            std::this_thread::sleep_for(25ms);
            auto second = slots.acquire_run(10, "test");
            require(held.slot() != second.slot(), "stale occupied slot was reused");
        }
        { auto exclusive = slots.acquire(0, "test"); slots.configure(1); }
        { auto lease = slots.acquire_run(0, "test");
          try { auto second = slots.acquire_run(20, "test"); require(false, "slots=1 allowed overlap"); }
          catch (const ServerBusyError &) {} }
        std::cout << "PASS concurrent slots, queue, timeout, exclusive unload, exception release, slots=1\n";
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
