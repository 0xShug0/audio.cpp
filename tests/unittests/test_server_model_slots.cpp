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

int main() {
    try {
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
