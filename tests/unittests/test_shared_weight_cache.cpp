#include "engine/framework/core/shared_weight_cache.h"

#include <atomic>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    try {
        std::weak_ptr<const int> retained;
        {
            engine::core::SharedWeightCache cache;
            std::atomic<int> loads{0};
            std::vector<std::future<std::shared_ptr<const int>>> requests;
            for (int i = 0; i < 16; ++i) {
                requests.push_back(std::async(std::launch::async, [&] {
                    return cache.get_or_load<int>("backbone:cuda0:bf16", [&] { ++loads; return 42; });
                }));
            }
            auto first = requests.front().get();
            retained = first;
            for (size_t i = 1; i < requests.size(); ++i) {
                if (requests[i].get() != first) throw std::runtime_error("slots do not share one allocation");
            }
            if (loads != 1 || *first != 42) throw std::runtime_error("concurrent factory ran more than once");
            if (cache.get_or_load<int>("backbone:cuda1:bf16", [] { return 7; }) == first ||
                cache.get_or_load<int>("backbone:cuda0:q8", [] { return 8; }) == first)
                throw std::runtime_error("device/storage keys share incompatible weights");
            bool failed = false;
            try { cache.get_or_load<int>("failed", []() -> int { throw std::runtime_error("load failed"); }); }
            catch (const std::runtime_error &) { failed = true; }
            if (!failed || *cache.get_or_load<int>("failed", [] { return 9; }) != 9)
                throw std::runtime_error("failed upload poisoned a cache entry");
            first.reset();
            if (retained.expired()) throw std::runtime_error("package lost weights when first slot closed");
        }
        if (!retained.expired()) throw std::runtime_error("package unload retained weights");
        std::cout << "shared_weight_cache_test: ok\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
