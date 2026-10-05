#include "engine/models/ace_step/backend_weights_cache.h"
#include "engine/framework/core/backend_weight_store.h"

#include <atomic>
#include <algorithm>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace engine;
using models::ace_step::AceStepBackendWeightsCache;
namespace {
void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}
struct Weights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue tensor;
};
}
int main(int argc, char ** argv) {
    try {
        core::BackendConfig config;
        config.type = argc > 1 && std::string(argv[1]) == "vulkan"
            ? core::BackendType::Vulkan : core::BackendType::Cuda;
        const auto devices = core::list_backend_devices();
        const std::string backend_name = config.type == core::BackendType::Vulkan ? "Vulkan" : "CUDA";
        auto selected = std::find_if(devices.begin(), devices.end(), [&](const auto & device) {
            return device.backend == backend_name && device.type == "GPU";
        });
        if (selected == devices.end()) selected = std::find_if(devices.begin(), devices.end(), [&](const auto & device) {
            return device.backend == backend_name;
        });
        if (selected == devices.end()) return 77;
        config.device = argc > 2 ? std::stoi(argv[2]) : selected->index;
        std::cout << "backend=" << backend_name << " device=" << config.device << '\n';
        AceStepBackendWeightsCache cache;
        std::atomic<int> loads{0};
        auto factory = [&](core::ExecutionContext & upload) {
            ++loads;
            auto weights = std::make_shared<Weights>();
            weights->store = std::make_shared<core::BackendWeightStore>(
                upload.backend(), upload.backend_type(), "ace-cache-test", 1024 * 1024);
            weights->tensor = weights->store->make_f32(core::TensorShape::from_dims({2}), {3.0F, 7.0F});
            weights->store->upload();
            return weights;
        };
        std::vector<std::unique_ptr<core::ExecutionContext>> slots;
        for (int i=0; i<8; ++i) slots.push_back(std::make_unique<core::ExecutionContext>(config));
        std::vector<std::future<std::shared_ptr<const Weights>>> jobs;
        for (auto & slot : slots) {
            auto * execution = slot.get();
            jobs.push_back(std::async(std::launch::async, [&, execution] {
                return cache.acquire<Weights>(*execution, assets::TensorStorageType::F32, factory);
            }));
        }
        std::vector<std::shared_ptr<const Weights>> shared;
        for (auto & job : jobs) shared.push_back(job.get());
        require(loads == 1, "concurrent cold slots duplicated weight uploads");
        for (auto & w : shared) require(w.get() == shared[0].get(), "slots did not share tensor owners");
        slots.clear();
        require(core::read_tensor_f32(shared[0]->tensor.tensor) == std::vector<float>({3.0F,7.0F}),
                "weights invalid after all slot backends were destroyed");
        core::ExecutionContext next(config);
        auto again=cache.acquire<Weights>(next, assets::TensorStorageType::F32, factory);
        require(loads == 1 && again.get() == shared[0].get(), "live owner was not reused");
        auto separate=cache.acquire<Weights>(next, assets::TensorStorageType::Native, factory);
        require(loads == 2 && separate.get() != again.get(), "different precision aliases an incompatible owner");
        std::weak_ptr<const Weights> expired=again;
        shared.clear(); again.reset();
        require(expired.expired(), "cache pins unleased weights after unload");
        auto retry=cache.acquire<Weights>(next, assets::TensorStorageType::F32, factory);
        require(loads == 3, "unloaded weights were not recreated");
        int failures=0;
        try {
            cache.acquire<Weights>(next, assets::TensorStorageType::BF16,
                [&](core::ExecutionContext &) -> std::shared_ptr<const Weights> {
                    ++failures; throw std::runtime_error("injected load failure");
                });
        } catch (const std::runtime_error &) {}
        auto recovered=cache.acquire<Weights>(next, assets::TensorStorageType::BF16, factory);
        require(failures==1 && recovered && loads==4, "failed upload poisoned future retry");
        require(core::read_tensor_f32(retry->tensor.tensor)==std::vector<float>({3.0F,7.0F}),
                "unrelated cached weights were modified");
        std::cout << "PASS: concurrent upload, precision isolation, independent backend lifetime, weak unload and retry\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
