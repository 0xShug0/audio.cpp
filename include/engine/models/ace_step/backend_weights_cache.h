#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <typeindex>
#include <utility>

namespace engine::models::ace_step {

// One cache per loaded package. Only immutable device weights are shared;
// execution backends, graph allocators and request caches remain slot-local.
class AceStepBackendWeightsCache {
public:
    template<class Weights, class Factory>
    std::shared_ptr<const Weights> acquire(
        core::ExecutionContext & execution,
        assets::TensorStorageType storage,
        Factory && factory) const {
        // TensorSource mmap/release is mutable even for const assets. Serialize
        // all imports, including Vulkan's CPU-prefill weights, with publication.
        std::lock_guard<std::mutex> lock(mutex_);
        const auto type = execution.backend_type();
        if (type != core::BackendType::Cuda && type != core::BackendType::Vulkan) {
            return factory(execution);
        }
        const Key key{type, execution.config().device, storage, std::type_index(typeid(Weights))};
        if (auto it = entries_.find(key); it != entries_.end()) {
            if (auto existing = it->second.lock()) {
                return std::static_pointer_cast<const Weights>(existing);
            }
            entries_.erase(it);
        }
        // The upload backend must outlive every borrowing slot. An independent
        // owner prevents slot zero's destruction leaving dangling device state.
        struct Owner {
            core::ExecutionContext execution;
            std::shared_ptr<const Weights> weights; // destroyed before execution
            explicit Owner(core::BackendConfig config) : execution(config) {}
        };
        auto config = execution.config();
        config.type = type;
        auto owner = std::make_shared<Owner>(config);
        owner->weights = factory(owner->execution);
        if (!owner->weights) {
            throw std::runtime_error("ACE-Step shared weight loader returned null");
        }
        // Publish only a fully uploaded value. Failed loads leave no cache entry.
        std::shared_ptr<const Weights> result(owner, owner->weights.get());
        entries_.emplace(key, result);
        return result;
    }

    template<class Factory>
    auto with_source_lock(Factory && factory) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return factory();
    }

private:
    using Key = std::tuple<core::BackendType, int, assets::TensorStorageType, std::type_index>;
    mutable std::mutex mutex_;
    // Weak ownership permits unload and memory-saver release to free weights;
    // also avoids a cycle when a weights runtime retains its package assets.
    mutable std::map<Key, std::weak_ptr<const void>> entries_;
};

} // namespace engine::models::ace_step
