#pragma once

#include "engine/framework/runtime/session.h"

#include <array>
#include <string_view>

namespace minitts::server {

struct AuditedModelSlots {
    std::string_view family;
    engine::runtime::VoiceTaskKind task;
    size_t capacity;
};

// Model-specific validation adds entries in the model-support follow-up.
// The common framework alone does not opt any legacy model into parallel runs.
// Optional test fixture defines tables here, inside minitts::server.
#if defined(AUDIOCPP_SLOT_VALIDATION_CAPACITY_HEADER)
#include AUDIOCPP_SLOT_VALIDATION_CAPACITY_HEADER
#else
inline constexpr std::array<AuditedModelSlots, 0> kAuditedCudaOfflineModels = {};
inline constexpr std::array<AuditedModelSlots, 0> kAuditedVulkanOfflineModels = {};
#endif

inline size_t audited_slot_capacity(
    std::string_view family,
    engine::runtime::VoiceTaskKind task,
    engine::core::BackendType backend,
    engine::runtime::RunMode mode) noexcept {
    if (mode != engine::runtime::RunMode::Offline) {
        return 1;
    }
    const auto lookup = [&](const auto & entries) {
        for (const auto & entry : entries) {
            if (entry.family == family && entry.task == task) {
                return entry.capacity;
            }
        }
        return size_t{1};
    };
    switch (backend) {
        case engine::core::BackendType::Cuda:
            return lookup(kAuditedCudaOfflineModels);
        case engine::core::BackendType::Vulkan:
            return lookup(kAuditedVulkanOfflineModels);
        default:
            return 1;
    }
}

} // namespace minitts::server
