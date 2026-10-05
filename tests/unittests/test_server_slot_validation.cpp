#include "audited_model_slots.h"

#include <iostream>
#include <stdexcept>

int main() {
    using engine::core::BackendType;
    using engine::runtime::RunMode;
    using engine::runtime::VoiceTaskKind;
    using minitts::server::audited_slot_capacity;
    const auto require = [](bool condition) {
        if (!condition) throw std::runtime_error("Validation admission scope changed");
    };
#if defined(AUDIOCPP_SLOT_VALIDATION_CAPACITY_HEADER)
    constexpr size_t gpu_capacity = 4;
#else
    constexpr size_t gpu_capacity = 1;
#endif
    for (auto backend : {BackendType::Cuda, BackendType::Vulkan}) {
        require(audited_slot_capacity("slot_validation_fixture", VoiceTaskKind::AudioGeneration,
                                     backend, RunMode::Offline) == gpu_capacity);
        require(audited_slot_capacity("slot_validation_fixture", VoiceTaskKind::AudioGeneration,
                                     backend, RunMode::Streaming) == 1);
        require(audited_slot_capacity("slot_validation_fixture", VoiceTaskKind::Asr,
                                     backend, RunMode::Offline) == 1);
        require(audited_slot_capacity("unvalidated_family", VoiceTaskKind::AudioGeneration,
                                     backend, RunMode::Offline) == 1);
    }
    require(audited_slot_capacity("slot_validation_fixture", VoiceTaskKind::AudioGeneration,
                                 BackendType::Cpu, RunMode::Offline) == 1);
    std::cout << "PASS: default/fixture admission remains backend/task/mode scoped\n";
}
