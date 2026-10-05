// Included inside minitts::server after AuditedModelSlots is declared.
inline constexpr std::array<AuditedModelSlots, 1> kAuditedCudaOfflineModels = {{
    {"slot_validation_fixture", engine::runtime::VoiceTaskKind::AudioGeneration, 4},
}};
inline constexpr std::array<AuditedModelSlots, 1> kAuditedVulkanOfflineModels = {{
    {"slot_validation_fixture", engine::runtime::VoiceTaskKind::AudioGeneration, 4},
}};
