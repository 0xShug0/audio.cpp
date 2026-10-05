// Validation fixture for the independent common slot-validation hooks.
// Included inside minitts::server, after AuditedModelSlots is declared.
// This is not production admission.
inline constexpr std::array<AuditedModelSlots, 1> kAuditedCudaOfflineModels = {{
    {"ace_step", engine::runtime::VoiceTaskKind::AudioGeneration, 4},
}};
inline constexpr std::array<AuditedModelSlots, 1> kAuditedVulkanOfflineModels = {{
    {"ace_step", engine::runtime::VoiceTaskKind::AudioGeneration, 4},
}};
