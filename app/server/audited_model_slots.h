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

// One package/task per family passed cold/warm output parity and overlapping
// CUDA requests. Each capacity is the highest consecutively validated count.
// See docs/reports/generic_cuda_slots_3_4_audit.md for VRAM and quality limits.
// Revalidated two-slot additions and their synchronization limits are in
// docs/reports/cuda_slots_failure_investigation.md.
// Codec parity and MOSS/AudioSR additions are documented in
// docs/reports/cuda_slots_parity_investigation.md.
// This does not certify other tasks, variants, options or larger pools.
inline constexpr std::array<AuditedModelSlots, 89> kAuditedCudaOfflineModels = {{
    {"ace_step", engine::runtime::VoiceTaskKind::AudioGeneration, 2},
    {"apollo", engine::runtime::VoiceTaskKind::SpeechToSpeech, 4},
    {"audio8_asr", engine::runtime::VoiceTaskKind::Asr, 4},
    {"audio8_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"audiosr", engine::runtime::VoiceTaskKind::SpeechToSpeech, 2},
    {"auk", engine::runtime::VoiceTaskKind::Tts, 4},
    {"breeze_tts", engine::runtime::VoiceTaskKind::VoiceDesign, 4},
    {"bs_roformer", engine::runtime::VoiceTaskKind::SourceSeparation, 4},
    {"canary_asr", engine::runtime::VoiceTaskKind::Asr, 4},
    {"chatterbox", engine::runtime::VoiceTaskKind::VoiceCloning, 2},
    {"chatterbox_turbo", engine::runtime::VoiceTaskKind::Tts, 2},
    {"citrinet_asr", engine::runtime::VoiceTaskKind::Asr, 4},
    {"cohere_asr", engine::runtime::VoiceTaskKind::Asr, 4},
    {"confucius4_r2t2", engine::runtime::VoiceTaskKind::Asr, 4},
    {"confucius4_tts", engine::runtime::VoiceTaskKind::VoiceCloning, 3},
    {"controlfoley", engine::runtime::VoiceTaskKind::AudioGeneration, 2},
    {"cosyvoice3", engine::runtime::VoiceTaskKind::VoiceCloning, 4},
    {"dots_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"echo_tts", engine::runtime::VoiceTaskKind::VoiceCloning, 3},
    {"f5_tts", engine::runtime::VoiceTaskKind::Tts, 2},
    {"firered_audio", engine::runtime::VoiceTaskKind::VoiceCloning, 2},
    {"fireredtts3", engine::runtime::VoiceTaskKind::VoiceCloning, 3},
    {"fish_audio", engine::runtime::VoiceTaskKind::Tts, 3},
    {"fun_asr_nano", engine::runtime::VoiceTaskKind::Asr, 4},
    {"glm_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"granite5asr", engine::runtime::VoiceTaskKind::Asr, 4},
    {"heartmula", engine::runtime::VoiceTaskKind::AudioGeneration, 2},
    {"higgs_audio_stt", engine::runtime::VoiceTaskKind::Asr, 4},
    {"higgs_audio_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"htdemucs", engine::runtime::VoiceTaskKind::SourceSeparation, 4},
    {"hviske_asr", engine::runtime::VoiceTaskKind::Asr, 4},
    {"index_tts2", engine::runtime::VoiceTaskKind::Tts, 4},
    {"inflect_v2", engine::runtime::VoiceTaskKind::Tts, 4},
    {"irodori_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"kitten_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"kokoro_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"kroko_asr", engine::runtime::VoiceTaskKind::Asr, 4},
    {"magpie_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"meanvc2", engine::runtime::VoiceTaskKind::VoiceConversion, 4},
    {"mel_band_roformer", engine::runtime::VoiceTaskKind::SourceSeparation, 4},
    {"midashenglm_gen", engine::runtime::VoiceTaskKind::AudioGeneration, 4},
    {"minimax_music3", engine::runtime::VoiceTaskKind::AudioGeneration, 2},
    {"miocodec", engine::runtime::VoiceTaskKind::VoiceConversion, 2},
    {"miotts", engine::runtime::VoiceTaskKind::Tts, 2},
    {"mira_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"mms_forced_aligner", engine::runtime::VoiceTaskKind::Alignment, 2},
    {"moonshine_asr", engine::runtime::VoiceTaskKind::Asr, 4},
    {"moss_transcribe_diarize", engine::runtime::VoiceTaskKind::Asr, 4},
    {"moss_tts_local", engine::runtime::VoiceTaskKind::Tts, 2},
    {"moss_tts_nano", engine::runtime::VoiceTaskKind::Tts, 4},
    {"moss_tts_v15", engine::runtime::VoiceTaskKind::Tts, 2},
    {"moss_ttsd", engine::runtime::VoiceTaskKind::Tts, 2},
    {"moss_voicegen", engine::runtime::VoiceTaskKind::VoiceDesign, 3},
    {"muscriptor", engine::runtime::VoiceTaskKind::Midi, 4},
    {"nemotron_3_diar", engine::runtime::VoiceTaskKind::Diarization, 4},
    {"nemotron_asr", engine::runtime::VoiceTaskKind::Asr, 4},
    {"neutts", engine::runtime::VoiceTaskKind::Tts, 2},
    {"omnivoice", engine::runtime::VoiceTaskKind::Tts, 4},
    {"outetts", engine::runtime::VoiceTaskKind::Tts, 3},
    {"parakeet_tdt", engine::runtime::VoiceTaskKind::Asr, 4},
    {"personaplex", engine::runtime::VoiceTaskKind::SpeechToSpeech, 2},
    {"piper_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"pocket_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"pulsevad", engine::runtime::VoiceTaskKind::Vad, 4},
    {"qwen3_asr", engine::runtime::VoiceTaskKind::Asr, 4},
    {"qwen3_forced_aligner", engine::runtime::VoiceTaskKind::Alignment, 4},
    {"qwen3_tts", engine::runtime::VoiceTaskKind::Tts, 3},
    {"rvc", engine::runtime::VoiceTaskKind::VoiceConversion, 3},
    {"sanotts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"seed_vc", engine::runtime::VoiceTaskKind::VoiceConversion, 2},
    {"sense_asr", engine::runtime::VoiceTaskKind::Asr, 4},
    {"sheetsage2", engine::runtime::VoiceTaskKind::Midi, 3},
    {"soprano_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"sopro_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"sortformer_diar", engine::runtime::VoiceTaskKind::Diarization, 4},
    {"sortformer_diar_v2", engine::runtime::VoiceTaskKind::Diarization, 4},
    {"stable_audio", engine::runtime::VoiceTaskKind::AudioGeneration, 4},
    {"supertonic", engine::runtime::VoiceTaskKind::Tts, 4},
    {"universr", engine::runtime::VoiceTaskKind::SpeechToSpeech, 4},
    {"vevo2", engine::runtime::VoiceTaskKind::VoiceConversion, 2},
    {"vibevoice", engine::runtime::VoiceTaskKind::Tts, 4},
    {"vibevoice_asr", engine::runtime::VoiceTaskKind::Asr, 2},
    {"vibevoice_asr_streaming", engine::runtime::VoiceTaskKind::Asr, 2},
    {"vieneu_v3_turbo", engine::runtime::VoiceTaskKind::Tts, 2},
    {"voxcpm1", engine::runtime::VoiceTaskKind::Tts, 4},
    {"voxcpm2", engine::runtime::VoiceTaskKind::Tts, 4},
    {"voxtral_realtime", engine::runtime::VoiceTaskKind::Asr, 4},
    {"yue2", engine::runtime::VoiceTaskKind::AudioGeneration, 4},
    {"zipvoice", engine::runtime::VoiceTaskKind::VoiceCloning, 4},
}};

// Vulkan has its own audited capacities; CUDA results do not imply Vulkan
// support. See docs/reports/generic_vulkan_slots_audit.md for tested workloads.
inline constexpr std::array<AuditedModelSlots, 67> kAuditedVulkanOfflineModels = {{
    {"apollo", engine::runtime::VoiceTaskKind::SpeechToSpeech, 4},
    {"audio8_asr", engine::runtime::VoiceTaskKind::Asr, 4},
    {"audio8_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"breeze_tts", engine::runtime::VoiceTaskKind::VoiceDesign, 4},
    {"bs_roformer", engine::runtime::VoiceTaskKind::SourceSeparation, 4},
    {"canary_asr", engine::runtime::VoiceTaskKind::Asr, 4},
    {"citrinet_asr", engine::runtime::VoiceTaskKind::Asr, 4},
    {"cohere_asr", engine::runtime::VoiceTaskKind::Asr, 4},
    {"confucius4_r2t2", engine::runtime::VoiceTaskKind::Asr, 4},
    {"confucius4_tts", engine::runtime::VoiceTaskKind::VoiceCloning, 3},
    {"cosyvoice3", engine::runtime::VoiceTaskKind::VoiceCloning, 4},
    {"dots_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"echo_tts", engine::runtime::VoiceTaskKind::VoiceCloning, 3},
    {"firered_audio", engine::runtime::VoiceTaskKind::VoiceCloning, 2},
    {"fireredtts3", engine::runtime::VoiceTaskKind::VoiceCloning, 3},
    {"fish_audio", engine::runtime::VoiceTaskKind::Tts, 3},
    {"fun_asr_nano", engine::runtime::VoiceTaskKind::Asr, 4},
    {"granite5asr", engine::runtime::VoiceTaskKind::Asr, 4},
    {"heartmula", engine::runtime::VoiceTaskKind::AudioGeneration, 2},
    {"higgs_audio_stt", engine::runtime::VoiceTaskKind::Asr, 4},
    {"higgs_audio_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"htdemucs", engine::runtime::VoiceTaskKind::SourceSeparation, 4},
    {"hviske_asr", engine::runtime::VoiceTaskKind::Asr, 4},
    {"index_tts2", engine::runtime::VoiceTaskKind::Tts, 3},
    {"irodori_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"kitten_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"kokoro_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"kroko_asr", engine::runtime::VoiceTaskKind::Asr, 4},
    {"magpie_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"meanvc2", engine::runtime::VoiceTaskKind::VoiceConversion, 4},
    {"midashenglm_gen", engine::runtime::VoiceTaskKind::AudioGeneration, 4},
    {"minimax_music3", engine::runtime::VoiceTaskKind::AudioGeneration, 2},
    {"mira_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"moonshine_asr", engine::runtime::VoiceTaskKind::Asr, 4},
    {"moss_transcribe_diarize", engine::runtime::VoiceTaskKind::Asr, 4},
    {"moss_tts_nano", engine::runtime::VoiceTaskKind::Tts, 4},
    {"moss_tts_v15", engine::runtime::VoiceTaskKind::Tts, 2},
    {"moss_voicegen", engine::runtime::VoiceTaskKind::VoiceDesign, 3},
    {"muscriptor", engine::runtime::VoiceTaskKind::Midi, 4},
    {"nemotron_3_diar", engine::runtime::VoiceTaskKind::Diarization, 4},
    {"nemotron_asr", engine::runtime::VoiceTaskKind::Asr, 4},
    {"omnivoice", engine::runtime::VoiceTaskKind::Tts, 4},
    {"parakeet_tdt", engine::runtime::VoiceTaskKind::Asr, 4},
    {"personaplex", engine::runtime::VoiceTaskKind::SpeechToSpeech, 2},
    {"piper_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"pocket_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"pulsevad", engine::runtime::VoiceTaskKind::Vad, 4},
    {"qwen3_asr", engine::runtime::VoiceTaskKind::Asr, 4},
    {"qwen3_forced_aligner", engine::runtime::VoiceTaskKind::Alignment, 4},
    {"qwen3_tts", engine::runtime::VoiceTaskKind::Tts, 3},
    {"rvc", engine::runtime::VoiceTaskKind::VoiceConversion, 3},
    {"sanotts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"sense_asr", engine::runtime::VoiceTaskKind::Asr, 4},
    {"soprano_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"sopro_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"sortformer_diar", engine::runtime::VoiceTaskKind::Diarization, 4},
    {"sortformer_diar_v2", engine::runtime::VoiceTaskKind::Diarization, 4},
    {"supertonic", engine::runtime::VoiceTaskKind::Tts, 4},
    {"universr", engine::runtime::VoiceTaskKind::SpeechToSpeech, 4},
    {"vibevoice", engine::runtime::VoiceTaskKind::Tts, 4},
    {"vibevoice_asr", engine::runtime::VoiceTaskKind::Asr, 2},
    {"vibevoice_asr_streaming", engine::runtime::VoiceTaskKind::Asr, 2},
    {"voxcpm1", engine::runtime::VoiceTaskKind::Tts, 4},
    {"voxcpm2", engine::runtime::VoiceTaskKind::Tts, 4},
    {"voxtral_realtime", engine::runtime::VoiceTaskKind::Asr, 4},
    {"yue2", engine::runtime::VoiceTaskKind::AudioGeneration, 3},
    {"zipvoice", engine::runtime::VoiceTaskKind::VoiceCloning, 2},
}};

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
