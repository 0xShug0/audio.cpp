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
// This does not certify other tasks, variants, options or larger pools.
inline constexpr std::array<AuditedModelSlots, 75> kAuditedCudaOfflineModels = {{
    {"apollo", engine::runtime::VoiceTaskKind::SpeechToSpeech, 4},
    {"audio8_asr", engine::runtime::VoiceTaskKind::Asr, 4},
    {"audio8_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"auk", engine::runtime::VoiceTaskKind::Tts, 4},
    {"breeze_tts", engine::runtime::VoiceTaskKind::VoiceDesign, 4},
    {"bs_roformer", engine::runtime::VoiceTaskKind::SourceSeparation, 4},
    {"canary_asr", engine::runtime::VoiceTaskKind::Asr, 4},
    {"citrinet_asr", engine::runtime::VoiceTaskKind::Asr, 4},
    {"cohere_asr", engine::runtime::VoiceTaskKind::Asr, 4},
    {"confucius4_r2t2", engine::runtime::VoiceTaskKind::Asr, 4},
    {"confucius4_tts", engine::runtime::VoiceTaskKind::VoiceCloning, 3},
    {"controlfoley", engine::runtime::VoiceTaskKind::AudioGeneration, 2},
    {"cosyvoice3", engine::runtime::VoiceTaskKind::VoiceCloning, 4},
    {"dots_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"echo_tts", engine::runtime::VoiceTaskKind::VoiceCloning, 3},
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
    {"sense_asr", engine::runtime::VoiceTaskKind::Asr, 4},
    {"sheetsage2", engine::runtime::VoiceTaskKind::Midi, 3},
    {"soprano_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"sopro_tts", engine::runtime::VoiceTaskKind::Tts, 4},
    {"sortformer_diar", engine::runtime::VoiceTaskKind::Diarization, 4},
    {"sortformer_diar_v2", engine::runtime::VoiceTaskKind::Diarization, 4},
    {"stable_audio", engine::runtime::VoiceTaskKind::AudioGeneration, 4},
    {"supertonic", engine::runtime::VoiceTaskKind::Tts, 4},
    {"universr", engine::runtime::VoiceTaskKind::SpeechToSpeech, 4},
    {"vibevoice", engine::runtime::VoiceTaskKind::Tts, 4},
    {"vibevoice_asr", engine::runtime::VoiceTaskKind::Asr, 2},
    {"vibevoice_asr_streaming", engine::runtime::VoiceTaskKind::Asr, 2},
    {"voxcpm1", engine::runtime::VoiceTaskKind::Tts, 4},
    {"voxcpm2", engine::runtime::VoiceTaskKind::Tts, 4},
    {"voxtral_realtime", engine::runtime::VoiceTaskKind::Asr, 4},
    {"yue2", engine::runtime::VoiceTaskKind::AudioGeneration, 4},
    {"zipvoice", engine::runtime::VoiceTaskKind::VoiceCloning, 4},
}};

inline size_t audited_cuda_slot_capacity(
    std::string_view family,
    engine::runtime::VoiceTaskKind task,
    engine::core::BackendType backend,
    engine::runtime::RunMode mode) noexcept {
    if (backend != engine::core::BackendType::Cuda ||
        mode != engine::runtime::RunMode::Offline) {
        return 1;
    }
    for (const auto & entry : kAuditedCudaOfflineModels) {
        if (entry.family == family && entry.task == task) {
            return entry.capacity;
        }
    }
    return 1;
}

} // namespace minitts::server
