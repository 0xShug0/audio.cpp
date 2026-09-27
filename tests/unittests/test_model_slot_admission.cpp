#include "audited_model_slots.h"
#include <iostream>
#include <stdexcept>

using namespace engine::runtime;
void require(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

void test_audited_model_policy() {
    using engine::core::BackendType;
    using minitts::server::audited_slot_capacity;
    auto capacity = [](std::string_view family, VoiceTaskKind task,
                       BackendType backend = BackendType::Cuda, RunMode mode = RunMode::Offline) {
        return audited_slot_capacity(family, task, backend, mode);
    };
    require(capacity("bs_roformer", VoiceTaskKind::SourceSeparation) == 4, "audited separation capacity wrong");
    require(capacity("qwen3_asr", VoiceTaskKind::Asr) == 4, "audited ASR capacity wrong");
    require(capacity("cosyvoice3", VoiceTaskKind::VoiceCloning) == 4, "audited clone capacity wrong");
    require(capacity("fireredtts3", VoiceTaskKind::VoiceCloning) == 4, "revalidated FireRedTTS3 capacity wrong");
    require(capacity("qwen3_tts", VoiceTaskKind::Tts) == 4, "guarded Qwen3 decoder capacity wrong");
    require(capacity("rvc", VoiceTaskKind::VoiceConversion) == 4, "revalidated RVC capacity wrong");
    require(capacity("confucius4_tts", VoiceTaskKind::VoiceCloning) == 4, "shared Confucius CUDA weights disabled");
    require(capacity("echo_tts", VoiceTaskKind::VoiceCloning) == 4, "shared Echo CUDA weights disabled");
    require(capacity("fish_audio", VoiceTaskKind::Tts) == 4, "shared Fish CUDA weights disabled");
    require(capacity("moss_voicegen", VoiceTaskKind::VoiceDesign) == 4, "shared MOSS VoiceGenerator CUDA weights disabled");
    require(capacity("outetts", VoiceTaskKind::Tts) == 4, "shared OuteTTS CUDA weights disabled");
    require(capacity("sheetsage2", VoiceTaskKind::Midi) == 4, "shared SheetSage CUDA weights disabled");
    require(capacity("controlfoley", VoiceTaskKind::AudioGeneration) == 2, "VRAM-limited family not restricted");
    require(capacity("cosyvoice3", VoiceTaskKind::Tts) == 1, "untested task enabled");
    require(capacity("bs_roformer", VoiceTaskKind::SourceSeparation, BackendType::Cpu) == 1, "CPU fallback enabled");
    require(capacity("bs_roformer", VoiceTaskKind::SourceSeparation, BackendType::Vulkan) == 4, "validated Vulkan separation disabled");
    require(capacity("index_tts2", VoiceTaskKind::Tts, BackendType::Vulkan) == 3, "Vulkan VRAM ceiling wrong");
    require(capacity("yue2", VoiceTaskKind::AudioGeneration, BackendType::Vulkan) == 4, "bounded Yue2 Vulkan capacity wrong");
    require(capacity("zipvoice", VoiceTaskKind::VoiceCloning, BackendType::Vulkan) == 4, "cold-load guarded ZipVoice capacity wrong");
    for (const auto & entry : minitts::server::kAuditedVulkanOfflineModels) {
        require(capacity(entry.family, entry.task, BackendType::Vulkan) == entry.capacity, "Vulkan table entry not admitted");
        require(capacity(entry.family, entry.task, BackendType::Vulkan, RunMode::Streaming) == 1, "Vulkan streaming fallback enabled");
        require(capacity(entry.family, entry.task, BackendType::Cpu) == 1, "CPU fallback enabled by Vulkan audit");
    }
    for (const auto & entry : minitts::server::kAuditedCudaOfflineModels) {
        require(capacity(entry.family, entry.task) == entry.capacity, "CUDA capacity changed");
    }
    require(capacity("firered_audio", VoiceTaskKind::VoiceCloning, BackendType::Vulkan) == 2, "shared FireRed Vulkan weights disabled");
    require(capacity("heartmula", VoiceTaskKind::AudioGeneration, BackendType::Vulkan) == 2, "shared HeartMuLa Vulkan weights disabled");
    require(capacity("moss_tts_v15", VoiceTaskKind::Tts, BackendType::Vulkan) == 2, "shared MOSS Vulkan backbone disabled");
    require(capacity("personaplex", VoiceTaskKind::SpeechToSpeech, BackendType::Vulkan) == 2, "shared PersonaPlex Vulkan weights disabled");
    require(capacity("glm_tts", VoiceTaskKind::Tts, BackendType::Vulkan) == 2, "validated GLM Vulkan slots disabled");
    require(capacity("outetts", VoiceTaskKind::Tts, BackendType::Vulkan) == 2, "validated OuteTTS Vulkan slots disabled");
    require(capacity("sheetsage2", VoiceTaskKind::Midi, BackendType::Vulkan) == 2, "validated SheetSage2 Vulkan slots disabled");
    require(capacity("stable_audio", VoiceTaskKind::AudioGeneration, BackendType::Vulkan) == 2, "validated Stable Audio Vulkan slots disabled");
    require(capacity("inflect_v2", VoiceTaskKind::Tts, BackendType::Vulkan) == 4, "aligned Inflect Vulkan slots disabled");
    require(capacity("mel_band_roformer", VoiceTaskKind::SourceSeparation, BackendType::Vulkan) == 4, "deterministic Mel-Band Vulkan slots disabled");
    for (const auto family : {"auk", "controlfoley"}) {
        for (const auto & entry : minitts::server::kAuditedCudaOfflineModels) {
            if (entry.family == family) {
                require(capacity(entry.family, entry.task, BackendType::Vulkan) == 1, "unvalidated Vulkan family enabled");
            }
        }
    }
    require(capacity("cosyvoice3", VoiceTaskKind::Tts, BackendType::Vulkan) == 1, "untested Vulkan task enabled");
    require(capacity("unknown-model", VoiceTaskKind::Tts, BackendType::Vulkan) == 1, "unknown Vulkan family enabled");
    require(capacity("qwen3_asr", VoiceTaskKind::Asr, BackendType::Cuda, RunMode::Streaming) == 1, "streaming fallback enabled");
    require(capacity("unknown-model", VoiceTaskKind::Tts) == 1, "unknown family enabled");
    require(capacity("f5_tts", VoiceTaskKind::Tts) == 2, "revalidated CUDA family disabled");
    require(capacity("chatterbox", VoiceTaskKind::VoiceCloning) == 2, "validated CUDA slots disabled");
    require(capacity("chatterbox_turbo", VoiceTaskKind::Tts) == 2, "validated CUDA slots disabled");
    require(capacity("mms_forced_aligner", VoiceTaskKind::Alignment) == 2, "validated CUDA slots disabled");
    require(capacity("seed_vc", VoiceTaskKind::VoiceConversion) == 2, "validated CUDA slots disabled");
    require(capacity("vevo2", VoiceTaskKind::VoiceConversion) == 2, "validated CUDA slots disabled");
    require(capacity("ace_step", VoiceTaskKind::AudioGeneration) == 2, "validated CUDA slots disabled");
    require(capacity("ace_step", VoiceTaskKind::AudioGeneration, BackendType::Vulkan) == 1, "unaudited ACE-Step Vulkan slots enabled");
    require(capacity("dramabox", VoiceTaskKind::Tts) == 1, "VRAM-blocked family enabled");
    require(capacity("liveavatar", VoiceTaskKind::AudioGeneration) == 1, "VRAM-blocked family enabled");
    require(capacity("audiosr", VoiceTaskKind::SpeechToSpeech) == 2, "guarded AudioSR CUDA slots disabled");
    require(capacity("miocodec", VoiceTaskKind::VoiceConversion) == 2, "deterministic MioCodec CUDA slots disabled");
    require(capacity("moss_tts_local", VoiceTaskKind::Tts) == 2, "revalidated CUDA slots disabled");
    require(capacity("moss_tts_local", VoiceTaskKind::Tts, BackendType::Vulkan) == 1, "unaudited Vulkan slots enabled");
    require(capacity("moss_ttsd", VoiceTaskKind::Tts) == 2, "revalidated CUDA slots disabled");
    require(capacity("moss_ttsd", VoiceTaskKind::Tts, BackendType::Vulkan) == 1, "unaudited Vulkan slots enabled");
    require(capacity("miotts", VoiceTaskKind::Tts) == 2, "revalidated CUDA slots disabled");
    require(capacity("miotts", VoiceTaskKind::Tts, BackendType::Vulkan) == 1, "unaudited Vulkan slots enabled");
    require(capacity("neutts", VoiceTaskKind::Tts) == 2, "revalidated CUDA slots disabled");
    require(capacity("neutts", VoiceTaskKind::Tts, BackendType::Vulkan) == 1, "unaudited Vulkan slots enabled");
    require(capacity("vieneu_v3_turbo", VoiceTaskKind::Tts) == 2, "revalidated CUDA slots disabled");
    require(capacity("vieneu_v3_turbo", VoiceTaskKind::Tts, BackendType::Vulkan) == 1, "unaudited Vulkan slots enabled");
    require(capacity("minimax_h3", VoiceTaskKind::AudioGeneration) == 1, "hardware-blocked family enabled");
}

int main() {
    try {
        test_audited_model_policy();
        std::cout << "PASS model-specific CUDA/Vulkan slot admission and fallback limits\n";
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
