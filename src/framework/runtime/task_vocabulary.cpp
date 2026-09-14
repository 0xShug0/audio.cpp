#include "engine/framework/runtime/task_vocabulary.h"

namespace engine::runtime {

namespace {

/// The vocabulary, in enum order.
///
/// Adding a task kind means adding one row here. Nothing else enumerates them,
/// which is the point: the four lists this replaced were edited independently
/// and drifted.
constexpr TaskVocabularyEntry kVocabulary[] = {
    {VoiceTaskKind::Vad, "vad", {"vad"}, 1},
    {VoiceTaskKind::Asr, "asr", {"asr"}, 1},
    {VoiceTaskKind::Diarization, "diar", {"diar"}, 1},
    {VoiceTaskKind::SourceSeparation, "sep", {"sep"}, 1},
    // The one genuinely many-to-one row: a spec says what the audio is for,
    // the runtime has a single generation kind.
    {VoiceTaskKind::AudioGeneration, "gen", {"music", "sfx", "edit", "audio_generation"}, 4},
    {VoiceTaskKind::Tts, "tts", {"tts"}, 1},
    {VoiceTaskKind::VoiceCloning, "clon", {"clone"}, 1},
    {VoiceTaskKind::VoiceConversion, "vc", {"vc"}, 1},
    {VoiceTaskKind::SpeechToSpeech, "s2s", {"s2s"}, 1},
    {VoiceTaskKind::Alignment, "align", {"align"}, 1},
    {VoiceTaskKind::VoiceDesign, "vdes", {"design"}, 1},
    {VoiceTaskKind::SpeakerRecognition, "spk", {"speaker"}, 1},
    {VoiceTaskKind::Svc, "svc", {"svc"}, 1},
    {VoiceTaskKind::Midi, "midi", {"midi"}, 1},
};

}  // namespace

const TaskVocabularyEntry * task_vocabulary(std::size_t & count) noexcept {
    count = sizeof(kVocabulary) / sizeof(kVocabulary[0]);
    return kVocabulary;
}

std::string_view task_token_for_spec_name(std::string_view spec_task) noexcept {
    for (const auto & entry : kVocabulary) {
        for (std::size_t i = 0; i < entry.alias_count; ++i) {
            if (entry.aliases[i] == spec_task) {
                return entry.token;
            }
        }
    }
    return {};
}

bool is_spec_task_name(std::string_view value) noexcept {
    return !task_token_for_spec_name(value).empty();
}

}  // namespace engine::runtime
