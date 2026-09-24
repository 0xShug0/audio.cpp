#include "engine/community_models/vieneu_v3_turbo/prompt_tts_voice_clone.h"

#include "engine/framework/audio/resampling.h"

#include <algorithm>

#include <stdexcept>
#include <string>

namespace engine::models::vieneu_v3_turbo {
namespace {

constexpr int64_t kCodeGroups = 16;
constexpr int64_t kCodecSamplesPerFrame = 3840;   // 80 ms at 48 kHz
constexpr double kMaxReferenceSeconds = 8.0;       // Python `_MAX_REF_SECONDS`

void require_text_token_limit(size_t actual, int64_t limit, const char * what) {
    if (limit <= 0) {
        throw std::runtime_error(std::string("VieNeu voice clone ") + what + " token limit must be positive");
    }
    if (actual > static_cast<size_t>(limit)) {
        throw std::runtime_error(
            std::string("VieNeu voice clone ") + what + " token count "
            + std::to_string(actual) + " exceeds limit " + std::to_string(limit));
    }
}

}  // namespace

VieNeuTTSVoiceClonePromptBuilder::VieNeuTTSVoiceClonePromptBuilder(
    const Qwen3TextTokenizer & tokenizer,
    engine::codecs::MossAudioTokenizerCodecRuntime * codec,
    const VieNeuSpeakerEncoderRuntime * speaker_encoder,
    int64_t text_token_limit)
    : tokenizer_(tokenizer),
      codec_(codec),
      speaker_encoder_(speaker_encoder),
      text_token_limit_(text_token_limit) {}

Qwen3VoiceClonePrompt VieNeuTTSVoiceClonePromptBuilder::build_voice_prompt(const Qwen3VoiceCloneInput & input) const {
    Qwen3VoiceClonePrompt prompt;
    if (input.speaker_embedding.has_value()) {
        prompt.speaker_embedding.dims = 192;
        prompt.speaker_embedding.values = *input.speaker_embedding;
    } else if (speaker_encoder_ != nullptr) {
        prompt.speaker_embedding = speaker_encoder_->encode(input.reference_audio);
    } else {
        prompt.speaker_embedding.dims = 192;
        prompt.speaker_embedding.values = std::vector<float>(192, 0.0f);
    }
    // v3 Turbo conditions on the reference audio codes only; the transcript is not part
    // of the prompt, so "ICL" here simply means the codec is available and the caller
    // did not ask for x-vector-only cloning.
    if (input.reference_codes.has_value() && input.reference_codes->frames > 0) {
        prompt.icl_mode = input.mode == Qwen3VoiceCloneMode::Icl;
        if (prompt.icl_mode) {
            prompt.reference_codes = *input.reference_codes;
        }
        return prompt;
    }
    prompt.icl_mode = (input.mode == Qwen3VoiceCloneMode::Icl) && (codec_ != nullptr);
    if (prompt.icl_mode) {
        prompt.reference_codes = encode_reference_codes(*codec_, input.reference_audio, kCodeGroups);
    }
    return prompt;
}

Qwen3SpeechCodes VieNeuTTSVoiceClonePromptBuilder::encode_reference_codes(
    engine::codecs::MossAudioTokenizerCodecRuntime & codec,
    const runtime::AudioBuffer & audio,
    int64_t code_groups) {
    const int channels = std::max(1, audio.channels);
    const int64_t frames = static_cast<int64_t>(audio.samples.size()) / channels;
    if (frames <= 0) {
        throw std::runtime_error("VieNeu-TTS voice reference audio is empty");
    }
    std::vector<float> mono(static_cast<size_t>(frames));
    for (int64_t i = 0; i < frames; ++i) {
        float sum = 0.0F;
        for (int c = 0; c < channels; ++c) {
            sum += audio.samples[static_cast<size_t>(i * channels + c)];
        }
        mono[static_cast<size_t>(i)] = sum / static_cast<float>(channels);
    }
    const int64_t sample_rate = codec.sampling_rate();
    if (audio.sample_rate <= 0) {
        throw std::runtime_error("VieNeu-TTS voice reference has an invalid sample rate");
    }
    if (audio.sample_rate != sample_rate) {
        mono = engine::audio::resample_mono_torchaudio_sinc_hann(mono, audio.sample_rate, static_cast<int>(sample_rate));
    }
    const size_t max_samples = static_cast<size_t>(kMaxReferenceSeconds * static_cast<double>(sample_rate));
    if (mono.size() > max_samples) {
        mono.resize(max_samples);
    }
    const size_t remainder = mono.size() % static_cast<size_t>(kCodecSamplesPerFrame);
    if (remainder != 0) {
        mono.resize(mono.size() + static_cast<size_t>(kCodecSamplesPerFrame) - remainder, 0.0F);
    }
    const int64_t n_frames = static_cast<int64_t>(mono.size()) / kCodecSamplesPerFrame;
    std::vector<std::vector<float>> stereo{mono, mono};
    const auto encoded = codec.encode(engine::codecs::MossAudioTokenizerAudio{sample_rate, std::move(stereo)});
    if (static_cast<int64_t>(encoded.codebooks.size()) < code_groups) {
        throw std::runtime_error("VieNeu-TTS reference encoder returned too few codebooks");
    }
    // The encoder emits one frame more than the padded input holds; that frame is the
    // audible pad artifact, so keep exactly n_frames (mirrors the Python engine).
    const int64_t keep = std::min<int64_t>(n_frames, encoded.frames);
    if (keep <= 0) {
        throw std::runtime_error("VieNeu-TTS reference encoder returned no frames");
    }
    Qwen3SpeechCodes out;
    out.frames = keep;
    out.code_groups = code_groups;
    out.codes.resize(static_cast<size_t>(keep * code_groups));
    for (int64_t f = 0; f < keep; ++f) {
        for (int64_t g = 0; g < code_groups; ++g) {
            out.codes[static_cast<size_t>(f * code_groups + g)] =
                encoded.codebooks[static_cast<size_t>(g)][static_cast<size_t>(f)];
        }
    }
    return out;
}

VieNeuTalkerPrefill VieNeuTTSVoiceClonePromptBuilder::build_prefill(
    const VieNeuTTSRequest & request,
    const Qwen3VoiceClonePrompt & prompt) const {
    VieNeuTalkerPrefill prefill;
    prefill.prompt_mode = VieNeuTalkerPromptMode::VoiceClone;
    prefill.input_ids = tokenizer_.encode(tokenizer_.build_assistant_prompt(request.text));
    require_text_token_limit(prefill.input_ids.size(), text_token_limit_, "text");
    prefill.reference_ids = prompt.reference_text_ids;
    prefill.reference_codes = prompt.reference_codes;
    prefill.speaker_embedding = prompt.speaker_embedding;
    prefill.language = request.language;
    prefill.icl_mode = prompt.icl_mode;
    prefill.x_vector_only_mode = !prompt.icl_mode;
    return prefill;
}

}  // namespace engine::models::vieneu_v3_turbo
