#include "engine/community_models/vietneu_tts/prompt_tts_voice_clone.h"

#include <stdexcept>
#include <string>

namespace engine::models::vietneu_tts {
namespace {

void require_text_token_limit(size_t actual, int64_t limit, const char * what) {
    if (limit <= 0) {
        throw std::runtime_error(std::string("Vietneu voice clone ") + what + " token limit must be positive");
    }
    if (actual > static_cast<size_t>(limit)) {
        throw std::runtime_error(
            std::string("Vietneu voice clone ") + what + " token count "
            + std::to_string(actual) + " exceeds limit " + std::to_string(limit));
    }
}

}  // namespace

VietneuTTSVoiceClonePromptBuilder::VietneuTTSVoiceClonePromptBuilder(
    const Qwen3TextTokenizer & tokenizer,
    const Qwen3SpeechTokenizerEncoderRuntime * speech_encoder,
    const VietneuSpeakerEncoderRuntime * speaker_encoder,
    int64_t text_token_limit)
    : tokenizer_(tokenizer),
      speech_encoder_(speech_encoder),
      speaker_encoder_(speaker_encoder),
      text_token_limit_(text_token_limit) {}

Qwen3VoiceClonePrompt VietneuTTSVoiceClonePromptBuilder::build_voice_prompt(const Qwen3VoiceCloneInput & input) const {
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
    prompt.icl_mode = (input.mode == Qwen3VoiceCloneMode::Icl) && (speech_encoder_ != nullptr);
    if (prompt.icl_mode) {
        if (input.reference_text.empty()) {
            throw std::runtime_error("Vietneu voice clone ICL mode requires reference text");
        }
        prompt.reference_codes = speech_encoder_->encode(input.reference_audio);
        prompt.reference_text_ids = tokenizer_.encode(tokenizer_.build_reference_prompt(input.reference_text));
        require_text_token_limit(prompt.reference_text_ids.size(), text_token_limit_, "reference text");
    }
    return prompt;
}

VietneuTalkerPrefill VietneuTTSVoiceClonePromptBuilder::build_prefill(
    const VietneuTTSRequest & request,
    const Qwen3VoiceClonePrompt & prompt) const {
    VietneuTalkerPrefill prefill;
    prefill.prompt_mode = VietneuTalkerPromptMode::VoiceClone;
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

}  // namespace engine::models::vietneu_tts
