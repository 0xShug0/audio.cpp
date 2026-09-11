#include "engine/models/firered_audio/tokenizer.h"

#include "engine/framework/tokenizers/llama_bpe.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace engine::models::firered_audio {
namespace {

std::string repeated(std::string_view value, int64_t count) {
    if (count < 0) {
        throw std::runtime_error("FireRedAudio prompt placeholder count must be non-negative");
    }
    std::string out;
    out.reserve(value.size() * static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        out += value;
    }
    return out;
}

std::string chatml(
    std::string_view system,
    std::string_view user,
    std::string_view assistant_prefix = {},
    bool enable_thinking = false) {
    std::string out;
    out += "<|im_start|>system\n";
    out += std::string(system);
    out += "<|im_end|>\n<|im_start|>user\n";
    out += std::string(user);
    out += "<|im_end|>\n<|im_start|>assistant\n";
    out += enable_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
    out += std::string(assistant_prefix);
    return out;
}

constexpr std::string_view kGenericSystemPrompt = "You are a helpful assistant.";
constexpr std::string_view kUnderstandingSystemPrompt =
    "You are an audio understanding expert. Please answer user questions based on the audio.";

std::string tts_prompt(
    std::string_view reference_text,
    std::string_view text,
    std::string_view language,
    int64_t reference_audio_patches) {
    const std::string sep = language == "en" ? " " : "";
    std::string user = "Convert text to speech.\n";
    user += std::string(reference_text);
    user += sep;
    user += std::string(text);
    std::string assistant = "<|sosp|>";
    assistant += repeated("<|AUDIO_NO_LATENT|>", reference_audio_patches);
    return chatml(kGenericSystemPrompt, user, assistant);
}

std::string voice_design_prompt(std::string_view instruction, std::string_view text) {
    std::string user = std::string(instruction);
    user += "\n\n根据上述音色描述，合成以下文本对应的音频：\n";
    user += std::string(text);
    return chatml(kGenericSystemPrompt, user);
}

std::string edit_prompt(
    std::string_view instruction,
    std::string_view edit_type,
    int64_t reference_audio_patches) {
    std::string user = "Audio 1: <|sosp|>";
    user += repeated("<|AUDIO_NO_LATENT|>", reference_audio_patches);
    user += "<|eosp|>\n";
    if (edit_type == "semantic") {
        user += "Identify the content of the audio. ";
    } else if (edit_type != "acoustic") {
        throw std::runtime_error("unknown FireRedAudio edit_type: " + std::string(edit_type));
    }
    user += std::string(instruction);
    return chatml(kGenericSystemPrompt, user);
}

std::string understanding_prompt(
    std::string_view prompt,
    int64_t audio_tokens,
    bool enable_thinking) {
    std::string user = "Audio 1: <|sosp|>";
    user += repeated("<|AUDIO|>", audio_tokens);
    user += "<|eosp|>\n";
    user += std::string(prompt);
    return chatml(kUnderstandingSystemPrompt, user, {}, enable_thinking);
}

FireRedAudioPromptEncoding encode_prompt(
    const engine::tokenizers::LlamaBpeTokenizer & tokenizer,
    const std::string & prompt,
    int64_t generation_audio_patches,
    int64_t understanding_audio_tokens,
    int32_t audio_id,
    int32_t audio_no_latent_id) {
    FireRedAudioPromptEncoding out;
    out.generation_audio_patches = generation_audio_patches;
    out.understanding_audio_tokens = understanding_audio_tokens;
    out.token_ids = tokenizer.encode(prompt, true);
    if (out.token_ids.empty()) {
        throw std::runtime_error("FireRedAudio tokenizer produced empty token sequence");
    }
    out.audio_mask.resize(out.token_ids.size(), 0);
    out.audio_no_latent_mask.resize(out.token_ids.size(), 0);
    int64_t found_audio = 0;
    int64_t found_audio_no_latent = 0;
    for (size_t i = 0; i < out.token_ids.size(); ++i) {
        if (out.token_ids[i] == audio_id) {
            out.audio_mask[i] = 1;
            ++found_audio;
        } else if (out.token_ids[i] == audio_no_latent_id) {
            out.audio_no_latent_mask[i] = 1;
            ++found_audio_no_latent;
        }
    }
    if (found_audio != understanding_audio_tokens || found_audio_no_latent != generation_audio_patches) {
        throw std::runtime_error("FireRedAudio prompt audio placeholder count mismatch");
    }
    return out;
}

}  // namespace

class FireRedAudioTokenizer::Impl {
public:
    explicit Impl(const FireRedAudioAssets & assets)
        : audio_id(assets.special_tokens.audio),
          audio_no_latent_id(assets.special_tokens.audio_no_latent) {
        engine::tokenizers::LlamaBpeTokenizerSpec spec;
        spec.tokenizer_json_path = assets.resources.require_file("tokenizer_json");
        spec.tokenizer_config_path = assets.resources.require_file("tokenizer_config");
        spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen35;
        tokenizer = engine::tokenizers::load_llama_bpe_tokenizer(spec);
    }

    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer;
    int32_t audio_id = 0;
    int32_t audio_no_latent_id = 0;
};

FireRedAudioTokenizer::FireRedAudioTokenizer(std::shared_ptr<const FireRedAudioAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("FireRedAudio tokenizer requires assets");
    }
    impl_ = std::make_shared<Impl>(*assets);
}

FireRedAudioTokenizer::~FireRedAudioTokenizer() = default;

FireRedAudioPromptEncoding FireRedAudioTokenizer::encode_tts_clone_prompt(
    std::string_view reference_text,
    std::string_view text,
    std::string_view language,
    int64_t reference_audio_patches) const {
    return encode_prompt(
        *impl_->tokenizer,
        tts_prompt(reference_text, text, language, reference_audio_patches),
        reference_audio_patches,
        0,
        impl_->audio_id,
        impl_->audio_no_latent_id);
}

FireRedAudioPromptEncoding FireRedAudioTokenizer::encode_voice_design_prompt(
    std::string_view instruction,
    std::string_view text) const {
    return encode_prompt(*impl_->tokenizer, voice_design_prompt(instruction, text), 0, 0, impl_->audio_id, impl_->audio_no_latent_id);
}

FireRedAudioPromptEncoding FireRedAudioTokenizer::encode_edit_prompt(
    std::string_view instruction,
    std::string_view edit_type,
    int64_t reference_audio_patches) const {
    return encode_prompt(
        *impl_->tokenizer,
        edit_prompt(instruction, edit_type, reference_audio_patches),
        reference_audio_patches,
        0,
        impl_->audio_id,
        impl_->audio_no_latent_id);
}

FireRedAudioPromptEncoding FireRedAudioTokenizer::encode_understanding_prompt(
    std::string_view prompt,
    int64_t audio_tokens,
    bool enable_thinking) const {
    return encode_prompt(
        *impl_->tokenizer,
        understanding_prompt(prompt, audio_tokens, enable_thinking),
        0,
        audio_tokens,
        impl_->audio_id,
        impl_->audio_no_latent_id);
}

std::string FireRedAudioTokenizer::decode(const std::vector<int32_t> & token_ids, bool skip_special_tokens) const {
    return impl_->tokenizer->decode(token_ids, skip_special_tokens);
}

}  // namespace engine::models::firered_audio
