#include "engine/community_models/glm_tts/prompt.h"

#include <cctype>
#include <limits>
#include <stdexcept>

namespace engine::models::glm_tts {
namespace {

bool ends_with_sentence_punctuation(const std::string & text) {
    if (text.empty()) return false;
    const unsigned char last =
        static_cast<unsigned char>(text.back());
    if (last == '.' || last == '?' || last == '!' ||
        last == ';' || last == ':') {
        return true;
    }
    static constexpr const char * kUtf8Punctuation[] = {
        "\xE3\x80\x82",  // 。
        "\xEF\xBC\x9F",  // ？
        "\xEF\xBC\x81",  // ！
        "\xEF\xBC\x9B",  // ；
        "\xEF\xBC\x9A",  // ：
    };
    for (const char * punctuation : kUtf8Punctuation) {
        const size_t length = std::char_traits<char>::length(punctuation);
        if (text.size() >= length &&
            text.compare(text.size() - length, length, punctuation) == 0) {
            return true;
        }
    }
    return false;
}

std::string normalize_text(std::string text) {
    std::string out;
    out.reserve(text.size() + 1);
    bool pending_space = false;
    for (unsigned char value : text) {
        if (std::isspace(value)) {
            pending_space = !out.empty();
            continue;
        }
        if (pending_space) {
            out.push_back(' ');
            pending_space = false;
        }
        out.push_back(
            value < 128
                ? static_cast<char>(std::tolower(value))
                : static_cast<char>(value));
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    if (!out.empty() && !ends_with_sentence_punctuation(out)) {
        out.push_back('.');
    }
    return out;
}

}  // namespace

GlmTTSPrompt build_glm_tts_prompt(
    const GlmTTSConfig & config,
    const GlmTTSTextTokenizer & tokenizer,
    const std::string & reference_text,
    const std::string & text,
    const std::vector<int32_t> & reference_speech_tokens) {
    if (reference_text.empty()) {
        throw std::runtime_error(
            "GLM-TTS voice cloning requires reference_text");
    }
    if (text.empty()) {
        throw std::runtime_error("GLM-TTS synthesis text must not be empty");
    }
    if (reference_speech_tokens.empty()) {
        throw std::runtime_error(
            "GLM-TTS voice cloning requires reference speech tokens");
    }

    std::string normalized_reference = normalize_text(reference_text);
    const std::string normalized_text = normalize_text(text);
    // Upstream tokenizes the normalized reference transcript with one
    // trailing space so the target text begins at a clean token boundary.
    normalized_reference.push_back(' ');
    const auto reference_text_ids = tokenizer.encode(normalized_reference);
    const auto text_ids = tokenizer.encode(normalized_text);
    if (reference_text_ids.empty() || text_ids.empty()) {
        throw std::runtime_error(
            "GLM-TTS tokenizer produced an empty text prompt");
    }

    GlmTTSPrompt out;
    out.text_token_count = static_cast<int64_t>(text_ids.size());
    if (out.text_token_count >
        std::numeric_limits<int64_t>::max() / 20) {
        throw std::runtime_error("GLM-TTS text token count is too large");
    }
    out.minimum_audio_tokens = out.text_token_count * 2;
    out.maximum_audio_tokens = out.text_token_count * 20;
    out.input_ids.reserve(
        reference_text_ids.size() + text_ids.size() + 1 +
        reference_speech_tokens.size());
    out.input_ids.insert(
        out.input_ids.end(),
        reference_text_ids.begin(),
        reference_text_ids.end());
    out.input_ids.insert(out.input_ids.end(), text_ids.begin(), text_ids.end());
    out.input_ids.push_back(static_cast<int32_t>(config.begin_audio_token));
    for (const int32_t token : reference_speech_tokens) {
        if (token < 0 ||
            static_cast<int64_t>(token) >
                config.audio_token_end - config.audio_token_start) {
            throw std::runtime_error(
                "GLM-TTS reference speech token is outside the VQ range");
        }
        out.input_ids.push_back(static_cast<int32_t>(
            config.audio_token_start + static_cast<int64_t>(token)));
    }
    if (static_cast<int64_t>(out.input_ids.size()) >=
        config.llama.max_position_embeddings) {
        throw std::runtime_error(
            "GLM-TTS voice prompt exceeds the Llama context window");
    }
    return out;
}

}  // namespace engine::models::glm_tts
