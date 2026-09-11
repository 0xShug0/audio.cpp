#include "engine/models/vevo2/prompt_builder.h"

#include <sstream>

namespace engine::models::vevo2 {
namespace {

std::string format_chat_prompt(
    const std::string & system_content,
    const std::string & user_content,
    bool add_assistant_token) {
    std::string prompt;
    prompt += "<|im_start|>system\n";
    prompt += system_content;
    prompt += "<|im_end|>\n";
    prompt += "<|im_start|>user\n";
    prompt += user_content;
    prompt += "<|im_end|>\n";
    if (add_assistant_token) {
        prompt += "<|im_start|>assistant\n";
    }
    return prompt;
}

std::string content_style_token_text(const Vevo2TokenSequence & tokens) {
    std::ostringstream out;
    for (const int32_t id : tokens.ids) {
        out << "<|content_style_" << id << "|>";
    }
    return out.str();
}

std::string prosody_token_text(const Vevo2TokenSequence & tokens) {
    std::ostringstream out;
    for (const int32_t id : tokens.ids) {
        out << "<|prosody_" << id << "|>";
    }
    return out.str();
}

std::string build_text_prompt(const Vevo2Request & request) {
    const char * instruction = request.generation.use_prosody_code
        ? "User will provide you with a text. Please first generate a good prosodic instruction, then vocalize the text based on it."
        : "User will provide you with a text. Please vocalize it with natural expression.";
    const std::string input_text = request.refs.style_ref_text + " " + request.refs.target_text;
    return format_chat_prompt(instruction, input_text, true);
}

std::string build_prosody_prompt(
    const Vevo2Request & request,
    const Vevo2TokenSequence & prosody_tokens) {
    if (!request.generation.use_prosody_code) {
        return "";
    }
    return "<|prosody_start|>" + prosody_token_text(prosody_tokens) + "<|prosody_end|>";
}

std::string build_content_style_prompt(const Vevo2TokenSequence & style_content_tokens) {
    return "<|content_style_start|>" + content_style_token_text(style_content_tokens);
}

}  // namespace

Vevo2PromptParts build_vevo2_prompt_parts(
    const Vevo2Request & request,
    const Vevo2TokenSequence & prosody_tokens,
    const Vevo2TokenSequence & style_content_tokens) {
    Vevo2PromptParts parts;
    parts.text_prompt = build_text_prompt(request);
    parts.prosody_prompt = build_prosody_prompt(request, prosody_tokens);
    parts.content_style_prompt = build_content_style_prompt(style_content_tokens);
    parts.full_prompt = parts.text_prompt + parts.prosody_prompt + parts.content_style_prompt;
    return parts;
}

}  // namespace engine::models::vevo2
