#include "engine/models/moss/moss_tts_nano/prompt_builder.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::models::moss_tts_nano {
namespace {

constexpr const char * kUserRolePrefix = "user\n";
constexpr const char * kUserTemplateReferencePrefix = "<user_inst>\n- Reference(s):\n";
constexpr const char * kUserTemplateAfterReference =
    "\n- Instruction:\nNone\n"
    "- Tokens:\nNone\n"
    "- Quality:\nNone\n"
    "- Sound Event:\nNone\n"
    "- Ambient Sound:\nNone\n"
    "- Language:\nNone\n"
    "- Text:\n";
constexpr const char * kUserTemplateSuffix = "\n</user_inst>";
constexpr const char * kAssistantTurnPrefix = "\n";
constexpr const char * kAssistantRolePrefix = "assistant\n";

void append_text_tokens(std::vector<int32_t> & out, const MossTTSNanoTextTokenizer & tokenizer, const char * text) {
    const auto tokens = tokenizer.encode(text);
    out.insert(out.end(), tokens.begin(), tokens.end());
}

void append_text_row(MossTTSNanoPrompt & prompt, int32_t text_token, int64_t audio_pad) {
    prompt.input_ids.push_back(text_token);
    for (int64_t i = 0; i < prompt.row_width - 1; ++i) {
        prompt.input_ids.push_back(static_cast<int32_t>(audio_pad));
    }
    prompt.attention_mask.push_back(1);
    ++prompt.rows;
}

void append_text_rows(MossTTSNanoPrompt & prompt, const std::vector<int32_t> & text_tokens, int64_t audio_pad) {
    for (const int32_t token : text_tokens) {
        append_text_row(prompt, token, audio_pad);
    }
}

void append_audio_rows(
    MossTTSNanoPrompt & prompt,
    const MossTTSNanoAudioCodes & codes,
    int64_t text_slot_token,
    int64_t audio_pad) {
    if (codes.codebooks != prompt.row_width - 1) {
        throw std::runtime_error("MOSS-TTS-Nano prompt audio codebook count does not match prompt row width");
    }
    for (int64_t frame = 0; frame < codes.frames; ++frame) {
        prompt.input_ids.push_back(static_cast<int32_t>(text_slot_token));
        for (int64_t codebook = 0; codebook < codes.codebooks; ++codebook) {
            const auto index = static_cast<size_t>(frame * codes.codebooks + codebook);
            const int32_t code = index < codes.token_ids.size() ? codes.token_ids[index] : static_cast<int32_t>(audio_pad);
            prompt.input_ids.push_back(code);
        }
        prompt.attention_mask.push_back(1);
        ++prompt.rows;
    }
}

std::vector<int32_t> build_user_prompt_prefix(
    const MossTTSNanoConfig & config,
    const MossTTSNanoTextTokenizer & tokenizer) {
    std::vector<int32_t> tokens{static_cast<int32_t>(config.im_start_token_id)};
    append_text_tokens(tokens, tokenizer, kUserRolePrefix);
    append_text_tokens(tokens, tokenizer, kUserTemplateReferencePrefix);
    return tokens;
}

std::vector<int32_t> build_assistant_prompt_prefix(
    const MossTTSNanoConfig & config,
    const MossTTSNanoTextTokenizer & tokenizer) {
    std::vector<int32_t> tokens;
    append_text_tokens(tokens, tokenizer, kUserTemplateSuffix);
    tokens.push_back(static_cast<int32_t>(config.im_end_token_id));
    append_text_tokens(tokens, tokenizer, kAssistantTurnPrefix);
    tokens.push_back(static_cast<int32_t>(config.im_start_token_id));
    append_text_tokens(tokens, tokenizer, kAssistantRolePrefix);
    return tokens;
}

}  // namespace

MossTTSNanoPromptBuilder::MossTTSNanoPromptBuilder(
    std::shared_ptr<const MossTTSNanoAssets> assets,
    const MossTTSNanoTextTokenizer & tokenizer)
    : assets_(std::move(assets)),
      tokenizer_(tokenizer) {
    if (assets_ == nullptr) {
        throw std::runtime_error("MOSS-TTS-Nano prompt builder requires assets");
    }
}

MossTTSNanoPrompt MossTTSNanoPromptBuilder::build(
    const MossTTSNanoRequest & request,
    const MossTTSNanoAudioCodes * prompt_codes) const {
    const auto & config = assets_->config;
    MossTTSNanoPrompt prompt;
    prompt.row_width = config.n_vq + 1;
    if (request.text.empty()) {
        throw std::runtime_error("MOSS-TTS-Nano prompt requires target text");
    }
    if (prompt_codes == nullptr) {
        if (!request.prompt_text.empty() || request.has_prompt_audio) {
            throw std::runtime_error("MOSS-TTS-Nano continuation prompt requires matching reference audio codes");
        }
        auto tokens = build_user_prompt_prefix(config, tokenizer_);
        append_text_tokens(tokens, tokenizer_, "None");
        append_text_tokens(tokens, tokenizer_, kUserTemplateAfterReference);
        const auto text_tokens = tokenizer_.encode(request.text);
        tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());
        const auto assistant = build_assistant_prompt_prefix(config, tokenizer_);
        tokens.insert(tokens.end(), assistant.begin(), assistant.end());
        tokens.push_back(static_cast<int32_t>(config.audio_start_token_id));
        append_text_rows(prompt, tokens, config.audio_pad_token_id);
        if (prompt.rows <= 0 || static_cast<int64_t>(prompt.input_ids.size()) != prompt.rows * prompt.row_width) {
            throw std::runtime_error("MOSS-TTS-Nano prompt builder produced invalid input shape");
        }
        return prompt;
    }
    auto prefix = build_user_prompt_prefix(config, tokenizer_);
    prefix.push_back(static_cast<int32_t>(config.audio_start_token_id));
    append_text_rows(prompt, prefix, config.audio_pad_token_id);
    append_audio_rows(prompt, *prompt_codes, config.audio_user_slot_token_id, config.audio_pad_token_id);
    std::vector<int32_t> suffix{static_cast<int32_t>(config.audio_end_token_id)};
    append_text_tokens(suffix, tokenizer_, kUserTemplateAfterReference);
    const auto text_tokens = tokenizer_.encode(request.text);
    suffix.insert(suffix.end(), text_tokens.begin(), text_tokens.end());
    const auto assistant = build_assistant_prompt_prefix(config, tokenizer_);
    suffix.insert(suffix.end(), assistant.begin(), assistant.end());
    suffix.push_back(static_cast<int32_t>(config.audio_start_token_id));
    append_text_rows(prompt, suffix, config.audio_pad_token_id);
    if (prompt.rows <= 0 || static_cast<int64_t>(prompt.input_ids.size()) != prompt.rows * prompt.row_width) {
        throw std::runtime_error("MOSS-TTS-Nano prompt builder produced invalid input shape");
    }
    return prompt;
}

}  // namespace engine::models::moss_tts_nano
