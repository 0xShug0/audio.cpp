#include "engine/models/fireredtts3/tokenizer_text.h"

#include "engine/framework/tokenizers/llama_bpe.h"

#include <array>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace engine::models::fireredtts3 {
namespace {

using engine::tokenizers::LlamaBpeAddedToken;

bool is_supported_language(std::string_view language) {
    return language == "Chinese" || language == "English";
}

constexpr int32_t kLatentInPadId = 151655;
constexpr int32_t kLatentOutPadId = 151656;
constexpr int32_t kFireRedAddedTokenStartId = 151669;

std::string repeated(std::string_view value, int64_t count) {
    if (count < 0) {
        throw std::runtime_error("FireRedTTS3 prompt placeholder count must be non-negative");
    }
    std::string out;
    out.reserve(value.size() * static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        out += value;
    }
    return out;
}

std::string convert_to_chatml(
    std::string_view text_in,
    int64_t latent_in_len,
    std::string_view text_out,
    int64_t latent_out_len) {
    std::string out;
    out += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
    out += "<|im_start|>user\n";
    if (latent_in_len > 0) {
        out += "<|sosp|>";
        out += repeated("<|image_pad|>", latent_in_len);
        out += "<|eosp|>\n";
    }
    out += std::string(text_in);
    out += " /no_think<|im_end|>\n";
    out += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    out += std::string(text_out);
    if (latent_out_len > 0) {
        out += "<|sosp|>";
        out += repeated("<|video_pad|>", latent_out_len);
        out += "<|eosp|>\n";
    }
    out += "<|im_end|>\n";
    return out;
}

void remove_suffix_or_throw(std::string & value, std::string_view suffix, const char * context) {
    if (value.size() < suffix.size() ||
        std::string_view(value).substr(value.size() - suffix.size()) != suffix) {
        throw std::runtime_error(std::string("FireRedTTS3 failed to trim ChatML suffix for ") + context);
    }
    value.erase(value.size() - suffix.size());
}

FireRedTTS3InstructTokens masks_from_tokens(std::vector<int32_t> tokens) {
    FireRedTTS3InstructTokens out;
    out.latent_in_mask.resize(tokens.size(), 0);
    out.latent_out_mask.resize(tokens.size(), 0);
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == kLatentInPadId) {
            out.latent_in_mask[i] = 1;
        }
        if (tokens[i] == kLatentOutPadId) {
            out.latent_out_mask[i] = 1;
        }
    }
    out.token_ids = std::move(tokens);
    if (out.token_ids.empty()) {
        throw std::runtime_error("FireRedTTS3 tokenizer produced empty token sequence");
    }
    return out;
}

std::vector<LlamaBpeAddedToken> firered_special_tokens() {
    std::vector<LlamaBpeAddedToken> tokens;
    tokens.reserve(15 + 192 + 24 + 21 + 3);
    int32_t token_id = kFireRedAddedTokenStartId;
    const auto add = [&tokens, &token_id](std::string content) {
        tokens.emplace_back(std::move(content), token_id++);
    };
    add("<|sosp|>");
    add("<|eosp|>");
    add("<|empty|>");
    add("<|Human|>");
    add("<|SpeechLM|>");
    add("<|sostm|>");
    add("<|eostm|>");
    add("<|sot|>");
    add("<|eot|>");
    add("<|TEXT_ONLY|>");
    add("<|AUDIO_ONLY|>");
    add("<|ASR|>");
    add("<|TTS|>");
    add("<|INTERLEAVE|>");
    add("<|UNDERSTANDING|>");
    for (int index = 1; index <= 192; ++index) {
        std::ostringstream token;
        token << "<|placeholder_" << std::setw(3) << std::setfill('0') << index << "|>";
        add(token.str());
    }
    constexpr std::array<std::string_view, 24> languages = {
        "<|Chinese|>",
        "<|English|>",
        "<|Cantonese|>",
        "<|Japanese|>",
        "<|Korean|>",
        "<|Spanish|>",
        "<|French|>",
        "<|Russian|>",
        "<|Arabic|>",
        "<|Turkish|>",
        "<|Indonesian|>",
        "<|Portuguese|>",
        "<|Italian|>",
        "<|Dutch|>",
        "<|Vietnamese|>",
        "<|German|>",
        "<|Ukrainian|>",
        "<|Thai|>",
        "<|Polish|>",
        "<|Romanian|>",
        "<|Greek|>",
        "<|Czech|>",
        "<|Finnish|>",
        "<|Hindi|>",
    };
    for (const auto language : languages) {
        add(std::string(language));
    }
    constexpr std::array<std::string_view, 21> dialects = {
        "<|ZH_Anhui|>",
        "<|ZH_Fujian|>",
        "<|ZH_Gansu|>",
        "<|ZH_Guizhou|>",
        "<|ZH_Hebei|>",
        "<|ZH_Henan|>",
        "<|ZH_Hubei|>",
        "<|ZH_Hunan|>",
        "<|ZH_Jiangxi|>",
        "<|ZH_Liaoning|>",
        "<|ZH_Minnan|>",
        "<|ZH_Ningxia|>",
        "<|ZH_Shaanxi|>",
        "<|ZH_Shandong|>",
        "<|ZH_Shanghai|>",
        "<|ZH_Shanxi|>",
        "<|ZH_Sichuan|>",
        "<|ZH_Tianjin|>",
        "<|ZH_Wenzhou|>",
        "<|ZH_Wu|>",
        "<|ZH_Yunnan|>",
    };
    for (const auto dialect : dialects) {
        add(std::string(dialect));
    }
    add("<|edit|>");
    add("<|frame_patch|>");
    add("<|end_edit|>");
    return tokens;
}

}  // namespace

class FireRedTTS3TextTokenizer::Impl {
public:
    explicit Impl(const FireRedTTS3Assets & assets) {
        engine::tokenizers::LlamaBpeTokenizerSpec spec;
        spec.tokenizer_json_path = assets.resources.require_file("tokenizer_json");
        spec.tokenizer_config_path = assets.resources.require_file("tokenizer_config");
        spec.additional_special_tokens = firered_special_tokens();
        spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen2;
        tokenizer = engine::tokenizers::load_llama_bpe_tokenizer(spec);
    }

    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer;
};

FireRedTTS3TextTokenizer::FireRedTTS3TextTokenizer(std::shared_ptr<const FireRedTTS3Assets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("FireRedTTS3 tokenizer requires assets");
    }
    impl_ = std::make_shared<Impl>(*assets);
}

FireRedTTS3TextTokenizer::~FireRedTTS3TextTokenizer() = default;

std::string FireRedTTS3TextTokenizer::build_base_prompt(
    std::string_view language,
    std::string_view reference_text,
    std::string_view text) const {
    if (!is_supported_language(language)) {
        throw std::runtime_error("FireRedTTS3 Base language must be Chinese or English");
    }
    return "<|" + std::string(language) + "|><|sot|>" +
        std::string(reference_text) + std::string(text) + "<|eot|>";
}

std::vector<int32_t> FireRedTTS3TextTokenizer::encode_base_prompt(
    std::string_view language,
    std::string_view reference_text,
    std::string_view text) const {
    auto prompt = build_base_prompt(language, reference_text, text);
    auto tokens = impl_->tokenizer->encode(prompt, true);
    if (tokens.empty()) {
        throw std::runtime_error("FireRedTTS3 tokenizer produced empty token sequence");
    }
    return tokens;
}

FireRedTTS3InstructTokens FireRedTTS3TextTokenizer::encode_instruct_clone_prompt(
    int64_t prompt_latent_patches,
    std::string_view reference_text,
    std::string_view text) const {
    std::string text_in = "Convert text to speech.\n";
    text_in += std::string(reference_text);
    text_in += std::string(text);
    auto prompt = convert_to_chatml(text_in, 0, "", prompt_latent_patches);
    remove_suffix_or_throw(prompt, "<|eosp|>\n<|im_end|>\n", "instruct clone");
    return masks_from_tokens(impl_->tokenizer->encode(prompt, true));
}

FireRedTTS3InstructTokens FireRedTTS3TextTokenizer::encode_voice_design_prompt(
    std::string_view instruction,
    std::string_view text) const {
    std::string text_in = std::string(instruction);
    text_in += "\n\n根据上述音色描述，首先整理成语音属性，再合成以下文本对应的音频：\n";
    text_in += std::string(text);
    auto prompt = convert_to_chatml(text_in, 0, "<|sot|>", 0);
    remove_suffix_or_throw(prompt, "<|im_end|>\n", "voice design");
    return masks_from_tokens(impl_->tokenizer->encode(prompt, true));
}

FireRedTTS3InstructTokens FireRedTTS3TextTokenizer::encode_semantic_edit_prompt(
    int64_t input_latent_patches,
    std::string_view instruction) const {
    std::string text_in = "Identify the content of the audio. ";
    text_in += std::string(instruction);
    auto prompt = convert_to_chatml(text_in, input_latent_patches, "<|sot|>", 0);
    remove_suffix_or_throw(prompt, "<|im_end|>\n", "semantic edit");
    return masks_from_tokens(impl_->tokenizer->encode(prompt, true));
}

FireRedTTS3InstructTokens FireRedTTS3TextTokenizer::encode_acoustic_edit_prompt(
    int64_t input_latent_patches,
    std::string_view instruction) const {
    auto prompt = convert_to_chatml(instruction, input_latent_patches, "", 1);
    remove_suffix_or_throw(prompt, "<|video_pad|><|eosp|>\n<|im_end|>\n", "acoustic edit");
    return masks_from_tokens(impl_->tokenizer->encode(prompt, true));
}

std::string FireRedTTS3TextTokenizer::decode(const std::vector<int32_t> & token_ids, bool skip_special_tokens) const {
    return impl_->tokenizer->decode(token_ids, skip_special_tokens);
}

}  // namespace engine::models::fireredtts3
