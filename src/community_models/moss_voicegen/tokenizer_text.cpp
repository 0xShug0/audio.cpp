#include "engine/community_models/moss_voicegen/tokenizer_text.h"

#include "engine/framework/tokenizers/llama_bpe.h"

#include <stdexcept>
#include <utility>

namespace engine::models::moss_voicegen {
namespace {

// Template fragments copied verbatim from MossTTSDelayProcessor so the encoded prompt
// matches the reference token-for-token.
constexpr const char * kUserRolePrefix = "user\n";
constexpr const char * kUserReferencePrefix = "<user_inst>\n- Reference(s):\n";
constexpr const char * kUserTextSuffix = "\n- Text:\n";
constexpr const char * kUserInstSuffix = "\n</user_inst>";
constexpr const char * kAssistantTurnPrefix = "\n";
constexpr const char * kAssistantRolePrefix = "assistant\n";
constexpr const char * kNoneValue = "None";
constexpr const char * kImStartToken = "<|im_start|>";
constexpr const char * kImEndToken = "<|im_end|>";

std::string normalize_template_value(const std::optional<std::string> & value) {
    if (!value.has_value()) {
        return kNoneValue;
    }
    std::string resolved = *value;
    const auto first = resolved.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return kNoneValue;
    }
    const auto last = resolved.find_last_not_of(" \t\r\n");
    resolved = resolved.substr(first, last - first + 1);
    return resolved.empty() ? kNoneValue : resolved;
}

// The fields between "- Reference(s):" and the target text. Voice design fills the
// instruction and the language; the remaining control slots stay "None".
std::string render_after_reference(
    const std::optional<std::string> & instruction,
    const std::optional<std::string> & language) {
    return std::string("\n- Instruction:\n") + normalize_template_value(instruction)
        + "\n- Tokens:\n" + kNoneValue
        + "\n- Quality:\n" + kNoneValue
        + "\n- Sound Event:\n" + kNoneValue
        + "\n- Ambient Sound:\n" + kNoneValue
        + "\n- Language:\n" + normalize_template_value(language)
        + kUserTextSuffix;
}

}  // namespace

struct MossVoiceGenTextProcessor::Impl {
    std::shared_ptr<const MossVoiceGenAssets> assets;
    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer;
    int32_t im_start_token_id = 0;
    int32_t im_end_token_id = 0;
};

MossVoiceGenTextProcessor::MossVoiceGenTextProcessor(std::shared_ptr<const MossVoiceGenAssets> assets)
    : impl_(std::make_unique<Impl>()) {
    if (assets == nullptr) {
        throw std::runtime_error("MOSS-VoiceGenerator text tokenizer requires assets");
    }
    engine::tokenizers::LlamaBpeTokenizerSpec spec;
    // The checkpoint ships tokenizer.json and merges.txt but no vocab.json. The framework
    // treats vocab/merges as a pair, so leave both unset and take the vocabulary and the
    // merge ranks from tokenizer.json, which carries them.
    spec.tokenizer_config_path = assets->resources.require_file("tokenizer_config");
    spec.tokenizer_json_path = assets->resources.require_file("tokenizer_json");
    spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen2;
    impl_->tokenizer = engine::tokenizers::load_llama_bpe_tokenizer(spec);

    // The config does not carry the chat-control ids; the reference processor resolves
    // them through the tokenizer, so do the same rather than hardcoding Qwen's values.
    const auto im_start = impl_->tokenizer->find_token_id(kImStartToken);
    const auto im_end = impl_->tokenizer->find_token_id(kImEndToken);
    if (!im_start.has_value() || !im_end.has_value()) {
        throw std::runtime_error("MOSS-VoiceGenerator tokenizer is missing the chat control tokens");
    }
    impl_->im_start_token_id = *im_start;
    impl_->im_end_token_id = *im_end;
    impl_->assets = std::move(assets);
}

MossVoiceGenTextProcessor::~MossVoiceGenTextProcessor() = default;

moss::TokenRows MossVoiceGenTextProcessor::build_generation_prefix(
    const std::string & text,
    const std::optional<std::string> & instruction,
    const std::optional<std::string> & language) const {
    const auto & config = impl_->assets->config;
    moss::TokenRowBuilder builder(config.num_codebooks, static_cast<int32_t>(config.audio_pad_code));

    // Render the whole turn as one string and encode it in a single pass, the way the
    // reference processor does. Encoding the fragments separately would split merges
    // across the seams — the text's trailing "." and the suffix's "\n" are one token in
    // the reference, two if the suffix is encoded on its own.
    const std::string prompt = std::string(kImStartToken) + kUserRolePrefix
        + kUserReferencePrefix + kNoneValue
        + render_after_reference(instruction, language)
        + text
        + kUserInstSuffix + kImEndToken
        + kAssistantTurnPrefix + kImStartToken + kAssistantRolePrefix;
    builder.push_text_tokens(impl_->tokenizer->encode(prompt, true));
    // Unlike moss_tts_local, the delay family does not seed the audio start token: the
    // model emits it itself on the first step, and generate() keys "is this a
    // continuation" off the last text token, so appending it here would be read as a
    // continuation prompt.

    return builder.finish();
}

}  // namespace engine::models::moss_voicegen
