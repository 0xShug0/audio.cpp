#include "engine/models/moss_tts_local/processor.h"

#include "engine/framework/tokenizers/llama_bpe.h"

#include <stdexcept>
#include <utility>

namespace engine::models::moss_tts_local {
namespace {

// User-template fragments, copied verbatim from MossTTSLocalProcessor so the encoded
// prompt matches the reference token-for-token.
constexpr const char * kUserRolePrefix = "user\n";
constexpr const char * kUserReferencePrefix = "<user_inst>\n- Reference(s):\n";
constexpr const char * kUserTextSuffix = "\n- Text:\n";
constexpr const char * kUserInstSuffix = "\n</user_inst>";
constexpr const char * kAssistantTurnPrefix = "\n";
constexpr const char * kAssistantRolePrefix = "assistant\n";
constexpr const char * kNoneValue = "None";

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

// Mirrors _render_user_prompt_after_reference for the text-only request: every optional
// field defaults to "None" and only the language slot can carry a caller value.
std::string render_after_reference(const std::optional<std::string> & language) {
    return std::string("\n- Instruction:\n") + kNoneValue
        + "\n- Tokens:\n" + kNoneValue
        + "\n- Quality:\n" + kNoneValue
        + "\n- Sound Event:\n" + kNoneValue
        + "\n- Ambient Sound:\n" + kNoneValue
        + "\n- Language:\n" + normalize_template_value(language)
        + kUserTextSuffix;
}

std::filesystem::path require_path(
    const std::optional<std::filesystem::path> & path,
    const char * label) {
    if (!path.has_value()) {
        throw std::runtime_error(std::string("MOSS-TTS-Local processor requires ") + label);
    }
    return *path;
}

}  // namespace

struct MossTextProcessor::Impl {
    std::shared_ptr<const MossTTSLocalAssets> assets;
    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer;

    void append_encoded(std::vector<int32_t> & tokens, const std::string & text) const {
        const auto ids = tokenizer->encode(text);
        tokens.insert(tokens.end(), ids.begin(), ids.end());
    }
};

MossTextProcessor::MossTextProcessor(std::shared_ptr<const MossTTSLocalAssets> assets)
    : impl_(std::make_unique<Impl>()) {
    if (assets == nullptr) {
        throw std::runtime_error("MOSS-TTS-Local processor requires assets");
    }
    engine::tokenizers::LlamaBpeTokenizerSpec spec;
    spec.vocab_path = require_path(assets->paths.tokenizer_vocab_path, "vocab.json");
    spec.merges_path = require_path(assets->paths.tokenizer_merges_path, "merges.txt");
    spec.tokenizer_config_path = require_path(assets->paths.tokenizer_config_path, "tokenizer_config.json");
    spec.tokenizer_json_path = assets->paths.tokenizer_json_path;
    spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen2;
    impl_->tokenizer = engine::tokenizers::load_llama_bpe_tokenizer(spec);
    impl_->assets = std::move(assets);
}

MossTextProcessor::~MossTextProcessor() = default;

MossGenerationPrefix MossTextProcessor::build_generation_prefix(
    const std::string & text,
    const std::optional<std::string> & language) const {
    const auto & config = impl_->assets->config;

    std::vector<int32_t> tokens;
    tokens.push_back(static_cast<int32_t>(config.im_start_token_id));
    impl_->append_encoded(tokens, kUserRolePrefix);
    impl_->append_encoded(tokens, kUserReferencePrefix);
    impl_->append_encoded(tokens, kNoneValue);
    impl_->append_encoded(tokens, render_after_reference(language));
    impl_->append_encoded(tokens, text);
    impl_->append_encoded(tokens, kUserInstSuffix);
    tokens.push_back(static_cast<int32_t>(config.im_end_token_id));
    impl_->append_encoded(tokens, kAssistantTurnPrefix);
    tokens.push_back(static_cast<int32_t>(config.im_start_token_id));
    impl_->append_encoded(tokens, kAssistantRolePrefix);
    tokens.push_back(static_cast<int32_t>(config.audio_start_token_id));

    MossGenerationPrefix prefix;
    prefix.text_tokens = std::move(tokens);
    prefix.audio_codes.assign(
        prefix.text_tokens.size() * static_cast<size_t>(config.num_codebooks),
        static_cast<int32_t>(config.audio_pad_token_id));
    return prefix;
}

}  // namespace engine::models::moss_tts_local
