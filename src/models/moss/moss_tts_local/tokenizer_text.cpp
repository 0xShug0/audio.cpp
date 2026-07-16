#include "engine/models/moss/moss_tts_local/tokenizer_text.h"

#include "engine/framework/tokenizers/llama_bpe.h"

#include <functional>
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

}  // namespace

struct MossTextProcessor::Impl {
    std::shared_ptr<const MossTTSLocalAssets> assets;
    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer;

    moss::TokenRowBuilder make_row_builder() const {
        return moss::TokenRowBuilder(
            assets->config.num_codebooks,
            static_cast<int32_t>(assets->config.audio_pad_token_id));
    }

    void push_text(moss::TokenRowBuilder & builder, const std::string & text) const {
        builder.push_text_tokens(tokenizer->encode(text));
    }

    // Emits the shared <user_inst> scaffold. `emit_reference` fills the "- Reference(s):"
    // slot: text-only passes "None"; the clone path emits the reference audio rows.
    MossGenerationPrefix build_prefix(
        const std::string & text,
        const std::optional<std::string> & language,
        const std::function<void(moss::TokenRowBuilder &)> & emit_reference) const {
        const auto & config = assets->config;
        moss::TokenRowBuilder builder = make_row_builder();
        builder.push_text_token(static_cast<int32_t>(config.im_start_token_id));
        push_text(builder, kUserRolePrefix);
        push_text(builder, kUserReferencePrefix);
        emit_reference(builder);
        push_text(builder, render_after_reference(language));
        push_text(builder, text);
        push_text(builder, kUserInstSuffix);
        builder.push_text_token(static_cast<int32_t>(config.im_end_token_id));
        push_text(builder, kAssistantTurnPrefix);
        builder.push_text_token(static_cast<int32_t>(config.im_start_token_id));
        push_text(builder, kAssistantRolePrefix);
        builder.push_text_token(static_cast<int32_t>(config.audio_start_token_id));

        return builder.finish();
    }
};

MossTextProcessor::MossTextProcessor(std::shared_ptr<const MossTTSLocalAssets> assets)
    : impl_(std::make_unique<Impl>()) {
    if (assets == nullptr) {
        throw std::runtime_error("MOSS-TTS-Local text tokenizer requires assets");
    }
    engine::tokenizers::LlamaBpeTokenizerSpec spec;
    spec.vocab_path = assets->resources.require_file("tokenizer_vocab");
    spec.merges_path = assets->resources.require_file("tokenizer_merges");
    spec.tokenizer_config_path = assets->resources.require_file("tokenizer_config");
    if (const auto * tokenizer_json = assets->resources.find_file("tokenizer_json")) {
        spec.tokenizer_json_path = *tokenizer_json;
    }
    spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen2;
    impl_->tokenizer = engine::tokenizers::load_llama_bpe_tokenizer(spec);
    impl_->assets = std::move(assets);
}

MossTextProcessor::~MossTextProcessor() = default;

MossGenerationPrefix MossTextProcessor::build_generation_prefix(
    const std::string & text,
    const std::optional<std::string> & language) const {
    return impl_->build_prefix(text, language, [this](moss::TokenRowBuilder & builder) {
        impl_->push_text(builder, kNoneValue);
    });
}

MossGenerationPrefix MossTextProcessor::build_clone_prefix(
    const std::string & text,
    const std::vector<std::vector<int32_t>> & reference_codes,
    const std::optional<std::string> & language) const {
    const auto & config = impl_->assets->config;
    const int64_t n_vq = config.num_codebooks;
    if (static_cast<int64_t>(reference_codes.size()) != n_vq) {
        throw std::runtime_error("MOSS-TTS-Local clone prefix expects num_codebooks reference rows");
    }
    const size_t frames = reference_codes.empty() ? 0 : reference_codes.front().size();
    if (frames == 0) {
        throw std::runtime_error("MOSS-TTS-Local clone prefix requires a non-empty reference");
    }
    for (const auto & row : reference_codes) {
        if (row.size() != frames) {
            throw std::runtime_error("MOSS-TTS-Local clone reference codebooks must be equal length");
        }
    }

    return impl_->build_prefix(text, language, [&](moss::TokenRowBuilder & builder) {
        // "- Reference(s):" slot -> audio_start, one audio_user_slot row per reference
        // frame carrying that frame's codes, then audio_end.
        builder.push_text_token(static_cast<int32_t>(config.audio_start_token_id));
        for (size_t frame = 0; frame < frames; ++frame) {
            builder.push_audio_row(
                static_cast<int32_t>(config.audio_user_slot_token_id),
                reference_codes,
                static_cast<int64_t>(frame));
        }
        builder.push_text_token(static_cast<int32_t>(config.audio_end_token_id));
    });
}

}  // namespace engine::models::moss_tts_local
