#include "engine/community_models/granite_speech5/prompt.h"

#include "engine/framework/io/json.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/io/text.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <utility>

namespace engine::community_models::granite_speech5 {
namespace json = engine::io::json;

// Pinned tokenizer.json pre-tokenizer (verbatim from the checkpoint; the C1
// unit test asserts framework encode == Python input_ids exactly, which pins
// this choice stronger than any string compare).
namespace {

constexpr const char * kPinnedSplitPattern =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| "
    "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

constexpr const char * kPinnedChatTemplate =
    "{% for message in messages %}{% if message['role'] == 'user' %}USER: "
    "{{ message['content'] }}\n ASSISTANT:{% elif message['role'] == "
    "'assistant' %}{{ message['content'] }}{% endif %}{% endfor %}";

}  // namespace

// Defined below; the loader calls it first so a drifted bundle fails closed.
void check_granite_speech5_prompt_assets(const assets::ResourceBundle & resources);

std::string granite_speech5_normalize_language_code(std::string_view language) {
    size_t begin = 0;
    while (begin < language.size() &&
           (language[begin] == ' ' || language[begin] == '\t' || language[begin] == '\n' ||
            language[begin] == '\r')) {
        ++begin;
    }
    size_t end = language.size();
    while (end > begin &&
           (language[end - 1] == ' ' || language[end - 1] == '\t' || language[end - 1] == '\n' ||
            language[end - 1] == '\r')) {
        --end;
    }
    std::string code;
    code.reserve(end - begin);
    for (size_t i = begin; i < end; ++i) {
        const unsigned char uch = static_cast<unsigned char>(language[i]);
        if (uch > 127) {
            throw std::runtime_error("granite_speech5 language hint must be ASCII");
        }
        code.push_back(static_cast<char>(std::tolower(uch)));
    }
    if (code.empty()) {
        return code;  // blank: automatic recognition, no hint text
    }
    (void) granite_speech5_language_name(code);  // rejects unsupported codes
    return code;
}

std::string_view granite_speech5_language_name(std::string_view normalized_code) {
    if (normalized_code == "en") {
        return "English";
    }
    if (normalized_code == "fr") {
        return "French";
    }
    if (normalized_code == "de") {
        return "German";
    }
    if (normalized_code == "es") {
        return "Spanish";
    }
    if (normalized_code == "pt") {
        return "Portuguese";
    }
    if (normalized_code == "ja") {
        return "Japanese";
    }
    throw runtime::InvalidRequestError(
        "granite_speech5 unsupported language hint: '" + std::string(normalized_code) + "'");
}

std::string granite_speech5_build_user_content(std::string_view normalized_code) {
    if (normalized_code.empty()) {
        return std::string(kGraniteSpeech5AudioMarker) + kGraniteSpeech5TranscribeInstruction;
    }
    const std::string_view name = granite_speech5_language_name(normalized_code);
    std::string content(kGraniteSpeech5HintPrefix);
    content += name;
    content += kGraniteSpeech5HintSuffix;
    content += kGraniteSpeech5AudioMarker;
    content += kGraniteSpeech5TranscribeInstruction;
    return content;
}

std::string granite_speech5_normalize_target_language_code(std::string_view language) {
    // Shared trim helper (C1/D ASCII whitespace rule reused for F codes).
    const std::string code = engine::io::trim_ascii_whitespace(std::string(language));
    std::string lowered;
    lowered.reserve(code.size());
    for (const char ch : code) {
        const unsigned char uch = static_cast<unsigned char>(ch);
        if (uch > 127) {
            throw std::runtime_error("granite_speech5 target language must be ASCII");
        }
        lowered.push_back(static_cast<char>(std::tolower(uch)));
    }
    if (lowered.empty()) {
        return lowered;  // blank: ASR, no translation target
    }
    (void) granite_speech5_target_language_name(lowered);  // rejects unsupported codes
    return lowered;
}

std::string_view granite_speech5_target_language_name(std::string_view normalized_code) {
    if (normalized_code == "en") {
        return "English";
    }
    if (normalized_code == "fr") {
        return "French";
    }
    if (normalized_code == "de") {
        return "German";
    }
    if (normalized_code == "es") {
        return "Spanish";
    }
    if (normalized_code == "pt") {
        return "Portuguese";
    }
    if (normalized_code == "ja") {
        return "Japanese";
    }
    if (normalized_code == "it") {
        return "Italian";
    }
    if (normalized_code == "zh") {
        return "Mandarin";
    }
    throw runtime::InvalidRequestError(
        "granite_speech5 unsupported target language: '" + std::string(normalized_code) + "'");
}

std::vector<std::string> granite_speech5_parse_keyword_terms(
    std::string_view keywords_text) {
    std::vector<std::string> terms;
    size_t start = 0;
    while (start <= keywords_text.size()) {
        size_t comma = keywords_text.find(',', start);
        if (comma == std::string_view::npos) {
            comma = keywords_text.size();
        }
        std::string term = engine::io::trim_ascii_whitespace(
            std::string(keywords_text.substr(start, comma - start)));
        if (!term.empty()) {
            terms.push_back(std::move(term));
        }
        start = comma + 1;
    }
    return terms;
}

std::string granite_speech5_join_keyword_terms(
    const std::vector<std::string> & terms) {
    std::string joined;
    for (size_t i = 0; i < terms.size(); ++i) {
        if (i > 0) {
            joined += ", ";
        }
        joined += terms[i];
    }
    return joined;
}

std::string granite_speech5_build_keyword_user_content(
    std::string_view normalized_source_code,
    const std::vector<std::string> & keyword_terms) {
    if (keyword_terms.empty()) {
        throw std::runtime_error("granite_speech5 keyword list is empty");
    }
    // Reuse the C1 hint validation (rejects unsupported source codes).
    std::string content;
    if (!normalized_source_code.empty()) {
        const std::string_view name = granite_speech5_language_name(normalized_source_code);
        content += kGraniteSpeech5HintPrefix;
        content += name;
        content += kGraniteSpeech5HintSuffix;
    }
    content += kGraniteSpeech5AudioMarker;
    content += kGraniteSpeech5KeywordInstruction;
    content += granite_speech5_join_keyword_terms(keyword_terms);
    return content;
}

std::string granite_speech5_build_translation_user_content(
    std::string_view normalized_source_code,
    std::string_view normalized_target_code,
    const std::vector<std::string> & keyword_terms) {
    const std::string_view target_name =
        granite_speech5_target_language_name(normalized_target_code);
    std::string content;
    if (!normalized_source_code.empty()) {
        // Bare source prefix: the C1 do-not-translate clause would contradict
        // the translation instruction, so only the language name is stated.
        const std::string_view source_name =
            granite_speech5_language_name(normalized_source_code);
        content += kGraniteSpeech5HintPrefix;
        content += source_name;
        content += kGraniteSpeech5TranslationHintSuffix;
    }
    content += kGraniteSpeech5AudioMarker;
    content += kGraniteSpeech5TranslationInstructionPrefix;
    content += target_name;
    content += kGraniteSpeech5TranslationInstructionSuffix;
    if (!keyword_terms.empty()) {
        content += kGraniteSpeech5TranslationKeywordsPrefix;
        content += granite_speech5_join_keyword_terms(keyword_terms);
    }
    return content;
}

std::vector<std::string> granite_speech5_reserved_markers_from_tokenizer_json(
    const std::filesystem::path & tokenizer_json_path) {
    const auto tokenizer_json = json::parse_file(tokenizer_json_path);
    const auto * added = tokenizer_json.find("added_tokens");
    if (added == nullptr || !added->is_array()) {
        throw std::runtime_error("granite_speech5 tokenizer.json has no added_tokens array");
    }
    std::vector<std::string> markers;
    for (const auto & token_config : added->as_array()) {
        const auto * content = token_config.find("content");
        const auto * special = token_config.find("special");
        if (content == nullptr || !content->is_string()) {
            continue;
        }
        if (special != nullptr && special->is_bool() && special->as_bool()) {
            markers.push_back(content->as_string());
        }
    }
    if (markers.empty()) {
        throw std::runtime_error("granite_speech5 tokenizer.json lists no special markers");
    }
    return markers;
}

void granite_speech5_require_no_reserved_markers(
    std::string_view text, const std::vector<std::string> & reserved_markers) {
    for (const auto & marker : reserved_markers) {
        if (!marker.empty() && text.find(marker) != std::string_view::npos) {
            throw std::runtime_error(
                "granite_speech5 text contains reserved marker: '" + marker + "'");
        }
    }
}

std::string granite_speech5_render_single_turn_prompt(std::string_view user_content) {
    std::string prompt(kGraniteSpeech5PromptPrefix);
    prompt += user_content;
    prompt += kGraniteSpeech5PromptSuffix;
    return prompt;
}

std::shared_ptr<tokenizers::LlamaBpeTokenizer> load_granite_speech5_tokenizer(
    const assets::ResourceBundle & resources) {
    // Fail closed at load: verify the staged template + tokenizer settings
    // before binding, so C2/D call sites cannot skip the drift check.
    check_granite_speech5_prompt_assets(resources);
    tokenizers::LlamaBpeTokenizerSpec spec;
    spec.tokenizer_config_path = resources.require_file("tokenizer_config");
    if (const auto * path = resources.find_file("tokenizer_json")) {
        spec.tokenizer_json_path = *path;
    } else {
        throw std::runtime_error("granite_speech5 resources lack tokenizer_json");
    }
    // Proven equal to the pinned HF split pattern (see header + C1 test).
    spec.pre_type = tokenizers::LlamaBpePreTokenizer::Llama3;
    auto tokenizer = tokenizers::load_llama_bpe_tokenizer(spec);
    const auto audio_id = tokenizer->find_token_id(kGraniteSpeech5AudioMarker);
    if (!audio_id.has_value() || *audio_id != kGraniteSpeech5AudioTokenId) {
        throw std::runtime_error(
            "granite_speech5 tokenizer audio marker does not resolve to " +
            std::to_string(kGraniteSpeech5AudioTokenId));
    }
    if (!tokenizer->is_control_token_id(kGraniteSpeech5AudioTokenId)) {
        throw std::runtime_error("granite_speech5 audio marker is not a control token");
    }
    // tokenizer_config pad must agree with the pinned generation pad: a
    // drifted pad would silently change any padded batch the probe stages.
    const auto pad_id = tokenizer->configured_pad_token_id();
    if (!pad_id.has_value() || *pad_id != kGraniteSpeech5PadTokenId) {
        throw std::runtime_error("granite_speech5 tokenizer pad does not resolve to " +
                                 std::to_string(kGraniteSpeech5PadTokenId));
    }
    return tokenizer;
}

void check_granite_speech5_prompt_assets(const assets::ResourceBundle & resources) {
    const auto template_text = resources.read_text("chat_template");
    if (template_text != kPinnedChatTemplate) {
        throw std::runtime_error(
            "granite_speech5 chat template drifted from the pinned single-turn shape");
    }
    const auto tokenizer_json = resources.parse_json("tokenizer_json");
    const auto & model = tokenizer_json.require("model");
    if (model.require("type").as_string() != "BPE") {
        throw std::runtime_error("granite_speech5 tokenizer model is not BPE");
    }
    if (const auto * ignore_merges = model.find("ignore_merges");
        ignore_merges != nullptr && ignore_merges->is_bool() && ignore_merges->as_bool()) {
        throw std::runtime_error("granite_speech5 tokenizer requires ignore_merges=false");
    }
    if (model.find("byte_fallback") == nullptr ||
        model.require("byte_fallback").as_bool()) {
        throw std::runtime_error("granite_speech5 tokenizer requires byte_fallback=false");
    }
    if (tokenizer_json.find("normalizer") == nullptr ||
        !tokenizer_json.find("normalizer")->is_null()) {
        throw std::runtime_error("granite_speech5 tokenizer must have no normalizer");
    }
    if (tokenizer_json.find("post_processor") == nullptr ||
        !tokenizer_json.find("post_processor")->is_null()) {
        throw std::runtime_error("granite_speech5 tokenizer must have no post-processor");
    }
    const auto & pre_tokenizer = tokenizer_json.require("pre_tokenizer");
    if (pre_tokenizer.require("type").as_string() != "Sequence") {
        throw std::runtime_error("granite_speech5 pre-tokenizer is not a Sequence");
    }
    const auto & steps = pre_tokenizer.require("pretokenizers").as_array();
    if (steps.size() != 2 || steps[0].require("type").as_string() != "Split" ||
        steps[1].require("type").as_string() != "ByteLevel") {
        throw std::runtime_error("granite_speech5 pre-tokenizer is not Split+ByteLevel");
    }
    const auto & split = steps[0].require("pattern");
    if (split.require("Regex").as_string() != kPinnedSplitPattern) {
        throw std::runtime_error("granite_speech5 Split pattern drifted from the pinned regex");
    }
    if (steps[1].find("add_prefix_space") == nullptr ||
        steps[1].require("add_prefix_space").as_bool()) {
        throw std::runtime_error("granite_speech5 ByteLevel must not add a prefix space");
    }
    if (steps[1].find("use_regex") == nullptr ||
        steps[1].require("use_regex").as_bool()) {
        throw std::runtime_error("granite_speech5 ByteLevel must not use regex splitting");
    }
    const auto * added = tokenizer_json.find("added_tokens");
    if (added == nullptr || !added->is_array()) {
        throw std::runtime_error("granite_speech5 tokenizer.json has no added_tokens array");
    }
    bool audio_ok = false;
    for (const auto & token_config : added->as_array()) {
        const auto * content = token_config.find("content");
        const auto * id = token_config.find("id");
        if (content == nullptr || !content->is_string() || id == nullptr || !id->is_number()) {
            continue;
        }
        // Every added token must be whitespace-neutral: the encoder splits
        // on these strings verbatim, so an lstrip/rstrip token would
        // silently eat caller whitespace.
        const auto * lstrip = token_config.find("lstrip");
        const auto * rstrip = token_config.find("rstrip");
        if (lstrip == nullptr || rstrip == nullptr || !lstrip->is_bool() ||
            !rstrip->is_bool() || lstrip->as_bool() || rstrip->as_bool()) {
            throw std::runtime_error(
                "granite_speech5 added token must have lstrip/rstrip=false: '" +
                content->as_string() + "'");
        }
        if (content->as_string() == kGraniteSpeech5AudioMarker) {
            const auto * special = token_config.find("special");
            audio_ok = id->as_i64() == kGraniteSpeech5AudioTokenId && special != nullptr &&
                special->is_bool() && special->as_bool();
        }
    }
    if (!audio_ok) {
        throw std::runtime_error("granite_speech5 <|audio|> is not the pinned special token");
    }
    const auto tokenizer_config = resources.parse_json("tokenizer_config");
    if (tokenizer_config.find("add_bos_token") == nullptr ||
        tokenizer_config.require("add_bos_token").as_bool()) {
        throw std::runtime_error("granite_speech5 requires add_bos_token=false");
    }
    if (tokenizer_config.find("add_prefix_space") == nullptr ||
        tokenizer_config.require("add_prefix_space").as_bool()) {
        throw std::runtime_error("granite_speech5 requires add_prefix_space=false");
    }
    const auto generation_config = resources.parse_json("generation_config");
    if (json::optional_i64(generation_config, "bos_token_id", -1) != kGraniteSpeech5BosTokenId ||
        json::optional_i64(generation_config, "eos_token_id", -1) != kGraniteSpeech5EosTokenId ||
        json::optional_i64(generation_config, "pad_token_id", -1) != kGraniteSpeech5PadTokenId) {
        throw std::runtime_error("granite_speech5 generation IDs disagree with the trace");
    }
}

GraniteSpeech5PromptIds granite_speech5_encode_single_marker_prompt(
    const tokenizers::LlamaBpeTokenizer & tokenizer,
    std::string_view user_content,
    int32_t audio_token_id,
    const std::vector<std::string> & reserved_markers) {
    if (!reserved_markers.empty()) {
        // Excise the single structural marker so the check covers caller
        // text only. A smuggled second marker still fails the exactly-one
        // count below; this names the offending marker instead.
        std::string checkable(user_content);
        const std::string_view structural = kGraniteSpeech5AudioMarker;
        if (const size_t at = checkable.find(structural); at != std::string::npos) {
            checkable.erase(at, structural.size());
        }
        granite_speech5_require_no_reserved_markers(checkable, reserved_markers);
    }
    const std::string prompt = granite_speech5_render_single_turn_prompt(user_content);
    GraniteSpeech5PromptIds out;
    out.prompt_text = prompt;
    out.input_ids = tokenizer.encode(prompt);
    const int64_t markers =
        static_cast<int64_t>(std::count(out.input_ids.begin(), out.input_ids.end(), audio_token_id));
    if (markers != 1) {
        throw std::runtime_error(
            "granite_speech5 single-turn prompt must carry exactly one audio marker, found " +
            std::to_string(markers));
    }
    out.audio_tokens = 0;
    return out;
}

GraniteSpeech5PromptIds granite_speech5_expand_audio_markers(
    const std::vector<int32_t> & one_marker_ids,
    int32_t audio_token_id,
    int64_t audio_tokens) {
    if (audio_tokens <= 0) {
        throw std::runtime_error("granite_speech5 expansion needs a positive audio token count");
    }
    // The single-turn prompt carries exactly one structural marker; splicing
    // a multi-marker run would silently misplace the injection positions.
    const int64_t markers = static_cast<int64_t>(
        std::count(one_marker_ids.begin(), one_marker_ids.end(), audio_token_id));
    if (markers != 1) {
        throw std::runtime_error(
            "granite_speech5 expansion needs exactly one audio marker, found " +
            std::to_string(markers));
    }
    GraniteSpeech5PromptIds out;
    out.audio_tokens = audio_tokens;
    out.input_ids.reserve(one_marker_ids.size() + static_cast<size_t>(audio_tokens - 1));
    for (const int32_t id : one_marker_ids) {
        if (id != audio_token_id) {
            out.input_ids.push_back(id);
            continue;
        }
        for (int64_t i = 0; i < audio_tokens; ++i) {
            out.audio_positions.push_back(static_cast<int32_t>(out.input_ids.size()));
            out.input_ids.push_back(audio_token_id);
        }
    }
    return out;
}

GraniteSpeech5PromptBudget granite_speech5_prompt_budget(
    const GraniteSpeech5PromptIds & expanded, int64_t max_position_embeddings) {
    GraniteSpeech5PromptBudget budget;
    budget.prompt_len = static_cast<int64_t>(expanded.input_ids.size());
    budget.audio_tokens = expanded.audio_tokens;
    budget.text_tokens = budget.prompt_len - budget.audio_tokens;
    budget.expansion_overhead = budget.audio_tokens > 0 ? budget.audio_tokens - 1 : 0;
    budget.max_position_embeddings = max_position_embeddings;
    if (budget.audio_tokens != static_cast<int64_t>(expanded.audio_positions.size())) {
        throw std::runtime_error("granite_speech5 budget disagrees with injection positions");
    }
    return budget;
}

void granite_speech5_require_prompt_fits(const GraniteSpeech5PromptBudget & budget) {
    if (budget.prompt_len <= 0) {
        throw std::runtime_error("granite_speech5 prompt is empty");
    }
    if (budget.max_position_embeddings <= 0 ||
        budget.prompt_len > budget.max_position_embeddings) {
        throw std::runtime_error(
            "granite_speech5 prompt length " + std::to_string(budget.prompt_len) +
            " exceeds max_position_embeddings " + std::to_string(budget.max_position_embeddings));
    }
}

runtime::GreedyCausalDecoderRuntime::Prompt granite_speech5_build_decoder_prompt(
    const GraniteSpeech5PromptIds & expanded,
    std::vector<float> projected_values,
    int64_t hidden_size,
    int32_t audio_token_id) {
    if (hidden_size <= 0) {
        throw std::runtime_error("granite_speech5 injection needs a positive hidden size");
    }
    const auto & positions = expanded.audio_positions;
    if (positions.empty()) {
        throw std::runtime_error("granite_speech5 injection needs audio positions");
    }
    for (size_t i = 1; i < positions.size(); ++i) {
        if (positions[i] != positions[i - 1] + 1) {
            throw std::runtime_error("granite_speech5 injection positions must be contiguous");
        }
    }
    const int64_t prompt_len = static_cast<int64_t>(expanded.input_ids.size());
    for (const int32_t position : positions) {
        if (position < 0 || static_cast<int64_t>(position) >= prompt_len) {
            throw std::runtime_error("granite_speech5 injection position out of range");
        }
        if (expanded.input_ids[static_cast<size_t>(position)] != audio_token_id) {
            throw std::runtime_error("granite_speech5 injection position is not an audio marker");
        }
    }
    if (static_cast<int64_t>(positions.size()) != expanded.audio_tokens) {
        throw std::runtime_error("granite_speech5 injection count disagrees with audio tokens");
    }
    const int64_t tokens = static_cast<int64_t>(positions.size());
    if (static_cast<int64_t>(projected_values.size()) != tokens * hidden_size) {
        throw std::runtime_error(
            "granite_speech5 projected values size " + std::to_string(projected_values.size()) +
            " != tokens(" + std::to_string(tokens) + ")*hidden(" + std::to_string(hidden_size) +
            ")");
    }
    runtime::GreedyCausalDecoderRuntime::Prompt prompt;
    prompt.input_ids = expanded.input_ids;
    prompt.injection.values = std::move(projected_values);
    prompt.injection.tokens = tokens;
    prompt.injection.positions = positions;
    return prompt;
}

std::vector<int32_t> granite_speech5_strip_observed_eos(
    std::vector<int32_t> token_ids, int32_t eos_token_id) {
    if (!token_ids.empty() && token_ids.back() == eos_token_id) {
        token_ids.pop_back();
    }
    return token_ids;
}

std::string granite_speech5_decode_generated(
    const tokenizers::LlamaBpeTokenizer & tokenizer,
    const std::vector<int32_t> & generated_ids,
    int32_t eos_token_id) {
    return tokenizer.decode(granite_speech5_strip_observed_eos(generated_ids, eos_token_id), true);
}

GraniteSpeech5PromptBuilder::GraniteSpeech5PromptBuilder(
    std::shared_ptr<const tokenizers::LlamaBpeTokenizer> tokenizer, int32_t audio_token_id)
    : tokenizer_(std::move(tokenizer)), audio_token_id_(audio_token_id) {
    if (tokenizer_ == nullptr) {
        throw std::runtime_error("granite_speech5 prompt builder requires a tokenizer");
    }
    const auto audio_id = tokenizer_->find_token_id(kGraniteSpeech5AudioMarker);
    if (!audio_id.has_value() || *audio_id != audio_token_id_) {
        throw std::runtime_error("granite_speech5 prompt builder audio marker mismatch");
    }
}

GraniteSpeech5PromptIds GraniteSpeech5PromptBuilder::build_prompt_ids(
    std::string_view normalized_language_code,
    int64_t audio_tokens,
    const std::vector<std::string> & reserved_markers) const {
    GraniteSpeech5PromptIds single = granite_speech5_encode_single_marker_prompt(
        *tokenizer_, granite_speech5_build_user_content(normalized_language_code),
        audio_token_id_, reserved_markers);
    GraniteSpeech5PromptIds expanded =
        granite_speech5_expand_audio_markers(single.input_ids, audio_token_id_, audio_tokens);
    expanded.prompt_text = single.prompt_text;
    return expanded;
}

GraniteSpeech5PromptIds GraniteSpeech5PromptBuilder::build_prompt_ids_with_context(
    std::string_view normalized_source_code,
    std::string_view normalized_target_code,
    const std::vector<std::string> & keyword_terms,
    int64_t audio_tokens,
    const std::vector<std::string> & reserved_markers) const {
    if (normalized_target_code.empty() && keyword_terms.empty()) {
        return build_prompt_ids(normalized_source_code, audio_tokens, reserved_markers);
    }
    std::string user_content;
    if (normalized_target_code.empty()) {
        user_content = granite_speech5_build_keyword_user_content(
            normalized_source_code, keyword_terms);
    } else {
        user_content = granite_speech5_build_translation_user_content(
            normalized_source_code, normalized_target_code, keyword_terms);
    }
    GraniteSpeech5PromptIds single = granite_speech5_encode_single_marker_prompt(
        *tokenizer_, user_content, audio_token_id_, reserved_markers);
    GraniteSpeech5PromptIds expanded =
        granite_speech5_expand_audio_markers(single.input_ids, audio_token_id_, audio_tokens);
    expanded.prompt_text = single.prompt_text;
    return expanded;
}

runtime::GreedyCausalDecoderRuntime::Prompt GraniteSpeech5PromptBuilder::build_decoder_prompt(
    std::string_view normalized_language_code,
    int64_t audio_tokens,
    std::vector<float> projected_values,
    const std::vector<std::string> & reserved_markers) const {
    return build_decoder_prompt_with_context(normalized_language_code, "", {},
                                             audio_tokens, std::move(projected_values),
                                             reserved_markers);
}

runtime::GreedyCausalDecoderRuntime::Prompt
GraniteSpeech5PromptBuilder::build_decoder_prompt_with_context(
    std::string_view normalized_source_code,
    std::string_view normalized_target_code,
    const std::vector<std::string> & keyword_terms,
    int64_t audio_tokens,
    std::vector<float> projected_values,
    const std::vector<std::string> & reserved_markers) const {
    return granite_speech5_build_decoder_prompt(
        build_prompt_ids_with_context(normalized_source_code, normalized_target_code,
                                      keyword_terms, audio_tokens, reserved_markers),
        std::move(projected_values), kGraniteSpeech5TextHiddenSize, audio_token_id_);
}

std::string GraniteSpeech5PromptBuilder::decode_generated(
    const std::vector<int32_t> & generated_ids) const {
    return granite_speech5_decode_generated(*tokenizer_, generated_ids);
}

}  // namespace engine::community_models::granite_speech5
