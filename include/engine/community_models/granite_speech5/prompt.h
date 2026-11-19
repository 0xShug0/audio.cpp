#pragma once

// Llama3 BPE tokenization, language/translation instructions and audio injection.
// A single structural audio marker expands into one ID per projected audio row.
// The checkpoint tokenizer.json provides the vocabulary and merge rules.

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/greedy_causal_decoder.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace engine::community_models::granite_speech5 {

// Pinned prompt/template bytes (from chat_template.jinja at the pinned
// revision; the trailing "\n ASSISTANT:" space is part of the template).
inline constexpr const char * kGraniteSpeech5AudioMarker = "<|audio|>";
inline constexpr const char * kGraniteSpeech5PromptPrefix = "USER: ";
inline constexpr const char * kGraniteSpeech5PromptSuffix = "\n ASSISTANT:";
inline constexpr const char * kGraniteSpeech5TranscribeInstruction =
    "transcribe the speech with proper punctuation and capitalization.";
inline constexpr const char * kGraniteSpeech5HintPrefix = "The speech is in ";
inline constexpr const char * kGraniteSpeech5HintSuffix =
    ". Transcribe it exactly; do not translate. ";

// Tokenizer/generation IDs (agree with trace.json model_config, which merges
// config.json text values with the resolved generation_config).
inline constexpr int32_t kGraniteSpeech5AudioTokenId = 100352;
inline constexpr int32_t kGraniteSpeech5BosTokenId = 100257;
inline constexpr int32_t kGraniteSpeech5EosTokenId = 100257;
inline constexpr int32_t kGraniteSpeech5PadTokenId = 100256;
inline constexpr int64_t kGraniteSpeech5TextHiddenSize = 2048;
inline constexpr int64_t kGraniteSpeech5MaxPositionEmbeddings = 16384;

// Supported source-language hint codes (reference.py --language choices).
// Normalized form is lowercase ASCII; "" means automatic recognition
// (no hint text, identical to the Python smoke prompt).
std::string granite_speech5_normalize_language_code(std::string_view language);
std::string_view granite_speech5_language_name(std::string_view normalized_code);
std::string granite_speech5_build_user_content(std::string_view normalized_code);

// Keyword biasing and translation: keyword biasing + speech translation (model-card preferred-prompt
// table, verified by prompt-ID parity against the pinned processor).
//
// IBM instruction text (verbatim from the model card):
// - keyword ASR: "transcribe the speech to text. Keywords: <kw1>, <kw2>, ..."
// - translation: "translate the speech to <language> with proper punctuation
//   and capitalization." (the card's punctuated AST row; the raw row without
//   the punctuation clause is not implemented — see the F report for the
//   measured comparison that selected this variant).
// - translation + keywords: the translation instruction + " Keywords: ..."
//   (the card's keyword-biasing AST row, same punctuation-clause choice).
//
// Composition follows the prompt pattern: an optional source-language hint, then
// the model-owned <|audio|> marker, then the task instruction. The source hint
// prefix ("The speech is in ... Transcribe it exactly; do not translate. ")
// is reused for keyword ASR; translation uses a bare source prefix ("The
// speech is in .... ") because the do-not-translate clause would contradict
// the translation instruction. Target names are the card's English names
// (English, French, German, Spanish, Portuguese, Japanese, Italian,
// Mandarin); "Portuguese" covers the pt target the prompt table omits but the
// eval-claims row ("to and from English for the same languages") includes.
inline constexpr const char * kGraniteSpeech5KeywordInstruction =
    "transcribe the speech to text. Keywords: ";
inline constexpr const char * kGraniteSpeech5TranslationInstructionPrefix =
    "translate the speech to ";
inline constexpr const char * kGraniteSpeech5TranslationInstructionSuffix =
    " with proper punctuation and capitalization.";
inline constexpr const char * kGraniteSpeech5TranslationKeywordsPrefix = " Keywords: ";
inline constexpr const char * kGraniteSpeech5TranslationHintSuffix = ". ";

// Supported translation target codes. Normalized form is lowercase ASCII;
// "" is not a target (absent target_language means ASR, never translation).
std::string granite_speech5_normalize_target_language_code(std::string_view language);
std::string_view granite_speech5_target_language_name(std::string_view normalized_code);

// Keyword parsing: split text_input.text on ',', trim ASCII whitespace per
// term, drop empties, preserve UTF-8 and multiword terms verbatim (no case
// folding, no dedup — the joined list round-trips the caller's text).
// Whitespace-only input yields no terms (plain path). Padding is ASCII
// whitespace only (space/tab/CR/LF): a fullwidth space (U+3000) or other
// Unicode padding is preserved inside the term verbatim, matching the
// no-folding rule — callers needing it trimmed must do so before sending.
// Reserved control markers are NOT stripped here: the per-chunk encode
// rejects them loudly.
std::vector<std::string> granite_speech5_parse_keyword_terms(
    std::string_view keywords_text);
std::string granite_speech5_join_keyword_terms(
    const std::vector<std::string> & terms);

// User-content composers. keyword_terms must be non-empty (the session routes
// empty lists to the plain ASR path); both throw on empty terms. Reserved
// markers inside terms are rejected by the encode step, not here.
std::string granite_speech5_build_keyword_user_content(
    std::string_view normalized_source_code,
    const std::vector<std::string> & keyword_terms);
std::string granite_speech5_build_translation_user_content(
    std::string_view normalized_source_code,
    std::string_view normalized_target_code,
    const std::vector<std::string> & keyword_terms);

// Reserved-marker rejection primitive (also used for text
// controls): throws std::runtime_error naming the offending marker when any
// entry of reserved_markers occurs in text. The default list is every
// special (control) added token from the checkpoint tokenizer.json, so
// caller text can never smuggle a structural ID through encode. F loads the
// list once per checkpoint with
// granite_speech5_reserved_markers_from_tokenizer_json and passes it to
// every per-chunk build call below.
std::vector<std::string> granite_speech5_reserved_markers_from_tokenizer_json(
    const std::filesystem::path & tokenizer_json_path);
void granite_speech5_require_no_reserved_markers(
    std::string_view text, const std::vector<std::string> & reserved_markers);

// Single-turn render: "USER: " + user_content + "\n ASSISTANT:" (exact bytes,
// verified against apply_chat_template in prompt parity tests). No validation
// here by design: F validates raw controls first, then composes.
std::string granite_speech5_render_single_turn_prompt(std::string_view user_content);

// Tokenizer load: Llama3 preset over tokenizer.json + tokenizer_config
// (pad ID). Throws when the audio marker does not resolve to 100352.
std::shared_ptr<tokenizers::LlamaBpeTokenizer> load_granite_speech5_tokenizer(
    const assets::ResourceBundle & resources);

// Prompt-side asset verification: checks the staged
// template + tokenizer settings this builder depends on. Throws
// std::runtime_error on any drift.
void check_granite_speech5_prompt_assets(const assets::ResourceBundle & resources);

// Expanded prompt: whole-string tokenization with exactly one structural
// audio marker spliced to audio_tokens consecutive IDs (mirrors the HF
// processor placeholder expansion). audio_positions are sorted/contiguous
// and satisfy input_ids[pos] == audio_token_id; count == audio_tokens.
struct GraniteSpeech5PromptIds {
    std::vector<int32_t> input_ids;
    std::vector<int32_t> audio_positions;
    int64_t audio_tokens = 0;
    std::string prompt_text;
};

GraniteSpeech5PromptIds granite_speech5_encode_single_marker_prompt(
    const tokenizers::LlamaBpeTokenizer & tokenizer,
    std::string_view user_content,
    int32_t audio_token_id = kGraniteSpeech5AudioTokenId,
    const std::vector<std::string> & reserved_markers = {});
GraniteSpeech5PromptIds granite_speech5_expand_audio_markers(
    const std::vector<int32_t> & one_marker_ids,
    int32_t audio_token_id,
    int64_t audio_tokens);

// Length/overhead reporting for decoder prompt budget inputs.
struct GraniteSpeech5PromptBudget {
    int64_t prompt_len = 0;
    int64_t audio_tokens = 0;
    int64_t text_tokens = 0;
    int64_t expansion_overhead = 0;  // prompt_len growth vs the single-marker form
    int64_t max_position_embeddings = kGraniteSpeech5MaxPositionEmbeddings;
};
GraniteSpeech5PromptBudget granite_speech5_prompt_budget(
    const GraniteSpeech5PromptIds & expanded,
    int64_t max_position_embeddings = kGraniteSpeech5MaxPositionEmbeddings);
void granite_speech5_require_prompt_fits(const GraniteSpeech5PromptBudget & budget);

// Decoder input for GreedyCausalDecoderRuntime::generate(..., reuse_graphs).
// projected_values are A token-major rows of hidden_size floats.
runtime::GreedyCausalDecoderRuntime::Prompt granite_speech5_build_decoder_prompt(
    const GraniteSpeech5PromptIds & expanded,
    std::vector<float> projected_values,
    int64_t hidden_size = kGraniteSpeech5TextHiddenSize,
    int32_t audio_token_id = kGraniteSpeech5AudioTokenId);

// Generation/tokenizer precedence: the Python trace preserves the observed
// stop (usually a trailing EOS 100257) while the shared C++ runtime excludes
// stop IDs from its return. Normalize both sides by stripping at most one
// trailing EOS before comparing visible output.
std::vector<int32_t> granite_speech5_strip_observed_eos(
    std::vector<int32_t> token_ids, int32_t eos_token_id = kGraniteSpeech5EosTokenId);
std::string granite_speech5_decode_generated(
    const tokenizers::LlamaBpeTokenizer & tokenizer,
    const std::vector<int32_t> & generated_ids,
    int32_t eos_token_id = kGraniteSpeech5EosTokenId);

// Per-chunk builder entry (the session applies the same hint to every chunk):
// owns the tokenizer and exposes the whole pipeline above.
class GraniteSpeech5PromptBuilder {
public:
    explicit GraniteSpeech5PromptBuilder(
        std::shared_ptr<const tokenizers::LlamaBpeTokenizer> tokenizer,
        int32_t audio_token_id = kGraniteSpeech5AudioTokenId);

    GraniteSpeech5PromptIds build_prompt_ids(
        std::string_view normalized_language_code,
        int64_t audio_tokens,
        const std::vector<std::string> & reserved_markers = {}) const;
    // Prompt entry: source hint ("", auto) + target ("" = ASR, else a
    // normalized target code) + keyword terms. Empty target + empty terms
    // reproduces build_prompt_ids exactly (the plain-ASR path is unchanged).
    GraniteSpeech5PromptIds build_prompt_ids_with_context(
        std::string_view normalized_source_code,
        std::string_view normalized_target_code,
        const std::vector<std::string> & keyword_terms,
        int64_t audio_tokens,
        const std::vector<std::string> & reserved_markers = {}) const;
    runtime::GreedyCausalDecoderRuntime::Prompt build_decoder_prompt(
        std::string_view normalized_language_code,
        int64_t audio_tokens,
        std::vector<float> projected_values,
        const std::vector<std::string> & reserved_markers = {}) const;
    // Prompt entry: the single decoder-prompt path for all callers. Routes
    // through build_prompt_ids_with_context, so translation/keyword callers
    // can never silently receive the plain ASR prompt; empty target + empty
    // terms reproduces build_decoder_prompt exactly. The plain overload above
    // delegates here (one path serves all callers).
    runtime::GreedyCausalDecoderRuntime::Prompt build_decoder_prompt_with_context(
        std::string_view normalized_source_code,
        std::string_view normalized_target_code,
        const std::vector<std::string> & keyword_terms,
        int64_t audio_tokens,
        std::vector<float> projected_values,
        const std::vector<std::string> & reserved_markers = {}) const;
    std::string decode_generated(const std::vector<int32_t> & generated_ids) const;

    const tokenizers::LlamaBpeTokenizer & tokenizer() const noexcept { return *tokenizer_; }
    int32_t audio_token_id() const noexcept { return audio_token_id_; }

private:
    std::shared_ptr<const tokenizers::LlamaBpeTokenizer> tokenizer_;
    int32_t audio_token_id_;
};

}  // namespace engine::community_models::granite_speech5
