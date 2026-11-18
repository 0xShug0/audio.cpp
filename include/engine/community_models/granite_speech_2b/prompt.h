#pragma once

// granite_speech_2b text-side prompt inputs (Task C1): tokenizer load/verify,
// single-turn prompt builder, language-hint mapping, expanded-ID + Injection
// builder, reserved-marker rejection, length/overhead reporting.
//
// The acoustic path (B1 frontend, B2 encoder/projector) produces A projected
// vectors (A = 3 * windows from granite_speech_2b_lengths_for_samples). This
// header builds the text-side Prompt the shared greedy decoder consumes:
// a pinned single-turn chat template with one structural <|audio|> marker,
// tokenized with the checkpoint tokenizer, expanded to A consecutive audio
// IDs, plus the sorted/contiguous Injection the C2 prefill path injects at.
//
// Tokenizer route (proven against the Python reference in the C1 unit test):
// LlamaBpeTokenizer over the checkpoint tokenizer.json with the Llama3
// pre-tokenizer preset. tokenizer.json carries no normalizer, a
// Split(GPT2-regex, Removed, invert) + ByteLevel(add_prefix_space=false,
// use_regex=false) pre-tokenizer, byte_fallback=false, ignore_merges=false,
// no post-processor, add_bos_token=false, all added tokens lstrip/rstrip
// false. The Llama3 regex is byte-identical to the pinned split pattern, so
// no model-owned splitter hook is needed. HuggingFaceTokenizerJson is
// decode-only and is NOT used for encode.
//
// C2 consumes: GraniteSpeech2BPromptBuilder, build_decoder_prompt (returns
// runtime::GreedyCausalDecoderRuntime::Prompt for generate(..., reuse_graphs
// = true)), check_granite_speech_2b_prompt_assets, and the EOS helpers below.

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

namespace engine::community_models::granite_speech_2b {

// Pinned prompt/template bytes (from chat_template.jinja at the pinned
// revision; the trailing "\n ASSISTANT:" space is part of the template).
inline constexpr const char * kGraniteSpeech2BAudioMarker = "<|audio|>";
inline constexpr const char * kGraniteSpeech2BPromptPrefix = "USER: ";
inline constexpr const char * kGraniteSpeech2BPromptSuffix = "\n ASSISTANT:";
inline constexpr const char * kGraniteSpeech2BTranscribeInstruction =
    "transcribe the speech with proper punctuation and capitalization.";
inline constexpr const char * kGraniteSpeech2BHintPrefix = "The speech is in ";
inline constexpr const char * kGraniteSpeech2BHintSuffix =
    ". Transcribe it exactly; do not translate. ";

// Tokenizer/generation IDs (agree with trace.json model_config, which merges
// config.json text values with the resolved generation_config).
inline constexpr int32_t kGraniteSpeech2BAudioTokenId = 100352;
inline constexpr int32_t kGraniteSpeech2BBosTokenId = 100257;
inline constexpr int32_t kGraniteSpeech2BEosTokenId = 100257;
inline constexpr int32_t kGraniteSpeech2BPadTokenId = 100256;
inline constexpr int64_t kGraniteSpeech2BTextHiddenSize = 2048;
inline constexpr int64_t kGraniteSpeech2BMaxPositionEmbeddings = 16384;

// Supported source-language hint codes (reference.py --language choices).
// Normalized form is lowercase ASCII; "" means automatic recognition
// (no hint text, identical to the Python smoke prompt).
std::string granite_speech_2b_normalize_language_code(std::string_view language);
std::string_view granite_speech_2b_language_name(std::string_view normalized_code);
std::string granite_speech_2b_build_user_content(std::string_view normalized_code);

// Task F: keyword biasing + speech translation (model-card preferred-prompt
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
// Composition follows the C1 pattern: an optional source-language hint, then
// the model-owned <|audio|> marker, then the task instruction. The C1 hint
// prefix ("The speech is in ... Transcribe it exactly; do not translate. ")
// is reused for keyword ASR; translation uses a bare source prefix ("The
// speech is in .... ") because the do-not-translate clause would contradict
// the translation instruction. Target names are the card's English names
// (English, French, German, Spanish, Portuguese, Japanese, Italian,
// Mandarin); "Portuguese" covers the pt target the prompt table omits but the
// eval-claims row ("to and from English for the same languages") includes.
inline constexpr const char * kGraniteSpeech2BKeywordInstruction =
    "transcribe the speech to text. Keywords: ";
inline constexpr const char * kGraniteSpeech2BTranslationInstructionPrefix =
    "translate the speech to ";
inline constexpr const char * kGraniteSpeech2BTranslationInstructionSuffix =
    " with proper punctuation and capitalization.";
inline constexpr const char * kGraniteSpeech2BTranslationKeywordsPrefix = " Keywords: ";
inline constexpr const char * kGraniteSpeech2BTranslationHintSuffix = ". ";

// Supported translation target codes. Normalized form is lowercase ASCII;
// "" is not a target (absent target_language means ASR, never translation).
std::string granite_speech_2b_normalize_target_language_code(std::string_view language);
std::string_view granite_speech_2b_target_language_name(std::string_view normalized_code);

// Keyword parsing: split text_input.text on ',', trim ASCII whitespace per
// term, drop empties, preserve UTF-8 and multiword terms verbatim (no case
// folding, no dedup — the joined list round-trips the caller's text).
// Whitespace-only input yields no terms (plain path). Reserved control
// markers are NOT stripped here: the per-chunk encode rejects them loudly.
std::vector<std::string> granite_speech_2b_parse_keyword_terms(
    std::string_view keywords_text);
std::string granite_speech_2b_join_keyword_terms(
    const std::vector<std::string> & terms);

// User-content composers. keyword_terms must be non-empty (the session routes
// empty lists to the plain ASR path); both throw on empty terms. Reserved
// markers inside terms are rejected by the encode step, not here.
std::string granite_speech_2b_build_keyword_user_content(
    std::string_view normalized_source_code,
    const std::vector<std::string> & keyword_terms);
std::string granite_speech_2b_build_translation_user_content(
    std::string_view normalized_source_code,
    std::string_view normalized_target_code,
    const std::vector<std::string> & keyword_terms);

// Reserved-marker rejection primitive (Task F reuses this for text
// controls): throws std::runtime_error naming the offending marker when any
// entry of reserved_markers occurs in text. The default list is every
// special (control) added token from the checkpoint tokenizer.json, so
// caller text can never smuggle a structural ID through encode. F loads the
// list once per checkpoint with
// granite_speech_2b_reserved_markers_from_tokenizer_json and passes it to
// every per-chunk build call below.
std::vector<std::string> granite_speech_2b_reserved_markers_from_tokenizer_json(
    const std::filesystem::path & tokenizer_json_path);
void granite_speech_2b_require_no_reserved_markers(
    std::string_view text, const std::vector<std::string> & reserved_markers);

// Single-turn render: "USER: " + user_content + "\n ASSISTANT:" (exact bytes,
// verified against apply_chat_template in the C1 unit test). No validation
// here by design: F validates raw controls first, then composes.
std::string granite_speech_2b_render_single_turn_prompt(std::string_view user_content);

// Tokenizer load: Llama3 preset over tokenizer.json + tokenizer_config
// (pad ID). Throws when the audio marker does not resolve to 100352.
std::shared_ptr<tokenizers::LlamaBpeTokenizer> load_granite_speech_2b_tokenizer(
    const assets::ResourceBundle & resources);

// Prompt-side asset verification (no B1 API changes): checks the staged
// template + tokenizer settings this builder depends on. Throws
// std::runtime_error on any drift.
void check_granite_speech_2b_prompt_assets(const assets::ResourceBundle & resources);

// Expanded prompt: whole-string tokenization with exactly one structural
// audio marker spliced to audio_tokens consecutive IDs (mirrors the HF
// processor placeholder expansion). audio_positions are sorted/contiguous
// and satisfy input_ids[pos] == audio_token_id; count == audio_tokens.
struct GraniteSpeech2BPromptIds {
    std::vector<int32_t> input_ids;
    std::vector<int32_t> audio_positions;
    int64_t audio_tokens = 0;
    std::string prompt_text;
};

GraniteSpeech2BPromptIds granite_speech_2b_encode_single_marker_prompt(
    const tokenizers::LlamaBpeTokenizer & tokenizer,
    std::string_view user_content,
    int32_t audio_token_id = kGraniteSpeech2BAudioTokenId,
    const std::vector<std::string> & reserved_markers = {});
GraniteSpeech2BPromptIds granite_speech_2b_expand_audio_markers(
    const std::vector<int32_t> & one_marker_ids,
    int32_t audio_token_id,
    int64_t audio_tokens);

// Length/overhead reporting for C2's prompt budget inputs.
struct GraniteSpeech2BPromptBudget {
    int64_t prompt_len = 0;
    int64_t audio_tokens = 0;
    int64_t text_tokens = 0;
    int64_t expansion_overhead = 0;  // prompt_len growth vs the single-marker form
    int64_t max_position_embeddings = kGraniteSpeech2BMaxPositionEmbeddings;
};
GraniteSpeech2BPromptBudget granite_speech_2b_prompt_budget(
    const GraniteSpeech2BPromptIds & expanded,
    int64_t max_position_embeddings = kGraniteSpeech2BMaxPositionEmbeddings);
void granite_speech_2b_require_prompt_fits(const GraniteSpeech2BPromptBudget & budget);

// Decoder input for C2's GreedyCausalDecoderRuntime::generate(..., reuse_graphs).
// projected_values are A token-major rows of hidden_size floats (B2 output).
runtime::GreedyCausalDecoderRuntime::Prompt granite_speech_2b_build_decoder_prompt(
    const GraniteSpeech2BPromptIds & expanded,
    std::vector<float> projected_values,
    int64_t hidden_size = kGraniteSpeech2BTextHiddenSize,
    int32_t audio_token_id = kGraniteSpeech2BAudioTokenId);

// Generation/tokenizer precedence: the Python trace preserves the observed
// stop (usually a trailing EOS 100257) while the shared C++ runtime excludes
// stop IDs from its return. Normalize both sides by stripping at most one
// trailing EOS before comparing visible output (used by C2).
std::vector<int32_t> granite_speech_2b_strip_observed_eos(
    std::vector<int32_t> token_ids, int32_t eos_token_id = kGraniteSpeech2BEosTokenId);
std::string granite_speech_2b_decode_generated(
    const tokenizers::LlamaBpeTokenizer & tokenizer,
    const std::vector<int32_t> & generated_ids,
    int32_t eos_token_id = kGraniteSpeech2BEosTokenId);

// Per-chunk builder entry (Task D applies the same hint to every chunk):
// owns the tokenizer and exposes the whole pipeline above.
class GraniteSpeech2BPromptBuilder {
public:
    explicit GraniteSpeech2BPromptBuilder(
        std::shared_ptr<const tokenizers::LlamaBpeTokenizer> tokenizer,
        int32_t audio_token_id = kGraniteSpeech2BAudioTokenId);

    GraniteSpeech2BPromptIds build_prompt_ids(
        std::string_view normalized_language_code,
        int64_t audio_tokens,
        const std::vector<std::string> & reserved_markers = {}) const;
    // Task F entry: source hint ("", auto) + target ("" = ASR, else a
    // normalized target code) + keyword terms. Empty target + empty terms
    // reproduces build_prompt_ids exactly (the plain-ASR path is unchanged).
    GraniteSpeech2BPromptIds build_prompt_ids_with_context(
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
    std::string decode_generated(const std::vector<int32_t> & generated_ids) const;

    const tokenizers::LlamaBpeTokenizer & tokenizer() const noexcept { return *tokenizer_; }
    int32_t audio_token_id() const noexcept { return audio_token_id_; }

private:
    std::shared_ptr<const tokenizers::LlamaBpeTokenizer> tokenizer_;
    int32_t audio_token_id_;
};

}  // namespace engine::community_models::granite_speech_2b
