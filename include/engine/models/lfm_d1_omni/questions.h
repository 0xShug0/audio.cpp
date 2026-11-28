#pragma once

#include "engine/framework/io/json.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace engine::models::lfm_d1_omni {

enum class QuestionKind { Noul, Choice, MultipleChoice, Score, Estimate };

// Selected by the checkpoint's decision contract; no sidecar files are required.
struct QuestionConfig {
    int codes = 1;
    int prompt = 1;
    bool numbers = false;
    // Exact special names from tokenizer metadata (including special added tokens).
    std::vector<std::string> special_tokens;
};

struct ParsedQuestion {
    QuestionKind kind = QuestionKind::Noul;
    std::string instructions;
    std::vector<std::string> labels;
    std::vector<std::optional<std::string>> descriptions;
    std::optional<std::string> yes_description, no_description;
    std::vector<double> grid;
    std::optional<std::string> unit;
};

struct QuestionSlot {
    // Relative to the suffix's last token (zero is that last token).
    int64_t offset = 0;
    std::vector<std::vector<int32_t>> candidate_groups;
    std::string temperature_bucket;
};

struct QuestionSlots {
    ParsedQuestion question;
    std::string suffix;
    std::vector<int32_t> suffix_ids;
    std::vector<QuestionSlot> slots;
};

struct NamedQuestion {
    std::string name;
    ParsedQuestion question;
};

// Raw JSON retains question and criteria order through cJSON's ordered children;
// json::Value supplies field validation. The named result preserves set grouping order.
// Callers form sets of 16 from this vector without sorting the names.
// Instructions are text; audio state/markup and media preparation belong to the caller.
std::vector<NamedQuestion> parse_questions(std::string_view named_questions_json);
QuestionSlots render_question(const ParsedQuestion & question,
                              const tokenizers::LlamaBpeTokenizer & tokenizer,
                              const QuestionConfig & config);

struct SlotScores {
    std::vector<int32_t> token_ids;
    std::vector<float> logits;  // same order as token_ids; full vocab or sparse selection
};

// Supply every candidate for every slot; missing/duplicate IDs are errors.
// Max-pool forms, then softmax. Typed estimate/multiple slots are untempered,
// matching upstream; their output tables are applied after the raw softmax.
io::json::Value decode_question(const QuestionSlots & question, const std::vector<SlotScores> & scores,
                                const io::json::Value & calibration,
                                const std::string & score_readout = "median");

}  // namespace engine::models::lfm_d1_omni
