#include "engine/community_models/audio8_tts/prompt_builder.h"

#include <regex>
#include <stdexcept>
#include <utility>

namespace engine::models::audio8_tts {
namespace {

void append_tokens(std::vector<int32_t> & out, const std::vector<int32_t> & tokens) {
    out.insert(out.end(), tokens.begin(), tokens.end());
}

struct CodeSpan {
    int64_t start = 0;
    const Audio8TtsCodes * codes = nullptr;
};

std::string reference_text_with_speakers(const std::string & text, int64_t speaker) {
    static const std::regex speaker_re(R"(<\|speaker:\d+\|>)");
    if (std::regex_search(text, speaker_re)) {
        return text;
    }
    return "<|speaker:" + std::to_string(speaker) + "|>" + text;
}

void append_code_span(
    std::vector<int32_t> & row0,
    std::vector<CodeSpan> & spans,
    const Audio8TtsTextTokenizer & tokenizer,
    const Audio8TtsCodes & codes,
    int64_t expected_codebooks) {
    if (codes.codebooks != expected_codebooks) {
        throw std::runtime_error("Audio8 TTS prompt codebook count mismatch");
    }
    const int64_t start = static_cast<int64_t>(row0.size());
    const int32_t semantic_begin = tokenizer.semantic_begin_id();
    for (int64_t frame = 0; frame < codes.frames; ++frame) {
        row0.push_back(semantic_begin + codes.codes[static_cast<size_t>(frame)]);
    }
    spans.push_back({start, &codes});
}

}  // namespace

Audio8TtsPromptBuilder::Audio8TtsPromptBuilder(
    std::shared_ptr<const Audio8TtsAssets> assets,
    Audio8TtsTextTokenizer tokenizer)
    : assets_(std::move(assets)),
      tokenizer_(std::move(tokenizer)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Audio8 TTS prompt builder requires assets");
    }
}

Audio8TtsPrompt Audio8TtsPromptBuilder::build(
    const Audio8TtsRequest & request,
    const std::vector<Audio8TtsCodes> & reference_codes,
    const std::optional<Audio8TtsConversationTurn> & previous_turn) const {
    if (request.text.empty()) {
        throw std::runtime_error("Audio8 TTS request text must not be empty");
    }
    const int64_t rows = assets_->config.fast.num_codebooks + 1;
    if (rows <= 1) {
        throw std::runtime_error("Audio8 TTS prompt rows are invalid");
    }

    std::vector<int32_t> row0;
    std::vector<CodeSpan> code_spans;
    if (!request.references.empty()) {
        if (reference_codes.size() != request.references.size()) {
            throw std::runtime_error("Audio8 TTS reference request requires one encoded code tensor per reference");
        }
        append_tokens(row0, tokenizer_.encode("<|im_start|>system\n"));
        append_tokens(row0, tokenizer_.encode("convert the provided text to speech reference to the following:\n\nText:\n"));
        for (size_t index = 0; index < request.references.size(); ++index) {
            if (index != 0) {
                append_tokens(row0, tokenizer_.encode("\n"));
            }
            append_tokens(
                row0,
                tokenizer_.encode(reference_text_with_speakers(
                    request.references[index].text,
                    static_cast<int64_t>(index))));
        }
        append_tokens(row0, tokenizer_.encode("\n\nSpeech:\n"));
        for (const auto & codes : reference_codes) {
            append_code_span(row0, code_spans, tokenizer_, codes, assets_->config.fast.num_codebooks);
        }
        append_tokens(row0, tokenizer_.encode("<|im_end|>\n"));
    } else {
        append_tokens(row0, tokenizer_.encode("<|im_start|>system\n"));
        append_tokens(row0, tokenizer_.encode("convert the provided text to speech"));
        append_tokens(row0, tokenizer_.encode("<|im_end|>\n"));
    }
    if (previous_turn.has_value()) {
        append_tokens(row0, tokenizer_.encode("<|im_start|>user\n"));
        append_tokens(row0, tokenizer_.encode(previous_turn->text));
        append_tokens(row0, tokenizer_.encode("<|im_end|>\n"));
        append_tokens(row0, tokenizer_.encode("<|im_start|>assistant\n<|voice|>"));
        append_code_span(row0, code_spans, tokenizer_, previous_turn->codes, assets_->config.fast.num_codebooks);
        append_tokens(row0, tokenizer_.encode("<|im_end|>\n"));
    }
    append_tokens(row0, tokenizer_.encode("<|im_start|>user\n"));
    append_tokens(row0, tokenizer_.encode(request.text));
    append_tokens(row0, tokenizer_.encode("<|im_end|>\n"));
    append_tokens(row0, tokenizer_.encode("<|im_start|>assistant\n<|voice|>"));

    Audio8TtsPrompt prompt;
    prompt.codebook_rows = rows;
    prompt.steps = static_cast<int64_t>(row0.size());
    prompt.text = request.text;
    prompt.matrix.assign(static_cast<size_t>(rows * prompt.steps), 0);
    for (int64_t step = 0; step < prompt.steps; ++step) {
        prompt.matrix[static_cast<size_t>(step)] = row0[static_cast<size_t>(step)];
    }
    for (const auto & span : code_spans) {
        if (span.codes == nullptr) {
            throw std::runtime_error("Audio8 TTS prompt code span is missing codes");
        }
        for (int64_t frame = 0; frame < span.codes->frames; ++frame) {
            const int64_t step = span.start + frame;
            if (step < 0 || step >= prompt.steps) {
                throw std::runtime_error("Audio8 TTS prompt code span exceeds prompt length");
            }
            for (int64_t codebook = 0; codebook < span.codes->codebooks; ++codebook) {
                prompt.matrix[static_cast<size_t>((codebook + 1) * prompt.steps + step)] =
                    span.codes->codes[static_cast<size_t>(codebook * span.codes->frames + frame)];
            }
        }
    }
    return prompt;
}

}  // namespace engine::models::audio8_tts
