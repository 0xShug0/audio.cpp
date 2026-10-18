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

// Golden parity with Audio8_TTS/audio8_tts_data.py + onnx_runtime/arktts_runtime/prompt.py
// CJK ranges from _CJK_RANGES, line-break set from _LINE_BREAK_RE.
bool is_cjk(uint32_t cp) {
    return (0x1100 <= cp && cp <= 0x11FF) || (0x2E80 <= cp && cp <= 0x2FDF) ||
           (0x3000 <= cp && cp <= 0x303F) || (0x3040 <= cp && cp <= 0x30FF) ||
           (0x3100 <= cp && cp <= 0x31FF) || (0x3400 <= cp && cp <= 0x4DBF) ||
           (0x4E00 <= cp && cp <= 0x9FFF) || (0xA960 <= cp && cp <= 0xA97F) ||
           (0xAC00 <= cp && cp <= 0xD7A3) || (0xD7B0 <= cp && cp <= 0xD7FF) ||
           (0xF900 <= cp && cp <= 0xFAFF) || (0xFE30 <= cp && cp <= 0xFE4F) ||
           (0xFF01 <= cp && cp <= 0xFF9F) || (0x20000 <= cp && cp <= 0x2FA1F);
}

bool is_line_break(uint32_t cp) {
    return cp == 0x0D || cp == 0x0A || cp == 0x0B || cp == 0x0C || cp == 0x1C ||
           cp == 0x1D || cp == 0x1E || cp == 0x85 || cp == 0x2028 || cp == 0x2029;
}

bool is_space(uint32_t cp) {
    if (cp == 0x20 || cp == 0x09 || cp == 0x0A || cp == 0x0B || cp == 0x0C ||
        cp == 0x0D || cp == 0x1C || cp == 0x1D || cp == 0x1E || cp == 0x85 ||
        cp == 0xA0 || cp == 0x1680 || cp == 0x2028 || cp == 0x2029 ||
        cp == 0x202F || cp == 0x205F || cp == 0x3000) {
        return true;
    }
    if (0x2000 <= cp && cp <= 0x200A) {
        return true;
    }
    return false;
}

bool is_control_category(uint32_t cp) {
    if (cp <= 0x1F) {
        return !is_space(cp);
    }
    if (cp == 0x7F) {
        return true;
    }
    if (0x80 <= cp && cp <= 0x9F) {
        return true;
    }
    if ((0x200B <= cp && cp <= 0x200F) || (0x202A <= cp && cp <= 0x202E) ||
        (0x2060 <= cp && cp <= 0x206F) || cp == 0xFEFF) {
        return true;
    }
    if (0xD800 <= cp && cp <= 0xDFFF) {
        return true;
    }
    return false;
}

std::vector<uint32_t> decode_utf8(const std::string & s) {
    std::vector<uint32_t> out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        uint32_t cp = 0;
        size_t len = 0;
        if (c < 0x80) {
            cp = c;
            len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp = c & 0x1F;
            len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0F;
            len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            cp = c & 0x07;
            len = 4;
        } else {
            cp = 0xFFFD;
            len = 1;
        }
        if (i + len > s.size()) {
            cp = 0xFFFD;
            len = 1;
        } else {
            for (size_t j = 1; j < len; ++j) {
                unsigned char cc = static_cast<unsigned char>(s[i + j]);
                if ((cc & 0xC0) != 0x80) {
                    cp = 0xFFFD;
                    len = j;
                    break;
                }
                cp = (cp << 6) | (cc & 0x3F);
            }
        }
        out.push_back(cp);
        i += len;
    }
    return out;
}

std::string encode_utf8(const std::vector<uint32_t> & cps) {
    std::string out;
    out.reserve(cps.size() * 2);
    for (uint32_t cp : cps) {
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    return out;
}

std::string normalize_whitespace(const std::string & text) {
    auto cps = decode_utf8(text);
    std::vector<uint32_t> out;
    out.reserve(cps.size());
    size_t i = 0;
    while (i < cps.size()) {
        if (is_space(cps[i])) {
            size_t j = i;
            bool has_line_break = false;
            while (j < cps.size() && is_space(cps[j])) {
                if (is_line_break(cps[j])) {
                    has_line_break = true;
                }
                ++j;
            }
            uint32_t left = (i > 0) ? cps[i - 1] : 0;
            uint32_t right = (j < cps.size()) ? cps[j] : 0;
            bool left_cjk = left != 0 && is_cjk(left);
            bool right_cjk = right != 0 && is_cjk(right);
            if (!(has_line_break && left_cjk && right_cjk)) {
                out.push_back(0x20);
            }
            i = j;
        } else {
            out.push_back(cps[i]);
            ++i;
        }
    }
    size_t start = 0;
    while (start < out.size() && out[start] == 0x20) {
        ++start;
    }
    size_t end = out.size();
    while (end > start && out[end - 1] == 0x20) {
        --end;
    }
    std::vector<uint32_t> trimmed(out.begin() + static_cast<std::ptrdiff_t>(start),
                                  out.begin() + static_cast<std::ptrdiff_t>(end));
    return encode_utf8(trimmed);
}

std::string clean_text(const std::string & value, const char * field_name = "text") {
    auto cps = decode_utf8(value);
    std::vector<uint32_t> filtered;
    filtered.reserve(cps.size());
    for (uint32_t cp : cps) {
        if (is_space(cp)) {
            filtered.push_back(cp);
        } else if (is_control_category(cp)) {
            continue;
        } else {
            filtered.push_back(cp);
        }
    }
    std::string intermediate = encode_utf8(filtered);
    std::string cleaned = normalize_whitespace(intermediate);
    if (cleaned.empty()) {
        throw std::runtime_error(std::string(field_name) + " must not be empty");
    }
    return cleaned;
}

std::string reference_text_with_speakers(const std::string & text, int64_t speaker) {
    std::string cleaned = clean_text(text, "reference_text");
    static const std::regex speaker_re(R"(<\|speaker:\d+\|>)");
    if (std::regex_search(cleaned, speaker_re)) {
        return cleaned;
    }
    return "<|speaker:" + std::to_string(speaker) + "|>" + cleaned;
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
        append_tokens(row0, tokenizer_.encode(clean_text(previous_turn->text)));
        append_tokens(row0, tokenizer_.encode("<|im_end|>\n"));
        append_tokens(row0, tokenizer_.encode("<|im_start|>assistant\n<|voice|>"));
        append_code_span(row0, code_spans, tokenizer_, previous_turn->codes, assets_->config.fast.num_codebooks);
        append_tokens(row0, tokenizer_.encode("<|im_end|>\n"));
    }
    append_tokens(row0, tokenizer_.encode("<|im_start|>user\n"));
    append_tokens(row0, tokenizer_.encode(clean_text(request.text)));
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
