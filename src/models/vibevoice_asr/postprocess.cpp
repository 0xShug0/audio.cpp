#include "engine/models/vibevoice_asr/postprocess.h"

#include "engine/framework/io/json.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace engine::models::vibevoice_asr {
namespace {

std::string trim(std::string value) {
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) {
        return std::isspace(ch) == 0;
    }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) {
        return std::isspace(ch) == 0;
    }).base(), value.end());
    return value;
}

std::string extract_json_payload(const std::string & text) {
    const size_t start = text.find('[') != std::string::npos ? text.find('[') : text.find('{');
    if (start == std::string::npos) {
        return {};
    }
    int depth = 0;
    for (size_t i = start; i < text.size(); ++i) {
        if (text[i] == '[' || text[i] == '{') {
            ++depth;
        } else if (text[i] == ']' || text[i] == '}') {
            --depth;
            if (depth == 0) {
                return text.substr(start, i - start + 1);
            }
        }
    }
    return {};
}

std::string optional_string_any_key(
    const engine::io::json::Value & object,
    std::initializer_list<const char *> keys) {
    for (const char * key : keys) {
        const auto * value = object.find(key);
        if (value == nullptr) {
            continue;
        }
        if (value->is_string()) {
            return value->as_string();
        }
        if (value->is_number()) {
            const double number = value->as_number();
            if (std::isfinite(number) && std::floor(number) == number) {
                return std::to_string(static_cast<int64_t>(number));
            }
            return std::to_string(number);
        }
    }
    return {};
}

double optional_f64_any_key(
    const engine::io::json::Value & object,
    std::initializer_list<const char *> keys) {
    const std::string value = optional_string_any_key(object, keys);
    if (value.empty()) {
        return 0.0;
    }
    return std::stod(value);
}

bool speaker_label_at(const std::string & text, size_t pos, size_t * label_end, std::string * speaker_id) {
    constexpr std::string_view prefix = "Speaker";
    if (text.compare(pos, prefix.size(), prefix) != 0) {
        return false;
    }
    pos += prefix.size();
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
        ++pos;
    }
    const size_t id_start = pos;
    while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos])) != 0) {
        ++pos;
    }
    if (pos == id_start) {
        return false;
    }
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
        ++pos;
    }
    if (pos >= text.size() || text[pos] != ':') {
        return false;
    }
    *label_end = pos + 1;
    *speaker_id = text.substr(id_start, pos - id_start);
    return true;
}

size_t find_speaker_label(const std::string & text, size_t start, size_t * label_end, std::string * speaker_id) {
    size_t pos = start;
    while ((pos = text.find("Speaker", pos)) != std::string::npos) {
        if (speaker_label_at(text, pos, label_end, speaker_id)) {
            return pos;
        }
        ++pos;
    }
    return std::string::npos;
}

}  // namespace

VibeVoiceASRPostprocessor::VibeVoiceASRPostprocessor(const VibeVoiceASRTextTokenizer & tokenizer)
    : tokenizer_(tokenizer) {}

VibeVoiceASRDecoded VibeVoiceASRPostprocessor::decode(const VibeVoiceASRGeneratedTokens & tokens) const {
    VibeVoiceASRDecoded out;
    out.raw_text = tokenizer_.decode(tokens.token_ids, true);
    const std::string payload = extract_json_payload(out.raw_text);
    if (!payload.empty()) {
        const auto root = engine::io::json::parse(payload);
        const auto parse_item = [&](const engine::io::json::Value & item) {
            if (!item.is_object()) {
                return;
            }
            VibeVoiceASRSegment segment;
            segment.start_time = optional_f64_any_key(item, {"Start time", "Start", "start_time"});
            segment.end_time = optional_f64_any_key(item, {"End time", "End", "end_time"});
            segment.speaker_id = optional_string_any_key(item, {"Speaker ID", "Speaker", "speaker_id"});
            segment.text = trim(optional_string_any_key(item, {"Content", "text"}));
            if (!segment.text.empty()) {
                out.segments.push_back(std::move(segment));
            }
        };
        if (root.is_array()) {
            for (const auto & item : root.as_array()) {
                parse_item(item);
            }
        } else {
            parse_item(root);
        }
    }
    for (const auto & segment : out.segments) {
        if (!out.text.empty()) {
            out.text += " ";
        }
        out.text += segment.text;
    }
    if (out.text.empty()) {
        out.text = trim(out.raw_text);
    }
    return out;
}

std::vector<VibeVoiceASRSegment> VibeVoiceASRPostprocessor::decode_speaker_attributed_text(
    const std::string & text) const {
    std::vector<VibeVoiceASRSegment> segments;
    size_t label_end = 0;
    std::string speaker_id;
    size_t label_pos = find_speaker_label(text, 0, &label_end, &speaker_id);
    while (label_pos != std::string::npos) {
        size_t next_label_end = 0;
        std::string next_speaker_id;
        const size_t next_label_pos = find_speaker_label(text, label_end, &next_label_end, &next_speaker_id);
        VibeVoiceASRSegment segment;
        segment.speaker_id = std::move(speaker_id);
        segment.text = trim(text.substr(
            label_end,
            next_label_pos == std::string::npos ? std::string::npos : next_label_pos - label_end));
        if (!segment.text.empty()) {
            segments.push_back(std::move(segment));
        }
        label_pos = next_label_pos;
        label_end = next_label_end;
        speaker_id = std::move(next_speaker_id);
    }
    return segments;
}

}  // namespace engine::models::vibevoice_asr
