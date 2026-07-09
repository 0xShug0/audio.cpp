#include "engine/models/vibevoice_asr/postprocess.h"

#include "engine/framework/io/json.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>
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

}  // namespace engine::models::vibevoice_asr
