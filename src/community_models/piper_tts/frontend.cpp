#include "engine/community_models/piper_tts/frontend.h"

#include "engine/framework/audio/espeak_phonemizer.h"

#include <algorithm>
#include <cctype>
#include <deque>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace engine::models::piper_tts {
namespace {

constexpr int kEspeakPhonemesIpaUnderscore = 0x02 | ('_' << 8);
constexpr std::string_view kPunctuationMarks = "!'(),-.:;?\"";

bool is_punctuation(char value) {
    return kPunctuationMarks.find(value) != std::string_view::npos;
}

bool is_space(char value) {
    return std::isspace(static_cast<unsigned char>(value)) != 0;
}

void replace_all(
    std::string & value,
    std::string_view from,
    std::string_view to) {
    size_t position = 0;
    while (!from.empty() &&
           (position = value.find(from, position)) != std::string::npos) {
        value.replace(position, from.size(), to);
        position += to.size();
    }
}

struct PunctuationMark {
    std::string value;
    char position = 'I';
};

std::pair<std::vector<std::string>, std::vector<PunctuationMark>>
preserve_punctuation(const std::string & text) {
    std::vector<std::pair<size_t, size_t>> runs;
    for (size_t index = 0; index < text.size();) {
        if (!is_space(text[index]) && !is_punctuation(text[index])) {
            ++index;
            continue;
        }
        size_t end = index;
        bool has_punctuation = false;
        while (end < text.size() &&
               (is_space(text[end]) || is_punctuation(text[end]))) {
            has_punctuation = has_punctuation || is_punctuation(text[end]);
            ++end;
        }
        if (has_punctuation) {
            runs.emplace_back(index, end);
        }
        index = end;
    }
    if (runs.empty()) {
        return {{text}, {}};
    }
    if (runs.size() == 1 && runs.front().first == 0 &&
        runs.front().second == text.size()) {
        return {{}, {{text, 'A'}}};
    }

    std::vector<PunctuationMark> marks;
    marks.reserve(runs.size());
    for (size_t index = 0; index < runs.size(); ++index) {
        char position = 'I';
        if (index == 0 && runs[index].first == 0) {
            position = 'B';
        } else if (index + 1 == runs.size() &&
                   runs[index].second == text.size()) {
            position = 'E';
        }
        marks.push_back({
            text.substr(
                runs[index].first,
                runs[index].second - runs[index].first),
            position,
        });
    }

    std::vector<std::string> chunks;
    std::string remaining = text;
    for (const auto & mark : marks) {
        const size_t position = remaining.find(mark.value);
        if (position == std::string::npos) {
            chunks.push_back(remaining);
            remaining.clear();
            continue;
        }
        chunks.push_back(remaining.substr(0, position));
        remaining.erase(0, position + mark.value.size());
    }
    chunks.push_back(remaining);
    chunks.erase(
        std::remove(chunks.begin(), chunks.end(), std::string{}),
        chunks.end());
    return {std::move(chunks), std::move(marks)};
}

std::string restore_punctuation(
    std::vector<std::string> chunks,
    std::vector<PunctuationMark> marks) {
    std::deque<std::string> text(chunks.begin(), chunks.end());
    std::deque<PunctuationMark> pending(marks.begin(), marks.end());
    std::vector<std::string> output;
    size_t position = 0;
    while (!text.empty() || !pending.empty()) {
        if (pending.empty()) {
            for (auto & line : text) {
                if (line.empty() || line.back() != ' ') {
                    line.push_back(' ');
                }
                output.push_back(std::move(line));
            }
            text.clear();
        } else if (text.empty()) {
            std::string joined;
            for (const auto & mark : pending) {
                joined += mark.value;
            }
            output.push_back(std::move(joined));
            pending.clear();
        } else if (position == 0) {
            const auto mark = pending.front();
            pending.pop_front();
            if (!text.front().empty() && text.front().back() == ' ') {
                text.front().pop_back();
            }
            const bool ends_with_separator =
                !mark.value.empty() && mark.value.back() == ' ';
            if (mark.position == 'B') {
                text.front() = mark.value + text.front();
            } else if (mark.position == 'E') {
                output.push_back(
                    text.front() + mark.value +
                    (ends_with_separator ? "" : " "));
                text.pop_front();
                ++position;
            } else if (mark.position == 'A') {
                output.push_back(
                    mark.value + (ends_with_separator ? "" : " "));
                ++position;
            } else if (text.size() == 1) {
                text.front() += mark.value;
            } else {
                auto first = std::move(text.front());
                text.pop_front();
                text.front() = first + mark.value + text.front();
            }
        } else {
            auto & line = text.front();
            if (line.empty() || line.back() != ' ') {
                line.push_back(' ');
            }
            output.push_back(std::move(line));
            text.pop_front();
            ++position;
        }
    }
    std::string restored;
    for (const auto & line : output) {
        restored += line;
    }
    return restored;
}

std::string postprocess_espeak(std::string text) {
    const auto not_space = [](unsigned char value) {
        return std::isspace(value) == 0;
    };
    text.erase(text.begin(), std::find_if(text.begin(), text.end(), not_space));
    text.erase(
        std::find_if(text.rbegin(), text.rend(), not_space).base(),
        text.end());
    std::replace(text.begin(), text.end(), '\n', ' ');
    while (text.find("  ") != std::string::npos) {
        replace_all(text, "  ", " ");
    }
    std::string squeezed;
    squeezed.reserve(text.size());
    for (const char value : text) {
        if (value == '_' && !squeezed.empty() && squeezed.back() == '_') {
            continue;
        }
        squeezed.push_back(value);
    }
    text = std::move(squeezed);
    replace_all(text, "_ ", " ");
    if (text.find('(') != std::string::npos) {
        std::string unflagged;
        for (size_t position = 0; position < text.size();) {
            if (text[position] == '(') {
                const size_t close = text.find(')', position + 1);
                if (close != std::string::npos) {
                    position = close + 1;
                    continue;
                }
            }
            unflagged.push_back(text[position++]);
        }
        text = std::move(unflagged);
    }
    if (text.empty()) {
        return text;
    }

    std::string output;
    for (size_t start = 0; start <= text.size();) {
        size_t end = text.find(' ', start);
        if (end == std::string::npos) {
            end = text.size();
        }
        auto word = text.substr(start, end - start);
        word.erase(std::remove(word.begin(), word.end(), '_'), word.end());
        output += word;
        output.push_back(' ');
        if (end == text.size()) {
            break;
        }
        start = end + 1;
    }
    return output;
}

size_t utf8_width(const std::string & text, size_t position) {
    const auto first = static_cast<unsigned char>(text[position]);
    size_t width = 0;
    if (first <= 0x7fU) {
        width = 1;
    } else if ((first & 0xe0U) == 0xc0U) {
        width = 2;
    } else if ((first & 0xf0U) == 0xe0U) {
        width = 3;
    } else if ((first & 0xf8U) == 0xf0U) {
        width = 4;
    } else {
        throw std::runtime_error("Piper TTS phonemes contain invalid UTF-8");
    }
    if (position + width > text.size()) {
        throw std::runtime_error("Piper TTS phonemes contain truncated UTF-8");
    }
    for (size_t offset = 1; offset < width; ++offset) {
        if ((static_cast<unsigned char>(text[position + offset]) & 0xc0U) !=
            0x80U) {
            throw std::runtime_error(
                "Piper TTS phonemes contain invalid UTF-8 continuation byte");
        }
    }
    return width;
}

}  // namespace

struct PiperTtsFrontend::Impl {
    audio::EspeakPhonemizer phonemizer;

    Impl(
        std::filesystem::path library,
        std::filesystem::path data,
        std::string voice)
        : phonemizer(
              std::move(library),
              std::move(data),
              {std::move(voice)}) {}
};

PiperTtsFrontend::PiperTtsFrontend(
    std::filesystem::path espeak_library_path,
    std::filesystem::path espeak_data_path,
    std::string espeak_voice,
    std::unordered_map<std::string, int32_t> phoneme_id_map,
    int64_t max_tokens)
    : impl_(std::make_unique<Impl>(
          std::move(espeak_library_path),
          std::move(espeak_data_path),
          std::move(espeak_voice))),
      id_map_(std::move(phoneme_id_map)),
      max_tokens_(std::max<int64_t>(max_tokens, 3)) {}

PiperTtsFrontend::~PiperTtsFrontend() = default;

PiperTtsEncoded PiperTtsFrontend::encode(const std::string & text) const {
    auto [chunks, marks] = preserve_punctuation(text);
    std::vector<std::string> phonemes;
    phonemes.reserve(chunks.size());
    for (const auto & chunk : chunks) {
        phonemes.push_back(postprocess_espeak(impl_->phonemizer.phonemize(
            chunk,
            kEspeakPhonemesIpaUnderscore)));
    }
    auto restored = restore_punctuation(
        std::move(phonemes),
        std::move(marks));
    while (!restored.empty() && is_space(restored.back())) {
        restored.pop_back();
    }

    PiperTtsEncoded output;
    output.token_ids = {1, 0};
    for (size_t position = 0; position < restored.size();) {
        const size_t width = utf8_width(restored, position);
        const auto found = id_map_.find(restored.substr(position, width));
        position += width;
        if (found == id_map_.end()) {
            continue;
        }
        output.token_ids.push_back(found->second);
        output.token_ids.push_back(0);
    }
    if (output.token_ids.size() == 2) {
        throw std::runtime_error(
            "Piper TTS phonemization produced no recognized symbols");
    }
    output.token_ids.push_back(2);
    if (static_cast<int64_t>(output.token_ids.size()) > max_tokens_) {
        throw std::runtime_error(
            "Piper TTS phoneme sequence exceeds the model token limit");
    }
    return output;
}

}  // namespace engine::models::piper_tts
