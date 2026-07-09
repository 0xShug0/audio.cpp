#include "engine/models/higgs_audio_stt/postprocess.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace engine::models::higgs_audio_stt {
namespace {

std::string trim(std::string value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    }).base();
    if (first >= last) {
        return {};
    }
    return std::string(first, last);
}

void erase_all_between(std::string & value, const std::string & begin, const std::string & end) {
    size_t start = value.find(begin);
    while (start != std::string::npos) {
        const size_t stop = value.find(end, start + begin.size());
        if (stop == std::string::npos) {
            break;
        }
        value.erase(start, stop + end.size() - start);
        start = value.find(begin, start);
    }
}

std::vector<std::string> split_words(const std::string & text) {
    std::vector<std::string> words;
    size_t pos = 0;
    while (pos < text.size()) {
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
            ++pos;
        }
        const size_t start = pos;
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) == 0) {
            ++pos;
        }
        if (start < pos) {
            words.push_back(text.substr(start, pos - start));
        }
    }
    return words;
}

std::string join_words(const std::vector<std::string> & words) {
    std::string out;
    for (const auto & word : words) {
        if (!out.empty()) {
            out.push_back(' ');
        }
        out += word;
    }
    return out;
}

std::vector<std::string> collapse_ngram(std::vector<std::string> words, int n, int max_rep) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < words.size()) {
        bool skip = false;
        if (out.size() >= static_cast<size_t>(n) && i + static_cast<size_t>(n) <= words.size()) {
            bool same = true;
            for (int j = 0; j < n; ++j) {
                if (words[i + static_cast<size_t>(j)] != out[out.size() - static_cast<size_t>(n) + static_cast<size_t>(j)]) {
                    same = false;
                    break;
                }
            }
            if (same) {
                int reps = 1;
                while (out.size() >= static_cast<size_t>(n * (reps + 1))) {
                    bool previous_same = true;
                    const size_t base = out.size() - static_cast<size_t>(n * (reps + 1));
                    const size_t ref = out.size() - static_cast<size_t>(n);
                    for (int j = 0; j < n; ++j) {
                        if (out[base + static_cast<size_t>(j)] != out[ref + static_cast<size_t>(j)]) {
                            previous_same = false;
                            break;
                        }
                    }
                    if (!previous_same) {
                        break;
                    }
                    ++reps;
                }
                if (reps >= max_rep) {
                    i += static_cast<size_t>(n);
                    skip = true;
                }
            }
        }
        if (!skip) {
            out.push_back(words[i]);
            ++i;
        }
    }
    return out;
}

std::string post_process_text(std::string text) {
    erase_all_between(text, "<think>", "</think>");
    const size_t open_think = text.find("<think>");
    if (open_think != std::string::npos) {
        text = text.substr(open_think + std::char_traits<char>::length("<think>"));
    }
    auto words = split_words(trim(text));
    for (int n = 16; n >= 1; --n) {
        words = collapse_ngram(std::move(words), n, n == 1 ? 3 : 2);
    }
    return join_words(words);
}

}  // namespace

HiggsAudioSTTPostprocessor::HiggsAudioSTTPostprocessor(const HiggsAudioSTTTextTokenizer & tokenizer)
    : tokenizer_(tokenizer) {}

HiggsAudioSTTResult HiggsAudioSTTPostprocessor::decode(
    const HiggsAudioSTTGeneratedTokens & tokens,
    const HiggsAudioSTTRequest & request) const {
    HiggsAudioSTTResult result;
    result.language = request.language.empty() ? "en" : request.language;
    result.text = post_process_text(tokenizer_.decode(tokens.token_ids));
    return result;
}

}  // namespace engine::models::higgs_audio_stt
