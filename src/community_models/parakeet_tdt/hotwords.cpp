#include "engine/community_models/parakeet_tdt/hotwords.h"
#include "engine/framework/io/json.h"
#include "unicode.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <queue>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace engine::community_models::parakeet_tdt {
namespace {
constexpr const char * boundary = "\xE2\x96\x81";
constexpr size_t max_words = 25, max_segmentations = 6, max_states = 8192;

bool whitespace(uint32_t c) {
    return unicode_cpt_flags_from_cpt(c).is_whitespace;
}

std::string clean(const std::string & text) {
    std::string result;
    bool space = false;
    for (auto c : unicode_cpts_from_utf8(text)) {
        if (whitespace(c)) { space = !result.empty(); continue; }
        if (space) result += ' ';
        space = false;
        result += unicode_cpt_to_utf8(c);
    }
    return result;
}

std::string lower(const std::string & text) {
    std::string out;
    for (auto c : unicode_cpts_from_utf8(text)) out += unicode_cpt_to_utf8(unicode_tolower(c));
    return out;
}

bool special(const std::string & piece) {
    return piece == "<unk>" || piece == "<pad>" ||
        (piece.size() >= 4 && piece.rfind("<|", 0) == 0 && piece.substr(piece.size() - 2) == "|>");
}
} // namespace

std::vector<PhononHotword> parse_phonon_hotwords(const std::string & value) {
    if (value.size() > 16384) throw std::runtime_error("Phonon hotwords exceed 16 KiB");
    const auto text = clean(value);
    std::vector<PhononHotword> out;
    if (!text.empty() && text.front() == '[') {
        const auto json = engine::io::json::parse(value);
        for (const auto & entry : json.as_array()) {
            PhononHotword item;
            if (entry.is_string()) item.word = entry.as_string();
            else if (entry.is_object()) {
                item.word = entry.require("word").as_string();
                if (const auto * spoken = entry.find("spoken")) {
                    for (const auto & v : spoken->as_array()) item.spoken.push_back(clean(v.as_string()));
                }
            } else throw std::runtime_error("Phonon hotwords must be strings or word/spoken objects");
            item.word = clean(item.word);
            out.push_back(std::move(item));
        }
    } else {
        // An explicit comma/newline/semicolon preserves multi-word terms.
        const bool separated = value.find_first_of(",\n;") != std::string::npos;
        std::string term;
        for (auto c : unicode_cpts_from_utf8(value)) {
            if (c == ',' || c == '\n' || c == ';' || (!separated && whitespace(c))) {
                if (!clean(term).empty()) out.push_back({clean(term), {}});
                term.clear();
            } else term += unicode_cpt_to_utf8(c);
        }
        if (!clean(term).empty()) out.push_back({clean(term), {}});
    }
    std::vector<PhononHotword> distinct;
    std::unordered_set<std::string> seen;
    for (auto & entry : out) {
        if (!entry.word.empty() && seen.insert(lower(entry.word)).second && distinct.size() < max_words) {
            if (entry.word.size() > 256 || entry.spoken.size() > 8)
                throw std::runtime_error("Phonon hotword term or pronunciation list is too long");
            for (const auto & v : entry.spoken) if (v.size() > 256)
                throw std::runtime_error("Phonon hotword pronunciation is too long");
            distinct.push_back(std::move(entry));
        }
    }
    return distinct;
}

PhononHotwordAutomaton::PhononHotwordAutomaton(
    const std::vector<PhononHotword> & words, const std::vector<std::string> & vocabulary,
    int32_t blank, float strength) : blank_(blank), strength_(strength) {
    if (!std::isfinite(strength) || strength < 0 || strength > 100)
        throw std::runtime_error("Phonon hotwords_score must be finite and between 0 and 100");
    if (strength == 0 || words.empty()) return;
    std::unordered_map<std::string, int32_t> pieces;
    std::unordered_map<uint32_t, uint32_t> upper;
    size_t max_length = 0;
    for (size_t i = 0; i < vocabulary.size(); ++i) {
        if (static_cast<int32_t>(i) == blank || special(vocabulary[i]) || vocabulary[i].empty()) continue;
        pieces.emplace(vocabulary[i], static_cast<int32_t>(i));
        max_length = std::max(max_length, vocabulary[i].size());
        // Only characters spellable by this vocabulary can match a hotword.
        for (auto c : unicode_cpts_from_utf8(vocabulary[i])) {
            if (unicode_cpt_flags_from_cpt(c).is_uppercase) upper.emplace(unicode_tolower(c), c);
        }
    }
    std::set<std::vector<int32_t>> phrases;
    for (size_t wi = 0; wi < std::min(words.size(), max_words); ++wi) {
        auto surfaces = words[wi].spoken;
        surfaces.insert(surfaces.begin(), words[wi].word);
        for (const auto & surface : surfaces) {
            const auto normalized = clean(surface);
            std::string capitalized;
            bool first = true;
            for (auto c : unicode_cpts_from_utf8(normalized)) {
                if (first) {
                    const auto found = upper.find(c);
                    if (found != upper.end()) c = found->second;
                }
                capitalized += unicode_cpt_to_utf8(c);
                first = c == ' ';
            }
            for (const auto & variant : std::set<std::string>{normalized, lower(normalized), capitalized}) {
                if (variant.empty()) continue;
                std::string text = boundary;
                for (char c : variant) text += c == ' ' ? std::string(boundary) : std::string(1, c);
                using Sequences = std::vector<std::vector<int32_t>>;
                std::unordered_map<size_t, Sequences> memo;
                memo[text.size()] = {{}};
                std::function<Sequences(size_t)> segment = [&](size_t pos) -> Sequences {
                    if (auto it = memo.find(pos); it != memo.end()) return it->second;
                    Sequences result;
                    for (size_t end = std::min(text.size(), pos + max_length); end > pos; --end) {
                        auto it = pieces.find(text.substr(pos, end - pos));
                        if (it == pieces.end()) continue;
                        for (auto tail : segment(end)) {
                            tail.insert(tail.begin(), it->second);
                            result.push_back(std::move(tail));
                            if (result.size() >= 4 * max_segmentations) break;
                        }
                        if (result.size() >= 4 * max_segmentations) break;
                    }
                    std::stable_sort(result.begin(), result.end(), [](const auto & a, const auto & b) { return a.size() < b.size(); });
                    if (result.size() > max_segmentations) result.resize(max_segmentations);
                    memo[pos] = result;
                    return result;
                };
                for (auto seq : segment(0)) phrases.insert(std::move(seq));
            }
        }
    }
    for (const auto & phrase : phrases) {
        int32_t s = 0;
        for (auto t : phrase) {
            auto it = nodes_[s].children.find(t);
            if (it == nodes_[s].children.end()) {
                if (nodes_.size() >= max_states) throw std::runtime_error("Phonon hotword automaton exceeds 8192 states");
                const auto next = static_cast<int32_t>(nodes_.size());
                const auto depth = nodes_[s].depth + 1;
                nodes_[s].children.emplace(t, next);
                nodes_.push_back({{}, 0, depth});
                s = next;
            } else s = it->second;
        }
    }
    std::queue<int32_t> queue;
    std::set<int32_t> active;
    for (const auto & node : nodes_) for (const auto & [token, _] : node.children) active.insert(token);
    active_tokens_.assign(active.begin(), active.end());
    for (const auto & [_, s] : nodes_[0].children) queue.push(s);
    while (!queue.empty()) {
        const auto s = queue.front(); queue.pop();
        for (const auto & [t, child] : nodes_[s].children) {
            const auto next = step(nodes_[s].fail, t);
            nodes_[child].fail = next == child ? 0 : next;
            queue.push(child);
        }
    }
}

int32_t PhononHotwordAutomaton::step(int32_t state, int32_t token) const {
    if (token == blank_ || token < 0) return state;
    while (state && !nodes_[state].children.count(token)) state = nodes_[state].fail;
    const auto it = nodes_[state].children.find(token);
    return it == nodes_[state].children.end() ? 0 : it->second;
}

float PhononHotwordAutomaton::bonus(int32_t state, int32_t token) const {
    if (token == blank_) return 0;
    return strength_ * static_cast<float>(std::max(0, nodes_[step(state, token)].depth - nodes_[state].depth));
}

int32_t PhononHotwordAutomaton::select(const std::vector<float> & scores, int32_t state) const {
    const auto begin = scores.begin();
    int32_t best = static_cast<int32_t>(std::max_element(begin, begin + blank_ + 1) - begin);
    float score = scores.at(best);
    // Only pieces present in the vocabulary list can receive a bonus. Keep
    // the ordinary argmax fast and preserve its lowest-ID tie convention.
    for (const auto t : active_tokens_) {
        const float candidate = scores.at(t) + bonus(state, t);
        if (candidate > score || (candidate == score && t < best)) { score = candidate; best = t; }
    }
    return best;
}
} // namespace engine::community_models::parakeet_tdt
