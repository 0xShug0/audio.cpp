#include "engine/models/lfm_d1_omni/questions.h"

#include "cJSON.h"
#include "unicode.h"
#include "engine/framework/runtime/errors.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace engine::models::lfm_d1_omni {
namespace {

namespace json = io::json;
using json::Value;
using Tokenizer = tokenizers::LlamaBpeTokenizer;
using JsonTree = std::unique_ptr<cJSON, decltype(&cJSON_Delete)>;

constexpr const char * kBreak = "\xE2\x80\x8B";
// The published ChatML template's non-thinking assistant lead, verbatim.
constexpr const char * kLead = "<think></think>";

bool whitespace(uint32_t cp) {
    // Python str.isspace includes these four C0 separators as well as Unicode whitespace.
    return unicode_cpt_flags_from_cpt(cp).is_whitespace || (cp >= 0x1c && cp <= 0x1f);
}

std::string stripped(const std::string & text) {
    size_t begin = text.size(), end = 0, at = 0;
    while (at < text.size()) {
        const size_t start = at;
        const auto cp = unicode_cpt_from_utf8(text, at);
        if (!whitespace(cp)) {
            if (begin == text.size()) { begin = start; }
            end = at;
        }
    }
    return end == 0 ? std::string{} : text.substr(begin, end - begin);
}

std::string general_number(double value) {
    // Python's :g default is six significant digits, independent of the process locale.
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(6) << std::defaultfloat << value;
    return out.str();
}

double round_even(double value) {
    const double base = std::floor(value);
    const double fraction = value - base;
    return fraction < 0.5 ? base : fraction > 0.5 ? base + 1 : std::fmod(base, 2.0) == 0 ? base : base + 1;
}

bool close(double a, double b, double absolute = 0.0) {
    return std::abs(a - b) <= std::max(absolute, 1e-9 * std::max(std::abs(a), std::abs(b)));
}

JsonTree ordered_tree(std::string_view text) {
    const char * end = nullptr;
    JsonTree tree(cJSON_ParseWithLengthOpts(text.data(), text.size(), &end, 0), &cJSON_Delete);
    if (!tree) { throw std::invalid_argument("D1 questions: invalid JSON"); }
    const auto * limit = text.data() + text.size();
    while (end < limit && (*end == ' ' || *end == '\n' || *end == '\r' || *end == '\t')) { ++end; }
    if (end != limit || !cJSON_IsObject(tree.get())) {
        throw std::invalid_argument("D1 questions: expected one JSON object");
    }
    return tree;
}

ParsedQuestion parse_fields(const Value & value, const cJSON * ordered) {
    if (!value.is_object()) { throw std::invalid_argument("D1 question must be an object"); }
    ParsedQuestion q;
    const auto & kind = value.require("type").as_string();
    if (kind == "noul") { q.kind = QuestionKind::Noul; }
    else if (kind == "choice") { q.kind = QuestionKind::Choice; }
    else if (kind == "score") { q.kind = QuestionKind::Score; }
    else if (kind == "number" && value.find("range")) { q.kind = QuestionKind::Estimate; }
    else { throw std::invalid_argument("D1 unsupported question type: " + kind); }

    const auto & instructions = value.require("instructions");
    if (!instructions.is_string() || stripped(instructions.as_string()).empty()) {
        throw std::invalid_argument("D1 instructions must be nonempty text; media instructions must be prepared outside questions");
    }
    q.instructions = instructions.as_string();  // Preserve the original whitespace.
    if (value.find("max")) { throw std::invalid_argument("D1 max is only supported by boxes questions"); }
    if (const auto * multiple = value.find("multiple")) {
        if (!multiple->is_bool()) { throw std::invalid_argument("D1 multiple must be a boolean"); }
        if (multiple->as_bool()) {
            if (q.kind != QuestionKind::Choice) { throw std::invalid_argument("D1 multiple belongs to choice"); }
            q.kind = QuestionKind::MultipleChoice;
        }
    }
    const auto * criteria = value.find("criteria");
    if (q.kind == QuestionKind::Choice || q.kind == QuestionKind::MultipleChoice) {
        if (!criteria || !criteria->is_object() || criteria->as_object().empty()) {
            throw std::invalid_argument("D1 choice criteria must be a nonempty object");
        }
        const cJSON * entries = cJSON_GetObjectItemCaseSensitive(ordered, "criteria");
        for (const cJSON * item = entries->child; item; item = item->next) {
            const std::string name(item->string);
            const auto & description = criteria->require(name);
            if (!description.is_null() && !description.is_string()) {
                throw std::invalid_argument("D1 choice descriptions must be text or null");
            }
            q.labels.push_back(name);
            q.descriptions.push_back(description.is_null() ? std::nullopt : std::optional<std::string>(description.as_string()));
        }
    } else if (q.kind == QuestionKind::Score) {
        if (!criteria || !criteria->is_array() || criteria->as_array().size() < 2 || criteria->as_array().size() > 10) {
            throw std::invalid_argument("D1 score criteria must list 2 to 10 levels, lowest first");
        }
        for (const auto & item : criteria->as_array()) { q.labels.push_back(item.as_string()); }
    } else if (q.kind == QuestionKind::Noul && criteria && !criteria->is_null()) {
        if (!criteria->is_object()) { throw std::invalid_argument("D1 noul criteria must be an object"); }
        if (!criteria->as_object().empty()) {
            for (auto entry : {std::pair{"true", &q.yes_description}, std::pair{"false", &q.no_description}}) {
                const auto * description = criteria->find(entry.first);
                if (description && !description->is_null() && !description->is_string()) {
                    throw std::invalid_argument("D1 noul descriptions must be text or null");
                }
                *entry.second = description && !description->is_null() ? description->as_string() : "None";
            }
        }
    }
    if (value.find("step") && !value.find("range")) { throw std::invalid_argument("D1 step needs a number range"); }
    if (value.find("range") && q.kind != QuestionKind::Estimate) { throw std::invalid_argument("D1 range belongs to number estimates"); }
    if (q.kind == QuestionKind::Estimate) {
        const auto & bounds = value.require("range").as_array();
        if (bounds.size() != 2) { throw std::invalid_argument("D1 range must contain low and high"); }
        const double low = bounds[0].as_number(), high = bounds[1].as_number();
        if (!std::isfinite(low) || !std::isfinite(high) || !(low < high) || !std::isfinite(high - low)) {
            throw std::invalid_argument("D1 estimate range must be finite and increasing");
        }
        size_t size = 128;
        if (const auto * step = value.find("step"); step && !step->is_null()) {
            const double spacing = step->as_number();
            const double intervals = (high - low) / spacing;
            if (!std::isfinite(spacing) || !(spacing > 0) || !std::isfinite(intervals) || !close(intervals, round_even(intervals))) {
                throw std::invalid_argument("D1 estimate step must be positive and divide its range");
            }
            const double count = round_even(intervals) + 1;
            if (count < 2) { throw std::invalid_argument("D1 estimate grid needs at least two values"); }
            size = static_cast<size_t>(std::min(128.0, count));
        }
        const double spacing = (high - low) / static_cast<double>(size - 1);
        for (size_t i = 0; i + 1 < size; ++i) { q.grid.push_back(low + static_cast<double>(i) * spacing); }
        q.grid.push_back(high);
        if (const auto * unit = value.find("unit"); unit && !unit->is_null()) { q.unit = unit->as_string(); }
    }
    return q;
}

std::vector<int32_t> forms(const Tokenizer & tokenizer, const std::vector<std::string> & texts) {
    std::vector<int32_t> ids;
    for (const auto & text : texts) {
        const auto encoded = tokenizer.encode(text, false);
        if (encoded.size() == 1 && std::find(ids.begin(), ids.end(), encoded[0]) == ids.end()) { ids.push_back(encoded[0]); }
    }
    return ids;
}

std::vector<std::pair<std::string, int32_t>> aliases(const Tokenizer & tokenizer, const std::vector<std::string> & labels, int scheme) {
    bool letters = !labels.empty();
    std::vector<std::string> codes;
    for (const auto & label : labels) {
        codes.push_back(stripped(label));
        const auto cps = unicode_cpts_from_utf8(codes.back());
        letters = letters && cps.size() == 1 && unicode_cpt_flags_from_cpt(cps[0]).is_letter;
    }
    if (!letters) {
        codes.clear();
        for (size_t i = 0; i < labels.size(); ++i) {
            if (labels.size() <= 26 || (scheme == 2 && i < 26)) { codes.push_back(std::string(1, static_cast<char>('A' + i))); }
            else if (scheme == 2) {
                if (i >= 702) { throw std::invalid_argument("D1 code scheme 2 supports at most 702 non-letter options"); }
                codes.push_back(std::string(1, static_cast<char>('A' + (i - 26) / 26)) + static_cast<char>('A' + (i - 26) % 26));
            } else { codes.push_back((i < 10 ? "0" : "") + std::to_string(i)); }
        }
    }
    std::vector<std::string> pool;
    for (char c = 'A'; c <= 'Z'; ++c) { pool.emplace_back(1, c); }
    for (int i = 0; i < 100; ++i) { pool.push_back((i < 10 ? "0" : "") + std::to_string(i)); }
    for (char c = 'a'; c <= 'z'; ++c) { pool.emplace_back(1, c); }
    for (int i = 0; i < 200; ++i) { pool.push_back("#" + std::to_string(i)); }
    for (char a = 'A'; a <= 'Z'; ++a) {
        for (char b = 'A'; b <= 'Z'; ++b) { pool.push_back(std::string(1, a) + b); }
    }
    std::unordered_set<int32_t> used;
    std::vector<std::pair<std::string, int32_t>> out;
    for (const auto & code : codes) {
        auto take = [&](const std::string & raw) {
            const auto ids = tokenizer.encode(raw, false);
            if (ids.size() != 1 || !used.insert(ids[0]).second) { return false; }
            out.emplace_back(raw, ids[0]);
            return true;
        };
        if (!take(code) && std::none_of(pool.begin(), pool.end(), take)) {
            throw std::runtime_error("D1 tokenizer has no distinct single-token alias left");
        }
    }
    return out;
}

std::string plain(const QuestionConfig & config, std::string text) {
    auto names = config.special_tokens;
    names.erase(std::remove_if(names.begin(), names.end(), [](const std::string & name) {
        return unicode_cpts_from_utf8(name).size() <= 1 || name.find(kBreak) != std::string::npos;
    }), names.end());
    std::stable_sort(names.begin(), names.end(), [](const std::string & a, const std::string & b) {
        return unicode_cpts_from_utf8(a).size() > unicode_cpts_from_utf8(b).size();
    });
    for (;;) {
        std::string escaped;
        bool changed = false;
        for (size_t at = 0; at < text.size();) {
            const auto match = std::find_if(names.begin(), names.end(), [&](const std::string & name) {
                return text.compare(at, name.size(), name) == 0;
            });
            if (match == names.end()) { escaped.push_back(text[at++]); continue; }
            size_t first_end = 0;
            (void)unicode_cpt_from_utf8(*match, first_end);
            escaped += match->substr(0, first_end) + kBreak + match->substr(first_end);
            at += match->size();
            changed = true;
        }
        if (!changed) { return text; }
        text = std::move(escaped);
    }
}

std::string bucket(QuestionKind kind, size_t count) {
    if (kind == QuestionKind::Noul || kind == QuestionKind::MultipleChoice) { return "noul"; }
    if (kind == QuestionKind::Score) { return "score:" + std::to_string(count); }
    return count <= 3 ? "choice:small" : count <= 8 ? "choice:mid" : count <= 26 ? "choice:large" : "choice:inventory";
}

double through(const Value & calibration, const std::string & name, double value, bool inverse = false) {
    if (calibration.is_null()) { return value; }
    const auto * tables = calibration.find("tables");
    if (tables && !tables->is_object()) { throw std::invalid_argument("D1 calibration tables must be an object"); }
    const auto * table = tables ? tables->find(name) : nullptr;
    if (!table || table->is_null()) { return value; }
    const auto & knots = table->as_array();
    if (knots.empty()) { return value; }
    const size_t x = inverse ? 1 : 0, y = inverse ? 0 : 1;
    std::vector<std::pair<double, double>> points;
    for (const auto & knot : knots) {
        const auto & pair = knot.as_array();
        if (pair.size() != 2) { throw std::invalid_argument("D1 calibration knot needs two values"); }
        const double a = pair[x].as_number(), b = pair[y].as_number();
        if (!std::isfinite(a) || !std::isfinite(b) || a < 0 || a > 1 || b < 0 || b > 1 ||
            (!points.empty() && (a < points.back().first || b < points.back().second))) {
            throw std::invalid_argument("D1 calibration knots must be finite monotone probability pairs");
        }
        points.emplace_back(a, b);
    }
    const auto at = std::lower_bound(points.begin(), points.end(), value,
        [](const auto & point, double v) { return point.first < v; });
    if (at == points.begin()) { return at->second; }
    if (at == points.end()) { return points.back().second; }
    const auto & before = *(at - 1);
    return before.second + (value - before.first) * (at->second - before.second) / (at->first - before.first);
}

}  // namespace

std::vector<NamedQuestion> parse_questions(std::string_view text) {
    try {
        const auto tree = ordered_tree(text);
        const auto value = json::parse(text);
        std::vector<NamedQuestion> out;
        for (const cJSON * item = tree->child; item; item = item->next) {
            const std::string name(item->string);
            out.push_back({name, parse_fields(value.require(name), item)});
        }
        return out;
    } catch (const std::invalid_argument & error) {
        throw runtime::InvalidRequestError(error.what());
    } catch (const std::runtime_error & error) {
        throw runtime::InvalidRequestError(error.what());
    }
}

QuestionSlots render_question(const ParsedQuestion & q, const Tokenizer & tokenizer, const QuestionConfig & config) {
    if ((config.codes != 1 && config.codes != 2) || (config.prompt != 1 && config.prompt != 2)) {
        throw std::invalid_argument("D1 unsupported code scheme or prompt version");
    }
    QuestionSlots out;
    out.question = q;
    std::string body = q.instructions;
    std::vector<std::vector<int32_t>> groups;
    std::vector<std::string> markers;
    QuestionKind read_kind = q.kind;
    auto yes_no = [&] {
        auto yes = forms(tokenizer, {"yes", "Yes", "YES"});
        auto no = forms(tokenizer, {"no", "No", "NO"});
        if (yes.empty() || no.empty()) { throw std::runtime_error("D1 tokenizer needs single-token yes and no"); }
        return std::vector<std::vector<int32_t>>{std::move(yes), std::move(no)};
    };
    if (q.kind == QuestionKind::Noul) {
        if (q.yes_description) { body += "\nYes: " + *q.yes_description + "\nNo: " + *q.no_description; }
        body += "\n\nReply with yes or no only.";
        groups = yes_no();
    } else if (q.kind == QuestionKind::MultipleChoice) {
        body += "\n\nOptions:";
        for (size_t i = 0; i < q.labels.size(); ++i) {
            const auto & desc = q.descriptions.at(i);
            const auto name = json::stringify_string(q.labels[i]);
            body += "\n" + name + ": " + (desc && !desc->empty() ? *desc : q.labels[i]);
            markers.push_back(name);
        }
        body += "\n\nReply with yes or no for each option.";
        groups = yes_no();
    } else if (q.kind == QuestionKind::Score) {
        body += "\n";
        for (size_t i = 0; i < q.labels.size(); ++i) {
            body += "\n" + std::to_string(i) + " " + q.labels[i];
            groups.push_back(forms(tokenizer, {std::to_string(i)}));
        }
        body += "\n\nReply with a single digit 0-" + std::to_string(q.labels.size() - 1) + " only.";
    } else {
        std::vector<std::string> labels = q.labels;
        std::vector<std::pair<std::string, int32_t>> codes;
        if (q.kind == QuestionKind::Estimate) {
            if (q.grid.size() < 2 || q.grid.size() > 128) { throw std::invalid_argument("D1 estimate needs 2 to 128 grid values"); }
            if (config.numbers) {
                for (const auto value : q.grid) {
                    const double integer = round_even(value);
                    if (!close(integer, value, 1e-6) || std::abs(integer) >= 9.22e18) { groups.clear(); break; }
                    const auto text = std::to_string(static_cast<int64_t>(integer));
                    auto group = forms(tokenizer, {text, " " + text});
                    if (tokenizer.encode(text, false).size() != 1 || group.empty()) { groups.clear(); break; }
                    groups.push_back(std::move(group));
                }
            }
            const auto low = general_number(q.grid.front()), high = general_number(q.grid.back());
            const auto step = general_number((q.grid.back() - q.grid.front()) / static_cast<double>(q.grid.size() - 1));
            const auto unit = q.unit ? " " + *q.unit : "";
            body += "\n\nEstimate from " + low + " to " + high + unit;
            if (!groups.empty()) { body += ", in steps of " + step + ". Reply with the number only."; }
            else {
                labels.clear();
                for (size_t i = 0; i < q.grid.size(); ++i) { labels.push_back(std::to_string(i)); }
                if (q.grid.size() <= 10) {
                    read_kind = QuestionKind::Score;
                    for (const auto & label : labels) { groups.push_back(forms(tokenizer, {label})); }
                } else { codes = aliases(tokenizer, labels, config.codes); }
                const auto first = codes.empty() ? "0" : codes.front().first;
                const auto last = codes.empty() ? std::to_string(q.grid.size() - 1) : codes.back().first;
                body += " on " + std::to_string(q.grid.size()) + " equally spaced values, step " + step + ": " +
                    first + " is " + low + ", " + last + " is " + high + ". Reply with the code only.";
            }
        } else {
            codes = aliases(tokenizer, labels, config.codes);
            body += "\n\nOptions:";
            for (size_t i = 0; i < labels.size(); ++i) {
                std::string text = q.descriptions.at(i).value_or("");
                if (text.empty()) {
                    text = labels[i];
                    std::replace(text.begin(), text.end(), '_', ' ');
                }
                body += "\n" + codes[i].first + " " + text;
            }
            body += "\n\nReply with the option code only.";
        }
        for (const auto & code : codes) {
            auto group = forms(tokenizer, {" " + code.first});
            group.erase(std::remove(group.begin(), group.end(), code.second), group.end());
            group.insert(group.begin(), code.second);
            groups.push_back(std::move(group));
        }
    }
    if (groups.empty() || std::any_of(groups.begin(), groups.end(), [](const auto & group) { return group.empty(); })) {
        throw std::runtime_error("D1 question has no single-token candidate for an option");
    }
    out.suffix = plain(config, body) + "<|im_end|>\n<|im_start|>assistant\n" + kLead;
    std::vector<size_t> colons;
    for (size_t i = 0; i < markers.size(); ++i) {
        out.suffix += (i ? "\n" : "") + plain(config, markers[i]) + ":";
        colons.push_back(out.suffix.size() - 1);
    }
    out.suffix_ids = tokenizer.encode(out.suffix, true);
    if (out.suffix_ids.empty()) { throw std::runtime_error("D1 suffix tokenization is empty"); }
    if (markers.empty()) { out.slots.push_back({0, groups, bucket(read_kind, groups.size())}); }
    else {
        // Byte-level BPE decoding reconstructs exact byte offsets, including colon+newline tokens.
        // Unlike separately encoding prefixes, this locates markers in the final complete suffix.
        std::string decoded;
        size_t next = 0;
        for (size_t i = 0; i < out.suffix_ids.size(); ++i) {
            const size_t begin = decoded.size();
            const auto piece = tokenizer.decode({out.suffix_ids[i]}, false);
            decoded += piece;
            if (next < colons.size() && begin <= colons[next] && colons[next] < decoded.size()) {
                if (!stripped(decoded.substr(colons[next] + 1)).empty()) {
                    throw std::runtime_error("D1 slot token must end in its colon and optional whitespace");
                }
                out.slots.push_back({static_cast<int64_t>(i) - static_cast<int64_t>(out.suffix_ids.size()) + 1, groups, "noul"});
                ++next;
            }
        }
        if (decoded != out.suffix || next != colons.size()) { throw std::runtime_error("D1 slot offsets require lossless byte-level token decoding"); }
    }
    return out;
}

Value decode_question(const QuestionSlots & question, const std::vector<SlotScores> & scores,
                      const Value & calibration, const std::string & score_readout) {
    if (score_readout != "mean" && score_readout != "median" && score_readout != "mode") {
        throw std::invalid_argument("D1 score_readout must be mean, median or mode");
    }
    if (!calibration.is_null() && !calibration.is_object()) { throw std::invalid_argument("D1 calibration must be an object or null"); }
    if (scores.size() != question.slots.size() || scores.empty()) { throw std::invalid_argument("D1 readout needs scores for every slot"); }
    const auto & q = question.question;
    std::vector<std::vector<double>> distributions;
    for (size_t slot = 0; slot < scores.size(); ++slot) {
        const auto & row = scores[slot];
        if (row.token_ids.size() != row.logits.size()) { throw std::invalid_argument("D1 token IDs and logits have different lengths"); }
        std::unordered_map<int32_t, double> lookup;
        for (size_t i = 0; i < row.token_ids.size(); ++i) {
            if (row.token_ids[i] < 0 || !lookup.emplace(row.token_ids[i], row.logits[i]).second) {
                throw std::invalid_argument("D1 score IDs must be nonnegative and unique");
            }
        }
        double temperature = 1.0;
        if (q.kind != QuestionKind::Estimate && q.kind != QuestionKind::MultipleChoice && !calibration.is_null()) {
            if (const auto * value = calibration.find("default")) { temperature = value->as_number(); }
            if (const auto * temperatures = calibration.find("temperatures")) {
                if (!temperatures->is_object()) { throw std::invalid_argument("D1 calibration temperatures must be an object"); }
                if (const auto * value = temperatures->find(question.slots[slot].temperature_bucket)) { temperature = value->as_number(); }
            }
        }
        if (!std::isfinite(temperature) || temperature <= 0) { throw std::invalid_argument("D1 temperature must be finite and positive"); }
        std::vector<double> p;
        for (const auto & group : question.slots[slot].candidate_groups) {
            if (group.empty()) { throw std::invalid_argument("D1 candidate group is empty"); }
            double best = -std::numeric_limits<double>::infinity();
            for (const auto id : group) {
                const auto found = lookup.find(id);
                if (found == lookup.end()) { throw std::invalid_argument("D1 readout is missing candidate token " + std::to_string(id)); }
                if (std::isnan(found->second) || found->second == std::numeric_limits<double>::infinity()) {
                    throw std::invalid_argument("D1 candidate logit is NaN or positive infinity");
                }
                best = std::max(best, found->second);
            }
            p.push_back(best / temperature);
        }
        if (p.empty()) { throw std::invalid_argument("D1 slot has no candidate groups"); }
        const double peak = *std::max_element(p.begin(), p.end());
        if (!std::isfinite(peak)) { throw std::invalid_argument("D1 slot has no finite candidate score"); }
        double total = 0;
        for (auto & value : p) { value = std::exp(value - peak); total += value; }
        for (auto & value : p) { value /= total; }
        distributions.push_back(std::move(p));
    }
    const auto & p = distributions.front();
    if (q.kind == QuestionKind::Noul) { return Value::make_object({{"type", Value::make_string("noul")}, {"noul", Value::make_number(p.at(0))}}); }
    if (q.kind == QuestionKind::MultipleChoice) {
        if (distributions.size() != q.labels.size()) { throw std::invalid_argument("D1 multiple choice needs one slot per criterion"); }
        Value::Object probabilities;
        Value::Array chosen;
        for (size_t i = 0; i < q.labels.size(); ++i) {
            const double yes = through(calibration, "multiple", distributions[i].at(0));
            probabilities.emplace(q.labels[i], Value::make_number(yes));
            if (yes >= 0.5) { chosen.push_back(Value::make_string(q.labels[i])); }
        }
        return Value::make_object({{"type", Value::make_string("choice")}, {"choices", Value::make_array(std::move(chosen))},
                                   {"probabilities", Value::make_object(std::move(probabilities))}});
    }
    const size_t best = static_cast<size_t>(std::max_element(p.begin(), p.end()) - p.begin());
    if (q.kind == QuestionKind::Estimate) {
        if (p.size() != q.grid.size()) { throw std::invalid_argument("D1 grid readout size mismatch"); }
        // Grid.read normalizes again with math.fsum before accumulating quantiles.
        // Positive terms permit compensated summation without a general sum utility.
        double total = 0, correction = 0;
        for (const auto value : p) {
            const double next = total + value;
            correction += total >= value ? (total - next) + value : (value - next) + total;
            total = next;
        }
        total += correction;
        Value::Array interval;
        double median = q.grid.back();
        for (const double share : {0.1, 0.5, 0.9}) {
            const double level = through(calibration, "estimate", share, true);
            double mass = 0, value = q.grid.back();
            for (size_t i = 0; i < p.size(); ++i) { mass += p[i] / total; if (mass >= level) { value = q.grid[i]; break; } }
            if (share == 0.5) { median = value; } else { interval.push_back(Value::make_number(value)); }
        }
        return Value::make_object({{"type", Value::make_string("number")}, {"number", Value::make_number(median)},
            {"interval", Value::make_array(std::move(interval))}, {"unit", q.unit ? Value::make_string(*q.unit) : Value::make_null()}});
    }
    if (p.size() != q.labels.size() || distributions.size() != 1) { throw std::invalid_argument("D1 option readout size mismatch"); }
    Value::Object probabilities;
    for (size_t i = 0; i < p.size(); ++i) { probabilities.emplace(q.kind == QuestionKind::Score ? std::to_string(i) : q.labels[i], Value::make_number(p[i])); }
    if (q.kind == QuestionKind::Choice) {
        return Value::make_object({{"type", Value::make_string("choice")}, {"choice", Value::make_string(q.labels[best])},
            {"confidence", Value::make_number(p[best])}, {"probabilities", Value::make_object(std::move(probabilities))}});
    }
    double level = static_cast<double>(best);
    if (score_readout == "mean") { level = 0; for (size_t i = 0; i < p.size(); ++i) { level += static_cast<double>(i) * p[i]; } }
    else if (score_readout == "median") {
        double total = 0;
        for (const auto value : p) { total += value; }
        double mass = 0;
        level = static_cast<double>(p.size() - 1);
        for (size_t i = 0; i < p.size(); ++i) { mass += p[i] / total; if (mass >= 0.5) { level = static_cast<double>(i); break; } }
    }
    Value::Object legend;
    for (size_t i = 0; i < q.labels.size(); ++i) { legend.emplace(std::to_string(i), Value::make_string(q.labels[i])); }
    return Value::make_object({{"type", Value::make_string("score")}, {"score", Value::make_number(level)},
        {"confidence", Value::make_number(p[best])}, {"probabilities", Value::make_object(std::move(probabilities))},
        {"legend", Value::make_object(std::move(legend))}});
}

}  // namespace engine::models::lfm_d1_omni
