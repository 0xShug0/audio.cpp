#include "align.h"

#include <algorithm>
#include <cstdlib>

namespace dizione {

namespace {

constexpr long kMaxRun = 4;
constexpr double kUnseen = -14.0;
constexpr double kLow = -1e18;

}  // namespace

std::string arpabet_base(std::string_view symbol) {
    size_t end = symbol.size();
    while (end > 0 && (symbol[end - 1] == '0' || symbol[end - 1] == '1' || symbol[end - 1] == '2')) {
        --end;
    }
    return std::string(symbol.substr(0, end));
}

Aligner::Aligner(const json::Value & table) {
    for (const auto & phone : table.members) {
        auto & runs = table_[phone.first];
        for (const auto & run : phone.second.members) {
            runs.emplace(run.first, run.second.number);
        }
    }
}

double Aligner::score(const std::string & phone, StrView run) const {
    const auto known = table_.find(phone);
    if (known != table_.end()) {
        const auto value = known->second.find(encode(run));
        if (value != known->second.end()) {
            return value->second;
        }
    }
    return kUnseen - 2.0 * static_cast<double>(std::labs(static_cast<long>(run.size()) - 1));
}

std::optional<std::vector<Str>> Aligner::align(StrView word, const std::vector<std::string> & symbols) const {
    std::vector<std::string> phones;
    phones.reserve(symbols.size());
    for (const auto & symbol : symbols) {
        phones.push_back(arpabet_base(symbol));
    }
    const long letters = static_cast<long>(word.size());
    const long count = static_cast<long>(phones.size());
    if (count == 0 || letters > kMaxRun * count) {
        return std::nullopt;
    }
    const auto columns = static_cast<size_t>(letters + 1);
    std::vector<std::vector<double>> best(static_cast<size_t>(count + 1), std::vector<double>(columns, kLow));
    std::vector<std::vector<long>> step(static_cast<size_t>(count + 1), std::vector<long>(columns, 0));
    best[0][0] = 0.0;
    for (long j = 1; j <= count; ++j) {
        const std::string & phone = phones[static_cast<size_t>(j - 1)];
        const auto & previous = best[static_cast<size_t>(j - 1)];
        auto & row = best[static_cast<size_t>(j)];
        auto & trace = step[static_cast<size_t>(j)];
        for (long i = 0; i <= letters; ++i) {
            for (long k = 0; k <= std::min(kMaxRun, i); ++k) {
                if (previous[static_cast<size_t>(i - k)] > kLow) {
                    const double value = previous[static_cast<size_t>(i - k)] +
                        score(phone, word.substr(static_cast<size_t>(i - k), static_cast<size_t>(k)));
                    if (value > row[static_cast<size_t>(i)]) {
                        row[static_cast<size_t>(i)] = value;
                        trace[static_cast<size_t>(i)] = k;
                    }
                }
            }
        }
    }
    if (best[static_cast<size_t>(count)][static_cast<size_t>(letters)] <= kLow) {
        return std::nullopt;
    }
    std::vector<Str> runs;
    long i = letters;
    for (long j = count; j >= 1; --j) {
        const long k = step[static_cast<size_t>(j)][static_cast<size_t>(i)];
        runs.emplace_back(word.substr(static_cast<size_t>(i - k), static_cast<size_t>(k)));
        i -= k;
    }
    std::reverse(runs.begin(), runs.end());
    return runs;
}

}  // namespace dizione
