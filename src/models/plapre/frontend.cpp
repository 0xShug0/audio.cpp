#include "engine/models/plapre/frontend.h"

#include <utf8proc.h>

#include <array>
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <regex>
#include <stdexcept>

namespace engine::models::plapre {
namespace {

std::string danish_integer(uint64_t value) {
    static const std::array<const char *, 20> units = {"nul", "et", "to", "tre", "fire", "fem", "seks", "syv",
        "otte", "ni", "ti", "elleve", "tolv", "tretten", "fjorten", "femten", "seksten", "sytten", "atten", "nitten"};
    static const std::array<const char *, 10> tens = {"", "", "tyve", "tredive", "fyrre", "halvtreds",
        "treds", "halvfjerds", "firs", "halvfems"};
    if (value < units.size()) return units[value];
    if (value < 100) {
        const auto rem = value % 10;
        return (rem ? std::string(rem == 1 ? "en" : units[rem]) + "og" : "") + tens[value / 10];
    }
    if (value < 1000) {
        return danish_integer(value / 100) + "hundrede" + (value % 100 ? " og " + danish_integer(value % 100) : "");
    }
    if (value < 1000000) {
        const auto head = danish_integer(value / 1000);
        if (value % 1000 == 0) return head + "tusind";
        return head + (head.find(' ') == std::string::npos ? "tusinde og " : "tusind") + danish_integer(value % 1000);
    }
    static const std::array<std::pair<uint64_t, const char *>, 5> scales = {{{1000000000000000000ULL, "trillioner"},
        {1000000000000000ULL, "billiarder"}, {1000000000000ULL, "billioner"}, {1000000000ULL, "milliarder"},
        {1000000ULL, "millioner"}}};
    for (const auto & [scale, name] : scales) {
        if (value >= scale) {
            const auto count = value / scale;
            return (count == 1 ? std::string("en") : danish_integer(count)) + " " + name +
                (value % scale ? " " + danish_integer(value % scale) : "");
        }
    }
    throw std::logic_error("unreachable Danish number scale");
}

}  // namespace

PlapreTextFrontend::PlapreTextFrontend(const assets::ResourceBundle & resources)
    : tokenizer_({{}, {}, resources.require_file("tokenizer_config"), resources.require_file("tokenizer_json"),
                  tokenizers::LlamaBpePreTokenizer::Gpt2}) {}

std::string PlapreTextFrontend::normalize(const std::string & text, bool preserve_line_breaks) const {
    utf8proc_uint8_t * output = nullptr;
    const auto size = utf8proc_map(reinterpret_cast<const utf8proc_uint8_t *>(text.data()),
        static_cast<utf8proc_ssize_t>(text.size()), &output,
        static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE));
    const std::unique_ptr<utf8proc_uint8_t, decltype(&std::free)> normalized(output, &std::free);
    if (size < 0) throw std::runtime_error("Plapre text normalization: " + std::string(utf8proc_errmsg(size)));
    const std::string nfc(reinterpret_cast<const char *>(output), static_cast<size_t>(size));
    const auto limit = nfc.find("--");
    std::string collapsed;
    bool space = false;
    for (size_t i = 0; i < std::min(nfc.size(), limit);) {
        utf8proc_int32_t cp = 0;
        const auto bytes = utf8proc_iterate(reinterpret_cast<const utf8proc_uint8_t *>(nfc.data() + i),
            static_cast<utf8proc_ssize_t>(nfc.size() - i), &cp);
        const auto category = utf8proc_category(cp);
        if ((cp >= 9 && cp <= 13) || cp == 0x85 || category == UTF8PROC_CATEGORY_ZS ||
            category == UTF8PROC_CATEGORY_ZL || category == UTF8PROC_CATEGORY_ZP) {
            if (preserve_line_breaks && (cp == '\n' || cp == '\r' ||
                category == UTF8PROC_CATEGORY_ZL || category == UTF8PROC_CATEGORY_ZP)) {
                collapsed += '\n';
                space = false;
            } else {
                space = !collapsed.empty();
            }
        } else {
            if (space) collapsed += ' ';
            collapsed.append(nfc, i, static_cast<size_t>(bytes));
            space = false;
        }
        i += static_cast<size_t>(bytes);
    }
    static const std::regex number("[0-9]+(?:[,.][0-9]+)?");
    std::string result;
    size_t offset = 0;
    for (auto it = std::sregex_iterator(collapsed.begin(), collapsed.end(), number); it != std::sregex_iterator(); ++it) {
        const auto & match = *it;
        result.append(collapsed, offset, static_cast<size_t>(match.position()) - offset);
        const auto raw = match.str();
        const auto separator = raw.find_first_of(",.");
        result += danish_integer(std::stoull(raw.substr(0, separator)));
        if (separator != std::string::npos) {
            auto fraction = raw.substr(separator + 1);
            while (!fraction.empty() && fraction.back() == '0') fraction.pop_back();
            if (!fraction.empty()) {
                result += " komma";
                for (const auto digit : fraction) result += " " + danish_integer(static_cast<uint64_t>(digit - '0'));
            }
        }
        offset = static_cast<size_t>(match.position() + match.length());
    }
    result.append(collapsed, offset, std::string::npos);
    return result;
}

std::vector<int32_t> PlapreTextFrontend::encode(const std::string & text, bool target) const {
    auto normalized = normalize(text);
    if (target && !normalized.empty()) {
        if (normalized.back() == ',') normalized.pop_back();
        if (!normalized.empty() && normalized.back() != '.' && normalized.back() != '!' &&
            normalized.back() != '?' && normalized.back() != ':' &&
            (normalized.size() < 3 || normalized.compare(normalized.size() - 3, 3, "\xe2\x80\xa6") != 0)) normalized += '.';
    }
    return tokenizer_.encode(normalized, false);
}

const tokenizers::LlamaBpeTokenizer & PlapreTextFrontend::tokenizer() const noexcept { return tokenizer_; }

}  // namespace engine::models::plapre
