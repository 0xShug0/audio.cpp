#include "engine/community_models/inflect_v2/frontend.h"

#include "engine/framework/io/dynamic_library.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace engine::models::inflect_v2 {
namespace {

using InitializeFn = int (*)(int, int, const char *, int);
using SetVoiceFn = int (*)(const char *);
using TextToPhonemesFn = const char * (*)(const void **, int, int);
using TerminateFn = int (*)();

constexpr int kEspeakSynchronous = 2;
constexpr int kEspeakCharsUtf8 = 1;
constexpr int kEspeakPhonemesIpa = 2;

void replace_all(std::string & value, std::string_view from, std::string_view to) {
    size_t position = 0;
    while (!from.empty() && (position = value.find(from, position)) != std::string::npos) {
        value.replace(position, from.size(), to);
        position += to.size();
    }
}

std::string_view preserved_punctuation_at(
    const std::string & text,
    size_t index) {
    static constexpr std::array<std::string_view, 5> multibyte = {
        "¡", "¿", "«", "»", "—",
    };
    if (std::string_view(";:,.!?\"").find(text[index]) !=
        std::string_view::npos) {
        return std::string_view(text.data() + index, 1);
    }
    for (const std::string_view punctuation : multibyte) {
        if (text.compare(index, punctuation.size(), punctuation) == 0) {
            return std::string_view(
                text.data() + index,
                punctuation.size());
        }
    }
    return {};
}

std::string trim(std::string value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    }).base();
    return first < last ? std::string(first, last) : std::string{};
}

std::string collapse_space(std::string value) {
    std::string out;
    out.reserve(value.size());
    bool pending = false;
    for (const unsigned char ch : value) {
        if (std::isspace(ch) != 0) {
            pending = !out.empty();
        } else {
            if (pending) {
                out.push_back(' ');
            }
            out.push_back(static_cast<char>(ch));
            pending = false;
        }
    }
    return trim(std::move(out));
}

template <typename Fn>
std::string regex_transform(const std::string & input, const std::regex & pattern, Fn fn) {
    std::string out;
    size_t cursor = 0;
    for (std::sregex_iterator it(input.begin(), input.end(), pattern), end; it != end; ++it) {
        const auto & match = *it;
        const size_t position = static_cast<size_t>(match.position());
        out.append(input, cursor, position - cursor);
        out += fn(match);
        cursor = position + static_cast<size_t>(match.length());
    }
    out.append(input, cursor, std::string::npos);
    return out;
}

const std::array<const char *, 20> kSmall = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
};
const std::array<const char *, 10> kTens = {
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
};

std::string cardinal(uint64_t value) {
    if (value < 20) {
        return kSmall[static_cast<size_t>(value)];
    }
    if (value < 100) {
        const auto ten = value / 10;
        const auto rest = value % 10;
        return std::string(kTens[static_cast<size_t>(ten)]) +
            (rest == 0 ? "" : " " + cardinal(rest));
    }
    if (value < 1000) {
        const auto rest = value % 100;
        return cardinal(value / 100) + " hundred" +
            (rest == 0 ? "" : " and " + cardinal(rest));
    }
    static constexpr std::array<std::pair<uint64_t, const char *>, 6> scales = {{
        {1000000000000000000ULL, "quintillion"},
        {1000000000000000ULL, "quadrillion"},
        {1000000000000ULL, "trillion"},
        {1000000000ULL, "billion"},
        {1000000ULL, "million"},
        {1000ULL, "thousand"},
    }};
    for (const auto & [scale, name] : scales) {
        if (value >= scale) {
            const auto rest = value % scale;
            std::string out = cardinal(value / scale) + " " + name;
            if (rest != 0) {
                out += rest < 100 ? " and " : " ";
                out += cardinal(rest);
            }
            return out;
        }
    }
    return {};
}

std::string ordinal(uint64_t value) {
    static const std::array<const char *, 20> small = {
        "zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
        "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth",
        "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth",
    };
    static const std::array<const char *, 10> tens = {
        "", "", "twentieth", "thirtieth", "fortieth", "fiftieth", "sixtieth",
        "seventieth", "eightieth", "ninetieth",
    };
    if (value < 20) {
        return small[static_cast<size_t>(value)];
    }
    if (value < 100) {
        return value % 10 == 0
            ? tens[static_cast<size_t>(value / 10)]
            : std::string(kTens[static_cast<size_t>(value / 10)]) + " " + ordinal(value % 10);
    }
    const std::array<std::pair<uint64_t, const char *>, 7> scales = {{
        {1000000000000000000ULL, "quintillionth"},
        {1000000000000000ULL, "quadrillionth"},
        {1000000000000ULL, "trillionth"},
        {1000000000ULL, "billionth"},
        {1000000ULL, "millionth"},
        {1000ULL, "thousandth"},
        {100ULL, "hundredth"},
    }};
    for (const auto & [scale, name] : scales) {
        if (value >= scale) {
            const auto rest = value % scale;
            return rest == 0
                ? cardinal(value / scale) + " " + name
                : cardinal(value - rest) + (rest < 100 ? " and " : " ") + ordinal(rest);
        }
    }
    return {};
}

uint64_t parse_integer(std::string value) {
    value.erase(std::remove(value.begin(), value.end(), ','), value.end());
    try {
        return std::stoull(value);
    } catch (...) {
        throw std::runtime_error("Inflect v2 number is outside the supported range: " + value);
    }
}

std::string digit_words(std::string_view digits, bool identifier = false) {
    std::string out;
    size_t index = 0;
    for (const char ch : digits) {
        if (!std::isdigit(static_cast<unsigned char>(ch))) {
            continue;
        }
        if (!out.empty()) {
            out.push_back(' ');
        }
        out += identifier && ch == '0' && index > 0 ? "oh" : kSmall[static_cast<size_t>(ch - '0')];
        ++index;
    }
    return out;
}

const std::unordered_map<char, std::string> & letter_names() {
    static const std::unordered_map<char, std::string> names = {
        {'A', "ay"}, {'B', "bee"}, {'C', "see"}, {'D', "dee"}, {'E', "ee"},
        {'F', "eff"}, {'G', "gee"}, {'H', "aitch"}, {'I', "eye"}, {'J', "jay"},
        {'K', "kay"}, {'L', "ell"}, {'M', "em"}, {'N', "en"}, {'O', "oh"},
        {'P', "pee"}, {'Q', "cue"}, {'R', "ar"}, {'S', "ess"}, {'T', "tee"},
        {'U', "you"}, {'V', "vee"}, {'W', "double you"}, {'X', "ex"},
        {'Y', "why"}, {'Z', "zee"},
    };
    return names;
}

std::string identifier_token(const std::string & token) {
    static const std::regex pattern("^([A-Za-z]?)([0-9]+)([A-Za-z]?)$");
    std::smatch match;
    if (!std::regex_match(token, match, pattern)) {
        return token;
    }
    std::vector<std::string> pieces;
    if (!match[1].str().empty()) {
        pieces.push_back(letter_names().at(static_cast<char>(std::toupper(match[1].str()[0]))));
    }
    const std::string digits = match[2].str();
    pieces.push_back(digits.size() == 3 || digits.front() == '0'
        ? digit_words(digits, true)
        : cardinal(parse_integer(digits)));
    if (!match[3].str().empty()) {
        pieces.push_back(letter_names().at(static_cast<char>(std::toupper(match[3].str()[0]))));
    }
    std::string out;
    for (const auto & piece : pieces) {
        if (!out.empty()) {
            out.push_back(' ');
        }
        out += piece;
    }
    return out;
}

bool valid_date(int year, int month, int day) {
    static const std::array<int, 12> days = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (month < 1 || month > 12 || day < 1) {
        return false;
    }
    int limit = days[static_cast<size_t>(month - 1)];
    const bool leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    if (month == 2 && leap) {
        ++limit;
    }
    return day <= limit;
}

std::vector<uint32_t> utf8_codepoints(std::string_view value) {
    std::vector<uint32_t> out;
    for (size_t index = 0; index < value.size();) {
        const auto first = static_cast<unsigned char>(value[index]);
        uint32_t codepoint = 0;
        size_t width = 0;
        if (first < 0x80U) {
            codepoint = first;
            width = 1;
        } else if ((first & 0xE0U) == 0xC0U) {
            codepoint = first & 0x1FU;
            width = 2;
        } else if ((first & 0xF0U) == 0xE0U) {
            codepoint = first & 0x0FU;
            width = 3;
        } else if ((first & 0xF8U) == 0xF0U) {
            codepoint = first & 0x07U;
            width = 4;
        } else {
            throw std::runtime_error("Inflect v2 frontend received invalid UTF-8");
        }
        if (index + width > value.size()) {
            throw std::runtime_error("Inflect v2 frontend received truncated UTF-8");
        }
        for (size_t offset = 1; offset < width; ++offset) {
            const auto continuation = static_cast<unsigned char>(value[index + offset]);
            if ((continuation & 0xC0U) != 0x80U) {
                throw std::runtime_error("Inflect v2 frontend received invalid UTF-8 continuation");
            }
            codepoint = (codepoint << 6U) | (continuation & 0x3FU);
        }
        out.push_back(codepoint);
        index += width;
    }
    return out;
}

const std::unordered_map<uint32_t, int32_t> & symbol_ids() {
    static const auto ids = [] {
        const std::string symbols =
            "_;:,.!?¡¿—…\"«»“” "
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ";
        std::unordered_map<uint32_t, int32_t> result;
        const auto points = utf8_codepoints(symbols);
        for (size_t index = 0; index < points.size(); ++index) {
            result.emplace(points[index], static_cast<int32_t>(index));
        }
        if (points.size() != 178) {
            throw std::runtime_error("Inflect v2 compiled symbol inventory is invalid");
        }
        return result;
    }();
    return ids;
}

struct EspeakApi {
    io::DynamicLibraryHandle library = nullptr;
    InitializeFn initialize = nullptr;
    SetVoiceFn set_voice = nullptr;
    TextToPhonemesFn text_to_phonemes = nullptr;
    TerminateFn terminate = nullptr;
    std::filesystem::path library_path;
    std::filesystem::path data_path;
    std::mutex call_mutex;

    EspeakApi(std::filesystem::path requested_library, std::filesystem::path requested_data)
        : library_path(std::move(requested_library)),
          data_path(std::move(requested_data)) {
        if (!library_path.empty() &&
            !std::filesystem::is_regular_file(library_path)) {
            throw std::runtime_error(
                "Inflect v2 eSpeak-ng library does not exist: " +
                library_path.string());
        }
        if (!data_path.empty() &&
            (!std::filesystem::is_directory(data_path) ||
             !std::filesystem::is_regular_file(data_path / "phontab"))) {
            throw std::runtime_error(
                "Inflect v2 eSpeak-ng data path is invalid; expected the "
                "espeak-ng-data directory containing phontab: " +
                data_path.string());
        }
        if (!library_path.empty()) {
            library = io::open_dynamic_library(library_path.string());
        } else {
            library = io::open_dynamic_library({
#ifdef _WIN32
                "espeak-ng.dll", "libespeak-ng.dll",
#elif __APPLE__
                "libespeak-ng.dylib", "libespeak-ng.1.dylib",
#else
                "libespeak-ng.so.1", "libespeak-ng.so",
#endif
            });
        }
        if (library == nullptr) {
            throw std::runtime_error(
                "Inflect v2 requires eSpeak-ng. Install it so its shared "
                "library is discoverable, or pass both --session-option "
                "inflect_v2.espeak_library_path=<library> and --session-option "
                "inflect_v2.espeak_data_path=<espeak-ng-data>");
        }
        initialize = symbol<InitializeFn>("espeak_Initialize");
        set_voice = symbol<SetVoiceFn>("espeak_SetVoiceByName");
        text_to_phonemes = symbol<TextToPhonemesFn>("espeak_TextToPhonemes");
        terminate = symbol<TerminateFn>("espeak_Terminate");
        const std::string data = data_path.empty() ? std::string{} : data_path.string();
        if (initialize(kEspeakSynchronous, 0, data.empty() ? nullptr : data.c_str(), 0) <= 0) {
            io::close_dynamic_library(library);
            library = nullptr;
            throw std::runtime_error(
                "Inflect v2 could not initialize eSpeak-ng data; set "
                "inflect_v2.espeak_data_path to the espeak-ng-data directory");
        }
        if (set_voice("en-us") != 0) {
            terminate();
            io::close_dynamic_library(library);
            library = nullptr;
            throw std::runtime_error("Inflect v2 eSpeak-ng installation has no en-us voice");
        }
    }

    ~EspeakApi() {
        if (library != nullptr) {
            terminate();
            io::close_dynamic_library(library);
        }
    }

    template <typename Fn>
    Fn symbol(const char * name) {
        auto * address = io::dynamic_library_symbol(library, name);
        if (address == nullptr) {
            io::close_dynamic_library(library);
            library = nullptr;
            throw std::runtime_error(std::string("Inflect v2 eSpeak-ng is missing symbol ") + name);
        }
        return reinterpret_cast<Fn>(address);
    }

    std::string phonemize_segment(const std::string & text) {
        const void * cursor = text.c_str();
        std::string out;
        while (cursor != nullptr && *static_cast<const char *>(cursor) != '\0') {
            const void * before = cursor;
            const char * clause = text_to_phonemes(
                &cursor,
                kEspeakCharsUtf8,
                kEspeakPhonemesIpa);
            if (clause != nullptr && *clause != '\0') {
                if (!out.empty() && !std::isspace(static_cast<unsigned char>(out.back()))) {
                    out.push_back(' ');
                }
                out += clause;
            }
            if (cursor == before) {
                break;
            }
        }
        return collapse_space(std::move(out));
    }

    std::string phonemize(const std::string & text) {
        std::lock_guard<std::mutex> lock(call_mutex);
        std::string out;
        size_t segment_start = 0;
        const auto append_segment = [&](size_t end) {
            const std::string raw = text.substr(segment_start, end - segment_start);
            const bool leading_space =
                !raw.empty() &&
                std::isspace(static_cast<unsigned char>(raw.front())) != 0;
            const bool trailing_space =
                !raw.empty() &&
                std::isspace(static_cast<unsigned char>(raw.back())) != 0;
            const std::string segment = trim(raw);
            if (segment.empty()) {
                if (!raw.empty() && !out.empty() &&
                    std::isspace(static_cast<unsigned char>(out.back())) == 0) {
                    out.push_back(' ');
                }
                return;
            }
            const std::string phonemes = phonemize_segment(segment);
            if (phonemes.empty()) {
                return;
            }
            if (leading_space && !out.empty() &&
                std::isspace(static_cast<unsigned char>(out.back())) == 0) {
                out.push_back(' ');
            }
            out += phonemes;
            if (trailing_space &&
                std::isspace(static_cast<unsigned char>(out.back())) == 0) {
                out.push_back(' ');
            }
        };
        for (size_t index = 0; index < text.size();) {
            const auto punctuation =
                preserved_punctuation_at(text, index);
            if (punctuation.empty()) {
                ++index;
                continue;
            }
            append_segment(index);
            out.append(punctuation);
            index += punctuation.size();
            segment_start = index;
        }
        append_segment(text.size());
        return collapse_space(std::move(out));
    }
};

std::mutex g_espeak_mutex;
std::shared_ptr<EspeakApi> g_espeak;

std::shared_ptr<EspeakApi> acquire_espeak(
    const std::filesystem::path & library_path,
    const std::filesystem::path & data_path) {
    std::lock_guard<std::mutex> lock(g_espeak_mutex);
    if (g_espeak != nullptr) {
        if ((!library_path.empty() && g_espeak->library_path != library_path) ||
            (!data_path.empty() && g_espeak->data_path != data_path)) {
            throw std::runtime_error(
                "Inflect v2 eSpeak-ng is already initialized with different paths");
        }
        return g_espeak;
    }
    auto created = std::make_shared<EspeakApi>(library_path, data_path);
    g_espeak = created;
    return created;
}

size_t utf8_prefix_bytes(const std::string & value, size_t codepoints) {
    size_t index = 0;
    for (size_t count = 0; count < codepoints && index < value.size(); ++count) {
        const auto ch = static_cast<unsigned char>(value[index]);
        index += ch < 0x80U ? 1 : (ch & 0xE0U) == 0xC0U ? 2 : (ch & 0xF0U) == 0xE0U ? 3 : 4;
    }
    return std::min(index, value.size());
}

}  // namespace

struct InflectV2Frontend::State {
    State(
        const std::filesystem::path & library_path,
        const std::filesystem::path & data_path)
        : espeak(acquire_espeak(library_path, data_path)) {}

    std::shared_ptr<EspeakApi> espeak;
};

InflectV2Frontend::InflectV2Frontend(
    std::filesystem::path espeak_library_path,
    std::filesystem::path espeak_data_path)
    : state_(std::make_unique<State>(espeak_library_path, espeak_data_path)) {}

InflectV2Frontend::~InflectV2Frontend() = default;
InflectV2Frontend::InflectV2Frontend(InflectV2Frontend &&) noexcept = default;
InflectV2Frontend & InflectV2Frontend::operator=(InflectV2Frontend &&) noexcept = default;

std::string InflectV2Frontend::normalize(const std::string & input) {
    std::string text = input;
    for (const auto & [from, to] : std::array<std::pair<const char *, const char *>, 12>{{
             {"‘", "'"}, {"’", "'"}, {"“", "\""}, {"”", "\""}, {"–", "-"},
             {"—", ", "}, {"…", "..."}, {"(", ", "}, {")", ", "}, {"[", ", "},
             {"]", ", "}, {"{", ", "},
         }}) {
        replace_all(text, from, to);
    }
    replace_all(text, "}", ", ");
    text = collapse_space(std::move(text));

    for (const auto & [from, to] : std::array<std::pair<const char *, const char *>, 10>{{
             {"RTX 3060", "ar tee ex thirty sixty"},
             {"RTX 3090", "ar tee ex thirty ninety"},
             {"RTX 4090", "ar tee ex forty ninety"},
             {"RTX 5080", "ar tee ex fifty eighty"},
             {"RTX 5090", "ar tee ex fifty ninety"},
             {"PyTorch", "pie torch"},
             {"SQLite", "ess cue lite"},
             {"USB-C", "you ess bee see"},
             {"Qwen3", "Qwen three"},
             {"Qwen", "Qwen"},
         }}) {
        text = std::regex_replace(
            text,
            std::regex("\\b" + std::string(from) + "\\b"),
            to);
    }
    for (const auto & [from, to] : std::array<std::pair<const char *, const char *>, 10>{{
             {"Dr\\.", "doctor"}, {"Mr\\.", "mister"}, {"Mrs\\.", "missus"},
             {"Ms\\.", "miss"}, {"Prof\\.", "professor"}, {"St\\.", "saint"},
             {"vs\\.", "versus"}, {"etc\\.", "et cetera"},
             {"e\\.g\\.", "for example"}, {"i\\.e\\.", "that is"},
         }}) {
        text = std::regex_replace(
            text,
            std::regex("\\b" + std::string(from), std::regex::icase),
            to);
    }

    text = regex_transform(text, std::regex("\\b(?:[A-Z]\\.){2,}"), [](const std::smatch & match) {
        std::string out;
        for (const char ch : match.str()) {
            if (std::isupper(static_cast<unsigned char>(ch)) != 0) {
                if (!out.empty()) {
                    out.push_back(' ');
                }
                out.push_back(ch);
            }
        }
        return out;
    });
    text = regex_transform(
        text,
        std::regex(
            "\\b(apartment|apt\\.?|suite|unit|room|flight|extension|order|invoice|locker|aisle|gate)\\s+([A-Za-z]?[0-9]{1,4}[A-Za-z]?)\\b",
            std::regex::icase),
        [](const std::smatch & match) {
            return match[1].str() + " " + identifier_token(match[2].str());
        });
    text = regex_transform(
        text,
        std::regex("\\b([0-9]{3})(\\s+(?:North|South|East|West)\\b)", std::regex::icase),
        [](const std::smatch & match) {
            return digit_words(match[1].str(), true) + match[2].str();
        });
    text = regex_transform(
        text,
        std::regex("\\$([0-9][0-9,]*(?:\\.[0-9]{1,2})?)"),
        [](const std::smatch & match) {
            std::string raw = match[1].str();
            raw.erase(std::remove(raw.begin(), raw.end(), ','), raw.end());
            const auto dot = raw.find('.');
            const uint64_t dollars = parse_integer(raw.substr(0, dot));
            std::string out = cardinal(dollars) + (dollars == 1 ? " dollar" : " dollars");
            if (dot != std::string::npos) {
                std::string cents_text = raw.substr(dot + 1);
                if (cents_text.size() == 1) {
                    cents_text.push_back('0');
                }
                const uint64_t cents = parse_integer(cents_text.substr(0, 2));
                if (cents != 0) {
                    out += " and " + cardinal(cents) + (cents == 1 ? " cent" : " cents");
                }
            }
            return out;
        });
    text = regex_transform(
        text,
        std::regex("\\b(0?[1-9]|1[0-2])/(0?[1-9]|[12][0-9]|3[01])/(20[0-9]{2}|19[0-9]{2})\\b"),
        [](const std::smatch & match) {
            static const std::array<const char *, 12> months = {
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December",
            };
            const int month = std::stoi(match[1].str());
            const int day = std::stoi(match[2].str());
            const int year = std::stoi(match[3].str());
            return valid_date(year, month, day)
                ? std::string(months[static_cast<size_t>(month - 1)]) + " " +
                    ordinal(static_cast<uint64_t>(day)) + " " + cardinal(static_cast<uint64_t>(year))
                : match.str();
        });
    text = regex_transform(
        text,
        std::regex("\\b([0-9]{1,2}):([0-9]{2})\\s*([AaPp]\\.?\\s*[Mm]\\.?)?\\b"),
        [](const std::smatch & match) {
            const int hour = std::stoi(match[1].str());
            const int minute = std::stoi(match[2].str());
            std::string out = cardinal(static_cast<uint64_t>(hour));
            out += minute == 0 ? " o clock" :
                minute < 10 ? " oh " + cardinal(static_cast<uint64_t>(minute)) :
                " " + cardinal(static_cast<uint64_t>(minute));
            std::string suffix = match[3].str();
            suffix.erase(std::remove_if(suffix.begin(), suffix.end(), [](unsigned char ch) {
                return std::isalpha(ch) == 0;
            }), suffix.end());
            for (char & ch : suffix) {
                ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            }
            for (const char ch : suffix) {
                out += " ";
                out.push_back(ch);
            }
            return out;
        });
    text = regex_transform(
        text,
        std::regex("\\b([0-9]{1,2})\\s*([AaPp]\\.?\\s*[Mm]\\.?)\\b"),
        [](const std::smatch & match) {
            std::string suffix = match[2].str();
            suffix.erase(std::remove_if(suffix.begin(), suffix.end(), [](unsigned char ch) {
                return std::isalpha(ch) == 0;
            }), suffix.end());
            std::string out = cardinal(parse_integer(match[1].str()));
            for (char ch : suffix) {
                out += " ";
                out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
            }
            return out;
        });
    text = regex_transform(
        text,
        std::regex("\\b([0-9]{3})-([0-9]{4})\\b"),
        [](const std::smatch & match) {
            return digit_words(match[1].str()) + ", " + digit_words(match[2].str());
        });
    text = regex_transform(
        text,
        std::regex("\\b[0-9]+(?:\\.[0-9]+){2,}\\b"),
        [](const std::smatch & match) {
            std::string out;
            size_t cursor = 0;
            const std::string value = match.str();
            while (cursor <= value.size()) {
                const size_t dot = value.find('.', cursor);
                if (!out.empty()) {
                    out += " point ";
                }
                out += cardinal(parse_integer(value.substr(cursor, dot - cursor)));
                if (dot == std::string::npos) {
                    break;
                }
                cursor = dot + 1;
            }
            return out;
        });
    text = regex_transform(
        text,
        std::regex("\\b([0-9]+)\\.([0-9]+)\\b"),
        [](const std::smatch & match) {
            return cardinal(parse_integer(match[1].str())) + " point " + digit_words(match[2].str());
        });
    text = regex_transform(
        text,
        std::regex("\\b([0-9]+)(st|nd|rd|th)\\b", std::regex::icase),
        [](const std::smatch & match) {
            return ordinal(parse_integer(match[1].str()));
        });
    text = regex_transform(
        text,
        std::regex("\\b[0-9][0-9,]*\\b"),
        [](const std::smatch & match) {
            std::string value = match.str();
            value.erase(std::remove(value.begin(), value.end(), ','), value.end());
            return value.size() >= 5 && value.rfind("20", 0) != 0
                ? digit_words(value)
                : cardinal(parse_integer(value));
        });
    text = regex_transform(
        text,
        std::regex("\\b[A-Z]{2,}\\b"),
        [](const std::smatch & match) {
            std::string out;
            for (const char ch : match.str()) {
                if (!out.empty()) {
                    out.push_back(' ');
                }
                out += letter_names().at(ch);
            }
            return out;
        });
    text = std::regex_replace(text, std::regex(",(?:\\s*,)+"), ",");
    text = std::regex_replace(text, std::regex(",\\s*([.!?])"), "$1");
    text = std::regex_replace(text, std::regex("\\s+([,;:.!?])"), "$1");
    text = std::regex_replace(text, std::regex("([,;:.!?])([^\\s])"), "$1 $2");
    return collapse_space(std::move(text));
}

InflectV2FrontendOutput InflectV2Frontend::encode(const std::string & text) const {
    InflectV2FrontendOutput out;
    out.normalized_text = normalize(text);
    if (out.normalized_text.empty()) {
        throw std::runtime_error("Inflect v2 text must not be empty");
    }
    out.phoneme_text = state_->espeak->phonemize(out.normalized_text);
    replace_all(out.phoneme_text, "sˈæskɐtʃˌuːən", "sɐskˈætʃəwən");
    replace_all(out.phoneme_text, "flʊɹɹˈɛsənt", "flʊˈɹɛsənt");
    out.phoneme_text = collapse_space(std::move(out.phoneme_text));
    if (out.phoneme_text.empty()) {
        throw std::runtime_error("Inflect v2 eSpeak-ng produced no phonemes");
    }
    out.token_ids = tokens_from_phonemes(out.phoneme_text);
    return out;
}

std::vector<int32_t> InflectV2Frontend::tokens_from_phonemes(
    const std::string & phoneme_text) {
    if (phoneme_text.empty()) {
        throw std::runtime_error("Inflect v2 phoneme input must not be empty");
    }
    const auto & ids = symbol_ids();
    const auto codepoints = utf8_codepoints(phoneme_text);
    std::vector<int32_t> token_ids;
    token_ids.reserve(codepoints.size() * 2 + 1);
    token_ids.push_back(0);
    for (const uint32_t codepoint : codepoints) {
        const auto found = ids.find(codepoint);
        if (found == ids.end()) {
            throw std::runtime_error(
                "Inflect v2 frontend produced unsupported phoneme codepoint " +
                std::to_string(codepoint));
        }
        token_ids.push_back(found->second);
        token_ids.push_back(0);
    }
    return token_ids;
}

std::vector<std::string> InflectV2Frontend::split_text(
    const std::string & input,
    int64_t limit) {
    if (limit <= 0) {
        throw std::runtime_error("Inflect v2 text_chunk_size must be positive");
    }
    const std::string normalized = collapse_space(input);
    if (normalized.empty()) {
        return {};
    }
    std::vector<std::string> sentences;
    size_t begin = 0;
    for (size_t index = 0; index < normalized.size(); ++index) {
        if ((normalized[index] == '.' || normalized[index] == '!' ||
             normalized[index] == '?' || normalized[index] == ';' ||
             normalized[index] == ':') &&
            index + 1 < normalized.size() &&
            std::isspace(static_cast<unsigned char>(normalized[index + 1])) != 0) {
            sentences.push_back(trim(normalized.substr(begin, index + 1 - begin)));
            begin = index + 2;
            index = begin - 1;
        }
    }
    if (begin < normalized.size()) {
        sentences.push_back(trim(normalized.substr(begin)));
    }

    std::vector<std::string> chunks;
    for (std::string sentence : sentences) {
        while (static_cast<int64_t>(utf8_codepoints(sentence).size()) > limit) {
            const size_t boundary = utf8_prefix_bytes(sentence, static_cast<size_t>(limit + 1));
            const std::string search = sentence.substr(0, boundary);
            size_t punctuation = std::string::npos;
            for (const char mark : {',', ';', ':'}) {
                const size_t position = search.rfind(mark);
                if (position != std::string::npos &&
                    (punctuation == std::string::npos || position > punctuation)) {
                    punctuation = position;
                }
            }
            size_t split_at = punctuation != std::string::npos &&
                    punctuation >= utf8_prefix_bytes(sentence, static_cast<size_t>(limit / 2))
                ? punctuation + 1
                : sentence.rfind(' ', boundary);
            const size_t halfway = utf8_prefix_bytes(sentence, static_cast<size_t>(limit / 2));
            if (split_at == std::string::npos || split_at < halfway) {
                split_at = utf8_prefix_bytes(sentence, static_cast<size_t>(limit));
            }
            chunks.push_back(trim(sentence.substr(0, split_at)));
            sentence = trim(sentence.substr(split_at));
        }
        if (!sentence.empty()) {
            chunks.push_back(std::move(sentence));
        }
    }
    return chunks;
}

double InflectV2Frontend::boundary_pause_seconds(const std::string & chunk) {
    const std::string value = trim(chunk);
    if (value.empty()) {
        return 0.08;
    }
    switch (value.back()) {
        case '?': return 0.28;
        case '!': return 0.24;
        case '.': return 0.22;
        case ';': return 0.16;
        case ':': return 0.13;
        case ',': return 0.09;
        default: return 0.08;
    }
}

}  // namespace engine::models::inflect_v2
