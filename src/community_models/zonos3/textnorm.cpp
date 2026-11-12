#include "engine/community_models/zonos3/textnorm.h"

#include "engine/framework/runtime/errors.h"

#if defined(AUDIOCPP_ZONOS3_HAVE_TEXTNORM)
#include "zyphra_textnorm.h"
#endif

#include <array>
#include <cstdint>
#include <memory>
#include <utility>

namespace engine::models::zonos3 {
namespace {

struct Language {
    std::string_view code;
    std::string_view nemo;  // zyphra-textnorm grammar
};

constexpr std::array<Language, 10> kLanguages{{
    {"en_us", "en"}, {"en_gb", "en"}, {"fr_fr", "fr"}, {"de", "de"}, {"es", "es"},
    {"it", "it"}, {"pt_br", "pt"}, {"ja", "ja"}, {"cmn", "zh"}, {"ko", "ko"},
}};

// Python str.isspace(): Unicode White_Space plus the U+001C..U+001F separators.
bool is_py_space(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20) || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
        (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
        cp == 0x3000;
}

// One code point starting at text[i]; a malformed byte is returned as a lone non-space unit.
uint32_t next_code_point(const std::string & text, size_t & i) {
    const auto c = static_cast<unsigned char>(text[i]);
    const size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
    if (len == 0 || i + len > text.size()) {
        ++i;
        return 0xFFFD;
    }
    uint32_t cp = len == 1 ? c : c & (0xFFu >> (len + 1));
    for (size_t k = 1; k < len; ++k) {
        const auto d = static_cast<unsigned char>(text[i + k]);
        if ((d & 0xC0) != 0x80) {
            ++i;
            return 0xFFFD;
        }
        cp = (cp << 6) | (d & 0x3F);
    }
    i += len;
    return cp;
}

#if defined(AUDIOCPP_ZONOS3_HAVE_TEXTNORM)
struct TextnormDeleter {
    void operator()(zyphra_textnorm * tn) const noexcept { zyphra_textnorm_delete(tn); }
};

const zyphra_textnorm & normalizer() {
    // ZONOS3's nonverbal markup tags stay as typed.
    static const std::unique_ptr<zyphra_textnorm, TextnormDeleter> tn = [] {
        static constexpr std::string_view tags = "laugh\nbreath\nsigh\ncough\npanting\nshouting\ncrying";
        std::unique_ptr<zyphra_textnorm, TextnormDeleter> out(zyphra_textnorm_new(tags.data(), tags.size()));
        return out;
    }();
    return *tn;
}

std::string take(char * text) {
    const std::unique_ptr<char, decltype(&zyphra_textnorm_free)> owned(text, &zyphra_textnorm_free);
    return owned ? std::string(owned.get()) : std::string();
}
#endif

}  // namespace

std::string zonos3_language(std::string_view language) {
    const std::string_view code = language == "en" ? "en_us"
        : language == "fr"                          ? "fr_fr"
        : language == "pt"                          ? "pt_br"
        : language == "zh"                          ? "cmn"
                                                    : language;
    for (const auto & entry : kLanguages) {
        if (entry.code == code) return std::string(code);
    }
    throw runtime::InvalidRequestError(
        "zonos3 language must be one of en_us, en_gb, fr_fr, de, es, it, pt_br, ja, cmn, ko (aliases en, fr, pt, zh)");
}

std::string zonos3_collapse_line_breaks(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    size_t run_start = std::string::npos;
    bool run_has_break = false;
    auto flush = [&](size_t end) {
        if (run_start == std::string::npos) return;
        if (run_has_break) out += ' ';
        else out.append(text, run_start, end - run_start);
        run_start = std::string::npos;
        run_has_break = false;
    };
    for (size_t i = 0; i < text.size();) {
        const size_t at = i;
        const uint32_t cp = next_code_point(text, i);
        if (is_py_space(cp)) {
            if (run_start == std::string::npos) run_start = at;
            run_has_break = run_has_break || cp != ' ';
        } else {
            flush(at);
            out.append(text, at, i - at);
        }
    }
    flush(text.size());
    return out;
}

bool zonos3_textnorm_linked() {
#if defined(AUDIOCPP_ZONOS3_HAVE_TEXTNORM)
    return true;
#else
    return false;
#endif
}

std::string zonos3_normalize_text(const std::string & text, const std::string & language, bool enabled) {
    if (!enabled) return text;
#if defined(AUDIOCPP_ZONOS3_HAVE_TEXTNORM)
    for (const auto & entry : kLanguages) {
        if (entry.code == language) {
            return take(zyphra_textnorm_normalize(
                &normalizer(), text.data(), text.size(), entry.nemo.data(), entry.nemo.size()));
        }
    }
    return take(zyphra_textnorm_collapse_line_breaks(text.data(), text.size()));
#else
    (void) language;
    return zonos3_collapse_line_breaks(text);
#endif
}

}  // namespace engine::models::zonos3
