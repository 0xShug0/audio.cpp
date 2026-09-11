#include "engine/models/magpie_tts/tokenizer_text.h"

#include "engine/framework/text/chunking.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <regex>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace engine::models::magpie_tts {
namespace {

namespace fs = std::filesystem;

struct IpaConfig {
    std::string tokenizer_name;
    int32_t offset = 0;
    std::string dict_hint;
    std::string heteronym_hint;
    std::string locale = "en-US";
    std::string grapheme_case = "upper";
    std::string grapheme_prefix;
    bool apostrophe = true;
    bool pad_with_space = false;
    double phoneme_probability = 1.0;
};

struct Utf8Char {
    uint32_t codepoint = 0;
    std::string text;
};

std::string read_file(const fs::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to read " + path.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<std::string> split_utf8(const std::string & text) {
    std::vector<std::string> out;
    for (size_t i = 0; i < text.size();) {
        const auto c = static_cast<unsigned char>(text[i]);
        size_t n = 1;
        if ((c & 0xe0U) == 0xc0U) {
            n = 2;
        } else if ((c & 0xf0U) == 0xe0U) {
            n = 3;
        } else if ((c & 0xf8U) == 0xf0U) {
            n = 4;
        }
        if (i + n > text.size()) {
            n = 1;
        }
        out.push_back(text.substr(i, n));
        i += n;
    }
    return out;
}

std::vector<Utf8Char> decode_utf8_chars(const std::string & text) {
    std::vector<Utf8Char> result;
    for (size_t i = 0; i < text.size();) {
        const auto first = static_cast<unsigned char>(text[i]);
        size_t length = 1;
        uint32_t codepoint = first;
        if ((first & 0xe0U) == 0xc0U && i + 1 < text.size()) {
            length = 2;
            codepoint = first & 0x1fU;
        } else if ((first & 0xf0U) == 0xe0U && i + 2 < text.size()) {
            length = 3;
            codepoint = first & 0x0fU;
        } else if ((first & 0xf8U) == 0xf0U && i + 3 < text.size()) {
            length = 4;
            codepoint = first & 0x07U;
        }
        bool valid = length > 1;
        for (size_t j = 1; j < length && valid; ++j) {
            const auto continuation = static_cast<unsigned char>(text[i + j]);
            valid = (continuation & 0xc0U) == 0x80U;
            codepoint = (codepoint << 6U) | (continuation & 0x3fU);
        }
        if (!valid && length > 1) {
            length = 1;
            codepoint = first;
        }
        result.push_back({codepoint, text.substr(i, length)});
        i += length;
    }
    return result;
}

bool is_han(uint32_t value) {
    return value == 0x3007 || (value >= 0xe815 && value <= 0xe864) || value == 0xfa18 ||
           (value >= 0x3400 && value <= 0x4dbf) || (value >= 0x4e00 && value <= 0x9fff) ||
           (value >= 0xf900 && value <= 0xfaff) || (value >= 0x20000 && value <= 0x2a6df) ||
           (value >= 0x2a703 && value <= 0x2b73f) || (value >= 0x2b740 && value <= 0x2b81d) ||
           (value >= 0x2b825 && value <= 0x2bf6e) || (value >= 0x2c029 && value <= 0x2ce93) ||
           value == 0x2d016 || (value >= 0x2d11b && value <= 0x2ebd9) ||
           (value >= 0x2f80a && value <= 0x2fa1f) || (value >= 0x30000 && value <= 0x3134a) ||
           (value >= 0x300f7 && value <= 0x31288) || value == 0x30edd || value == 0x30ede ||
           (value >= 0x31350 && value <= 0x32389);
}

bool is_ascii_alnum(char c) {
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

std::string ascii_upper(std::string text) {
    std::string out;
    for (const auto & ch : split_utf8(text)) {
        if (ch.size() == 1) {
            const auto c = static_cast<unsigned char>(ch[0]);
            out.push_back(c >= 'a' && c <= 'z' ? static_cast<char>(c - 32U) : static_cast<char>(c));
        } else if (ch == "ä") {
            out += "Ä";
        } else if (ch == "ö") {
            out += "Ö";
        } else if (ch == "ü") {
            out += "Ü";
        } else if (ch == "ß") {
            out += "ẞ";
        } else if (ch == "é") {
            out += "É";
        } else if (ch == "è") {
            out += "È";
        } else if (ch == "ê") {
            out += "Ê";
        } else if (ch == "à") {
            out += "À";
        } else if (ch == "á") {
            out += "Á";
        } else {
            out += ch;
        }
    }
    return out;
}

std::string set_grapheme_case(const std::string & text, const std::string & mode) {
    return mode == "upper" ? ascii_upper(text) : text;
}

std::string replace_all(std::string text, const std::string & from, const std::string & to) {
    size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
    return text;
}

std::vector<std::string> default_punct() {
    return {"!", "\"", "(", ")", ",", "-", ".", "/", ":", ";", "?", "[", "]", "{", "}"};
}

std::vector<std::string> ipa_punct(const std::string & locale) {
    const auto base = default_punct();
    std::set<std::string> punct(base.begin(), base.end());
    if (locale == "de-DE" || locale == "es-ES") {
        punct.insert("«");
        punct.insert("»");
        punct.insert("‹");
        punct.insert("›");
    }
    if (locale == "de-DE") {
        punct.insert("„");
        punct.insert("“");
        punct.insert("‚");
        punct.insert("‘");
        punct.insert("‒");
        punct.insert("–");
        punct.insert("—");
    } else if (locale == "es-ES") {
        punct.insert("¿");
        punct.insert("¡");
    }
    return std::vector<std::string>(punct.begin(), punct.end());
}

std::vector<std::string> exact_ipa_tokens(const std::string & tokenizer_name) {
    if (tokenizer_name == "english_phoneme") {
        return {
            "!", "\"", "'", "(", ")", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6",
            "7", "8", "9", ":", ";", "?", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
            "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
            "[", "]", "a", "b", "d", "e", "f", "h", "i", "j", "k", "l", "m", "n", "o", "p",
            "s", "t", "u", "v", "w", "z", "{", "}", "À", "É", "æ", "ð", "ŋ", "ɑ", "ɔ", "ə",
            "ɚ", "ɛ", "ɝ", "ɡ", "ɪ", "ɹ", "ʃ", "ʊ", "ʌ", "ʒ", "ˈ", "ˌ", "θ", " ", "<pad>", "<oov>",
        };
    }
    if (tokenizer_name == "german_phoneme") {
        return {
            "!", "\"", "#A", "#B", "#C", "#D", "#E", "#F", "#G", "#H", "#I", "#J", "#K",
            "#L", "#M", "#N", "#O", "#P", "#Q", "#R", "#S", "#T", "#U", "#V", "#W", "#X",
            "#Y", "#Z", "#a", "#b", "#c", "#d", "#e", "#f", "#g", "#h", "#i", "#j", "#k",
            "#l", "#m", "#n", "#o", "#p", "#q", "#r", "#s", "#t", "#u", "#v", "#w", "#x",
            "#y", "#z", "#Ä", "#Å", "#Ö", "#Ü", "#ß", "#à", "#á", "#ä", "#å", "#ç", "#è",
            "#é", "#ê", "#ë", "#í", "#ó", "#ö", "#ø", "#ü", "'", "(", ")", ",", "-",
            ".", "/", "1", ":", ";", "?", "[", "]", "a", "b", "d", "e", "f",
            "h", "i", "j", "k", "l", "m", "n", "o", "p", "r", "s", "t", "u",
            "v", "w", "x", "y", "z", "{", "}", "«", "»", "ç", "ð", "ø", "ŋ",
            "œ", "ɐ", "ɑ", "ɒ", "ɔ", "ə", "ɛ", "ɜ", "ɡ", "ɪ", "ɹ", "ɾ", "ʃ",
            "ʊ", "ʌ", "ʒ", "ˈ", "ˌ", "ː", "̃", "θ", "‒", "–", "—", "‘", "‚",
            "“", "„", "‹", "›", " ", "<pad>", "<oov>",
        };
    }
    if (tokenizer_name == "spanish_phoneme") {
        return {
            "!", "\"", "'", "(", ")", ",", "-", ".", "/", ":", ";", "?", "A", "B", "C", "D",
            "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
            "T", "U", "V", "W", "X", "Y", "Z", "[", "]", "a", "b", "d", "e", "f", "h",
            "i", "j", "k", "l", "m", "n", "o", "p", "r", "s", "t", "u", "w", "x", "{",
            "}", "¡", "«", "»", "¿", "Á", "Ç", "É", "Í", "Î", "Ñ", "Ó", "Ö", "Ú", "Ü",
            "Ý", "ð", "ŋ", "ɛ", "ɟ", "ɡ", "ɣ", "ɪ", "ɲ", "ɾ", "ʃ", "ʊ", "ʎ", "ʒ",
            "ʝ", "ˈ", "ˌ", "ː", "̯", "͡", "β", "θ", "‹", "›", " ", "<pad>", "<oov>",
        };
    }
    if (tokenizer_name == "portuguese_Brazilian_phoneme") {
        return {
            "!", "\"", "#A", "#B", "#C", "#D", "#E", "#F", "#G", "#H", "#I", "#J",
            "#K", "#L", "#M", "#N", "#O", "#P", "#Q", "#R", "#S", "#T", "#U", "#V",
            "#W", "#X", "#Y", "#Z", "#À", "#Á", "#Â", "#Ã", "#Ç", "#É", "#Ê", "#Í",
            "#Ó", "#Ô", "#Õ", "#Ú", "#Ü", "'", "(", ")", ",", "-", ".", "/", ":", ";",
            "?", "[", "]", "a", "b", "d", "e", "f", "h", "i", "j", "k", "l", "m", "n",
            "o", "p", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "}", "ð", "õ",
            "ĩ", "ŋ", "ũ", "ɐ", "ɑ", "ɒ", "ɔ", "ə", "ɛ", "ɜ", "ɡ", "ɪ", "ɲ", "ɹ", "ɾ",
            "ʁ", "ʃ", "ʊ", "ʌ", "ʎ", "ʒ", "ʲ", "ˈ", "ˌ", "ː", "̃", "θ", "ẽ", " ",
            "<pad>", "<oov>",
        };
    }
    if (tokenizer_name == "hindi_phoneme") {
        return {
            "!", "\"", "'", "(", ")", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5",
            "6", "7", "8", "9", ":", ";", "?", "A", "B", "C", "D", "E", "F", "G", "H",
            "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
            "W", "X", "Y", "Z", "[", "]", "a", "b", "c", "d", "e", "f", "h", "i",
            "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w",
            "x", "z", "{", "}", "À", "É", "ã", "æ", "ð", "õ", "ĩ", "ŋ", "ũ", "ɑ",
            "ɔ", "ɖ", "ə", "ɚ", "ɛ", "ɝ", "ɟ", "ɡ", "ɣ", "ɪ", "ɭ", "ɲ", "ɳ", "ɹ",
            "ɾ", "ʂ", "ʃ", "ʈ", "ʊ", "ʋ", "ʌ", "ʒ", "ʰ", "ˈ", "ˌ", "ː", "̃", "̩",
            "θ", "χ", "ँ", "ं", "ः", "अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ", "ऌ", "ऍ",
            "ऎ", "ए", "ऐ", "ऑ", "ओ", "औ", "क", "ख", "ग", "घ", "ङ", "च", "छ", "ज",
            "झ", "ञ", "ट", "ठ", "ड", "ढ", "ण", "त", "थ", "द", "ध", "न", "ऩ", "प",
            "फ", "ब", "भ", "म", "य", "र", "ऱ", "ल", "ळ", "ऴ", "व", "श", "ष", "स",
            "ह", "ऺ", "़", "ऽ", "ा", "ि", "ी", "ु", "ू", "ृ", "ॅ", "ॆ", "े", "ै",
            "ॉ", "ॊ", "ो", "ौ", "्", "ॐ", "॓", "ॠ", "ॡ", "ॢ", "।", "॥", "॰", "ẽ",
            " ", "<pad>", "<oov>",
        };
    }
    throw std::runtime_error("MagpieTTS has no IPA token table for " + tokenizer_name);
}

IpaConfig ipa_config_for_language(const std::string & language) {
    if (language == "en") {
        return {"english_phoneme", 0, "ipa_cmudict", "heteronyms-052722", "en-US", "upper", "", true, false, 1.0};
    }
    if (language == "de") {
        return {
            "german_phoneme",
            583,
            "de_nv230119.dict",
            "de_nv230119.heteronym",
            "de-DE",
            "mixed",
            "#",
            true,
            true,
            1.0,
        };
    }
    if (language == "es") {
        return {"spanish_phoneme", 480, "es_ES_nv230301.dict", "", "es-ES", "upper", "", true, true, 1.0};
    }
    if (language == "pt-BR") {
        return {
            "portuguese_Brazilian_phoneme",
            1017,
            "pt_br_prondict",
            "",
            "pt-BR",
            "upper",
            "#",
            true,
            true,
            1.0,
        };
    }
    if (language == "hi") {
        return {
            "hindi_phoneme",
            1128,
            "hindi_phoneme_merged_phoneme_dict",
            "",
            "hi-IN",
            "upper",
            "",
            true,
            true,
            1.0,
        };
    }
    throw std::runtime_error("MagpieTTS native tokenizer does not support " + language);
}

std::optional<int32_t> byt5_aggregate_offset_for_language(const std::string & language) {
    if (language == "fr") {
        return 1821;
    }
    if (language == "it") {
        return 2205;
    }
    if (language == "vi") {
        return 2589;
    }
    if (language == "ko") {
        return 2973;
    }
    return std::nullopt;
}

std::string byt5_tokenizer_name_for_language(const std::string & language) {
    if (language == "fr") {
        return "french_chartokenizer";
    }
    if (language == "it") {
        return "italian_chartokenizer";
    }
    if (language == "vi") {
        return "vietnamese_chartokenizer";
    }
    if (language == "ko") {
        return "korean_chartokenizer";
    }
    throw std::runtime_error("MagpieTTS language does not use a ByT5 tokenizer: " + language);
}

std::vector<int32_t> encode_byt5_chars(const std::string & text, int32_t aggregate_offset) {
    std::vector<int32_t> ids;
    ids.reserve(text.size() + 1);
    for (const unsigned char byte : text) {
        ids.push_back(aggregate_offset + 3 + static_cast<int32_t>(byte));
    }
    ids.push_back(aggregate_offset + 1);
    return ids;
}

std::vector<std::string> exact_char_tokens(const std::string & tokenizer_name) {
    if (
        tokenizer_name == "arabic_AE_chartokenizer" ||
        tokenizer_name == "arabic_SA_chartokenizer" ||
        tokenizer_name == "arabic_MSA_chartokenizer") {
        return {
            " ", "ء", "آ", "أ", "إ", "ؤ", "ئ", "ا", "ب", "ة", "ت", "ث", "ج", "ح",
            "خ", "د", "ذ", "ر", "ز", "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ", "ف",
            "ق", "ك", "ل", "م", "ن", "ه", "و", "ى", "ي", "ً", "ٌ", "ٍ", "َ", "ُ",
            "ِ", "ّ", "ٰ", "ْ", "ء", "آ", "أ", "إ", "ؤ", "ئ", "ا", "ب", "ة", "ت",
            "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س", "ش", "ص", "ض", "ط", "ظ",
            "ع", "غ", "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ى", "ي", "ً", "ٌ",
            "ٍ", "َ", "ُ", "ِ", "ّ", "ٰ", "ْ", "a", "b", "c", "d", "e", "f", "g",
            "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
            "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I",
            "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
            "W", "X", "Y", "Z", "'", "!", "\"", "(", ")", ",", "-", ".", "/",
            ":", ";", "?", "[", "]", "{", "}", "،", "؛", "؟", "<pad>", "<oov>",
        };
    }
    throw std::runtime_error("MagpieTTS has no character token table for " + tokenizer_name);
}

std::optional<std::pair<std::string, int32_t>> char_tokenizer_for_language(const std::string & language) {
    if (language == "ar-AE") {
        return std::make_pair(std::string("arabic_AE_chartokenizer"), 1329);
    }
    if (language == "ar-SA") {
        return std::make_pair(std::string("arabic_SA_chartokenizer"), 1493);
    }
    if (language == "ar-MSA" || language == "ar") {
        return std::make_pair(std::string("arabic_MSA_chartokenizer"), 1657);
    }
    return std::nullopt;
}

fs::path find_file_containing(const fs::path & root, const std::string & needle);

std::vector<int32_t> encode_char_tokens(
    const std::string & text,
    const std::vector<std::string> & tokens,
    int32_t offset) {
    std::unordered_map<std::string, int32_t> token_to_id;
    for (size_t index = 0; index < tokens.size(); ++index) {
        token_to_id[tokens[index]] = static_cast<int32_t>(index);
    }

    std::vector<std::string> symbols;
    for (const auto & c : split_utf8(replace_all(text, "’", "'"))) {
        if (c == " ") {
            if (!symbols.empty() && symbols.back() != " ") {
                symbols.push_back(c);
            }
        } else if (token_to_id.count(c) != 0) {
            symbols.push_back(c);
        }
    }
    while (!symbols.empty() && symbols.back() == " ") {
        symbols.pop_back();
    }
    symbols.insert(symbols.begin(), " ");
    symbols.push_back(" ");

    std::vector<int32_t> ids;
    ids.reserve(symbols.size());
    for (const auto & symbol : symbols) {
        const auto it = token_to_id.find(symbol);
        if (it != token_to_id.end()) {
            ids.push_back(offset + it->second);
        }
    }
    return ids;
}

std::vector<std::string> split_ascii_words(const std::string & text) {
    std::vector<std::string> out;
    std::istringstream in(text);
    std::string value;
    while (in >> value) {
        out.push_back(value);
    }
    return out;
}

std::vector<std::string> exact_mandarin_tokens() {
    return {
        " ", "a", "ai", "au", "e", "ei", "f", "i", "j", "k", "kʰ", "l", "m", "n", "o",
        "ou", "p", "pʰ", "s", "t", "ts", "tsʰ", "tɕ", "tɕʰ", "tʰ", "u", "w", "x",
        "y", "ŋ", "ɕ", "ə", "ɚ", "ɛ", "ɤ", "ɥ", "ʂ", "ʈʂ", "ʈʂʰ", "ʊ", "ʐ",
        "#1", "#2", "#3", "#4", "#5", "A", "B", "C", "D", "E", "F", "G", "H", "I",
        "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
        "Y", "Z", "'", "，", "。", "？", "！", "；", "：", "、", "‘", "’", "“", "”",
        "（", "）", "【", "】", "「", "」", "《", "》", ",", ".", "!", "?", "-", ":",
        ";", "/", "\"", "(", ")", "[", "]", "{", "}", "<pad>", "<oov>",
    };
}

fs::path find_mandarin_data_dir(const fs::path & root) {
    const std::vector<fs::path> candidates = {
        root / "mandarin_data",
        root / "mandarin_g2p",
        root,
    };
    for (const auto & path : candidates) {
        if (fs::is_regular_file(path / "pinyin_chars.tsv") &&
            fs::is_regular_file(path / "pinyin_phrases.tsv")) {
            return path;
        }
    }
    return {};
}

class MandarinTokenizer {
public:
    explicit MandarinTokenizer(fs::path root)
        : root_(std::move(root)) {
        load();
    }

    std::vector<int32_t> encode(const std::string & raw_text) const {
        std::vector<std::string> pinyin;
        const auto chars = decode_utf8_chars(ascii_upper(raw_text));
        for (size_t begin = 0; begin < chars.size();) {
            const bool han = is_han(chars[begin].codepoint);
            size_t end = begin + 1;
            while (end < chars.size() && is_han(chars[end].codepoint) == han) {
                ++end;
            }
            if (han) {
                append_han_pinyin(chars, begin, end, pinyin);
            } else {
                for (size_t index = begin; index < end; ++index) {
                    pinyin.push_back(chars[index].text);
                }
            }
            begin = end;
        }

        std::vector<std::string> symbols;
        for (const auto & value : pinyin) {
            if (!value.empty() && value.back() >= '1' && value.back() <= '5') {
                const auto syllable = value.substr(0, value.size() - 1);
                const auto found = phonemes_.find(syllable);
                if (found == phonemes_.end()) {
                    continue;
                }
                symbols.insert(symbols.end(), found->second.begin(), found->second.end());
                symbols.push_back("#" + value.substr(value.size() - 1));
            } else {
                symbols.push_back(value);
            }
        }

        std::vector<std::string> kept;
        for (const auto & symbol : symbols) {
            if (symbol == " ") {
                if (!kept.empty() && kept.back() != " ") {
                    kept.push_back(symbol);
                }
            } else if (token_to_id_.count(symbol) != 0) {
                kept.push_back(symbol);
            }
        }
        while (!kept.empty() && kept.back() == " ") {
            kept.pop_back();
        }
        kept.insert(kept.begin(), " ");
        kept.push_back(" ");

        std::vector<int32_t> ids;
        ids.reserve(kept.size());
        for (const auto & symbol : kept) {
            ids.push_back(token_to_id_.at(symbol));
        }
        return ids;
    }

    int32_t pad_id() const {
        return token_to_id_.at("<pad>");
    }

private:
    fs::path root_;
    fs::path data_dir_;
    std::unordered_map<uint32_t, std::string> character_pinyin_;
    std::unordered_map<std::string, std::vector<std::string>> phrase_pinyin_;
    size_t max_phrase_chars_ = 0;
    std::unordered_map<std::string, std::vector<std::string>> phonemes_;
    std::unordered_map<std::string, int32_t> token_to_id_;

    void load() {
        data_dir_ = find_mandarin_data_dir(root_);
        if (data_dir_.empty()) {
            throw std::runtime_error("MagpieTTS Mandarin tokenizer data is missing");
        }
        const auto phoneme_path = find_file_containing(root_, "ipa_dict");
        if (phoneme_path.empty()) {
            throw std::runtime_error("MagpieTTS Mandarin pinyin dictionary is missing");
        }
        load_chars(data_dir_ / "pinyin_chars.tsv");
        load_phrases(data_dir_ / "pinyin_phrases.tsv");
        load_phonemes(phoneme_path);
        const auto tokens = exact_mandarin_tokens();
        for (size_t index = 0; index < tokens.size(); ++index) {
            token_to_id_[tokens[index]] = 733 + static_cast<int32_t>(index);
        }
    }

    void load_chars(const fs::path & path) {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("failed to read " + path.string());
        }
        std::string line;
        while (std::getline(input, line)) {
            const auto tab = line.find('\t');
            if (tab == std::string::npos) {
                continue;
            }
            const auto codepoint = static_cast<uint32_t>(std::stoul(line.substr(0, tab), nullptr, 16));
            character_pinyin_[codepoint] = line.substr(tab + 1);
        }
    }

    void load_phrases(const fs::path & path) {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("failed to read " + path.string());
        }
        std::string line;
        while (std::getline(input, line)) {
            const auto tab = line.find('\t');
            if (tab == std::string::npos) {
                continue;
            }
            const std::string phrase = line.substr(0, tab);
            phrase_pinyin_[phrase] = split_ascii_words(line.substr(tab + 1));
            max_phrase_chars_ = std::max(max_phrase_chars_, decode_utf8_chars(phrase).size());
        }
    }

    void load_phonemes(const fs::path & path) {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("failed to read " + path.string());
        }
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty() || line.rfind(";;;", 0) == 0) {
                continue;
            }
            const auto tab = line.find('\t');
            if (tab == std::string::npos) {
                continue;
            }
            std::string syllable = line.substr(0, tab);
            std::transform(syllable.begin(), syllable.end(), syllable.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            std::string pronunciation = line.substr(tab + 1);
            std::transform(pronunciation.begin(), pronunciation.end(), pronunciation.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            phonemes_[syllable] = split_ascii_words(pronunciation);
        }
    }

    void append_han_pinyin(
        const std::vector<Utf8Char> & chars,
        size_t begin,
        size_t end,
        std::vector<std::string> & output) const {
        for (size_t position = begin; position < end;) {
            size_t matched_length = 0;
            const std::vector<std::string> * matched = nullptr;
            std::string candidate;
            const size_t limit = std::min(end, position + max_phrase_chars_);
            for (size_t cursor = position; cursor < limit; ++cursor) {
                candidate += chars[cursor].text;
                const auto found = phrase_pinyin_.find(candidate);
                if (found != phrase_pinyin_.end()) {
                    matched_length = cursor - position + 1;
                    matched = &found->second;
                }
            }
            if (matched != nullptr) {
                output.insert(output.end(), matched->begin(), matched->end());
                position += matched_length;
                continue;
            }
            const auto found = character_pinyin_.find(chars[position].codepoint);
            output.push_back(found == character_pinyin_.end() ? chars[position].text : found->second);
            ++position;
        }
    }
};

fs::path find_file_containing(const fs::path & root, const std::string & needle) {
    if (!fs::is_directory(root)) {
        return {};
    }
    for (const auto & entry : fs::directory_iterator(root)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto name = entry.path().filename().string();
        if (name.find(needle) != std::string::npos) {
            return entry.path();
        }
    }
    return {};
}

bool is_space_char(const std::string & c) {
    return c == " " || c == "\t" || c == "\n" || c == "\r";
}

std::string trim_ascii_space(const std::string & text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

std::vector<std::string> tokenizer_units(const std::string & text, int64_t chunk_size, engine::text::TextChunkMode mode) {
    auto chunks = engine::text::split_text_chunks(text, chunk_size, mode);
    if (chunks.empty()) {
        return {text};
    }
    return chunks;
}

int32_t token_id_for_symbol(const std::vector<std::string> & tokens, int32_t offset, const std::string & symbol) {
    const auto it = std::find(tokens.begin(), tokens.end(), symbol);
    if (it == tokens.end()) {
        throw std::runtime_error("MagpieTTS tokenizer missing symbol " + symbol);
    }
    return offset + static_cast<int32_t>(std::distance(tokens.begin(), it));
}

void pad_short_text_chunk_before_eos(std::vector<int32_t> & tokens, int32_t eos_id, int32_t pad_id) {
    constexpr size_t kMinTokensExcludingFirstAndEos = 4;
    if (tokens.size() < 2 || tokens.back() != eos_id) {
        return;
    }
    const size_t middle = tokens.size() - 2;
    if (middle >= kMinTokensExcludingFirstAndEos) {
        return;
    }
    tokens.insert(tokens.begin() + 1, kMinTokensExcludingFirstAndEos - middle, pad_id);
}

class IpaTokenizer {
public:
    IpaTokenizer(fs::path root, IpaConfig config)
        : root_(std::move(root)),
          config_(std::move(config)) {
        load();
    }

    std::vector<int32_t> encode(const std::string & raw_text, std::mt19937_64 & rng) const {
        std::string text = replace_all(raw_text, "’", "'");
        const auto pieces = lexical_pieces(text);
        std::vector<std::string> g2p;
        for (size_t index = 0; index < pieces.size(); ++index) {
            const auto & piece = pieces[index];
            if (piece == " ") {
                g2p.push_back(piece);
            } else if (is_word_piece(piece)) {
                const bool phonemize_single =
                    is_single_character_word_piece(piece) && !has_adjacent_single_character_word_piece(pieces, index);
                const auto parsed = parse_word(piece, phonemize_single, rng);
                g2p.insert(g2p.end(), parsed.begin(), parsed.end());
            } else {
                const auto chars = split_utf8(piece);
                g2p.insert(g2p.end(), chars.begin(), chars.end());
            }
        }

        std::vector<std::string> symbols;
        std::unordered_set<std::string> token_set(tokens_.begin(), tokens_.end());
        for (const auto & symbol : g2p) {
            if (symbol == " " && !symbols.empty() && symbols.back() != " ") {
                symbols.push_back(symbol);
            } else if (token_set.count(symbol) != 0 || punct_.count(symbol) != 0) {
                symbols.push_back(symbol);
            }
        }
        while (!symbols.empty() && symbols.back() == " ") {
            symbols.pop_back();
        }
        if (config_.pad_with_space) {
            symbols.insert(symbols.begin(), " ");
            symbols.push_back(" ");
        }

        std::vector<int32_t> ids;
        ids.reserve(symbols.size());
        for (const auto & symbol : symbols) {
            const auto it = token_to_id_.find(symbol);
            if (it != token_to_id_.end()) {
                ids.push_back(config_.offset + it->second);
            }
        }
        return ids;
    }

    int32_t pad_id() const {
        return token_id_for_symbol(tokens_, config_.offset, "<pad>");
    }

    const std::string & tokenizer_name() const noexcept {
        return config_.tokenizer_name;
    }

private:
    fs::path root_;
    IpaConfig config_;
    std::map<std::string, std::vector<std::vector<std::string>>> dict_;
    std::unordered_set<std::string> heteronyms_;
    std::vector<std::string> tokens_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_set<std::string> punct_;

    void load() {
        const auto dict_path = find_file_containing(root_, config_.dict_hint);
        if (dict_path.empty()) {
            throw std::runtime_error("MagpieTTS tokenizer dictionary is missing: " + config_.dict_hint);
        }
        for (const auto & punct : ipa_punct(config_.locale)) {
            punct_.insert(punct);
        }
        if (!config_.heteronym_hint.empty()) {
            const auto path = find_file_containing(root_, config_.heteronym_hint);
            if (!path.empty()) {
                std::istringstream in(read_file(path));
                std::string line;
                while (std::getline(in, line)) {
                    if (!line.empty() && line.back() == '\r') {
                        line.pop_back();
                    }
                    line = trim_ascii_space(line);
                    if (!line.empty()) {
                        heteronyms_.insert(set_grapheme_case(line, config_.grapheme_case));
                    }
                }
            }
        }
        std::istringstream in(read_file(dict_path));
        std::string line;
        const std::regex alt_re("\\([0-9]+\\)");
        while (std::getline(in, line)) {
            if (line.empty()) {
                continue;
            }
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            const auto first = static_cast<unsigned char>(line[0]);
            if (!(first == '\'' || first >= 0x80U || is_ascii_alnum(static_cast<char>(first)))) {
                continue;
            }
            std::istringstream ls(line);
            std::string word;
            ls >> word;
            std::string pron;
            std::getline(ls, pron);
            if (word.empty() || pron.empty()) {
                continue;
            }
            word = std::regex_replace(word, alt_re, "");
            word = set_grapheme_case(word, config_.grapheme_case);
            pron.erase(
                std::remove_if(pron.begin(), pron.end(), [](unsigned char c) { return c == ' ' || c == '\t'; }),
                pron.end());
            auto pron_chars = split_utf8(pron);
            dict_[word].push_back(pron_chars);
            if (config_.grapheme_case == "mixed") {
                const auto upper = ascii_upper(word);
                if (upper != word) {
                    dict_[upper].push_back(pron_chars);
                }
            }
        }
        tokens_ = exact_ipa_tokens(config_.tokenizer_name);
        for (size_t index = 0; index < tokens_.size(); ++index) {
            token_to_id_[tokens_[index]] = static_cast<int32_t>(index);
        }
    }

    static bool is_word_char(const std::string & c) {
        if (c.size() == 1) {
            const char ch = c[0];
            return is_ascii_alnum(ch) || ch == '\'' || ch == '-';
        }
        static const std::unordered_set<std::string> utf8_punct = {
            "¡", "«", "»", "¿", "‘", "’", "‚", "“", "”", "„", "‹", "›", "‒", "–", "—",
        };
        return !is_space_char(c) && utf8_punct.count(c) == 0;
    }

    static bool is_word_piece(const std::string & piece) {
        for (const auto & c : split_utf8(piece)) {
            if (is_word_char(c) && c != "-") {
                return true;
            }
        }
        return false;
    }

    static bool is_single_character_word_piece(const std::string & piece) {
        const auto chars = split_utf8(piece);
        return chars.size() == 1 && is_word_char(chars.front()) && chars.front() != "'" && chars.front() != "-";
    }

    static bool has_adjacent_single_character_word_piece(const std::vector<std::string> & pieces, size_t index) {
        if (index >= 2 && pieces[index - 1] == " " && is_single_character_word_piece(pieces[index - 2])) {
            return true;
        }
        if (index + 2 < pieces.size() && pieces[index + 1] == " " && is_single_character_word_piece(pieces[index + 2])) {
            return true;
        }
        return false;
    }

    static std::vector<std::string> lexical_pieces(const std::string & text) {
        std::vector<std::string> pieces;
        std::string current;
        enum class Kind { None, Word, Other };
        Kind current_kind = Kind::None;
        for (const auto & c : split_utf8(text)) {
            if (is_space_char(c)) {
                if (!current.empty()) {
                    pieces.push_back(current);
                    current.clear();
                }
                if (pieces.empty() || pieces.back() != " ") {
                    pieces.push_back(" ");
                }
                current_kind = Kind::None;
                continue;
            }
            const Kind kind = is_word_char(c) ? Kind::Word : Kind::Other;
            if (kind != current_kind && !current.empty()) {
                pieces.push_back(current);
                current = c;
            } else {
                current += c;
            }
            current_kind = kind;
        }
        if (!current.empty()) {
            pieces.push_back(current);
        }
        return pieces;
    }

    std::vector<std::string> prepend_grapheme_prefix(const std::string & word) const {
        std::vector<std::string> out;
        for (const auto & c : split_utf8(word)) {
            out.push_back(config_.grapheme_prefix + c);
        }
        return out;
    }

    bool dict_has_unique(const std::string & word) const {
        const auto it = dict_.find(word);
        return it != dict_.end() && it->second.size() == 1;
    }

    std::vector<std::string> parse_word_direct(
        const std::string & word,
        bool & handled,
        bool phonemize_single_char,
        std::mt19937_64 & rng) const {
        handled = true;
        const auto chars = split_utf8(word);
        bool has_alnum = false;
        for (const auto & c : chars) {
            if (c.size() != 1 || is_ascii_alnum(c[0])) {
                has_alnum = true;
                break;
            }
        }
        if (!has_alnum) {
            return chars;
        }
        if (config_.phoneme_probability < 1.0) {
            const double draw = std::generate_canonical<double, 53>(rng);
            if (draw > config_.phoneme_probability) {
                return prepend_grapheme_prefix(word);
            }
        }
        if (chars.size() == 1 && is_word_char(chars.front()) && !phonemize_single_char) {
            return prepend_grapheme_prefix(word);
        }
        if (heteronyms_.count(word) != 0) {
            return prepend_grapheme_prefix(word);
        }
        if (config_.locale == "en-US") {
            const auto suffix_lookup = [&](const std::string & suffix, std::vector<std::string> append) {
                if (word.size() <= suffix.size() || word.rfind(suffix) != word.size() - suffix.size()) {
                    return std::vector<std::string>{};
                }
                if (dict_.count(word) != 0) {
                    return std::vector<std::string>{};
                }
                const std::string base = word.substr(0, word.size() - suffix.size());
                if (!dict_has_unique(base) && dict_.count(base) == 0) {
                    return std::vector<std::string>{};
                }
                const auto it = dict_.find(base);
                if (it == dict_.end()) {
                    return std::vector<std::string>{};
                }
                auto out = it->second.front();
                out.insert(out.end(), append.begin(), append.end());
                return out;
            };
            auto out = suffix_lookup("'S", {"z"});
            if (!out.empty()) {
                return out;
            }
            out = suffix_lookup("S", {"z"});
            if (!out.empty()) {
                return out;
            }
        }
        auto it = dict_.find(word);
        if (it != dict_.end()) {
            return it->second.front();
        }
        if (config_.grapheme_case == "mixed") {
            const auto upper = ascii_upper(word);
            it = dict_.find(upper);
            if (it != dict_.end()) {
                return it->second.front();
            }
        }
        handled = false;
        return prepend_grapheme_prefix(word);
    }

    std::vector<std::string> parse_word(
        const std::string & word,
        bool phonemize_single_char,
        std::mt19937_64 & rng) const {
        const auto cased = set_grapheme_case(word, config_.grapheme_case);
        bool handled = false;
        auto out = parse_word_direct(cased, handled, phonemize_single_char, rng);
        if (!handled && cased.find('-') != std::string::npos) {
            std::vector<std::string> joined;
            std::stringstream ss(cased);
            std::string sub;
            bool first = true;
            while (std::getline(ss, sub, '-')) {
                if (sub.empty()) {
                    continue;
                }
                bool sub_handled = false;
                auto sub_out = parse_word_direct(sub, sub_handled, phonemize_single_char, rng);
                if (!first) {
                    joined.push_back("-");
                }
                joined.insert(joined.end(), sub_out.begin(), sub_out.end());
                first = false;
            }
            if (!joined.empty()) {
                return joined;
            }
        }
        return out;
    }
};

}  // namespace

class MagpieTextTokenizer::Impl {
public:
    explicit Impl(std::filesystem::path root)
        : root_(std::move(root)) {}

    MagpieTokenizationResult tokenize(const std::string & text, const MagpieTTSGenerationOptions & options) const {
        const std::string language = MagpieTextTokenizer::normalize_language(options.language);
        std::mt19937_64 rng(options.seed);
        MagpieTokenizationResult out;
        out.language = language;
        constexpr int32_t kTextEosId = 3358;
        if (const auto char_tokenizer = char_tokenizer_for_language(language)) {
            out.tokenizer_name = char_tokenizer->first;
            const auto tokens = exact_char_tokens(out.tokenizer_name);
            const int32_t pad_id = token_id_for_symbol(tokens, char_tokenizer->second, "<pad>");
            for (const auto & unit : tokenizer_units(text, options.text_chunk_size, options.text_chunk_mode)) {
                MagpieTokenChunk chunk;
                chunk.text = unit;
                chunk.tokens = encode_char_tokens(unit, tokens, char_tokenizer->second);
                chunk.tokens.push_back(kTextEosId);
                pad_short_text_chunk_before_eos(chunk.tokens, kTextEosId, pad_id);
                out.tokens.insert(out.tokens.end(), chunk.tokens.begin(), chunk.tokens.end());
                out.chunks.push_back(std::move(chunk));
            }
        } else if (language == "zh") {
            const MandarinTokenizer & tokenizer = mandarin_tokenizer();
            out.tokenizer_name = "mandarin_phoneme";
            for (const auto & unit : tokenizer_units(text, options.text_chunk_size, options.text_chunk_mode)) {
                MagpieTokenChunk chunk;
                chunk.text = unit;
                chunk.tokens = tokenizer.encode(unit);
                chunk.tokens.push_back(kTextEosId);
                pad_short_text_chunk_before_eos(chunk.tokens, kTextEosId, tokenizer.pad_id());
                out.tokens.insert(out.tokens.end(), chunk.tokens.begin(), chunk.tokens.end());
                out.chunks.push_back(std::move(chunk));
            }
        } else if (const auto byt5_offset = byt5_aggregate_offset_for_language(language)) {
            out.tokenizer_name = byt5_tokenizer_name_for_language(language);
            for (const auto & unit : tokenizer_units(text, options.text_chunk_size, options.text_chunk_mode)) {
                MagpieTokenChunk chunk;
                chunk.text = unit;
                chunk.tokens = encode_byt5_chars(unit, *byt5_offset);
                chunk.tokens.push_back(kTextEosId);
                pad_short_text_chunk_before_eos(chunk.tokens, kTextEosId, *byt5_offset);
                out.tokens.insert(out.tokens.end(), chunk.tokens.begin(), chunk.tokens.end());
                out.chunks.push_back(std::move(chunk));
            }
        } else {
            const auto config = ipa_config_for_language(language);
            const IpaTokenizer & tokenizer = tokenizer_for_config(language, config);
            out.tokenizer_name = tokenizer.tokenizer_name();
            for (const auto & unit : tokenizer_units(text, options.text_chunk_size, options.text_chunk_mode)) {
                MagpieTokenChunk chunk;
                chunk.text = unit;
                chunk.tokens = tokenizer.encode(unit, rng);
                chunk.tokens.push_back(kTextEosId);
                pad_short_text_chunk_before_eos(chunk.tokens, kTextEosId, tokenizer.pad_id());
                out.tokens.insert(out.tokens.end(), chunk.tokens.begin(), chunk.tokens.end());
                out.chunks.push_back(std::move(chunk));
            }
        }
        if (out.tokens.empty()) {
            throw std::runtime_error("MagpieTTS tokenizer produced no text tokens");
        }
        return out;
    }

private:
    const IpaTokenizer & tokenizer_for_config(const std::string & language, const IpaConfig & config) const {
        auto it = ipa_tokenizers_.find(language);
        if (it == ipa_tokenizers_.end()) {
            it = ipa_tokenizers_.emplace(language, std::make_unique<IpaTokenizer>(root_, config)).first;
        }
        return *it->second;
    }

    const MandarinTokenizer & mandarin_tokenizer() const {
        if (!mandarin_tokenizer_) {
            mandarin_tokenizer_ = std::make_unique<MandarinTokenizer>(root_);
        }
        return *mandarin_tokenizer_;
    }

    std::filesystem::path root_;
    mutable std::unordered_map<std::string, std::unique_ptr<IpaTokenizer>> ipa_tokenizers_;
    mutable std::unique_ptr<MandarinTokenizer> mandarin_tokenizer_;
};

MagpieTextTokenizer::MagpieTextTokenizer(std::filesystem::path resource_root)
    : impl_(std::make_unique<Impl>(std::move(resource_root))) {}

MagpieTextTokenizer::~MagpieTextTokenizer() = default;

MagpieTokenizationResult MagpieTextTokenizer::tokenize(
    const std::string & text,
    const MagpieTTSGenerationOptions & options) const {
    return impl_->tokenize(text, options);
}

std::string MagpieTextTokenizer::normalize_language(std::string language) {
    if (language.empty()) {
        return "en";
    }
    std::replace(language.begin(), language.end(), '_', '-');
    std::transform(language.begin(), language.end(), language.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (language == "pt" || language == "pt-br" || language == "pt-brasil" || language == "portuguese") {
        return "pt-BR";
    }
    if (language == "ar" || language == "ar-msa" || language == "arabic") {
        return "ar-MSA";
    }
    if (language == "ar-ae") {
        return "ar-AE";
    }
    if (language == "ar-sa") {
        return "ar-SA";
    }
    const auto dash = language.find('-');
    if (dash != std::string::npos) {
        language.resize(dash);
    }
    return language;
}

std::vector<std::string> MagpieTextTokenizer::supported_native_languages() {
    return {"ar-AE", "ar-MSA", "ar-SA", "de", "en", "es", "fr", "hi", "it", "ko", "pt-BR", "vi", "zh"};
}

}  // namespace engine::models::magpie_tts
