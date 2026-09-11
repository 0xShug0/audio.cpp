#include "engine/models/kokoro_tts/g2p_multilingual.h"
#include "engine/framework/audio/espeak_phonemizer.h"
#include "engine/framework/io/json.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <codecvt>
#include <cstdlib>
#include <locale>
#include <map>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace engine::models::kokoro_tts {
namespace {
using engine::io::json::Value;
using U = std::u32string;
U decode(const std::string & s) { return std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t>{}.from_bytes(s); }
std::string encode(const U & s) { return std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t>{}.to_bytes(s); }
void replace(std::string & s, const std::string & from, const std::string & to) {
    size_t p = 0;
    while ((p = s.find(from, p)) != std::string::npos) { s.replace(p, from.size(), to); p += to.size(); }
}
std::string spaces(const std::string & s) {
    auto out = std::regex_replace(s, std::regex("[ \\t\\r\\n]+"), " ");
    auto start = out.find_first_not_of(' ');
    return start == std::string::npos ? "" : out.substr(start, out.find_last_not_of(' ') - start + 1);
}
bool han(char32_t c) { return c >= 0x4e00 && c <= 0x9fff; }

class Library {
#ifdef _WIN32
    HMODULE handle_ = nullptr;
#else
    void * handle_ = nullptr;
#endif
public:
    Library(const char * env, const char * windows_name, const char * unix_name) {
        const char * override_path = std::getenv(env);
#ifdef _WIN32
        std::filesystem::path path;
        if (override_path && *override_path) path = std::filesystem::u8path(override_path);
        else {
            std::wstring exe(32768, L'\0');
            const auto n = GetModuleFileNameW(nullptr, exe.data(), static_cast<DWORD>(exe.size()));
            if (!n || n >= exe.size()) throw std::runtime_error("Cannot locate the audio.cpp executable");
            exe.resize(n);
            path = std::filesystem::path(exe).parent_path() / windows_name;
        }
        handle_ = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
#else
        (void) windows_name;
        handle_ = dlopen(override_path && *override_path ? override_path : unix_name, RTLD_NOW | RTLD_LOCAL);
#endif
        if (!handle_) throw std::runtime_error(std::string("Missing Kokoro pronunciation library; set ") + env);
    }
    ~Library() {
#ifdef _WIN32
        if (handle_) FreeLibrary(handle_);
#else
        if (handle_) dlclose(handle_);
#endif
    }
    template<class F> F symbol(const char * name) {
#ifdef _WIN32
        auto p = GetProcAddress(handle_, name);
#else
        auto p = dlsym(handle_, name);
#endif
        if (!p) throw std::runtime_error(std::string("Missing pronunciation API: ") + name);
        return reinterpret_cast<F>(p);
    }
};

std::string espeak_text(const std::string & text, const std::string & language, const std::filesystem::path & root) {
    const auto * library_override = std::getenv("AUDIOCPP_ESPEAK_LIBRARY");
    const auto * data_override = std::getenv("AUDIOCPP_ESPEAK_DATA");
    const auto library = library_override && *library_override
        ? std::filesystem::u8path(library_override) : std::filesystem::path{};
    std::filesystem::path data;
    if (data_override && *data_override) data = std::filesystem::u8path(data_override);
#ifdef AUDIOCPP_STATIC_ESPEAK
    else if (!library.empty()) data = root / "espeak-ng-data";
#else
    else data = root / "espeak-ng-data";
#endif
    engine::audio::EspeakPhonemizer phonemizer(library, data, {language == "fr-fr" ? "fr" : language});
    // Preserve punctuation ourselves: TextToPhonemes consumes clause punctuation.
    const U punctuation = U";:,.!?¡¿—…\"«»“”()";
    std::string out, chunk;
    auto flush = [&] {
        if (chunk.empty()) return;
        const auto ps = phonemizer.phonemize(chunk, 2 | (1 << 7) | ('^' << 8));
        if (!out.empty() && out.back() != ' ' && out != u8"¿" && out != u8"¡") out += ' ';
        out += ps;
        chunk.clear();
    };
    for (char32_t c : decode(text)) {
        if (punctuation.find(c) != U::npos) {
            const bool spaced = !chunk.empty() && chunk.back() == ' ';
            flush(); if (spaced && !out.empty() && out.back() != ' ') out += ' ';
            out += encode(U(1, c));
        }
        else chunk += encode(U(1, c));
    }
    flush();
    out = std::regex_replace(out, std::regex("\\([a-z-]+\\)"), "");
    for (const auto & pair : std::vector<std::pair<std::string, std::string>>{
        {u8"a^ɪ", "I"}, {u8"a^ʊ", "W"}, {"d^z", u8"ʣ"}, {u8"d^ʒ", u8"ʤ"},
        {u8"e^ɪ", "A"}, {u8"o^ʊ", "O"}, {u8"ə^ʊ", "Q"}, {"s^s", "S"},
        {"t^s", u8"ʦ"}, {u8"t^ʃ", u8"ʧ"}, {u8"ɔ^ɪ", "Y"}}) replace(out, pair.first, pair.second);
    replace(out, "^", ""); replace(out, "-", "");
    return spaces(out);
}

std::vector<std::string> split(const std::string & s, char delim) {
    std::vector<std::string> out;
    std::istringstream in(s); std::string item;
    while (std::getline(in, item, delim)) out.push_back(item);
    return out;
}
std::string cjk_number(uint64_t n, bool ja) {
    const U digits = ja ? U"零一二三四五六七八九" : U"零一二三四五六七八九";
    if (n < 10) return encode(U(1, digits[n]));
    const std::vector<std::pair<uint64_t, U>> units = {
        {100000000, ja ? U"億" : U"亿"}, {10000, U"万"}, {1000, U"千"}, {100, U"百"}, {10, U"十"}};
    for (const auto & [unit, name] : units) if (n >= unit) {
        auto head = n / unit; auto rest = n % unit;
        auto out = ((head == 1 && (ja ? unit < 10000 : unit == 10)) ? "" : cjk_number(head, ja)) + encode(name);
        if (rest) {
            if (!ja && unit >= 100 && rest < unit / 10) out += u8"零";
            out += cjk_number(rest, ja);
        }
        return out;
    }
    return "";
}
std::string normalize_numbers(const std::string & text, bool ja) {
    std::string out;
    const auto s = decode(text);
    for (size_t i = 0; i < s.size();) {
        char32_t c = s[i];
        if (c >= 0xff01 && c <= 0xff5e) c -= 0xfee0;
        if (c < U'0' || c > U'9') { out += encode(U(1, c)); ++i; continue; }
        std::string number;
        while (i < s.size()) {
            c = s[i]; if (c >= 0xff10 && c <= 0xff19) c -= 0xfee0;
            if (c < U'0' || c > U'9') break;
            number += static_cast<char>(c); ++i;
        }
        if (number.size() > 12) throw std::runtime_error("Kokoro CJK number exceeds 12 digits; spell it out");
        out += cjk_number(std::stoull(number), ja);
    }
    return out;
}
}

struct MultilingualG2P::Impl {
    std::filesystem::path root;
    mutable std::once_flag ja_once, zh_once;
    mutable std::unordered_map<U, U> kana;
    mutable std::set<std::string> ja_words;
    mutable Value zh;
    explicit Impl(std::filesystem::path path) : root(std::move(path)) {}
    void load_ja() const {
        std::call_once(ja_once, [&] {
            auto value = engine::io::json::parse_file(root / "g2p/ja.json");
            for (const auto & [k, v] : value.require("kana").as_object()) kana.emplace(decode(k), decode(v.as_string()));
            for (const auto & v : value.require("words").as_array()) ja_words.insert(v.as_string());
        });
    }
    std::string japanese(const std::string & text) const {
        load_ja();
        static Library lib("AUDIOCPP_MECAB_LIBRARY", "libmecab.dll", "libmecab.so.2");
        auto create = lib.symbol<void * (*)(int, char **)>("mecab_new");
        auto destroy = lib.symbol<void (*)(void *)>("mecab_destroy");
        // Prefix of MeCab's stable C node ABI; later cost fields are not accessed.
        struct Node {
            Node * prev; Node * next; Node * enext; Node * bnext;
            void * rpath; void * lpath;
            const char * surface; const char * feature;
            unsigned int id;
            unsigned short length, rlength, rcAttr, lcAttr, posid;
            unsigned char char_type, stat;
        };
        auto parse = lib.symbol<const Node * (*)(void *, const char *)>("mecab_sparse_tonode");
        auto error = lib.symbol<const char * (*)(void *)>("mecab_strerror");
        std::vector<std::string> args = {"mecab", "-r", (root / "unidic/dicrc").u8string(),
            "-d", (root / "unidic").u8string()};
        std::vector<char *> argv; for (auto & arg : args) argv.push_back(arg.data());
        void * tagger = create(static_cast<int>(argv.size()), argv.data());
        if (!tagger) throw std::runtime_error(std::string("Cannot load Japanese dictionary: ") + error(nullptr));
        struct End { void * p; void (*fn)(void *); ~End() { fn(p); } } end{tagger, destroy};
        const auto normalized = normalize_numbers(text, true);
        const Node * result = parse(tagger, normalized.c_str());
        if (!result) throw std::runtime_error("Japanese tokenization failed");
        struct Word { std::string surface; U reading; int type; };
        std::vector<Word> words;
        for (auto * node = result; node; node = node->next) {
            if (node->stat >= 2) continue;
            std::string surface(node->surface, node->length);
            auto fields = split(node->feature, ',');
            auto reading_field = [&](size_t index) {
                return index < fields.size() ? fields[index] : std::string{};
            };
            auto pron = reading_field(9); if (pron.empty()) pron = reading_field(17);
            if (pron.empty()) pron = surface;
            if (pron == "*") continue; // UniDic's no-pronunciation marker, as in Misaki.
            U reading = decode(pron);
            for (auto & c : reading) if (c >= 0x30a1 && c <= 0x30f6) c -= 0x60;
            int type = node->char_type;
            words.push_back({surface, std::move(reading), type == 7 || node->stat == 0 ? 6 : type});
        }
        // Preserve Misaki's lexical grouping before inserting spaces.
        for (size_t i = 0; i < words.size(); ++i) {
            std::string combined; size_t last = i;
            for (size_t j = i; j < words.size() && words[j].type == words[i].type; ++j) {
                combined += words[j].surface;
                if (ja_words.count(combined)) last = j;
            }
            while (last > i) {
                words[i].surface += words[i + 1].surface;
                words[i].reading += words[i + 1].reading;
                words.erase(words.begin() + i + 1); --last;
            }
        }
        auto mapping = [&](const U & key) { auto it = kana.find(key); return it == kana.end() ? U{} : it->second; };
        U output;
        for (const auto & w : words) {
            U phonemes;
            auto surface = decode(w.surface);
            bool ascii = std::all_of(surface.begin(), surface.end(), [](char32_t c) { return c < 128; });
            if (ascii) phonemes = surface;
            else if (w.type != 6 && w.type != 3) continue;
            else for (size_t i = 0; i < w.reading.size(); ++i) {
                auto c = w.reading[i]; U key(1, c);
                auto prev = i ? w.reading.substr(i - 1, 2) : U{};
                auto next = w.reading.substr(i, 2);
                if (prev.size() == 2 && kana.count(prev)) { phonemes += mapping(prev); continue; }
                if (next.size() == 2 && kana.count(next)) continue;
                if (c == U'ー') { phonemes += U"ː"; continue; }
                if (c == U'っ') { phonemes += U"ʔ"; continue; }
                if (c == U'ん') {
                    auto after = i + 1 < w.reading.size() ? mapping(w.reading.substr(i + 1, 1)) : U{};
                    phonemes += after.empty() ? U"ɴ" : U(U"mpb").find(after[0]) != U::npos ? U"m" :
                        U(U"kɡ").find(after[0]) != U::npos ? U"ŋ" : U(U"ɲʨʥ").find(after[0]) != U::npos ? U"ɲ" :
                        U(U"ntdɾz").find(after[0]) != U::npos ? U"n" : U"ɴ";
                    continue;
                }
                if (U(U"ゃゅょぁぃぅぇぉ").find(c) != U::npos) continue;
                auto ps = mapping(key);
                if (ps.empty() && c != U' ' && c != 0x3099 && c != 0x309a)
                    throw std::runtime_error("Japanese pronunciation missing for: " + encode(key));
                phonemes += ps;
            }
            if (phonemes.empty()) continue;
            if (phonemes.size() == 1 && U(U"]).,?!:").find(phonemes[0]) != U::npos)
                while (!output.empty() && output.back() == U' ') output.pop_back();
            output += phonemes;
            if (!(phonemes.size() == 1 && U(U"([“").find(phonemes[0]) != U::npos)) output += U' ';
        }
        auto out = spaces(encode(output));
        replace(out, "(", u8"«"); replace(out, ")", u8"»");
        return out;
    }

    std::string chinese(const std::string & text) const {
        std::call_once(zh_once, [&] { zh = engine::io::json::parse_file(root / "g2p/zh.json"); });
        const auto & frequencies = zh.require("frequency").as_object();
        const auto & characters = zh.require("chars").as_object();
        const auto & phrases = zh.require("phrases").as_object();
        const auto & ipa = zh.require("ipa").as_object();
        const double log_total = std::log(static_cast<double>(zh.require("total").as_i64()));
        U text32 = decode(normalize_numbers(text, false));
        std::string out;
        for (size_t start = 0; start < text32.size();) {
            if (!han(text32[start])) {
                const U punct = U"、，。．！：；？《》「」（）";
                const U mapped = U",,..!:;?“”“”()";
                auto p = punct.find(text32[start]);
                out += encode(U(1, p == U::npos ? text32[start] : mapped[p]));
                if (p != U::npos || U(U",.!?:;").find(text32[start]) != U::npos) out += ' ';
                ++start; continue;
            }
            size_t end = start; while (end < text32.size() && han(text32[end])) ++end;
            U span = text32.substr(start, end - start);
            // Jieba's maximum-probability dictionary segmentation.
            std::vector<double> score(span.size() + 1, 0);
            std::vector<size_t> next(span.size());
            for (size_t i = span.size(); i-- > 0;) {
                score[i] = -1e300; next[i] = i + 1;
                for (size_t j = i + 1; j <= span.size() && j - i <= 32; ++j) {
                    auto found = frequencies.find(encode(span.substr(i, j - i)));
                    if (j != i + 1 && found == frequencies.end()) continue;
                    double frequency = found == frequencies.end() ? 1 : static_cast<double>(found->second.as_i64());
                    double candidate = std::log(frequency) - log_total + score[j];
                    if (candidate >= score[i]) { score[i] = candidate; next[i] = j; }
                }
            }
            // Run the same four-state Jieba HMM on consecutive singleton DAG words.
            std::vector<U> words;
            auto flush_singletons = [&](const U & buffer) {
                if (buffer.empty()) return;
                if (buffer.size() == 1 || frequencies.count(encode(buffer))) {
                    for (char32_t c : buffer) words.push_back(U(1, c));
                    return;
                }
                const std::string states = "BMES";
                const std::array<std::string, 4> previous = {"ES", "MB", "BM", "SE"};
                std::vector<std::array<double, 4>> probs(buffer.size());
                std::vector<std::array<int, 4>> back(buffer.size());
                auto probability = [](const Value & table, const std::string & key) {
                    const auto * v = table.find(key); return v ? v->as_number() : -3.14e100;
                };
                const auto & emission = *zh.find("emission");
                const auto & transition = *zh.find("transition");
                for (int y = 0; y < 4; ++y)
                    probs[0][y] = probability(*zh.find("start"), states.substr(y, 1)) +
                        probability(*emission.find(states.substr(y, 1)), encode(buffer.substr(0, 1)));
                for (size_t t = 1; t < buffer.size(); ++t) for (int y = 0; y < 4; ++y) {
                    const auto state = states.substr(y, 1);
                    const auto emit = probability(*emission.find(state), encode(buffer.substr(t, 1)));
                    double best = -1e300; char best_char = 0; int best_index = 0;
                    for (char prev : previous[y]) {
                        int k = static_cast<int>(states.find(prev));
                        double value = probs[t - 1][k] + probability(*transition.find(std::string(1, prev)), state) + emit;
                        if (value > best || (value == best && prev > best_char)) {
                            best = value; best_char = prev; best_index = k;
                        }
                    }
                    probs[t][y] = best; back[t][y] = best_index;
                }
                int state = probs.back()[2] > probs.back()[3] ? 2 : 3;
                std::vector<int> path(buffer.size());
                for (size_t t = buffer.size(); t-- > 0;) { path[t] = state; if (t) state = back[t][state]; }
                size_t begin = 0, consumed = 0;
                for (size_t t = 0; t < path.size(); ++t) {
                    if (states[path[t]] == 'B') begin = t;
                    else if (states[path[t]] == 'E') { words.push_back(buffer.substr(begin, t - begin + 1)); consumed = t + 1; }
                    else if (states[path[t]] == 'S') { words.push_back(buffer.substr(t, 1)); consumed = t + 1; }
                }
                if (consumed < buffer.size()) words.push_back(buffer.substr(consumed));
            };
            U buffer;
            for (size_t i = 0; i < span.size();) {
                U word = span.substr(i, next[i] - i);
                if (word.size() == 1) buffer += word;
                else { flush_singletons(buffer); buffer.clear(); words.push_back(word); }
                i = next[i];
            }
            flush_singletons(buffer);
            for (size_t i = 0; i < words.size(); ++i) {
                const U & word = words[i];
                const auto phrase = phrases.find(encode(word));
                for (size_t k = 0; k < word.size(); ++k) {
                    std::string syllable;
                    if (phrase != phrases.end()) syllable = phrase->second.as_array().at(k).as_string();
                    else {
                        auto ch = characters.find(encode(U(1, word[k])));
                        if (ch == characters.end()) throw std::runtime_error("Missing Chinese pronunciation: " + encode(U(1, word[k])));
                        syllable = ch->second.as_string();
                    }
                    auto found = ipa.find(syllable);
                    if (found == ipa.end()) throw std::runtime_error("Missing Chinese pinyin mapping: " + syllable);
                    out += found->second.as_string();
                }
                if (i + 1 < words.size()) out += ' ';
            }
            start = end;
        }
        return spaces(out);
    }
};
MultilingualG2P::MultilingualG2P(const std::filesystem::path & root) : impl_(std::make_unique<Impl>(root)) {}
MultilingualG2P::~MultilingualG2P() = default;
std::string MultilingualG2P::phonemize(const std::string & text, const std::string & language) const {
    if (language == "j") return impl_->japanese(text);
    if (language == "z") return impl_->chinese(text);
    static const std::map<std::string, std::string> langs = {{"e", "es"}, {"f", "fr-fr"},
        {"h", "hi"}, {"i", "it"}, {"p", "pt-br"}};
    auto it = langs.find(language);
    if (it == langs.end()) throw std::runtime_error("Unsupported Kokoro language: " + language);
    return espeak_text(text, it->second, impl_->root);
}
}
