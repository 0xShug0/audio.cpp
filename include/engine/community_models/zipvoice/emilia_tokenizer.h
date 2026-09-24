#pragma once

// Chinese/English mixed text frontend for ZipVoice, mirroring the upstream
// EmiliaTokenizer (zipvoice/tokenizer/tokenizer.py). Chinese segments are
// converted with baked pypinyin tables (tone3 syllables split into
// initial+"0" / final+tone tokens, e.g. "w0 o3") plus run-time tone sandhi;
// English segments are phonemized with eSpeak-ng; <pin3yin1> overrides use
// the bare-syllable table. Output token ids follow tokens.txt with OOV
// tokens skipped, exactly like the reference.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::audio {
class EspeakPhonemizer;
}

namespace engine::models::zipvoice {

class JiebaSegmenter;

class EmiliaTokenizer {
public:
    struct EspeakConfig {
        std::string library_path;  // optional: libespeak-ng path
        std::string data_path;     // optional: espeak-ng data directory
        std::string lang = "en-us";
    };

    // Frontend table locations. The spec-registered resource bundle hands
    // over materialized GGUF-embedded sidecars; development directories and
    // direct API callers fill the paths from the model directory.
    struct TablePaths {
        std::filesystem::path chars;       // zh_chars.tsv
        std::filesystem::path phrases;     // zh_phrases.tsv
        std::filesystem::path syllables;   // zh_syllables.tsv
        std::filesystem::path jieba_dict;  // zh_jieba_dict.txt
        std::filesystem::path hmm_model;   // zh_hmm_model.txt

        // Loose-file layout: everything in one model directory.
        static TablePaths from_model_dir(const std::filesystem::path & dir) {
            return {dir / "zh_chars.tsv", dir / "zh_phrases.tsv",
                    dir / "zh_syllables.tsv", dir / "zh_jieba_dict.txt",
                    dir / "zh_hmm_model.txt"};
        }
    };

    // `vocab` is the tokens.txt token -> id map; OOV phones are skipped.
    // The Chinese tables are optional only if no Chinese text is encoded.
    EmiliaTokenizer(const TablePaths & tables,
                    const std::unordered_map<std::string, int32_t> & vocab,
                    const EspeakConfig & espeak);
    ~EmiliaTokenizer();  // out-of-line: owns an incomplete-type unique_ptr

    std::vector<int32_t> encode(const std::string & text) const;

private:
    std::unordered_map<std::string, int32_t> vocab_;
    TablePaths tables_;
    std::unordered_map<std::string, std::vector<std::string>> chars_;
    std::unordered_map<std::string, std::vector<std::string>> phrases_;
    std::unordered_map<std::string, std::vector<std::string>> syllables_;
    size_t max_phrase_codepoints_ = 0;
    EspeakConfig espeak_;
    // Lazily created so a pure-Chinese workload never touches eSpeak.
    mutable std::unique_ptr<audio::EspeakPhonemizer> phonemizer_;
    // Lazily created jieba segmenter (model-local MixSegment port = python
    // jieba.cut with HMM): word boundaries define reading lookup and the
    // scope of tone-sandhi application, mirroring the reference pipeline
    // lazy_pinyin(jieba.cut(text), tone_sandhi=True).
    mutable std::unique_ptr<JiebaSegmenter> segmenter_;
    mutable std::mutex segmenter_mutex_;
};

}  // namespace engine::models::zipvoice
