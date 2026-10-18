#pragma once

// SPDX-License-Identifier: MIT
//
// Minimal Jieba word segmentation for the ZipVoice Chinese frontend.
//
// This is a focused port of the parts of cppjieba (Copyright (c) 2013 Yanyi
// Wu, MIT) and jieba (Copyright (c) 2012 Sun Junyi, MIT) that the frontend
// needs: the PreFilter separator pass, the maximum-probability DAG
// segmentation over jieba.dict.utf8, and the four-state BMES HMM over
// hmm_model.utf8. It reads the same dictionary files that are embedded in the
// model GGUF, so no language-specific library is vendored.
//
// cppjieba: https://github.com/yanyiwu/cppjieba (MIT)
// jieba:    https://github.com/fxsjy/jieba   (MIT)

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::zipvoice {

// Word segmentation equivalent to cppjieba::MixSegment (HMM enabled), which in
// turn matches python jieba.cut for the same dictionaries.
class JiebaSegmenter {
public:
    JiebaSegmenter(const std::filesystem::path & dict_path,
                   const std::filesystem::path & hmm_path);

    // Word boundaries for `text` (UTF-8). Separators become single-character
    // words; runs of dictionary-missed single characters are re-segmented with
    // the BMES HMM, exactly like MixSegment.
    std::vector<std::string> cut(const std::string & text) const;

private:
    struct Rune {
        uint32_t cp = 0;
        uint32_t offset = 0;  // byte offset in the source string
        uint32_t len = 0;     // byte length
    };

    static std::vector<Rune> decode(const std::string & text);
    static std::string slice(const std::string & text, const std::vector<Rune> & runes,
                             size_t begin, size_t end_inclusive);
    static bool is_separator(uint32_t cp);
    static size_t sequential_letter_rule(const std::vector<Rune> & runes, size_t begin, size_t end);
    static size_t numbers_rule(const std::vector<Rune> & runes, size_t begin, size_t end);
    // MP DAG + HMM over the rune range [begin, end).
    void segment_range(const std::string & text, const std::vector<Rune> & runes,
                       size_t begin, size_t end, std::vector<std::string> & words) const;
    void hmm_cut(const std::string & text, const std::vector<Rune> & runes,
                 size_t begin, size_t end, std::vector<std::string> & words) const;
    void viterbi_cut(const std::string & text, const std::vector<Rune> & runes,
                     size_t begin, size_t end, std::vector<std::string> & words) const;
    double emit_probability(size_t state, uint32_t cp) const;

    std::unordered_map<std::string, double> weights_;  // word -> log(freq/freq_sum)
    double min_weight_ = 0.0;
    size_t max_word_len_ = 1;
    double start_prob_[4] = {};
    double trans_prob_[4][4] = {};
    std::unordered_map<uint32_t, double> emit_prob_[4];
};

}  // namespace engine::models::zipvoice
