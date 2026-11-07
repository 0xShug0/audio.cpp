#pragma once

// Word spans for sanoTTS word timestamps. Pure functions of the token stream
// and the chunk text, so they can be tested without the phonemizer.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace engine::models::sanotts::word_timestamps {

/** Frame range [start, end) in decoder frames. */
struct FrameRange {
    int64_t start = 0;
    int64_t end = 0;
};

/** A run of phonemes between spaces and word-end marks. */
struct PhonemeGroup {
    FrameRange frames;
    std::string phonemes;   // the group's symbols, concatenated
};

struct LabeledRange {
    FrameRange frames;
    std::string label;
};

/** Token id -> vocabulary symbol; empty for specials and unknown ids. */
using SymbolLookup = std::function<std::string(int32_t)>;

/** Word -> its phonemes (concatenated symbols) when pronounced on its own. */
using WordPhonemizer = std::function<std::string(const std::string & word)>;

/** True for symbols that carry sound, i.e. not a space, word-end mark, or
 *  empty special. Apostrophes and hyphens are excluded too. */
bool is_phoneme_symbol(const std::string & symbol);

/** Spoken words of the chunk text: runs between whitespace and word-end
 *  marks (! " ( ) , . : ; ? em dash, curly quotes). Apostrophes and hyphens
 *  stay inside a word. */
std::vector<std::string> text_words(const std::string & text);

/** Phoneme groups of the token stream. Token t covers frames
 *  [sum(d[<t]), sum(d[<=t])). */
std::vector<PhonemeGroup> phoneme_word_ranges(
    const std::vector<int32_t> & token_ids,
    const std::vector<int64_t> & token_frames,
    const SymbolLookup & symbol_for);

/** Labels each word with its frame range. Equal counts pair by index. When
 *  the counts differ, each word is phonemized alone and its phonemes are
 *  aligned to the stream's phonemes by edit distance (espeak can merge or
 *  split words). Frames are spread evenly over each group's phonemes. If the
 *  alignment is poor, returns one range covering all groups, labeled with the
 *  joined words. */
std::vector<LabeledRange> label_ranges(
    const std::vector<PhonemeGroup> & groups,
    const std::vector<std::string> & words,
    const WordPhonemizer & phonemize_word);

}  // namespace engine::models::sanotts::word_timestamps
