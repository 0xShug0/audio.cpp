#pragma once

#include "engine/community_models/sense_asr/types.h"
#include "engine/framework/decoders/ctc_host_decoder.h"
#include "engine/framework/runtime/session.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// Word timestamps from SenseVoice's CTC token spans.
namespace engine::community_models::sense_asr {

inline constexpr std::string_view kWordBoundary = "\xe2\x96\x81";

// The first code point of a UTF-8 string, or 0 when it is empty or invalid.
inline uint32_t first_code_point(const std::string &text) {
  if (text.empty()) {
    return 0;
  }
  const auto *bytes = reinterpret_cast<const unsigned char *>(text.data());
  const unsigned char lead = bytes[0];
  int length = 0;
  uint32_t value = 0;
  if (lead < 0x80) {
    return lead;
  } else if ((lead & 0xE0) == 0xC0) {
    length = 2;
    value = lead & 0x1F;
  } else if ((lead & 0xF0) == 0xE0) {
    length = 3;
    value = lead & 0x0F;
  } else if ((lead & 0xF8) == 0xF0) {
    length = 4;
    value = lead & 0x07;
  } else {
    return 0;
  }
  if (text.size() < static_cast<size_t>(length)) {
    return 0;
  }
  for (int i = 1; i < length; ++i) {
    if ((bytes[i] & 0xC0) != 0x80) {
      return 0;
    }
    value = (value << 6) | (bytes[i] & 0x3F);
  }
  return value;
}

// Scripts written without spaces between words: every token is its own
// timed unit, as SenseVoice's tokens there are single characters.
inline bool unspaced_script(uint32_t cp) {
  return (cp >= 0x3040 && cp <= 0x30FF) ||   // Hiragana, Katakana
         (cp >= 0x3400 && cp <= 0x4DBF) ||   // CJK Extension A
         (cp >= 0x4E00 && cp <= 0x9FFF) ||   // CJK Unified Ideographs
         (cp >= 0xF900 && cp <= 0xFAFF) ||   // CJK Compatibility Ideographs
         (cp >= 0x20000 && cp <= 0x3134F) || // CJK Extensions B-G
         (cp >= 0x0E00 && cp <= 0x0E7F);     // Thai
}

// Punctuation carries no timing of its own; it joins the preceding word.
inline bool punctuation(uint32_t cp) {
  return (cp >= 0x21 && cp <= 0x2F) || (cp >= 0x3A && cp <= 0x40) ||
         (cp >= 0x5B && cp <= 0x60) || (cp >= 0x7B && cp <= 0x7E) ||
         (cp >= 0x2010 && cp <= 0x205E) || // General Punctuation
         (cp >= 0x3000 && cp <= 0x303F) || // CJK Symbols and Punctuation
         (cp >= 0xFF01 && cp <= 0xFF0F) || (cp >= 0xFF1A && cp <= 0xFF20) ||
         (cp >= 0xFF3B && cp <= 0xFF40) || (cp >= 0xFF5B && cp <= 0xFF65);
}

// Groups the decoded text tokens into words. A token opens a new word when
// it starts with the SentencePiece boundary, follows or is a character of an
// unspaced script; otherwise it continues the open word. Frames are shifted
// by the query-token prefix so 0 is the first audio frame.
inline std::vector<SenseAsrWordFrames>
group_words(const std::vector<decoders::CtcTokenSpan> &spans,
            const std::vector<std::string> &vocab, int64_t prefix_frames) {
  std::vector<SenseAsrWordFrames> words;
  bool open_word = false; // The last word may take more subword tokens.
  for (const auto &span : spans) {
    if (span.id < 0 || span.id >= static_cast<int32_t>(vocab.size())) {
      continue;
    }
    std::string piece = vocab[static_cast<size_t>(span.id)];
    if (piece.size() >= 2 && piece[0] == '<' && piece[1] == '|') {
      continue; // Language, emotion, event and ITN tags.
    }
    bool boundary = false;
    while (piece.compare(0, kWordBoundary.size(), kWordBoundary) == 0) {
      boundary = true;
      piece.erase(0, kWordBoundary.size());
    }
    if (piece.empty()) {
      open_word = false;
      continue;
    }
    const int64_t start = std::max<int64_t>(0, span.start_frame - prefix_frames);
    const int64_t end = std::max<int64_t>(start + 1, span.end_frame - prefix_frames);
    const uint32_t cp = first_code_point(piece);
    if (punctuation(cp)) {
      // An open word stays open, so "I" "'" "ve" or "e" "-" "mail" stay one
      // word; the next boundary marker still ends it.
      if (!words.empty()) {
        words.back().text += piece;
      }
      continue;
    }
    if (unspaced_script(cp)) {
      words.push_back({piece, start, end});
      open_word = false;
      continue;
    }
    if (open_word && !boundary) {
      words.back().text += piece;
      words.back().end_frame = end;
      continue;
    }
    words.push_back({piece, start, end});
    open_word = true;
  }
  return words;
}

// Converts word frames to samples of audio at sample_rate, each frame being
// frame_ms long, clamped to the total_samples of the transcribed audio.
// Times are where CTC emitted the tokens, not a forced alignment: starts
// track onsets closely, ends fall at the last emission frame.
inline std::vector<runtime::WordTimestamp>
word_timestamps_from_frames(const std::vector<SenseAsrWordFrames> &words,
                            int64_t frame_ms, int sample_rate,
                            int64_t total_samples) {
  const double frame_samples = static_cast<double>(frame_ms) *
                               static_cast<double>(sample_rate) / 1000.0;
  std::vector<runtime::WordTimestamp> out;
  out.reserve(words.size());
  for (const auto &word : words) {
    const int64_t start = std::min<int64_t>(
        total_samples,
        std::llround(static_cast<double>(word.start_frame) * frame_samples));
    const int64_t end = std::min<int64_t>(
        total_samples,
        std::llround(static_cast<double>(word.end_frame) * frame_samples));
    out.push_back({{start, std::max(start, end)}, word.text, 0.0F});
  }
  return out;
}

} // namespace engine::community_models::sense_asr
