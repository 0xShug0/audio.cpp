// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Adapted for audio.cpp from NVIDIA/NeMo-Speech.cpp, revision 66d50d91b75a406becd95fbaf39d2c9c13e95d7c.
// Host side of SALM word timestamps, ported from NeMo's MultiSpeakerSOTWordTimestampAligner
// (nemo/collections/speechlm2/parts/ctc_timestamp_utils.py, main 6f558408db) and the vLLM
// plugin's public result (speechlm2/vllm/salm/ctc_timestamps.py): transcript words and speaker
// streams, their CTC targets, the token-distribution speaker mapping, the anchor-banded max-sum CTC
// search, word rows, diarization-assisted word refinement and the 10 ms diarization segments. Pure
// functions over host buffers; the CTC head that produces the log-probabilities is in
// ctc_timestamp.h.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sentencepiece {
class SentencePieceProcessor;
}

namespace engine::models::nemotron_transcribe::timestamps {

// ---- transcript ----

// Python str.split(): runs of Unicode whitespace separate words; invalid UTF-8 bytes are word
// bytes.
std::vector<std::string> split_whitespace(const std::string& text);

// What a timestamped word is: a whitespace word, one character of a Chinese or Japanese
// transcript, or one character of a Thai transcript with its combining marks.
enum class UnitKind : uint8_t { word, character, cluster };

struct TranscriptWord {
    std::string word;
    int tag = -1;   // <spk:N> number in force, -1 = none (NeMo's speaker_tag None)
    int index = 0;  // position in the transcript's word list
    UnitKind kind = UnitKind::word;
    // The unit in running text when that differs from `word` (a character keeps the punctuation
    // around it), and whether it follows the previous unit without a space (one word's characters).
    std::string display;
    bool joins_previous = false;
};

// NeMo's clean_sot_transcript: every special token of the decoding tokenizer except speaker
// (<spk:N>), event (<ev:...>) and style (<sty:...>) tags becomes a space, then the words are
// joined by single spaces.
std::string clean_sot_transcript(
    const std::string& transcript, const std::vector<std::string>& special_tokens);

// NeMo's parse_sot_words: em dashes (U+2014) separate words; each <spk:N> tag (`<spk:(\d+)>`,
// case-insensitive, Unicode decimal digits) starts a speaker turn. Without a tag every word has tag
// 0; with tags, words before the first one have none (-1). Any other text, malformed tags included,
// is words.
std::vector<TranscriptWord> parse_sot_words(const std::string& transcript);

// "spk:N", or "unknown" for tag < 0.
std::string speaker_label(int tag);

// ---- alignment units ----

// NeMo's _lexical_characters: the code points of general category L*, M* or N*. NeMo takes them
// from the NFC text; the decoder's text is used as it is (invalid bytes become U+FFFD).
std::vector<std::string> lexical_characters(const std::string& text);

// NeMo's script detection over the words that are not event tags (<...>): "ko" with any Hangul,
// else "ja" with kana, "zh" with CJK ideographs, "th" with Thai, else "".
std::string timestamp_script(const std::vector<TranscriptWord>& words);

// NeMo's _timestamp_units: in a Chinese or Japanese transcript every lexical character of every
// word is a unit (punctuation is not), event tags stay whole; a Thai transcript has character
// units too, each with its combining marks (NeMo's newmm words need a dictionary); any other
// transcript keeps its words. Units are renumbered and carry their running text in `display`.
std::vector<TranscriptWord> timestamp_units(const std::vector<TranscriptWord>& words);

// The CTC head's SentencePiece model from a file (tests; the model reads it from its GGUF).
// NeMo's SentencePieceTokenizer with legacy=false encodes with the model alone: its special
// tokens are the model's user-defined pieces, matched atomically.
std::shared_ptr<const sentencepiece::SentencePieceProcessor> load_tokenizer(
    const std::string& path);

// NeMo's _character_token_ids: the unit's own piece, else its encoding without bare U+2581
// pieces; the ids must be known, below `blank` and decode to the unit. `normalized` compares
// both sides after the tokenizer's normalization (Thai SARA AM decomposes under NFKC). Throws
// std::runtime_error ("Unrepresentable character") otherwise.
std::vector<int> character_token_ids(
    const sentencepiece::SentencePieceProcessor& tokenizer, const std::string& unit, int blank,
    bool normalized = false);

// ---- CTC targets ----

// One speaker's words (NeMo's _group_words_by_speaker) and their blank-expanded CTC target.
struct SpeakerStream {
    int tag = -1;
    std::vector<int> words;          // unit indices, transcript order
    std::vector<int> labels;         // compact columns (blank = 0): blank, tok, blank, ..., blank
    std::vector<int> state_to_word;  // position in `words` per state, -1 for blank states
};

struct AlignmentTargets {
    std::vector<TranscriptWord> words;   // the transcript's units
    std::vector<SpeakerStream> streams;  // in first-appearance order of the tags
    // The CTC classes the alignment reads (vocabulary ids): blank, then every stream token
    // ascending. The streams' labels index this list.
    std::vector<int> columns;
};

// NeMo's _prepare for one transcript: words (parse_sot_words, timestamp_units), speaker streams,
// each tokenized on its own (_tokenize_word_stream: the stream's words joined by single spaces,
// which must equal the words tokenized one by one; each unit alone with character units) and
// blank-expanded, the minimum-frames check of every stream against `ctc_frames`, and the compact
// columns. Throws std::runtime_error with NeMo's messages; NeMo rejects the whole transcript then.
AlignmentTargets alignment_targets(
    const sentencepiece::SentencePieceProcessor& tokenizer, const std::string& transcript,
    int blank, int ctc_frames);

// One frame per token plus one per repeated adjacent token.
int minimum_frames(const std::vector<int>& labels);

// ---- CTC head windows ----

struct HeadWindow {
    int start = 0, end = 0;                  // frames kept from this window
    int context_start = 0, context_end = 0;  // frames the head reads
};
// Head windows over `frames` joined states: one window when frames <= core (or core <= 0), else
// cores of `core` frames with `left` / `right` context frames, clipped.
std::vector<HeadWindow> head_windows(int frames, int core, int left, int right);

// ---- numerics of NeMo's libraries ----

// numpy.log of a float32 array (NumPy 2.x SIMD kernel, AVX2/AVX512F: rational minimax polynomial
// with FMA); equal to NumPy's for every float32.
float numpy_log(float x);

// NeMo's _resample_speaker_probs over speaker probabilities clamped to [0, 1] (rows x speakers):
// unchanged with `frames` rows, repeated from one row, else linearly interpolated at
// torch.linspace(0, rows - 1, frames) in float32, as PyTorch computes it.
std::vector<float> resample_speaker_probs(const float* probs, int rows, int speakers, int frames);

// SciPy's linear_sum_assignment (rectangular shortest augmenting path, Crouse) of a rows x cols
// cost matrix: the assigned (row, column) pairs, rows ascending. Ties resolve as in SciPy.
std::vector<std::pair<int, int>> linear_sum_assignment(
    const std::vector<double>& cost, int rows, int cols);

// ---- alignment ----

struct AlignmentConfig {
    float speaker_activity_threshold = 0.5f;
    float speaker_logprob_weight = 1.0f;
    double maximum_token_len = 1.0;  // seconds
    // Refinement cap in CTC frames: 12 encoder frames times the head's upsampling factor.
    int max_word_frames = 12;
    float epsilon = 1.0e-6f;
};

// NeMo's _one_pass_speaker_mapping: greedy token onsets counted per speaker column while it is
// active, matched to each stream's target tokens (IDF-weighted cosine of square-root counts minus a
// penalty on the log ratio of the totals) by linear assignment. `greedy` is the per-frame argmax
// column, `speaker` frames x speakers on the CTC grid. Column per stream, -1 = none.
std::vector<int> speaker_mapping(
    const std::vector<int>& greedy, int columns, const std::vector<SpeakerStream>& streams,
    const float* speaker, int frames, int speakers, float threshold);

struct BandedPath {
    std::vector<int> path;  // target state per searched frame, all -1 when no path exists
    float score = 0.0f;     // -inf when no path exists
};

// NeMo's _banded_ctc_search: max-sum CTC over `positions` (indices of the searched frames of
// `log_probs`, frames x columns, blank column 0) in a band of states (every state under 1000
// searched frames, else a half-width of ceil(0.1 * states + 100)) centered through unique trigram
// anchors of the greedy onsets and the target. `prior` (per frame, nullable) is added to non-blank
// states; ties prefer stay, step, skip; float32 throughout.
BandedPath banded_ctc_search(
    const float* log_probs, int columns, const std::vector<int>& labels,
    const std::vector<int>& positions, const float* prior, const std::vector<int>& greedy);

struct WordRow {
    int word = 0;  // unit index
    int tag = -1;
    double start = 0.0, end = 0.0;  // seconds
    int start_frame = 0, end_frame = 0;
    int column = -1;                   // Sortformer column of the tag, -1 = none
    float speaker_confidence = -1.0f;  // mean column activity over the word's frames, < 0 = none
};

struct StreamFailure {
    int tag = -1;
    std::string reason;  // "no_legal_path_in_anchor_band"
    std::vector<int> words;
};

// NeMo's diarization_segments; speaker = label row (Sortformer column).
struct Segment {
    int speaker = 0;
    double start = 0.0, end = 0.0;
};

struct RefinementStats {
    double max_seconds = 0.0;
    int words = 0, onsets_moved = 0, unmatched_words = 0;
};

struct AlignmentResult {
    double ctc_step = 0.0;  // seconds per CTC frame: duration / frames
    // Tag -> Sortformer column (-1 = none), one per stream in stream order.
    std::vector<std::pair<int, int>> mapping;
    std::vector<float> scores;  // path score per stream; -inf for a failed stream
    std::vector<StreamFailure> failed_streams;
    // speaker_word_timestamps after refinement: the aligned streams in stream order, each stream's
    // words by start time.
    std::vector<WordRow> words;
    std::vector<Segment> diarization;
    RefinementStats refinement;
};

struct AlignmentInputs {
    const float* log_probs = nullptr;  // frames x targets.columns.size(), frame-major
    int frames = 0;                    // CTC frames
    // Sortformer probabilities (speaker_frames x speakers), nullptr = align without speakers.
    const float* speaker_probs = nullptr;
    int speaker_frames = 0, speakers = 0;
    // 10 ms speaker activity (label_frames x label_speakers), labelled active above 0.5;
    // nullptr = no diarization segments.
    const float* activity10 = nullptr;
    int label_frames = 0, label_speakers = 0;
    double label_seconds = 0.01;
    double duration = 0.0;  // audio seconds
    double time_offset = 0.0;
};

// NeMo's _align_banded_ctc for one record: speaker mapping, one banded search per stream over the
// frames where its column's activity is at least 0.01 with the log activity as the prior, word
// rows, the diarization segments and _refine_word_timestamps. A stream without a legal path is
// reported in failed_streams and its words are left out. Throws std::invalid_argument on
// inconsistent inputs.
AlignmentResult align_record(
    const AlignmentTargets& targets, const AlignmentInputs& inputs,
    const AlignmentConfig& config = {});

// NeMo's diarization_labels (10 ms activity > 0.5) as diarization_segments: per row the
// contiguous active runs, ends clipped to the duration, empty ones dropped, sorted by
// (start, end, speaker).
std::vector<Segment> diarization_segments(
    const float* activity10, int frames, int speakers, double frame_seconds, double duration,
    double time_offset = 0.0);

// NeMo's _refine_word_timestamps over `words` (stream groups as in AlignmentResult::words).
RefinementStats refine_word_timestamps(
    std::vector<WordRow>& words, const std::vector<std::pair<int, int>>& mapping,
    const std::vector<Segment>& diarization, double ctc_step, int max_frames);

// The plugin's _public_result order: every word with its times rounded to milliseconds, sorted by
// (start, end) (stable over the stream groups).
std::vector<WordRow> public_words(const std::vector<WordRow>& words);

// Python round(x, decimals): the decimal value nearest the exact binary value, ties to even.
double round_decimals(double x, int decimals);

}  // namespace engine::models::nemotron_transcribe::timestamps
