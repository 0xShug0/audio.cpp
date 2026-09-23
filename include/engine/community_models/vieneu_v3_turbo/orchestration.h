#pragma once

// What turns one long text into several generated chunks that sound like one
// utterance: where to cut, how long a pause belongs at each seam, and whether a
// chunk came back wrong and should be generated again.
//
// Ported from `vieneu_utils.core_utils` in the Python package (and its Rust port
// in the VieNeu desktop app), which is where these rules were measured. The two
// halves differ in how portable they are:
//
//   * pauses and the babble guard need no language knowledge — they work on the
//     waveform and on a syllable count — so they are ported here in full;
//   * the *text* chunker does: the Python one cuts at connectives and never
//     between two number words, rules that live next to the normaliser. This
//     file only cuts the phoneme string at punctuation, which survives
//     phonemisation, and packs to a budget. That covers ordinary prose; a text
//     front end (sea-g2p) would bring the full rules with it rather than have a
//     third copy of them drift here.

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::vieneu_v3_turbo {

/// The seam between two consecutive chunks.
enum class Gap {
    /// Paragraph break (a newline in the source text).
    Para,
    /// The previous chunk ended a sentence (`. ! ? …`).
    Sentence,
    /// Anything else: a comma, or a cut forced by the budget.
    Minor,
};

/// Minimum audible pause per seam, in seconds (`V3_GAP_SILENCE`). Measured on
/// the preset voices with a whole sentence in one chunk: the model itself pauses
/// ~0.30–0.45 s at commas and ~0.50–0.65 s at sentence ends, so a forced cut is
/// padded to the same rhythm. It is a MINIMUM — a longer natural tail is kept.
double gap_pause_sec(Gap gap);

/// The seam after `chunk`, from its final punctuation.
Gap classify_gap(const std::string & chunk);

struct PhonemeChunk {
    std::string phonemes;
    /// The seam before this chunk; the first chunk has none.
    Gap gap_before = Gap::Minor;
};

/// Cuts a phoneme string into chunks of at most `max_chars` (counted in
/// characters, the budget the frame ceiling is derived from): paragraphs, then
/// sentences, then minor punctuation, then whitespace. Chunks shorter than
/// `min_chars` join a neighbour, because a two-syllable chunk of its own reads
/// as a stutter.
std::vector<PhonemeChunk> split_phoneme_chunks(
    const std::string & phonemes,
    int64_t max_chars,
    int64_t min_chars);

/// `(lead, tail)` silent samples at the two ends of `samples`; an all-silent
/// buffer reports `(size, 0)`. 10 ms mean-|x| envelope, −45 dB threshold.
void edge_silence(const std::vector<float> & samples, int64_t sample_rate, int64_t & lead, int64_t & tail);

/// Zeros to insert between `previous` and `next` so that the audible pause
/// (tail silence of `previous` + zeros + lead silence of `next`) reaches
/// `pause_seconds`; 0 when the model already left enough.
int64_t pause_pad_samples(
    const std::vector<float> & previous,
    const std::vector<float> & next,
    int64_t sample_rate,
    double pause_seconds);

/// Energy bursts in `samples` — one Vietnamese syllable is one burst.
int64_t count_speech_bursts(const std::vector<float> & samples, int64_t sample_rate);

/// Why a chunk is suspected of having kept talking past its text.
struct BabbleVerdict {
    bool suspect = false;
    int64_t syllables = 0;
    int64_t bursts = 0;
    int64_t frames = 0;
};

/// Judge a decoded chunk. `cap_frames` is the ceiling it was generated under,
/// `frames` what it produced. Only short chunks are judged: longer ones have
/// merged bursts that cannot be counted reliably.
BabbleVerdict babble_suspect(
    const std::vector<float> & samples,
    int64_t sample_rate,
    const std::string & phonemes,
    int64_t cap_frames,
    int64_t frames);

/// Should a regenerated chunk replace the current one? Not-suspect beats
/// suspect; then fewer bursts; then shorter.
bool babble_prefer(const BabbleVerdict & candidate, const BabbleVerdict & current);

/// Regenerations allowed for a suspect chunk that has words.
inline constexpr int64_t kBabbleMaxRetries = 2;
/// Regenerations for a standalone cue: a lone `[cười]` runs away far more often
/// and each attempt costs at most ~13 frames.
inline constexpr int64_t kBabbleMaxRetriesCue = 5;

}  // namespace engine::models::vieneu_v3_turbo
