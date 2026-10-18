#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// Speaker-tagged transcription: diarizer probability input, word attribution and
// the NeMo SegLST segment builder. See docs/asr.md, "Speaker-tagged transcription".
namespace engine::models::nemotron_asr {

inline constexpr int64_t kSpeakerNativeHopSamples = 160;   // 10 ms diarizer rows
inline constexpr int64_t kSpeakerFrameSamples = 1280;      // 80 ms ASR encoder frames
inline constexpr int64_t kSpeakerRowsPerFrame = kSpeakerFrameSamples / kSpeakerNativeHopSamples;
inline constexpr double kSpeakerFrameSeconds = 0.08;

// nemotron_asr request options for speaker tagging.
struct SpeakerTaggingOptions {
    std::filesystem::path probabilities;
    bool masked = false;
    bool audio_mask = false;
    double gap_sec = 1.0;
    double max_event_sec = 10.0;
};

// A nemotron_3_diar speaker_probabilities.safetensors timeline.
struct SpeakerProbabilities {
    int64_t rows = 0;      // native 10 ms rows
    int64_t speakers = 0;
    int64_t frames = 0;           // 80 ms frames, ceil(rows / 8)
    std::vector<float> frames80;  // [frames, speakers]; a partial last group averages its valid rows
    std::unordered_map<std::string, std::string> metadata;
};

SpeakerProbabilities load_speaker_probabilities(const std::filesystem::path & path);
// native is [rows, speakers]; only the 80 ms means are kept.
SpeakerProbabilities make_speaker_probabilities(
    const std::vector<float> & native, int64_t rows, int64_t speakers,
    std::unordered_map<std::string, std::string> metadata = {});

// The file must describe the same audio: floor(samples / 160) rows.
void require_speaker_probability_rows(const SpeakerProbabilities & probabilities, int64_t samples);

// Masked mode picks the ASR lookahead that matches the diarizer geometry (plan D6):
// chunk_len - 1 when chunk_right_context == 0 and that value is supported. An explicit
// request wins; a missing or unmatched geometry falls back to 13. Warnings explain
// every case where the masks are not NeMo-exact.
struct LookaheadChoice {
    int64_t lookahead = 13;
    std::vector<std::string> warnings;
};
LookaheadChoice resolve_masked_lookahead(
    const std::unordered_map<std::string, std::string> & metadata,
    std::optional<int64_t> requested,
    const std::vector<int64_t> & supported);

// NeMo decode_tokens_to_str_with_strip_punctuation: drop one space before a supported punctuation
// mark, then, with strip_tags, remove language tags and surrounding whitespace.
std::string nemo_hypothesis_text(const std::string & decoded, bool strip_tags);

// Tokens grouped into words: a word starts at a token with a leading space or at
// the first token; punctuation-only tokens stay with the preceding word.
struct TaggedWord {
    std::string text;        // as decoded, including its leading space
    int64_t first_frame = 0; // 80 ms encoder frames
    int64_t last_frame = 0;
};
std::vector<TaggedWord> group_words(const std::vector<runtime::WordTimestamp> & tokens);

// NeMo get_word_dict_content_offline: the speaker with the highest mean activity over
// 80 ms frames [first - 1, last + 1), clamped to [0.01, 1]; ties go to the lowest index.
int attribute_speaker(const SpeakerProbabilities & probabilities, const TaggedWord & word);

struct SpeakerSegment {
    int speaker = 0;
    double start = 0.0;
    double end = 0.0;
    std::string words;
};

// Port of NeMo MultiTalkerInstanceManager.ASRState.update_sessionwise_seglsts_for_parallel:
// per-speaker sentences that break on a gap longer than gap_sec, with word
// continuations and leading punctuation kept on the previous sentence.
class SpeakerSegmentBuilder {
public:
    // capitalize: NeMo uppercase_first_letter (on in the multitalker pipeline).
    explicit SpeakerSegmentBuilder(double gap_sec, double max_event_sec = 10.0, bool capitalize = true);

    // One streaming step for one speaker: the cumulative hypothesis text, the
    // cumulative encoder frame of every emitted token, the speaker's decoded frame
    // count after the step, and the chunk start time.
    void update_hypothesis(
        int speaker,
        const std::string & text,
        const std::vector<int64_t> & token_frames,
        int64_t decoded_frames,
        double offset_sec);
    // Attribution mode: one word of the single transcript, already timed.
    void append_word(int speaker, const std::string & word, double start, double end);

    // Final segments, merged over speakers and sorted by start time.
    std::vector<SpeakerSegment> segments() const;

    // Streaming events: segments that can no longer change (a newer segment
    // exists, or no speech for gap_sec before stream_time), plus open segments
    // force-closed after max_event_sec. Each call returns only new pieces; with
    // final=true everything left is returned.
    std::vector<SpeakerSegment> take_events(double stream_time, bool final);

private:
    struct Speaker {
        std::vector<SpeakerSegment> sentences;
        std::string previous_text;
        size_t previous_tokens = 0;
        int64_t previous_decoded = 0;
        size_t emitted_sentences = 0;
        size_t emitted_chars = 0;
        double piece_start = 0.0;
    };
    Speaker & speaker(int index);
    void apply(Speaker & state, int index, std::string diff, double start, double end, bool no_timing,
               bool continues_word);

    double gap_sec_;
    double max_event_sec_;
    bool capitalize_;
    std::vector<Speaker> speakers_;
};

std::string seglst_json(const std::vector<SpeakerSegment> & segments, const std::string & session_id);
std::vector<runtime::SpeakerTurn> segments_to_turns(const std::vector<SpeakerSegment> & segments);
// Time-ordered "speaker_k: words" lines.
std::string segments_to_lines(const std::vector<SpeakerSegment> & segments);
std::string speaker_label(int speaker);

}  // namespace engine::models::nemotron_asr
