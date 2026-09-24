#include "engine/models/nemotron_asr/speaker_tagging.h"

#include "engine/framework/io/json.h"
#include "engine/framework/io/safetensors.h"
#include "engine/framework/io/text.h"
#include "engine/framework/runtime/options.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace engine::models::nemotron_asr {
namespace {

constexpr int64_t kMaxSpeakers = 8;
constexpr float kMinSpeakerProbability = 0.01F;  // NeMo get_simulated_softmax min_sigmoid_val

bool is_space(char c) {
    return std::isspace(static_cast<unsigned char>(c)) != 0;
}

std::string strip(std::string text) {
    return engine::io::trim_ascii_whitespace(std::move(text));
}

std::string lstrip(const std::string & text) {
    size_t begin = 0;
    while (begin < text.size() && is_space(text[begin])) ++begin;
    return text.substr(begin);
}

std::string strip_language_tags(const std::string & text) {
    static const std::regex pattern(R"(\s*<[a-z]{2}-[A-Z]{2}>)");
    return std::regex_replace(text, pattern, "");
}

// Python: words = text.split(); words[0] = words[0].capitalize(); " ".join(words)
// ponytail: ASCII case mapping only; str.capitalize also lowercases non-ASCII letters.
std::string capitalize_first_word(const std::string & text) {
    std::istringstream stream(text);
    std::string word;
    std::string out;
    bool first = true;
    while (stream >> word) {
        if (first) {
            for (size_t i = 0; i < word.size(); ++i) {
                const auto c = static_cast<unsigned char>(word[i]);
                if (c < 0x80) word[i] = static_cast<char>(i == 0 ? std::toupper(c) : std::tolower(c));
            }
            first = false;
        } else {
            out += ' ';
        }
        out += word;
    }
    return out;
}

// Python round(x, 3) for the SegLST times.
double round3(double value) {
    return std::nearbyint(value * 1000.0) / 1000.0;
}

int64_t metadata_int(const std::unordered_map<std::string, std::string> & metadata, const char * key, int64_t fallback) {
    return runtime::parse_i64_option(metadata, {key}).value_or(fallback);
}

}  // namespace

SpeakerProbabilities make_speaker_probabilities(
    const std::vector<float> & native, int64_t rows, int64_t speakers,
    std::unordered_map<std::string, std::string> metadata) {
    if (rows <= 0 || speakers < 1 || speakers > kMaxSpeakers ||
        native.size() != static_cast<size_t>(rows * speakers)) {
        throw std::runtime_error("speaker_probabilities must be [frames, speakers] with 1 to 8 speakers");
    }
    for (const float value : native) {
        if (!std::isfinite(value) || value < 0.0F || value > 1.0F) {
            throw std::runtime_error("speaker_probabilities values must be finite and in [0, 1]");
        }
    }
    SpeakerProbabilities out;
    out.rows = rows;
    out.speakers = speakers;
    out.metadata = std::move(metadata);
    out.frames = (rows + kSpeakerRowsPerFrame - 1) / kSpeakerRowsPerFrame;
    out.frames80.assign(static_cast<size_t>(out.frames * speakers), 0.0F);
    for (int64_t frame = 0; frame < out.frames; ++frame) {
        const int64_t begin = frame * kSpeakerRowsPerFrame;
        const int64_t end = std::min(rows, begin + kSpeakerRowsPerFrame);
        for (int64_t speaker = 0; speaker < speakers; ++speaker) {
            float sum = 0.0F;
            for (int64_t row = begin; row < end; ++row) sum += native[static_cast<size_t>(row * speakers + speaker)];
            out.frames80[static_cast<size_t>(frame * speakers + speaker)] = sum / static_cast<float>(end - begin);
        }
    }
    return out;
}

SpeakerProbabilities load_speaker_probabilities(const std::filesystem::path & path) {
    const auto index = engine::io::load_safetensors_index(path);
    const auto tensor = index.tensors.find("speaker_probabilities");
    if (index.tensors.size() != 1 || tensor == index.tensors.end()) {
        throw std::runtime_error(
            "speaker_probabilities file must hold exactly one tensor named speaker_probabilities: " + path.string());
    }
    const auto & info = tensor->second;
    if (info.dtype != "F32" || info.shape.size() != 2) {
        throw std::runtime_error("speaker_probabilities must be an F32 [frames, speakers] tensor: " + path.string());
    }
    if (metadata_int(index.metadata, "frame_hop_samples", kSpeakerNativeHopSamples) != kSpeakerNativeHopSamples ||
        metadata_int(index.metadata, "sample_rate", 16000) != 16000) {
        throw std::runtime_error("speaker_probabilities must be 10 ms frames at 16 kHz: " + path.string());
    }
    const int64_t rows = info.shape[0];
    const int64_t speakers = info.shape[1];
    if (metadata_int(index.metadata, "frames", rows) != rows) {
        throw std::runtime_error("speaker_probabilities metadata frames does not match the tensor: " + path.string());
    }
    std::vector<float> native(static_cast<size_t>(std::max<int64_t>(rows, 0) * std::max<int64_t>(speakers, 0)));
    if (info.data_end - info.data_begin != native.size() * sizeof(float)) {
        throw std::runtime_error("speaker_probabilities data size does not match its shape: " + path.string());
    }
    std::ifstream input(path, std::ios::binary);
    input.seekg(static_cast<std::streamoff>(index.header_bytes + info.data_begin));
    input.read(reinterpret_cast<char *>(native.data()), static_cast<std::streamsize>(native.size() * sizeof(float)));
    if (!input) throw std::runtime_error("failed to read speaker_probabilities data: " + path.string());
    return make_speaker_probabilities(native, rows, speakers, index.metadata);
}

void require_speaker_probability_rows(const SpeakerProbabilities & probabilities, int64_t samples) {
    const int64_t expected = samples / kSpeakerNativeHopSamples;
    if (probabilities.rows != expected) {
        throw std::runtime_error(
            "speaker_probabilities has " + std::to_string(probabilities.rows) + " frames but the audio needs " +
            std::to_string(expected) + " (floor(samples / 160)); diarize the same audio file");
    }
}

LookaheadChoice resolve_masked_lookahead(
    const std::unordered_map<std::string, std::string> & metadata,
    std::optional<int64_t> requested,
    const std::vector<int64_t> & supported) {
    std::optional<int64_t> matched;
    if (metadata_int(metadata, "chunk_right_context", -1) == 0) {
        const int64_t candidate = metadata_int(metadata, "chunk_len", 0) - 1;
        if (std::find(supported.begin(), supported.end(), candidate) != supported.end()) matched = candidate;
    }
    const auto profile_it = metadata.find("latency_profile");
    const std::string profile = profile_it == metadata.end() ? "unknown" : profile_it->second;
    LookaheadChoice out;
    if (requested.has_value()) {
        out.lookahead = *requested;
        if (matched.has_value() && *matched != *requested) {
            out.warnings.push_back(
                "lookahead_tokens=" + std::to_string(*requested) + " overrides the diarizer file's lookahead " +
                std::to_string(*matched) + " (latency_profile " + profile + "); speaker masks are not NeMo-exact");
        } else if (!matched.has_value()) {
            out.warnings.push_back(
                "the diarizer file (latency_profile " + profile + ") does not match an ASR lookahead; speaker masks are "
                "not NeMo-exact. Diarize with latency_profile=asr_la" + std::to_string(*requested) + " for exact masks");
        }
        return out;
    }
    if (matched.has_value()) {
        out.lookahead = *matched;
        return out;
    }
    out.warnings.push_back(
        "the diarizer file (latency_profile " + profile + ") does not match an ASR lookahead; using lookahead_tokens=13 "
        "and speaker masks are not NeMo-exact. Diarize with latency_profile=asr_la13 for exact masks");
    return out;
}

std::string nemo_hypothesis_text(const std::string & decoded, bool strip_tags) {
    // RNNTDecoding.space_before_punct_pattern for this model's supported_punctuation
    // (the single-character punctuation tokens of its vocabulary):
    // ？ , । ! ؟ 、 ' " ⁇ ， ? 〜 - · . ، ¿ 。 as UTF-8 bytes, since the model
    // libraries do not compile with /utf-8.
    static const std::regex space_before_punct(
        "(\\s)(" "\xEF\xBC\x9F" "|,|" "\xE0\xA5\xA4" "|!|" "\xD8\x9F" "|" "\xE3\x80\x81" "|'|\"|"
        "\xE2\x81\x87" "|" "\xEF\xBC\x8C" "|\\?|" "\xE3\x80\x9C" "|-|" "\xC2\xB7" "|\\.|" "\xD8\x8C"
        "|" "\xC2\xBF" "|" "\xE3\x80\x82" ")");
    std::string text = std::regex_replace(decoded, space_before_punct, "$2");
    if (strip_tags) text = strip(strip_language_tags(text));
    return text;
}

std::vector<TaggedWord> group_words(const std::vector<runtime::WordTimestamp> & tokens) {
    std::vector<TaggedWord> words;
    for (const auto & token : tokens) {
        const int64_t frame = token.span.start_sample / kSpeakerFrameSamples;
        if (words.empty() || (!token.word.empty() && is_space(token.word.front()))) {
            if (!words.empty() && strip(words.back().text).empty()) words.pop_back();
            words.push_back({token.word, frame, frame});
        } else {
            words.back().text += token.word;
            words.back().last_frame = frame;
        }
    }
    if (!words.empty() && strip(words.back().text).empty()) words.pop_back();
    return words;
}

int attribute_speaker(const SpeakerProbabilities & probabilities, const TaggedWord & word) {
    const int64_t speakers = probabilities.speakers;
    const int64_t frames = probabilities.frames;
    std::vector<double> mean(static_cast<size_t>(speakers), 0.0);
    int64_t begin = std::max<int64_t>(word.first_frame - 1, 0);
    int64_t end = std::min<int64_t>(word.last_frame + 1, frames);
    if (begin >= end) {  // past the diarized frontier: use the last frame
        begin = frames - 1;
        end = frames;
    }
    for (int64_t frame = begin; frame < end; ++frame) {
        for (int64_t speaker = 0; speaker < speakers; ++speaker) {
            mean[static_cast<size_t>(speaker)] += probabilities.frames80[static_cast<size_t>(frame * speakers + speaker)];
        }
    }
    for (auto & value : mean) {
        value = std::clamp(value / static_cast<double>(end - begin), static_cast<double>(kMinSpeakerProbability), 1.0);
    }
    int best = 0;
    for (int64_t speaker = 1; speaker < speakers; ++speaker) {
        if (mean[static_cast<size_t>(speaker)] > mean[static_cast<size_t>(best)]) best = static_cast<int>(speaker);
    }
    return best;
}

SpeakerSegmentBuilder::SpeakerSegmentBuilder(double gap_sec, double max_event_sec, bool capitalize)
    : gap_sec_(gap_sec), max_event_sec_(max_event_sec), capitalize_(capitalize) {}

SpeakerSegmentBuilder::Speaker & SpeakerSegmentBuilder::speaker(int index) {
    if (index < 0 || index >= kMaxSpeakers) throw std::runtime_error("speaker index out of range");
    if (static_cast<size_t>(index) >= speakers_.size()) speakers_.resize(static_cast<size_t>(index) + 1);
    return speakers_[static_cast<size_t>(index)];
}

void SpeakerSegmentBuilder::apply(
    Speaker & state, int index, std::string diff, double start, double end, bool no_timing, bool continues_word) {
    auto & sentences = state.sentences;
    auto update_last = [&](std::optional<double> new_end, const std::string & text) {
        if (sentences.empty()) return;  // NeMo would raise here; nothing to attach to
        auto & last = sentences.back();
        if (new_end.has_value()) last.end = std::max(*new_end, last.start + kSpeakerFrameSeconds);
        last.words += text;  // already left-stripped: only the tail can need trimming
        while (!last.words.empty() && is_space(last.words.back())) last.words.pop_back();
    };
    const double last_end = sentences.empty() ? 0.0 : sentences.back().end;
    const bool has_gap = last_end != 0.0 && start > last_end + gap_sec_;
    if (continues_word && has_gap) {
        // A word continued after a long gap keeps its suffix on the previous sentence.
        const size_t space = diff.find(' ');
        update_last(std::nullopt, diff.substr(0, space));
        diff = space == std::string::npos ? std::string() : diff.substr(space);
        continues_word = false;
    }
    if (!continues_word && (no_timing || last_end == 0.0 || has_gap)) {
        const std::string stripped = strip(diff);
        if (!stripped.empty() && std::strchr(".,?!", stripped.front()) != nullptr) {
            update_last(std::nullopt, stripped.substr(0, 1));
            diff = stripped.substr(1);
        }
        std::string words = lstrip(diff);
        if (!strip(words).empty()) {
            if (capitalize_) words = capitalize_first_word(words);
            sentences.push_back({index, round3(start), round3(end), std::move(words)});
        }
    } else {
        update_last(end, diff);
    }
}

void SpeakerSegmentBuilder::update_hypothesis(
    int index,
    const std::string & text,
    const std::vector<int64_t> & token_frames,
    int64_t decoded_frames,
    double offset_sec) {
    auto & state = speaker(index);
    const std::string & previous = state.previous_text;
    if (text != previous) {
        const bool prefix_extension = text.rfind(previous, 0) == 0;
        const std::string diff = prefix_extension ? text.substr(previous.size()) : strip(text);
        double start = offset_sec;
        double end = offset_sec;
        const bool no_timing = token_frames.size() <= state.previous_tokens;
        if (!no_timing) {
            const int64_t start_local = token_frames[state.previous_tokens] - state.previous_decoded;
            const int64_t end_local = token_frames.back() - state.previous_decoded;
            start = offset_sec + static_cast<double>(start_local) * kSpeakerFrameSeconds;
            end = offset_sec + static_cast<double>(end_local + 1) * kSpeakerFrameSeconds;
        }
        const bool continues = !previous.empty() && prefix_extension && !diff.empty() &&
            !is_space(previous.back()) && !is_space(diff.front()) && !state.sentences.empty();
        apply(state, index, diff, start, end, no_timing, continues);
        state.previous_text = text;
    }
    state.previous_decoded = decoded_frames;
    state.previous_tokens = token_frames.size();
}

void SpeakerSegmentBuilder::append_word(int index, const std::string & word, double start, double end) {
    auto & state = speaker(index);
    const bool continues = !state.previous_text.empty() && !word.empty() &&
        !is_space(state.previous_text.back()) && !is_space(word.front()) && !state.sentences.empty();
    apply(state, index, word, start, end, false, continues);
    state.previous_text += word;
}

std::vector<SpeakerSegment> SpeakerSegmentBuilder::segments() const {
    std::vector<SpeakerSegment> out;
    for (const auto & state : speakers_) {
        for (const auto & sentence : state.sentences) {
            if (!strip(sentence.words).empty()) out.push_back(sentence);
        }
    }
    std::stable_sort(out.begin(), out.end(), [](const auto & a, const auto & b) { return a.start < b.start; });
    return out;
}

std::vector<SpeakerSegment> SpeakerSegmentBuilder::take_events(double stream_time, bool final) {
    std::vector<SpeakerSegment> out;
    auto emit = [&](Speaker & state, const SpeakerSegment & sentence, size_t until) {
        const size_t from = std::min(state.emitted_chars, until);
        const std::string text = strip(sentence.words.substr(from, until - from));
        if (!text.empty()) out.push_back({sentence.speaker, state.piece_start, sentence.end, text});
        state.emitted_chars = until;
        state.piece_start = sentence.end;
    };
    for (auto & state : speakers_) {
        const size_t count = state.sentences.size();
        if (count == 0) continue;
        const bool paused = stream_time - state.sentences.back().end > gap_sec_;
        const size_t closed = final || paused ? count : count - 1;
        while (state.emitted_sentences < count) {
            const auto & sentence = state.sentences[state.emitted_sentences];
            if (state.emitted_chars == 0) state.piece_start = sentence.start;
            if (state.emitted_sentences < closed) {
                emit(state, sentence, sentence.words.size());
                ++state.emitted_sentences;
                state.emitted_chars = 0;
                continue;
            }
            // Open sentence: force-close a piece after max_event_sec at a word boundary.
            // ponytail: the piece end is the sentence end so far, not the cut word's end.
            if (sentence.end - state.piece_start >= max_event_sec_) {
                const size_t cut = sentence.words.rfind(' ');
                if (cut != std::string::npos && cut > state.emitted_chars) emit(state, sentence, cut);
            }
            break;
        }
    }
    std::stable_sort(out.begin(), out.end(), [](const auto & a, const auto & b) { return a.start < b.start; });
    return out;
}

std::string speaker_label(int speaker) {
    return "speaker_" + std::to_string(speaker);
}

std::string seglst_json(const std::vector<SpeakerSegment> & segments, const std::string & session_id) {
    std::string out = "[";
    char times[64];
    for (size_t i = 0; i < segments.size(); ++i) {
        const auto & segment = segments[i];
        std::snprintf(times, sizeof(times), "\"start_time\": %.3f, \"end_time\": %.3f", segment.start, segment.end);
        out += i == 0 ? "\n  {" : ",\n  {";
        out += "\"session_id\": " + engine::io::json::stringify_string(session_id) +
            ", \"speaker\": " + engine::io::json::stringify_string(speaker_label(segment.speaker)) + ", " + times +
            ", \"words\": " + engine::io::json::stringify_string(segment.words) + "}";
    }
    out += segments.empty() ? "]\n" : "\n]\n";
    return out;
}

std::vector<runtime::SpeakerTurn> segments_to_turns(const std::vector<SpeakerSegment> & segments) {
    std::vector<runtime::SpeakerTurn> turns;
    turns.reserve(segments.size());
    for (const auto & segment : segments) {
        runtime::SpeakerTurn turn;
        turn.span.start_sample = std::llround(segment.start * 16000.0);
        turn.span.end_sample = std::llround(segment.end * 16000.0);
        turn.speaker_id = speaker_label(segment.speaker);
        turn.text = segment.words;
        turns.push_back(std::move(turn));
    }
    return turns;
}

std::string segments_to_lines(const std::vector<SpeakerSegment> & segments) {
    std::string out;
    for (const auto & segment : segments) {
        out += speaker_label(segment.speaker) + ": " + segment.words + "\n";
    }
    return out;
}

}  // namespace engine::models::nemotron_asr
