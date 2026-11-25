#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// Text and audio preparation around the Reverb generator, ported from reverb_tts/text.py and
// reverb_tts/audio.py.
namespace engine::community_models::reverb {

inline constexpr int64_t kFrameRate = 25;          // codec frames per second
inline constexpr int64_t kLongestFrames = 30 * 25;  // a text estimated longer than this is spoken in pieces
inline constexpr int64_t kPieceFrames = 15 * 25;    // of about this length
inline constexpr double kEndCutSeconds = 0.2;      // cut from every take's end (codec noise burst)

// The seven languages Reverb speaks; anything else is rejected.
bool is_supported_language(std::string_view language);
const std::vector<std::string> & supported_languages();

// Upstream verbalizes text with Verbalis (numbers, dates, symbols read out as words) before this
// step; audio.cpp does not, so callers should pass spoken-form text. This keeps the rest of
// prepare_text: brackets dropped and whitespace collapsed to single spaces.
std::string prepare_text(std::string_view text);

// The target length in codec frames: the text's spoken weight at the rate the reference speaks its
// transcript, with short texts given relatively more time (text.estimate_frames).
int64_t estimate_frames(std::string_view text, std::string_view reference_text, int64_t reference_frames);

// The text whole when estimated at most `longest` frames, else cut at sentence ends into pieces of
// about `piece` frames (text.split_text).
std::vector<std::string> split_text(
    std::string_view text,
    std::string_view reference_text,
    int64_t reference_frames,
    int64_t longest = kLongestFrames,
    int64_t piece = kPieceFrames);

// Python's round(): halves go to the even neighbour.
int64_t round_half_even(double value);

// audio.raise_quiet: a recording quieter than -40 dBFS RMS raised to -20 dBFS, its peak kept <= 0.99.
std::vector<float> raise_quiet(std::vector<float> audio);

// audio.speech_length: seconds of a reference that set the speaker's pace.
double speech_length(const std::vector<float> & audio, int sample_rate);

// audio.trim_silence: cuts leading and trailing silence, keeping `keep` seconds of it.
std::vector<float> trim_silence(const std::vector<float> & audio, int sample_rate, double keep = 0.1);

// torchaudio.functional.resample with upstream's librosa kaiser_best settings (sinc_interp_kaiser,
// lowpass_filter_width 64, rolloff 0.9475937, beta 14.7696565), in float32.
std::vector<float> resample_kaiser_best(const std::vector<float> & audio, int source_rate, int target_rate);

}  // namespace engine::community_models::reverb
