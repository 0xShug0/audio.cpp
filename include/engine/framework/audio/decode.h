#pragma once

#include "engine/framework/audio/wav_reader.h"

#include <string_view>

namespace engine::audio {

// Decodes an arbitrary in-memory audio upload into interleaved f32 PCM.
//
// RIFF/WAVE bytes are handed straight to read_wav_f32() so existing WAV
// callers see no behavior change. Other containers (MP3, FLAC today) are
// decoded through the vendored miniaudio backend at their native sample
// rate and channel count, matching what read_wav_f32() returns for WAV.
//
// Throws std::runtime_error with a client-actionable message when the
// upload is a container audio.cpp cannot decode yet (for example Ogg or
// WebM carrying Opus, which miniaudio does not decode without an
// additional codec backend) or when the bytes are not recognizable audio
// at all, instead of failing deep inside a codec parser.
WavData decode_audio_upload_f32(std::string_view bytes);

}  // namespace engine::audio
