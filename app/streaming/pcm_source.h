#pragma once

#include "streaming.h"

#include <istream>
#include <string>
#include <string_view>

namespace minitts::app {

// Sample encodings accepted for raw (headerless) PCM input.
enum class PcmSampleFormat {
    S16LE,
    F32LE,
};

PcmSampleFormat parse_pcm_sample_format(std::string_view name);
std::string to_string(PcmSampleFormat format);

// Streams raw interleaved PCM from `input` without buffering the whole stream. Blocks until a
// full block is available, a short block can be completed at end of input, or the stream ends.
// `input` must outlive the returned stream.
AudioChunkStream make_pcm_chunk_stream(
    std::istream & input,
    AudioStreamFormat format,
    PcmSampleFormat sample_format);

// Streams raw interleaved PCM from process stdin, switching the handle to binary mode first
// (required on Windows, where the default text mode mangles PCM bytes).
AudioChunkStream make_stdin_pcm_stream(AudioStreamFormat format, PcmSampleFormat sample_format);

}  // namespace minitts::app
