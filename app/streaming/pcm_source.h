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
int pcm_sample_format_bytes(PcmSampleFormat format);

// Streams raw interleaved PCM from `input` without buffering the whole stream. Blocks until a
// full block is available, a short block can be completed at end of input, or the stream ends.
// `input` must outlive the returned stream.
AudioChunkStream make_pcm_chunk_stream(
    std::istream & input,
    AudioStreamFormat format,
    PcmSampleFormat sample_format);

// Puts the process stdin handle into binary mode. Required on Windows, where the default text
// mode mangles PCM bytes; a no-op elsewhere.
void set_stdin_binary_mode();

}  // namespace minitts::app
