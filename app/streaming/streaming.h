#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace minitts::app {

using StreamEventSink = std::function<void(const engine::runtime::StreamEvent &)>;

// Format of an incrementally supplied audio stream. A live source has to declare it up front
// because the samples themselves only arrive later.
struct AudioStreamFormat {
    int sample_rate = 0;
    int channels = 1;
};

// Pulls the next block of interleaved samples into `samples`, at most `max_samples` values
// (`channels * frames`). Returns false once the stream has ended and no further audio will
// arrive; the final block may be short and can be returned together with either result.
// Implementations are expected to block while waiting for a live source.
using AudioChunkReader = std::function<bool(int64_t max_samples, std::vector<float> & samples)>;

struct AudioChunkStream {
    AudioStreamFormat format;
    AudioChunkReader read;
};

engine::runtime::TaskResult run_streaming_task(
    engine::runtime::IStreamingVoiceTaskSession & session,
    const engine::runtime::TaskRequest & request,
    const StreamEventSink & sink);

// Streams audio from `stream` rather than from `request.audio_input`, so the audio never has to
// be fully materialized. `request` still supplies text, options and the audio format contract
// used to prepare the session.
engine::runtime::TaskResult run_streaming_task(
    engine::runtime::IStreamingVoiceTaskSession & session,
    const engine::runtime::TaskRequest & request,
    const StreamEventSink & sink,
    const AudioChunkStream & stream);

}  // namespace minitts::app
