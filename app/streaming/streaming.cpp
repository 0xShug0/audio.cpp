#include "streaming.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace minitts::app {
namespace {

void emit_if_nonempty(const engine::runtime::StreamEvent & event, const StreamEventSink & sink) {
    if (event.voice_activity.empty() &&
        !event.partial_text.has_value() &&
        !event.audio_output.has_value() &&
        event.named_audio_outputs.empty() &&
        event.speaker_turns.empty() &&
        event.word_timestamps.empty() &&
        event.output_artifacts.empty() &&
        !event.is_final) {
        return;
    }
    if (sink) {
        sink(event);
    }
}

int64_t resolve_chunk_samples(
    const engine::runtime::StreamingPolicy & policy,
    const AudioStreamFormat & format) {
    if (policy.preferred_audio_chunk_seconds > 0.0) {
        return static_cast<int64_t>(std::llround(
            policy.preferred_audio_chunk_seconds * static_cast<double>(format.sample_rate))) *
            static_cast<int64_t>(format.channels);
    }
    return policy.preferred_audio_chunk_samples;
}

void feed_audio_stream(
    engine::runtime::IStreamingVoiceTaskSession & session,
    const AudioChunkStream & stream,
    int64_t chunk_samples,
    const StreamEventSink & sink) {
    if (chunk_samples <= 0) {
        throw std::runtime_error("streaming audio chunk size must be positive");
    }
    if (stream.format.sample_rate <= 0 || stream.format.channels <= 0) {
        throw std::runtime_error("streaming audio input requires positive sample rate and channels");
    }
    if (!stream.read) {
        throw std::runtime_error("streaming audio input requires a chunk reader");
    }

    int64_t start_sample = 0;
    std::vector<float> samples;
    while (true) {
        samples.clear();
        const bool has_more = stream.read(chunk_samples, samples);
        if (samples.size() % static_cast<size_t>(stream.format.channels) != 0) {
            throw std::runtime_error("streaming audio input sample count must be divisible by channels");
        }
        if (!samples.empty()) {
            const auto consumed = static_cast<int64_t>(samples.size());
            emit_if_nonempty(
                session.process_audio_chunk({
                    stream.format.sample_rate,
                    stream.format.channels,
                    start_sample,
                    std::move(samples),
                }),
                sink);
            start_sample += consumed;
        }
        if (!has_more) {
            break;
        }
    }
}

// Replays an already materialized buffer through the same path a live source takes, so the two
// input routes cannot drift apart. `audio` must outlive the returned stream.
AudioChunkStream buffer_audio_stream(const engine::runtime::AudioBuffer & audio) {
    // The whole length is known here, unlike a live source, so reject a malformed buffer before
    // the session is touched rather than partway through it on the final short chunk.
    if (audio.channels <= 0 || audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("streaming audio input sample count must be divisible by channels");
    }
    AudioChunkStream stream;
    stream.format = AudioStreamFormat{audio.sample_rate, audio.channels};
    stream.read = [source = &audio, offset = size_t{0}](
        int64_t max_samples, std::vector<float> & samples) mutable {
        const size_t step = static_cast<size_t>(max_samples);
        const size_t available = std::min(step, source->samples.size() - offset);
        samples.assign(
            source->samples.begin() + static_cast<std::ptrdiff_t>(offset),
            source->samples.begin() + static_cast<std::ptrdiff_t>(offset + available));
        offset += available;
        return offset < source->samples.size();
    };
    return stream;
}

void pull_stream_events(
    engine::runtime::IStreamingVoiceTaskSession & session,
    const StreamEventSink & sink) {
    while (true) {
        auto event = session.next_stream_event();
        if (!event.has_value()) {
            break;
        }
        emit_if_nonempty(*event, sink);
    }
}

engine::runtime::TaskResult run_stream(
    engine::runtime::IStreamingVoiceTaskSession & session,
    const engine::runtime::TaskRequest & request,
    const StreamEventSink & sink,
    const AudioChunkStream * stream) {
    const auto policy = session.streaming_policy();
    session.set_stream_event_sink(sink);
    try {
        session.start_stream(request);
        if (policy.input == engine::runtime::StreamingInputKind::AudioChunks) {
            if (stream == nullptr) {
                throw std::runtime_error(
                    "streaming audio input mode requires samples in audio_input, or an "
                    "AudioChunkStream for a live source");
            }
            feed_audio_stream(session, *stream, resolve_chunk_samples(policy, stream->format), sink);
        }
        if (policy.output == engine::runtime::StreamingOutputKind::PullEvents) {
            pull_stream_events(session, sink);
        }
        auto result = session.finish_stream();
        session.set_stream_event_sink(nullptr);
        return result;
    } catch (...) {
        session.set_stream_event_sink(nullptr);
        throw;
    }
}

}  // namespace

engine::runtime::TaskResult run_streaming_task(
    engine::runtime::IStreamingVoiceTaskSession & session,
    const engine::runtime::TaskRequest & request,
    const StreamEventSink & sink) {
    // A live source carries its format in audio_input but no samples, and must be supplied as an
    // AudioChunkStream instead. Treating that as an empty buffer here would feed the session
    // nothing and fail much later, so leave it to the null-stream check in run_stream.
    if (!request.audio_input.has_value() || request.audio_input->samples.empty()) {
        return run_stream(session, request, sink, nullptr);
    }
    const auto stream = buffer_audio_stream(*request.audio_input);
    return run_stream(session, request, sink, &stream);
}

engine::runtime::TaskResult run_streaming_task(
    engine::runtime::IStreamingVoiceTaskSession & session,
    const engine::runtime::TaskRequest & request,
    const StreamEventSink & sink,
    const AudioChunkStream & stream) {
    return run_stream(session, request, sink, &stream);
}

}  // namespace minitts::app
