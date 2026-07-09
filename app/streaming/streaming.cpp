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

void feed_audio_chunks(
    engine::runtime::IStreamingVoiceTaskSession & session,
    const engine::runtime::TaskRequest & request,
    int64_t chunk_samples,
    const StreamEventSink & sink) {
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("streaming audio input mode requires audio_input");
    }
    if (chunk_samples <= 0) {
        throw std::runtime_error("streaming audio chunk size must be positive");
    }
    const auto & audio = *request.audio_input;
    if (audio.sample_rate <= 0 || audio.channels <= 0) {
        throw std::runtime_error("streaming audio input requires positive sample rate and channels");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("streaming audio input sample count must be divisible by channels");
    }

    const size_t step = static_cast<size_t>(chunk_samples);
    for (size_t offset = 0; offset < audio.samples.size(); offset += step) {
        const size_t available = std::min(step, audio.samples.size() - offset);
        std::vector<float> samples;
        samples.reserve(available);
        samples.insert(
            samples.end(),
            audio.samples.begin() + static_cast<std::ptrdiff_t>(offset),
            audio.samples.begin() + static_cast<std::ptrdiff_t>(offset + available));
        emit_if_nonempty(
            session.process_audio_chunk({
                audio.sample_rate,
                audio.channels,
                static_cast<int64_t>(offset),
                std::move(samples),
            }),
            sink);
    }
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

}  // namespace

engine::runtime::TaskResult run_streaming_task(
    engine::runtime::IStreamingVoiceTaskSession & session,
    const engine::runtime::TaskRequest & request,
    const StreamEventSink & sink) {
    const auto policy = session.streaming_policy();
    session.set_stream_event_sink(sink);
    try {
        session.start_stream(request);
        if (policy.input == engine::runtime::StreamingInputKind::AudioChunks) {
            int64_t chunk_samples = policy.preferred_audio_chunk_samples;
            if (policy.preferred_audio_chunk_seconds > 0.0) {
                if (!request.audio_input.has_value()) {
                    throw std::runtime_error("streaming audio input mode requires audio_input");
                }
                const auto & audio = *request.audio_input;
                chunk_samples = static_cast<int64_t>(
                    std::llround(policy.preferred_audio_chunk_seconds * static_cast<double>(audio.sample_rate))) *
                    static_cast<int64_t>(audio.channels);
            }
            feed_audio_chunks(session, request, chunk_samples, sink);
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

}  // namespace minitts::app
