#include "engine/community_models/parakeet_tdt/phonon_streaming.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::community_models::parakeet_tdt {
namespace {
constexpr int64_t block_samples = 800, first_partial = 5600, partial_every = 8000;
constexpr int64_t silence_close = 11200, segment_cap = 480000;
}

void PhononStreamingSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (!request.audio || request.audio->sample_rate != 16000 || request.audio->channels != 1)
        throw std::runtime_error("Phonon streaming requires mono 16 kHz audio");
    encoder_->prepare_capacity(first_partial / assets_->config.frontend.hop_length + 1,
                               assets_->config.frontend.feature_size);
    decoder_->prepare();
    mark_prepared();
    reset();
}

runtime::StreamingPolicy PhononStreamingSession::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::AudioChunks;
    policy.output = runtime::StreamingOutputKind::FinalResult;
    policy.preferred_audio_chunk_samples = block_samples;
    policy.preferred_audio_chunk_seconds = 0.05;
    return policy;
}

void PhononStreamingSession::reset_segment() {
    segment_.samples.clear();
    voiced_ = false;
    last_voice_ = 0;
    peak_rms_ = 0;
    next_partial_ = first_partial;
}

void PhononStreamingSession::reset() {
    require_prepared("Phonon streaming reset()");
    reset_segment();
    pending_.clear();
    pending_.reserve(block_samples);
    received_ = segment_start_ = committed_tokens_ = 0;
    text_.clear(); words_.clear(); options_ = {};
    decoder_->reset_state();
    active_ = true;
    wall_start_ = std::chrono::steady_clock::now();
}

void PhononStreamingSession::abort() {
    reset();
    active_ = false;
}

void PhononStreamingSession::start_stream(const runtime::TaskRequest & request) {
    reset();
    try {
        runtime::validate_spec_backed_request_options(request.options, *contract_, "Phonon");
        options_ = decode_options_for_request(request);
    } catch (...) { abort(); throw; }
}

void PhononStreamingSession::publish(const runtime::StreamEvent & event) {
    if (sink_ && (event.partial_text || event.partial_text_snapshot)) sink_(event);
}

ParakeetDecodedText PhononStreamingSession::decode_segment() {
    auto options = options_;
    if (options.max_tokens > 0) {
        options.max_tokens -= committed_tokens_;
        if (options.max_tokens <= 0) return {};
    }
    options.audio_end_sample = static_cast<int64_t>(segment_.samples.size());
    return decoder_->decode(encoder_->encode(frontend_.extract(segment_, true)), options);
}

runtime::StreamEvent PhononStreamingSession::finish_segment() {
    auto decoded = decode_segment();
    const auto count = static_cast<int64_t>(segment_.samples.size());
    runtime::StreamEvent event;
    if (!decoded.text.empty()) {
        const std::string delta = (text_.empty() ? "" : " ") + decoded.text;
        text_ += delta;
        event.partial_text = runtime::Transcript{delta, "en"};
        event.partial_text_snapshot = runtime::Transcript{text_, "en"};
        for (auto word : decoded.word_timestamps) {
            word.span.start_sample += segment_start_;
            word.span.end_sample += segment_start_;
            words_.push_back(word);
            event.word_timestamps.push_back(std::move(word));
        }
    }
    committed_tokens_ += static_cast<int64_t>(decoded.token_ids.size());
    segment_start_ += count;
    reset_segment();
    publish(event);
    return event;
}

runtime::StreamEvent PhononStreamingSession::feed_block(const std::vector<float> & block) {
    if (block.empty()) return {};
    segment_.samples.insert(segment_.samples.end(), block.begin(), block.end());
    double power = 0;
    for (float sample : block) {
        if (!std::isfinite(sample)) throw std::runtime_error("Phonon streaming audio contains nonfinite samples");
        power += static_cast<double>(sample) * sample;
    }
    const float rms = static_cast<float>(std::sqrt(power / block.size()));
    peak_rms_ = std::max(peak_rms_, rms);
    if (rms > std::max(0.004f, 0.18f * peak_rms_)) {
        voiced_ = true;
        last_voice_ = static_cast<int64_t>(segment_.samples.size());
    }
    const auto count = static_cast<int64_t>(segment_.samples.size());
    if (voiced_ && count - last_voice_ >= silence_close) return finish_segment();
    if (count >= segment_cap) {
        if (voiced_) return finish_segment();
        std::vector<float> keep(segment_.samples.end() - 16000, segment_.samples.end());
        segment_start_ += count - 16000;
        reset_segment();
        segment_.samples = std::move(keep);
    } else if (voiced_ && count >= next_partial_) {
        auto decoded = decode_segment();
        runtime::StreamEvent event;
        if (!decoded.text.empty()) {
            event.partial_text_snapshot = runtime::Transcript{ text_ + (text_.empty() ? "" : " ") + decoded.text, "en" };
            publish(event);
        }
        next_partial_ = count + partial_every;
        return event;
    }
    return {};
}

runtime::StreamEvent PhononStreamingSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    require_prepared("Phonon process_audio_chunk()");
    if (!active_) throw std::runtime_error("Phonon chunk received outside an active stream");
    // Validate before mutation, so a malformed chunk can be retried.
    if (chunk.sample_rate != 16000 || chunk.channels != 1 || chunk.start_sample != received_)
        throw std::runtime_error("Phonon streaming requires contiguous mono 16 kHz chunks");
    try {
        runtime::StreamEvent last;
        for (float sample : chunk.samples) {
            pending_.push_back(sample);
            if (pending_.size() == block_samples) {
                auto event = feed_block(pending_);
                pending_.clear();
                if (event.partial_text) {
                    if (!last.partial_text) last.partial_text = runtime::Transcript{"", "en"};
                    last.partial_text->text += event.partial_text->text;
                }
                if (event.partial_text_snapshot) last.partial_text_snapshot = std::move(event.partial_text_snapshot);
                last.word_timestamps.insert(last.word_timestamps.end(), event.word_timestamps.begin(), event.word_timestamps.end());
            }
        }
        received_ += static_cast<int64_t>(chunk.samples.size());
        // A registered sink has already received every event, including when
        // one caller chunk crosses multiple phrase boundaries. Do not replay
        // the final event through the application's returned-event path.
        return sink_ ? runtime::StreamEvent{} : last;
    } catch (...) { abort(); throw; }
}

runtime::TaskResult PhononStreamingSession::finalize() {
    require_prepared("Phonon finalize()");
    if (!active_) throw std::runtime_error("Phonon finalize() requires an active stream");
    try {
        if (!pending_.empty()) { feed_block(pending_); pending_.clear(); }
        if (voiced_ && !segment_.samples.empty()) finish_segment();
        else reset_segment();
        runtime::TaskResult result;
        result.text_output = runtime::Transcript{text_, "en"};
        result.word_timestamps = words_;
        active_ = false;
        options_ = {}; // Release request-local hotwords without dropping cached graphs.
        decoder_->reset_state();
        debug::timing_log_scalar("session.wall_ms",
            debug::elapsed_ms(wall_start_, std::chrono::steady_clock::now()));
        return result;
    } catch (...) { abort(); throw; }
}
} // namespace engine::community_models::parakeet_tdt
