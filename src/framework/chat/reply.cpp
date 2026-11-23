#include "engine/framework/chat/reply.h"

#include "engine/framework/runtime/partial_text.h"
#include "engine/framework/debug/trace.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace engine::chat {

struct Reply::Impl {
    ReplySource source;
    ReplyCodec codec;
    ReplyConfig config;
    enum class State { Ready, Streaming, Complete, Failed } state = State::Ready;
    runtime::AudioBuffer audio;
    std::string pending_text;
    std::string text;
    std::vector<ReplyStep> history;
    std::optional<HistoryFormat> history_format;
    bool emit_artifact = false;
    bool record_history = false;
    bool released = false;
    size_t frames = 0;
    size_t tokens = 0;
    size_t generated_steps = 0;
    std::optional<TextSpeechSource> text_speech;
    bool source_exhausted = false;
    bool saw_eos = false;
    bool text_complete = false;
    bool text_ended = false;

    struct Batch {
        std::vector<int32_t> tokens;
        AudioCodes frames;
        bool exhausted = false;
    };

    Batch collect(bool streaming) {
        Batch batch;
        while (!streaming || batch.frames.size() < config.frames_per_event) {
            if (config.max_steps > 0 && generated_steps >= static_cast<size_t>(config.max_steps)) {
                batch.exhausted = true;
                break;
            }
            auto step = source.next();
            if (!step) {
                source_exhausted = true;
                batch.exhausted = true;
                break;
            }
            if (step->codes.empty() && std::find(config.eos_tokens.begin(), config.eos_tokens.end(), step->token) != config.eos_tokens.end()) {
                saw_eos = true;
                batch.exhausted = true;
                break;
            }
            ++generated_steps;
            if (record_history && (!source.history || !config.eos_tokens.empty())) history.push_back(*step);
            if (step->codes.empty()) {
                batch.tokens.push_back(step->token);
                ++tokens;
                // A text token must not wait for the next audio inference.
                if (streaming && config.publish_text_immediately && batch.frames.empty()) break;
            } else if (std::find(step->codes.begin(), step->codes.end(), config.end_of_audio) == step->codes.end()) {
                batch.frames.push_back(std::move(step->codes));
                ++frames;
            }
        }
        return batch;
    }

    std::string consume_text(const std::vector<int32_t> & ids) {
        pending_text += source.decode_text(ids);
        auto delta = runtime::consume_utf8_prefix(pending_text);
        text += delta;
        return delta;
    }

    bool ended() const {
        if (text_complete) return text_ended;
        return saw_eos || (source.ended ? source.ended() : source_exhausted);
    }

    void complete() const {
        if (config.require_audio && frames == 0) {
            throw std::runtime_error(ended() ? "Chat reply produced no speech" :
                "Chat reply reached max_tokens before any speech; raise max_tokens");
        }
        if (!ended()) {
            std::cerr << "[warning][chat] Reply reached max_tokens=" << config.max_steps
                      << " and is cut off; raise max_tokens for the rest\n";
        }
    }

    void generate_text() {
        source = text_speech->text_source();
        if (!source.next || !source.decode_text) throw std::invalid_argument("Chat text stage requires token inference and decoding");
        auto batch = collect(false);
        if (!batch.frames.empty()) throw std::logic_error("Chat text stage returned audio codes");
        complete();
        consume_text(batch.tokens);
        text_ended = ended();
        text_complete = true;
        if (record_history && source.history && config.eos_tokens.empty()) history = source.history();
        // End the text-stage resource lifetime before synthesis allocates its
        // workspace. Model graphs themselves remain session-owned.
        source = {};
        if (text.empty()) throw std::runtime_error("Chat speech synthesis requires nonempty reply text");
    }

    void generate_speech() {
        audio.samples = text_speech->synthesize(text);
        state = State::Complete;
    }

    runtime::TaskResult result() const {
        runtime::TaskResult result;
        result.audio_output = audio;
        result.text_output = runtime::Transcript{text, config.language};
        if (emit_artifact) result.output_artifacts.push_back(encode_reply(*history_format, source.history ? source.history() : history, ended()));
        return result;
    }
};

Reply::Reply(ReplySource source, ReplyCodec codec, ReplyConfig config)
    : impl_(std::make_unique<Impl>()) {
    if (!source.next || !source.ended || !source.decode_text || !codec.decode || !codec.begin || !codec.push || !codec.finish ||
        codec.sample_rate <= 0 || codec.channels <= 0 || config.frames_per_event == 0) {
        throw std::invalid_argument("Chat reply requires a source, codec and positive audio/event dimensions");
    }
    impl_->source = std::move(source);
    impl_->codec = std::move(codec);
    impl_->config = config;
    impl_->audio = {impl_->codec.sample_rate, impl_->codec.channels, {}};
}

Reply::~Reply() = default;

Reply::Reply(TextSpeechSource source) : impl_(std::make_unique<Impl>()) {
    if (!source.text_source || !source.synthesize || source.sample_rate <= 0) {
        throw std::invalid_argument("Text/speech reply requires text inference, synthesis and a positive sample rate");
    }
    impl_->audio = {source.sample_rate, 1, {}};
    impl_->config.language = source.language;
    impl_->config.max_steps = source.max_steps;
    impl_->config.eos_tokens = source.eos_tokens;
    impl_->config.require_audio = false;
    impl_->text_speech = std::move(source);
}

void Reply::run() {
    auto & m = *impl_;
    if (m.released || m.state != Impl::State::Ready) throw std::logic_error("Chat reply has already started");
    try {
        if (m.text_speech) {
            m.generate_text();
            m.generate_speech();
            return;
        }
        auto batch = m.collect(false);
        m.complete();
        m.consume_text(batch.tokens);
        m.audio.samples = m.codec.decode(batch.frames);
        m.state = Impl::State::Complete;
    } catch (...) {
        m.state = Impl::State::Failed;
        throw;
    }
}

std::optional<runtime::StreamEvent> Reply::next() {
    auto & m = *impl_;
    if (m.released) throw std::logic_error("Chat reply has been released");
    if (m.state == Impl::State::Complete) return std::nullopt;
    if (m.state == Impl::State::Failed) throw std::logic_error("Chat reply failed; begin a new reply before continuing");
    try {
        if (m.text_speech) {
            runtime::StreamEvent event;
            if (!m.text_complete) {
                m.state = Impl::State::Streaming;
                m.generate_text();
                event.partial_text = runtime::Transcript{m.text, m.config.language};
            } else {
                m.generate_speech();
                event.audio_output = m.audio;
            }
            return event;
        }
        if (m.state == Impl::State::Ready) {
            m.codec.begin();
            m.state = Impl::State::Streaming;
        }
        auto batch = m.collect(true);
        auto samples = batch.frames.empty() ? std::vector<float>{} : m.codec.push(batch.frames);
        if (batch.exhausted) {
            m.complete();
            auto tail = m.codec.finish();
            samples.insert(samples.end(), tail.begin(), tail.end());
            m.state = Impl::State::Complete;
        }
        runtime::StreamEvent event;
        auto delta = m.consume_text(batch.tokens);
        if (!delta.empty()) event.partial_text = runtime::Transcript{std::move(delta), m.config.language};
        if (!samples.empty()) {
            event.audio_output = runtime::AudioBuffer{m.audio.sample_rate, m.audio.channels, std::move(samples)};
            runtime::append_audio_buffer(m.audio, *event.audio_output);
        }
        return event;
    } catch (...) {
        m.state = Impl::State::Failed;
        throw;
    }
}

runtime::TaskResult Reply::finalize() {
    auto & m = *impl_;
    if (m.released || m.state == Impl::State::Failed) throw std::logic_error("Cannot finish a released or failed chat reply");
    runtime::TaskResult result;
    m.released = true;
    result.audio_output = std::move(m.audio);
    result.text_output = runtime::Transcript{std::move(m.text), m.config.language};
    if (m.emit_artifact) result.output_artifacts.push_back(encode_reply(*m.history_format, m.source.history ? m.source.history() : m.history, m.ended()));
    return result;
}
runtime::TaskResult Reply::result() const { return impl_->result(); }
bool Reply::done() const { return impl_->state == Impl::State::Complete; }
bool Reply::reached_eos() const { return impl_->ended(); }
size_t Reply::audio_frames() const { return impl_->frames; }
size_t Reply::text_tokens() const { return impl_->tokens; }
const std::vector<ReplyStep> & Reply::steps() const { return impl_->source.history ? impl_->source.history() : impl_->history; }
void Reply::retain_history() {
    if (impl_->state != Impl::State::Ready) throw std::logic_error("Configure reply history before starting inference");
    impl_->record_history = true;
}
void Reply::enable_history(HistoryFormat format, bool emit_artifact) {
    retain_history();
    impl_->history_format = std::move(format);
    impl_->emit_artifact = emit_artifact;
}

}  // namespace engine::chat
