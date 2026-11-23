#include "engine/framework/chat/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/audio/conversion.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <stdexcept>
#include <utility>

namespace engine::chat {

namespace {
void require_audio_room(const std::string & model, int64_t frames, int rate, double limit,
                        const std::string & what, const std::string & remedy) {
    if (limit > 0 && frames > std::llround(limit * rate)) {
        const auto tenths = (frames * 10 + rate - 1) / rate;
        char maximum[32];
        char actual[32];
        std::snprintf(maximum, sizeof(maximum), "%.1f", limit);
        std::snprintf(actual, sizeof(actual), "%.1f", static_cast<double>(tenths) / 10);
        throw runtime::CapacityError(model + " encodes at most " + maximum + " s of audio in one pass, and " +
            what + " " + actual + " s; " + remedy);
    }
}
}  // namespace

struct Duplex::Impl {
    DuplexSource source;
    enum class Encoding { Undecided, Complete, Incremental, Prefix } encoding = Encoding::Undecided;
    int input_rate = 0;
    int input_channels = 0;
    int64_t source_frames = 0;
    int64_t consumed_frames = 0;
    int64_t code_base = 0;
    int64_t emitted_samples = 0;
    std::vector<int32_t> codes;
    runtime::AudioBuffer raw;
    runtime::AudioBuffer output;
    bool started = false;
    bool used = false;
    bool input_closed = false;

    int64_t target_samples() const {
        return static_cast<int64_t>(std::llround(static_cast<double>(source_frames) * source.sample_rate / input_rate));
    }

    int64_t frames(const std::vector<int32_t> & values) const {
        if (values.size() % static_cast<size_t>(source.codebooks) != 0) {
            throw std::runtime_error("Chat codec returned an incomplete code frame");
        }
        return static_cast<int64_t>(values.size() / static_cast<size_t>(source.codebooks));
    }

    std::vector<int32_t> predict(const int32_t * input, int64_t count) {
        std::vector<int32_t> output_codes;
        output_codes.reserve(static_cast<size_t>(count * source.codebooks));
        std::vector<int32_t> frame(static_cast<size_t>(source.codebooks));
        for (int64_t i = 0; i < count; ++i) {
            if (source.predict(input + i * source.codebooks, frame.data())) {
                output_codes.insert(output_codes.end(), frame.begin(), frame.end());
            }
        }
        return output_codes;
    }

    runtime::StreamEvent advance(bool flush) {
        runtime::StreamEvent event;
        if (source_frames == 0) return event;
        const auto target = target_samples();
        const auto wanted = flush ? (target + source.frame_samples - 1) / source.frame_samples : target / source.frame_samples;
        std::vector<int32_t> prefix;
        const auto * available_codes = &codes;
        int64_t base = code_base;
        if (encoding == Encoding::Prefix && wanted > consumed_frames) {
            auto mono = audio::convert_interleaved_audio_to_mono_linear_resampled(raw.samples, raw.sample_rate, raw.channels, source.sample_rate);
            mono.resize(static_cast<size_t>(wanted * source.frame_samples), 0.0f);
            prefix = source.encode({source.sample_rate, 1, std::move(mono)});
            if (frames(prefix) != wanted) throw std::runtime_error("Chat prefix encoder frame count mismatch");
            available_codes = &prefix;
            base = 0;
        }
        const int64_t available = std::min(wanted, base + frames(*available_codes));
        std::vector<int32_t> generated;
        if (available > consumed_frames) {
            generated = predict(available_codes->data() + (consumed_frames - base) * source.codebooks, available - consumed_frames);
            consumed_frames = available;
        }
        if (encoding == Encoding::Incremental && consumed_frames > code_base) {
            codes.erase(codes.begin(), codes.begin() + (consumed_frames - code_base) * source.codebooks);
            code_base = consumed_frames;
        }
        if (!generated.empty()) {
            auto decoded = source.decode_stream(generated, frames(generated));
            runtime::append_audio_buffer(output, decoded);
        }
        if (flush && static_cast<int64_t>(output.samples.size()) > target) output.samples.resize(static_cast<size_t>(target));
        const auto visible = static_cast<int64_t>(output.samples.size());
        if (visible > emitted_samples) {
            event.audio_output = runtime::AudioBuffer{output.sample_rate, output.channels,
                std::vector<float>(output.samples.begin() + emitted_samples, output.samples.end())};
            emitted_samples = visible;
        }
        return event;
    }
};

Duplex::Duplex(DuplexSource source) : impl_(std::make_unique<Impl>()) {
    if (source.sample_rate <= 0 || source.frame_samples <= 0 || source.codebooks <= 0 || !source.encode || !source.encode_stream ||
        !source.reset_codec || !source.predict || !source.decode || !source.decode_stream) {
        throw std::invalid_argument("Duplex chat requires codec dimensions, encoder, predictor and decoder bindings");
    }
    impl_->source = std::move(source);
    impl_->output = {impl_->source.sample_rate, 1, {}};
}
Duplex::~Duplex() = default;

void Duplex::begin(const std::optional<runtime::AudioBuffer> & complete_input) {
    auto & m = *impl_;
    if (m.used) throw std::logic_error("Duplex execution has already been used; start a new conversation execution");
    m.used = true;
    m.source.reset_codec();
    if (complete_input && !complete_input->samples.empty()) {
        m.codes = m.source.encode(*complete_input);
        m.frames(m.codes);
        m.encoding = Impl::Encoding::Complete;
    }
    m.started = true;
}

runtime::StreamEvent Duplex::push(const runtime::AudioChunk & chunk) {
    auto & m = *impl_;
    if (!m.started || m.input_closed) throw std::logic_error("Duplex chat input is not open");
    if (chunk.sample_rate <= 0 || chunk.channels <= 0 || chunk.samples.size() % static_cast<size_t>(chunk.channels) != 0) {
        throw runtime::InvalidRequestError("Duplex chat audio requires a positive format and whole sample frames");
    }
    if (m.input_rate == 0) {
        m.input_rate = chunk.sample_rate;
        m.input_channels = chunk.channels;
        if (m.encoding == Impl::Encoding::Undecided) {
            m.encoding = chunk.sample_rate == m.source.sample_rate && chunk.channels == 1 ?
                Impl::Encoding::Incremental : Impl::Encoding::Prefix;
        }
    } else if (m.input_rate != chunk.sample_rate || m.input_channels != chunk.channels) {
        throw runtime::InvalidRequestError("Duplex chat input format changed during the conversation");
    }
    m.source_frames += static_cast<int64_t>(chunk.samples.size() / static_cast<size_t>(chunk.channels));
    if (m.encoding == Impl::Encoding::Incremental) {
        const auto encoded = m.source.encode_stream({chunk.sample_rate, chunk.channels, chunk.samples}, false);
        m.frames(encoded);
        m.codes.insert(m.codes.end(), encoded.begin(), encoded.end());
    } else if (m.encoding == Impl::Encoding::Prefix) {
        runtime::append_audio_buffer(m.raw, {chunk.sample_rate, chunk.channels, chunk.samples});
    }
    return m.advance(false);
}

runtime::StreamEvent Duplex::end_input() {
    auto & m = *impl_;
    if (!m.started || m.source_frames == 0) throw runtime::InvalidRequestError("Duplex chat requires audio before completion");
    if (m.input_closed) return {};
    if (m.encoding == Impl::Encoding::Incremental) {
        const auto tail = m.source.encode_stream({m.source.sample_rate, 1, {}}, true);
        m.frames(tail);
        m.codes.insert(m.codes.end(), tail.begin(), tail.end());
    }
    auto event = m.advance(true);
    m.input_closed = true;
    return event;
}

runtime::TaskResult Duplex::finalize() {
    (void)end_input();
    auto & m = *impl_;
    runtime::TaskResult result;
    if (!m.output.samples.empty()) result.audio_output = std::move(m.output);
    m.started = false;
    return result;
}

runtime::TaskResult Duplex::interrupt() {
    auto & m = *impl_;
    if (!m.started) throw std::logic_error("Duplex chat has not been started");
    runtime::TaskResult result;
    if (!m.output.samples.empty()) result.audio_output = std::move(m.output);
    m.started = false;
    return result;
}

runtime::TaskResult Duplex::run(const runtime::AudioBuffer & input) {
    auto & m = *impl_;
    if (m.used) throw std::logic_error("Duplex execution has already been used; start a new conversation execution");
    m.used = true;
    const auto encoded = m.source.encode(input);
    const auto generated = m.predict(encoded.data(), m.frames(encoded));
    runtime::TaskResult result;
    if (!generated.empty()) {
        result.audio_output = m.source.decode(generated, m.frames(generated));
        const auto input_frames = input.samples.size() / static_cast<size_t>(input.channels);
        const auto target = static_cast<int64_t>(std::llround(static_cast<double>(input_frames) * m.source.sample_rate / input.sample_rate));
        if (target > 0) result.audio_output->samples.resize(static_cast<size_t>(target), 0.0f);
    }
    return result;
}

struct Session::Impl {
    std::string model_name;
    OpenTurn open;
    std::vector<Exchange> history;
    bool limited = false;
    struct Active {
        runtime::TaskRequest user;
        bool retain_history = false;
        UserTurn turn;
        runtime::AudioBuffer input;
        std::unique_ptr<Reply> reply;
        std::unique_ptr<Duplex> duplex;
        std::vector<ConversationTurn> history;
        size_t imported_turns = 0;
        std::unique_ptr<std::deque<runtime::StreamEvent>> ready;
        size_t published_samples = 0;
        int64_t received_frames = 0;
    };
    std::unique_ptr<Active> active;

    Active & require_active() {
        if (!active) throw std::logic_error(model_name + " chat has not been started");
        return *active;
    }

    void commit_exchange(Exchange exchange) {
        auto & a = require_active();
        if (a.imported_turns == 0) {
            history.push_back(std::move(exchange));
            return;
        }
        // The decoded prefix precedes existing native history. Commit it only
        // with the new exchange, so cancellation/failure cannot import history.
        std::vector<Exchange> retained;
        retained.reserve(a.imported_turns + history.size() + 1);
        for (size_t index = 0; index < a.imported_turns; ++index) {
            // decode_conversation validated ordered question/reply pairs.
            const auto & artifact = exchange.user.input_artifacts[2 * index + 1];
            Exchange imported;
            imported.user.audio_input = a.history[index].question;
            imported.assistant.output_artifacts.push_back(artifact);
            imported.complete = artifact.meta.at("ended") == "true";
            imported.steps = a.history[index].reply;
            retained.push_back(std::move(imported));
        }
        for (auto & previous : history) retained.push_back(std::move(previous));
        retained.push_back(std::move(exchange));
        history = std::move(retained);
    }

    void start_reply(const runtime::AudioBuffer & input) {
        auto & a = require_active();
        if (a.reply) return;
        if (input.samples.empty()) throw runtime::InvalidRequestError(model_name + " s2s streaming received no audio");
        a.reply = a.turn.reply(input, a.history);
        if (!a.reply) throw std::logic_error(model_name + " returned an empty chat reply binding");
        if (a.retain_history) a.reply->retain_history();
        if (a.turn.return_codes) a.reply->enable_history(*a.turn.history_format, true);
    }
};

Session::Session(std::string model_name, OpenTurn open)
    : impl_(std::make_unique<Impl>()) {
    if (!open) throw std::invalid_argument("Chat session requires an inference binding");
    impl_->model_name = std::move(model_name);
    impl_->open = std::move(open);
}
Session::~Session() = default;

void Session::begin(const runtime::TaskRequest & request, HistoryMode history) { begin_user(request, true, history); }

void Session::begin_user(const runtime::TaskRequest & request, bool streaming, HistoryMode history) {
    if (impl_->active) throw std::logic_error("Finish or interrupt the current user turn before starting another");
    auto next = std::make_unique<Impl::Active>();
    next->retain_history = history == HistoryMode::Conversation;
    if (next->retain_history) {
        next->user = request;
        next->user.audio_input.reset();
    }
    next->turn = impl_->open(request, history);
    if (static_cast<bool>(next->turn.reply) == static_cast<bool>(next->turn.duplex) || next->turn.output_sample_rate <= 0) {
        throw std::invalid_argument("Chat turn must bind exactly one execution path and a positive output sample rate");
    }
    if (next->turn.return_codes && !next->turn.history_format) {
        throw std::invalid_argument("Chat artifact export requires a declared history wire format");
    }
    if (next->turn.history_format) {
        next->history = decode_conversation(*next->turn.history_format, request.input_artifacts, next->turn.max_reply_steps);
        next->imported_turns = next->history.size();
    }
    if (next->retain_history) {
        for (const auto & exchange : impl_->history) {
            if (exchange.user.audio_input) next->history.push_back({*exchange.user.audio_input, exchange.steps});
        }
    }
    if (next->retain_history && next->turn.history_format) {
        const auto & format = *next->turn.history_format;
        int64_t steps = 0;
        require_conversation_room(format, steps, true, next->turn.max_reply_steps);
        for (const auto & turn : next->history) {
            if (format.question_steps) {
                steps += format.question_steps(turn.question);
                require_conversation_room(format, steps, true, next->turn.max_reply_steps);
            }
            steps += static_cast<int64_t>(turn.reply.size());
            require_conversation_room(format, steps, true, next->turn.max_reply_steps);
        }
    }
    for (size_t index = 0; index < next->history.size(); ++index) {
        const auto & question = next->history[index].question;
        require_audio_room(impl_->model_name, static_cast<int64_t>(question.samples.size() / static_cast<size_t>(question.channels)),
            question.sample_rate, next->turn.max_audio_seconds, "the question of earlier turn " + std::to_string(index + 1) + " is",
            "leave that turn out, or raise the limit");
    }
    if (next->turn.history_prefix_steps && !next->history.empty()) {
        const auto prefix_steps = next->turn.history_prefix_steps(next->history);
        if (!next->turn.history_format) throw std::invalid_argument("Chat prompt preflight requires a context budget");
        require_conversation_room(*next->turn.history_format, prefix_steps, true, next->turn.max_reply_steps);
    }
    if (next->turn.duplex) {
        next->duplex = next->turn.duplex();
        next->ready = std::make_unique<std::deque<runtime::StreamEvent>>();
        if (streaming) next->duplex->begin(request.audio_input);
    }
    impl_->limited = false;
    impl_->active = std::move(next);
}

void Session::push_audio(const runtime::AudioChunk & chunk) {
    auto & a = impl_->require_active();
    if (a.reply) throw std::logic_error(impl_->model_name + " s2s takes no more audio once the reply has started");
    if (chunk.sample_rate <= 0 || chunk.channels <= 0 || chunk.samples.size() % static_cast<size_t>(chunk.channels) != 0) {
        throw runtime::InvalidRequestError("Chat audio chunks require a positive format and whole sample frames");
    }
    if (a.input.sample_rate == 0) {
        a.input.sample_rate = chunk.sample_rate;
        a.input.channels = chunk.channels;
    } else if (a.input.sample_rate != chunk.sample_rate || a.input.channels != chunk.channels) {
        throw runtime::InvalidRequestError("Chat audio chunk format changed during the turn");
    }
    const auto frames = static_cast<int64_t>(chunk.samples.size() / static_cast<size_t>(chunk.channels));
    require_audio_room(impl_->model_name, a.received_frames + frames, chunk.sample_rate,
        a.turn.max_audio_seconds, "the question so far is", "send a shorter one, or raise the limit");
    if (a.duplex) {
        auto event = a.duplex->push(chunk);
        if (a.retain_history) a.input.samples.insert(a.input.samples.end(), chunk.samples.begin(), chunk.samples.end());
        if (event.audio_output || event.partial_text) a.ready->push_back(std::move(event));
    } else {
        a.input.samples.insert(a.input.samples.end(), chunk.samples.begin(), chunk.samples.end());
    }
    a.received_frames += frames;
}

void Session::end_turn() {
    auto & active = impl_->require_active();
    if (!active.duplex) impl_->start_reply(active.input);
}
void Session::end_input() {
    auto & active = impl_->require_active();
    if (active.duplex) {
        auto event = active.duplex->end_input();
        if (event.audio_output || event.partial_text) active.ready->push_back(std::move(event));
    } else {
        end_turn();
    }
}

std::optional<runtime::StreamEvent> Session::poll() {
    auto & active = impl_->require_active();
    if (active.duplex) {
        if (active.ready->empty()) return std::nullopt;
        auto event = std::move(active.ready->front());
        active.ready->pop_front();
        if (event.audio_output) active.published_samples += event.audio_output->samples.size();
        return event;
    }
    if (!active.reply) return std::nullopt;
    return active.reply->next();
}

runtime::TaskResult Session::finalize() {
    auto & a = impl_->require_active();
    runtime::TaskResult result;
    if (a.duplex) {
        result = a.duplex->finalize();
    } else if (a.reply) {
        impl_->limited = a.reply->done() && !a.reply->reached_eos();
        result = a.reply->finalize();
    } else {
        result.audio_output = runtime::AudioBuffer{a.turn.output_sample_rate, 1, {}};
        result.text_output = runtime::Transcript{"", a.turn.language};
        if (a.turn.return_codes && a.turn.history_format) {
            result.output_artifacts.push_back(encode_reply(*a.turn.history_format, {}, false));
        }
    }
    if (a.retain_history && !a.input.samples.empty()) {
        a.user.audio_input = std::move(a.input);
        impl_->commit_exchange({std::move(a.user), result, a.duplex || (a.reply && a.reply->reached_eos()), a.reply ? a.reply->steps() : std::vector<ReplyStep>{}});
    }
    impl_->active.reset();
    return result;
}

runtime::TaskResult Session::run(const runtime::TaskRequest & request, HistoryMode history) {
    if (!request.audio_input) throw runtime::InvalidRequestError(impl_->model_name + " s2s requires audio_input");
    const auto & audio = *request.audio_input;
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty() || audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw runtime::InvalidRequestError(impl_->model_name + " chat requires nonempty audio with a valid format");
    }
    begin_user(request, false, history);
    try {
        const auto & input = *request.audio_input;
        if (impl_->active->duplex) {
            auto result = impl_->active->duplex->run(input);
            if (impl_->active->retain_history) impl_->commit_exchange({request, result, true, {}});
            impl_->active.reset();
            return result;
        }
        auto & active = *impl_->active;
        require_audio_room(impl_->model_name, static_cast<int64_t>(input.samples.size() / static_cast<size_t>(input.channels)),
            input.sample_rate, active.turn.max_audio_seconds, "the question is", "send a shorter one, or raise the limit");
        if (active.retain_history) active.input = input;
        impl_->start_reply(input);
        active.reply->run();
        return finalize();
    } catch (...) {
        impl_->active.reset();
        throw;
    }
}

void Session::interrupt() {
    if (impl_->active && impl_->active->retain_history && impl_->active->duplex && !impl_->active->input.samples.empty()) {
        auto & a = *impl_->active;
        auto prefix = a.duplex->interrupt();
        if (prefix.audio_output) prefix.audio_output->samples.resize(a.published_samples);
        a.user.audio_input = std::move(a.input);
        impl_->commit_exchange({std::move(a.user), std::move(prefix), false, {}});
        impl_->active.reset();
    } else if (impl_->active && impl_->active->retain_history && impl_->active->reply) {
        (void)finalize();
    } else {
        impl_->active.reset();
    }
}
void Session::cancel() { impl_->active.reset(); }

void Session::reset() {
    impl_->active.reset();
    impl_->history.clear();
    impl_->limited = false;
}
void Session::clear_history() {
    if (impl_->active) throw std::logic_error("Interrupt the current turn before clearing chat history");
    impl_->history.clear();
}
const std::vector<Exchange> & Session::history() const { return impl_->history; }

void TaskSession::bind_chat(std::string model_name, OpenTurn open, runtime::StreamingPolicy policy) {
    if (chat_) throw std::logic_error("Chat task binding is already installed");
    policy_ = policy;
    chat_ = std::make_unique<Session>(std::move(model_name), std::move(open));
}
Session & TaskSession::conversation() { return *chat_; }

runtime::TaskResult TaskSession::run(const runtime::TaskRequest & request) {
    const auto start = std::chrono::steady_clock::now();
    auto result = chat_->run(request, HistoryMode::Request);
    debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(start));
    return result;
}
runtime::StreamingPolicy TaskSession::streaming_policy() const { return policy_; }
bool TaskSession::reply_limited() const { return chat_->reply_limited(); }
bool Session::reply_limited() const {
    const auto & a = impl_->active;
    return a && a->reply ? a->reply->done() && !a->reply->reached_eos() : impl_->limited;
}
void TaskSession::start_stream(const runtime::TaskRequest & request) {
    if (run_mode() != runtime::RunMode::Streaming) throw runtime::InvalidRequestError("Chat streaming requires a streaming session");
    reset();
    stream_start_ = std::chrono::steady_clock::now();
    chat_->begin(request, HistoryMode::Request);
}
std::optional<runtime::StreamEvent> TaskSession::next_stream_event() { chat_->end_turn(); return chat_->poll(); }
void TaskSession::set_stream_event_sink(runtime::StreamEventCallback) {} // The task driver forwards returned events.
runtime::TaskResult TaskSession::finish_stream() {
    auto result = chat_->finalize();
    debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(stream_start_));
    return result;
}
void TaskSession::reset() { chat_->cancel(); }
runtime::StreamEvent TaskSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    chat_->push_audio(chunk);
    auto event = chat_->poll();
    return event ? std::move(*event) : runtime::StreamEvent{};
}
runtime::TaskResult TaskSession::finalize() { return policy_.output == runtime::StreamingOutputKind::FinalResult ? finish_stream() : runtime::TaskResult{}; }

}  // namespace engine::chat
