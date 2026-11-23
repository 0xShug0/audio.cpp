#include "engine/framework/chat/session.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/errors.h"
#include "test_assert.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace engine;
using test::require;
using test::require_eq;

struct Counts { size_t invocations = 0; size_t next = 0; int begin = 0; int push = 0; int flush = 0; };

chat::HistoryFormat format() {
    return {"Synthetic", "test.question", "test.reply", "test.reply/1", 2, 10, 256, 64, {}, {}};
}

std::unique_ptr<chat::Reply> make_reply(const std::shared_ptr<Counts> & counts, bool ended = true, int64_t max_steps = 8, bool publish_text_immediately = true) {
    const std::vector<chat::ReplyStep> sequence = {{1, {}}, {2, {}}, {3, {}}, {-1, {5, 6}}, {-1, {9, 8}}, {4, {}}};
    chat::ReplySource source;
    source.next = [counts, sequence]() -> std::optional<chat::ReplyStep> {
        ++counts->invocations;
        if (counts->next == sequence.size()) return std::nullopt;
        return sequence[counts->next++];
    };
    source.ended = [counts, sequence, ended] { return ended && counts->next == sequence.size(); };
    source.decode_text = [](const std::vector<int32_t> & ids) {
        const std::string bytes[] = {"", "\xe4", "\xb8", "\x80", "\xff"};
        std::string text;
        for (auto id : ids) text += bytes[id];
        return text;
    };
    chat::ReplyCodec codec;
    codec.sample_rate = 8;
    codec.decode = [](const chat::AudioCodes & frames) {
        require_eq(frames.size(), size_t{1}, "EOS frame excluded from whole decode");
        require(frames[0] == std::vector<int32_t>({5, 6}), "correct audio frame");
        return std::vector<float>{5, 9};
    };
    codec.begin = [counts] { ++counts->begin; };
    codec.push = [counts](const chat::AudioCodes & frames) {
        ++counts->push;
        require(frames == chat::AudioCodes{{5, 6}}, "only spoken frames sent to codec");
        return std::vector<float>{5};
    };
    codec.finish = [counts] { ++counts->flush; return std::vector<float>{9}; };
    return std::make_unique<chat::Reply>(std::move(source), std::move(codec), chat::ReplyConfig{9, 2, "en", max_steps, true, {}, publish_text_immediately});
}

void test_reply_and_wire_history() {
    auto counts = std::make_shared<Counts>();
    auto reply = make_reply(counts);
    reply->enable_history(format(), true);
    std::string transcript;
    std::vector<float> pcm;
    for (int i = 0; i < 3; ++i) {
        const auto event = reply->next();
        require(event.has_value(), "text update remains available");
        require_eq(counts->next, static_cast<size_t>(i + 1), "no inference ahead of text publication");
        if (event->partial_text) transcript += event->partial_text->text;
    }
    require_eq(transcript, std::string("\xe4\xb8\x80"), "complete UTF-8 published independently");
    while (const auto event = reply->next()) {
        if (event->partial_text) transcript += event->partial_text->text;
        if (event->audio_output) pcm.insert(pcm.end(), event->audio_output->samples.begin(), event->audio_output->samples.end());
    }
    require_eq(transcript, std::string("\xe4\xb8\x80\xef\xbf\xbd"), "malformed maximal subpart replaced");
    require(pcm == std::vector<float>({5, 9}), "streamed PCM includes flush tail");
    require_eq(counts->begin, 1, "codec started once");
    require_eq(counts->push, 1, "text-only updates do not run audio decoding");
    require_eq(counts->flush, 1, "codec flushed once");
    const auto output = reply->finalize();
    const auto & artifact = output.output_artifacts.at(0);
    // Independent wire fixture: four text tokens and two two-codebook frames,
    // with the EOS frame retained for replay despite not being decoded.
    const int32_t values[] = {1, 2, 3, -1, 5, 6, -1, 9, 8, 4};
    std::vector<std::byte> expected;
    for (int32_t value : values) {
        const auto bits = static_cast<uint32_t>(value);
        for (int shift : {0, 8, 16, 24}) expected.push_back(static_cast<std::byte>((bits >> shift) & 255));
    }
    require(artifact.payload == expected, "little-endian reply wire fixture");
    require_eq(artifact.meta.at("steps"), std::string("6"), "all generated steps retained");
    require_eq(artifact.meta.at("ended"), std::string("true"), "natural completion");
    auto incompatible = format(); incompatible.codebooks = 3;
    bool rejected = false;
    try { chat::decode_reply(incompatible, artifact); } catch (const runtime::InvalidRequestError &) { rejected = true; }
    require(rejected, "incompatible checkpoint rejected");
    rejected = false;
    try { chat::decode_conversation(format(), {artifact}, 6); } catch (const runtime::InvalidRequestError &) { rejected = true; }
    require(rejected, "reply without user question rejected");

    auto offline = make_reply(std::make_shared<Counts>());
    offline->run();
    const auto offline_result = offline->finalize();
    require(offline_result.audio_output->samples == pcm, "independent whole decoder agrees with streamed PCM");
    require_eq(offline_result.text_output->text, transcript, "offline and streaming text");
}

void test_audio_batched_text_publication() {
    auto counts = std::make_shared<Counts>();
    auto reply = make_reply(counts, true, 8, false);
    const auto event = reply->next();
    require(event && event->audio_output && event->partial_text, "batched text arrives with audio");
    require_eq(event->partial_text->text, std::string("\xe4\xb8\x80\xef\xbf\xbd"), "all batch text retained");
    require(event->audio_output->samples == std::vector<float>({5, 9}), "batch includes speech and flush tail");
    require_eq(counts->invocations, size_t{7}, "collect until frame batch or exhaustion");
    require(!reply->next(), "one batch produces exactly one event");
    require_eq(counts->push, 1, "codec receives spoken frames once");
    require_eq(counts->flush, 1, "codec flushes once");
}

void test_partial_and_limited_completion() {
    auto counts = std::make_shared<Counts>();
    auto partial = make_reply(counts);
    partial->enable_history(format(), true);
    partial->next();
    const auto early = partial->finalize();
    require_eq(early.output_artifacts[0].meta.at("ended"), std::string("false"), "early finish exports incomplete reply");
    require_eq(early.output_artifacts[0].meta.at("steps"), std::string("1"), "early finish captures only consumed step");
    require_eq(counts->flush, 0, "early finish does not generate a new tail");
    auto limited_counts = std::make_shared<Counts>();
    auto limited = make_reply(limited_counts, false, 6);
    limited->enable_history(format(), true);
    limited->run();
    require_eq(limited_counts->invocations, size_t{6}, "no unused inference after the generation cap");
    require(limited->done() && !limited->reached_eos(), "generation-cap completion distinguished from EOS");
    require_eq(limited->finalize().output_artifacts[0].meta.at("ended"), std::string("false"), "capped reply remains replayable");
}

void test_no_speech_fails_explicitly() {
    int decodes = 0;
    chat::ReplySource source;
    source.next = []() -> std::optional<chat::ReplyStep> { return std::nullopt; };
    source.ended = [] { return true; };
    source.decode_text = [](const std::vector<int32_t> &) { return std::string{}; };
    chat::ReplyCodec codec;
    codec.sample_rate = 8;
    codec.decode = [&](const chat::AudioCodes &) { ++decodes; return std::vector<float>{}; };
    codec.begin = [] {};
    codec.push = [](const chat::AudioCodes &) { return std::vector<float>{}; };
    codec.finish = [] { return std::vector<float>{}; };
    chat::Reply reply(std::move(source), std::move(codec), {});
    bool rejected = false;
    try { reply.run(); } catch (const std::runtime_error & error) {
        rejected = std::string(error.what()).find("no speech") != std::string::npos;
    }
    require(rejected && decodes == 0, "no-speech reply is an explicit failure before waveform decoding");
}

void test_conversation_lifecycle_and_endpoint() {
    std::vector<size_t> supplied_history;
    auto counts = std::make_shared<Counts>();
    bool limit_reply = false;
    chat::Session session("Synthetic", [&](const runtime::TaskRequest &, chat::HistoryMode) {
        chat::UserTurn turn;
        turn.output_sample_rate = 8;
        turn.max_audio_seconds = 2;
        turn.max_reply_steps = 6;
        turn.history_format = format();
        turn.return_codes = limit_reply;
        turn.reply = [&](const runtime::AudioBuffer &, const std::vector<chat::ConversationTurn> & history) {
            supplied_history.push_back(history.size());
            counts = std::make_shared<Counts>();
            return make_reply(counts, !limit_reply, limit_reply ? 6 : 8);
        };
        return turn;
    });
    runtime::TaskRequest request;
    request.audio_input = runtime::AudioBuffer{8, 1, {1, 2, 3, 4}};
    session.begin(request);
    session.push_audio({8, 1, 0, {1, 2}});
    require(!session.poll(), "polling is not a user endpoint");
    require(supplied_history.empty(), "no prefill before explicit end turn");
    session.push_audio({8, 1, 2, {3, 4}});
    session.end_turn();
    while (session.poll()) {}
    const auto first = session.finalize();
    require(!first.audio_output->samples.empty(), "completed chat has audio");
    require_eq(session.history().size(), size_t{1}, "conversation owns completed exchange");
    require(session.history()[0].complete, "complete reply recorded");
    require(session.history()[0].steps.size() == 6, "native history retained without requesting artifact export");
    session.run(request);
    session.run(request, chat::HistoryMode::Request);
    require(supplied_history == std::vector<size_t>({0, 1, 0}), "native history and legacy request isolation");
    require_eq(session.history().size(), size_t{2}, "legacy request does not mutate native conversation history");
    session.begin(request);
    session.push_audio({8, 1, 0, {1}});
    session.end_turn();
    session.poll();
    session.interrupt();
    require_eq(session.history().size(), size_t{3}, "interrupted prefix committed");
    require(!session.history().back().complete && session.history().back().steps.size() == 1, "no imaginary steps after interruption");
    session.reset();
    require(session.history().empty(), "reset clears conversation");
    session.run(request);
    require_eq(supplied_history.back(), size_t{0}, "clean generation after reset");
    limit_reply = true;
    const auto capped = session.run(request);
    require(!session.history().back().complete, "token-limited native reply is incomplete");
    require_eq(capped.output_artifacts.at(0).meta.at("ended"), std::string("false"), "token-limited wire reply is incomplete");
}

void test_native_history_needs_no_wire_format() {
    std::vector<size_t> history_sizes;
    chat::Session session("Native", [&](const runtime::TaskRequest &, chat::HistoryMode) {
        chat::UserTurn turn;
        turn.output_sample_rate = 8;
        turn.reply = [&](const runtime::AudioBuffer &, const std::vector<chat::ConversationTurn> & history) {
            history_sizes.push_back(history.size());
            if (!history.empty()) require_eq(history.back().reply.size(), size_t{6}, "native replay retains generated steps");
            return make_reply(std::make_shared<Counts>());
        };
        return turn;
    });
    runtime::TaskRequest request;
    request.audio_input = runtime::AudioBuffer{8, 1, {1, 2}};
    require(session.run(request).output_artifacts.empty(), "native conversation does not need wire artifacts");
    require_eq(session.history().back().steps.size(), size_t{6}, "capture independent of legacy artifact schema");
    session.run(request);
    require(history_sizes == std::vector<size_t>({0, 1}), "native replay independent of transport serialization");
}

void test_imported_history_continuation() {
    // Independent PCM16 WAV: two samples, +0.25 and -0.25, at 8 Hz.
    const char wav[] = "RIFF\x28\0\0\0WAVEfmt \x10\0\0\0\x01\0\x01\0"
                       "\x08\0\0\0\x10\0\0\0\x02\0\x10\0data\x04\0\0\0\0\x20\0\xe0";
    const auto * bytes = reinterpret_cast<const std::byte *>(wav);
    const auto question = runtime::make_voice_artifact(runtime::ArtifactKind::Custom, "test.question",
        std::vector<std::byte>(bytes, bytes + sizeof(wav) - 1));
    runtime::TaskRequest imported;
    imported.audio_input = runtime::AudioBuffer{8, 1, {0.5f, 0.75f}};
    imported.input_artifacts = {question, chat::encode_reply(format(), {{42, {}}}, false)};
    runtime::TaskRequest next;
    next.audio_input = runtime::AudioBuffer{8, 1, {-0.5f, -0.75f}};

    for (int completion = 0; completion < 3; ++completion) {
        imported.input_artifacts[1] = chat::encode_reply(format(), {{42, {}}}, completion != 2);
        std::vector<std::vector<chat::ConversationTurn>> observed;
        chat::Session session("Imported", [&](const runtime::TaskRequest &, chat::HistoryMode) {
            chat::UserTurn turn;
            turn.output_sample_rate = 8;
            turn.history_format = format();
            turn.reply = [&](const runtime::AudioBuffer &, const std::vector<chat::ConversationTurn> & history) {
                observed.push_back(history);
                return make_reply(std::make_shared<Counts>());
            };
            return turn;
        });
        if (completion == 0) {
            session.run(imported);
        } else {
            session.begin(imported);
            session.push_audio({8, 1, 0, imported.audio_input->samples});
            session.end_turn();
            if (completion == 1) {
                while (session.poll()) {}
                session.finalize();
            } else {
                session.poll();
                session.interrupt();
            }
        }
        session.run(next);
        require_eq(observed[0].size(), size_t{1}, "B receives imported A");
        require_eq(observed[1].size(), size_t{2}, "C receives imported A and retained B");
        require(observed[1][0].question.samples == std::vector<float>({0.25f, -0.25f}), "imported question retained exactly");
        require_eq(observed[1][0].reply.at(0).token, int32_t{42}, "imported reply retained exactly");
        require(observed[1][1].question.samples == imported.audio_input->samples, "new exchange follows imported prefix");
        require_eq(observed[1][1].reply.size(), completion == 2 ? size_t{1} : size_t{6}, "only generated B steps retained");
        require_eq(session.history().size(), size_t{3}, "native history exposes A, B and C once each");
        require_eq(session.history()[0].complete, completion != 2, "imported completion status preserved");

        // A later import must preserve already-retained native exchanges too.
        imported.input_artifacts[1] = chat::encode_reply(format(), {{43, {}}}, false);
        session.run(imported);
        session.run(next);
        require_eq(observed.back().size(), size_t{5}, "later import retains prefix and existing conversation");
        require_eq(observed.back()[0].reply.at(0).token, int32_t{43}, "new imported prefix stays first");
        require_eq(observed.back()[1].reply.at(0).token, int32_t{42}, "earlier imported prefix survives another import");

        session.reset();
        session.begin(imported);
        session.push_audio({8, 1, 0, imported.audio_input->samples});
        session.end_turn();
        session.poll();
        session.cancel();
        require(session.history().empty(), "cancellation does not commit imported prefix");
        session.run(next);
        require(observed.back().empty(), "cancelled import does not leak into continuation");

        const auto retained = session.history().size();
        session.run(imported, chat::HistoryMode::Request);
        require_eq(observed.back().size(), size_t{1}, "legacy request uses only explicit artifacts");
        session.run(next, chat::HistoryMode::Request);
        require(observed.back().empty(), "legacy request does not retain imported artifacts");
        require_eq(session.history().size(), retained, "legacy requests leave native history untouched");
        imported.input_artifacts[1] = chat::encode_reply(format(), {{42, {}}}, false);
    }
}

void test_text_stage_is_shared_and_released_before_speech() {
    struct Lease {
        bool & alive;
        explicit Lease(bool & value) : alive(value) { alive = true; }
        ~Lease() { alive = false; }
    };
    bool text_alive = false;
    size_t calls = 0;
    int synth_calls = 0;
    chat::TextSpeechSource pipeline;
    pipeline.sample_rate = 8;
    pipeline.max_steps = 8;
    pipeline.eos_tokens = {7};
    pipeline.text_source = [&] {
        auto lease = std::make_shared<Lease>(text_alive);
        chat::ReplySource source;
        source.next = [&, lease]() -> std::optional<chat::ReplyStep> {
            const int32_t tokens[] = {1, 2, 7};
            require(calls < 3, "no inference after EOS");
            return chat::ReplyStep{tokens[calls++], {}};
        };
        source.decode_text = [](const std::vector<int32_t> & tokens) {
            require(tokens == std::vector<int32_t>({1, 2}), "shared text loop excludes EOS");
            return std::string("hi");
        };
        return source;
    };
    pipeline.synthesize = [&](const std::string & text) {
        ++synth_calls;
        require(!text_alive, "prepared text resources released before speech allocation");
        require_eq(text, std::string("hi"), "speech conditioning is the generated text");
        return std::vector<float>{4, 5};
    };
    auto streaming_pipeline = pipeline;
    chat::Reply reply(std::move(pipeline));
    reply.retain_history();
    reply.run();
    require(reply.done() && reply.reached_eos(), "framework owns text-stage natural completion");
    require_eq(calls, size_t{3}, "exact token-step count including EOS");
    require_eq(reply.steps().size(), size_t{2}, "text history contains emitted tokens");
    const auto result = reply.finalize();
    require(result.audio_output->samples == std::vector<float>({4, 5}), "synthesis output retained");
    calls = 0;
    synth_calls = 0;
    chat::Reply streamed(std::move(streaming_pipeline));
    const auto text_event = streamed.next();
    require(text_event && text_event->partial_text && text_event->partial_text->text == "hi" && !text_event->audio_output,
        "staged reply publishes text before speech inference");
    require(synth_calls == 0 && !text_alive, "text publication does not hold the prepared text stage or run speech");
    const auto audio_event = streamed.next();
    require(audio_event && !audio_event->partial_text && audio_event->audio_output->samples == std::vector<float>({4, 5}),
        "second stage publishes independently");
    require(synth_calls == 1 && !streamed.next(), "synthesis and completion occur exactly once");

}

void test_task_adapter_preserves_native_history() {
    class Task final : public chat::TaskSession {
    public:
        Task() {
            bind_chat("Task", [](const runtime::TaskRequest &, chat::HistoryMode) {
                chat::UserTurn turn;
                turn.output_sample_rate = 8;
                turn.reply = [](const runtime::AudioBuffer &, const std::vector<chat::ConversationTurn> &) {
                    return make_reply(std::make_shared<Counts>());
                };
                return turn;
            });
        }
        std::string family() const override { return "task_fixture"; }
        runtime::VoiceTaskKind task_kind() const override { return runtime::VoiceTaskKind::SpeechToSpeech; }
        runtime::RunMode run_mode() const override { return runtime::RunMode::Streaming; }
        void prepare(const runtime::SessionPreparationRequest &) override {}
    } task;
    runtime::TaskRequest request;
    request.audio_input = runtime::AudioBuffer{8, 1, {1, 2}};
    auto & native = dynamic_cast<chat::IConversationSession &>(task).conversation();
    const auto first = native.run(request);
    task.start_stream(request);
    task.process_audio_chunk({8, 1, 0, {1, 2}});
    while (task.next_stream_event()) {}
    const auto legacy = task.finish_stream();
    require(legacy.audio_output->samples == first.audio_output->samples, "legacy adapter uses the same pipeline");
    require_eq(native.history().size(), size_t{1}, "legacy streaming does not clear or append native history");
    task.reset();
    require_eq(native.history().size(), size_t{1}, "legacy cancellation preserves completed exchanges");
    native.reset();
    require(native.history().empty(), "native reset explicitly clears history");
}

void test_duplex_clock_and_bounded_reuse() {
    std::vector<int32_t> seen;
    chat::DuplexSource source;
    source.sample_rate = 8; source.frame_samples = 2; source.codebooks = 1;
    source.reset_codec = [] {};
    source.encode = [](const runtime::AudioBuffer & input) {
        std::vector<int32_t> result;
        for (size_t i = 0; i < input.samples.size(); i += 2) result.push_back(static_cast<int32_t>(input.samples[i]));
        return result;
    };
    source.encode_stream = [](const runtime::AudioBuffer & input, bool) {
        std::vector<int32_t> result;
        for (size_t i = 0; i < input.samples.size(); i += 2) result.push_back(static_cast<int32_t>(input.samples[i]));
        return result;
    };
    source.predict = [&seen](const int32_t * input, int32_t * output) { seen.push_back(*input); *output = *input + 10; return true; };
    source.decode = source.decode_stream = [](const std::vector<int32_t> & codes, int64_t frames) {
        require_eq(codes.size(), static_cast<size_t>(frames), "decode shape");
        runtime::AudioBuffer output{8, 1, {}};
        for (auto code : codes) { output.samples.push_back(static_cast<float>(code)); output.samples.push_back(static_cast<float>(code)); }
        return output;
    };
    chat::Duplex duplex(source);
    duplex.begin(std::nullopt);
    for (int i = 1; i <= 5; ++i) {
        const auto event = duplex.push({8, 1, (i - 1) * 2, {static_cast<float>(i), 0}});
        require(event.audio_output && event.audio_output->samples == std::vector<float>(2, static_cast<float>(i + 10)), "live frame emitted promptly");
    }
    const auto output = duplex.finalize();
    require(seen == std::vector<int32_t>({1, 2, 3, 4, 5}), "consumed frame queue offsets remain correct across compaction");
    require_eq(output.audio_output->samples.size(), size_t{10}, "live output sample count");
    seen.clear();
    chat::Duplex complete(source);
    const runtime::AudioBuffer input{8, 1, {1, 0, 2, 0, 3}};
    complete.begin(input);
    complete.push({8, 1, 0, {1, 0}});
    require(seen == std::vector<int32_t>{1}, "preencoded input still follows received sample clock");
    complete.push({8, 1, 2, {2, 0, 3}});
    require(seen == std::vector<int32_t>({1, 2}), "partial final frame waits for completion");
    const auto tail = complete.finalize();
    require(seen == std::vector<int32_t>({1, 2, 3}), "final partial frame consumed exactly once");
    require_eq(tail.audio_output->samples.size(), size_t{5}, "codec padding removed at final input boundary");
    // The conversation API queues events for both execution paths. Closing
    // input flushes a partial frame before result collection.
    chat::Session queued("Duplex", [&](const runtime::TaskRequest &, chat::HistoryMode) {
        chat::UserTurn turn;
        turn.output_sample_rate = 8;
        turn.duplex = [source] { return std::make_unique<chat::Duplex>(source); };
        return turn;
    });
    queued.begin({});
    queued.push_audio({8, 1, 0, {1, 0, 2, 0, 3}});
    auto ready = queued.poll();
    require(ready && ready->audio_output->samples == std::vector<float>({11, 11, 12, 12}), "duplex uses the shared poll API");
    queued.end_input();
    bool closed_rejected = false;
    try { queued.push_audio({8, 1, 5, {99, 0}}); } catch (const std::logic_error &) { closed_rejected = true; }
    require(closed_rejected, "closed input rejects additional chunks");
    auto final_frame = queued.poll();
    require(final_frame && final_frame->audio_output->samples == std::vector<float>{13}, "partial codec tail published once");
    require(!queued.poll(), "closed duplex event queue drains");
    require_eq(queued.finalize().audio_output->samples.size(), size_t{5}, "queued result includes the complete output");
    require_eq(queued.history().back().user.audio_input->samples.size(), size_t{5}, "rejected chunks do not enter user history");
    queued.begin({});
    queued.push_audio({8, 1, 0, {1, 0}});
    queued.push_audio({8, 1, 2, {2, 0}});
    queued.poll();
    queued.interrupt();
    require(queued.history().back().assistant.audio_output->samples == std::vector<float>({11, 11}),
        "interruption commits only published duplex output");
    const auto exchanges = queued.history().size();
    queued.begin({});
    queued.interrupt();
    require_eq(queued.history().size(), exchanges, "an unopened user message does not poison history");

}
}  // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--log") engine::debug::configure_logging({true, std::nullopt});
    try {
        test_reply_and_wire_history();
        test_audio_batched_text_publication();
        test_partial_and_limited_completion();
        test_no_speech_fails_explicitly();
        test_conversation_lifecycle_and_endpoint();
        test_native_history_needs_no_wire_format();
        test_imported_history_continuation();
        test_text_stage_is_shared_and_released_before_speech();
        test_task_adapter_preserves_native_history();
        test_duplex_clock_and_bounded_reuse();
        std::cout << "chat_session_test: ok\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "chat_session_test: " << error.what() << '\n';
        return 1;
    }
}
