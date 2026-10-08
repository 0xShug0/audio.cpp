#include "engine/framework/runtime/streaming_asr_session.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/debug/trace.h"
#include "test_assert.h"

#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>

namespace {
using namespace engine::runtime;
using engine::test::require;

class Session final : public StreamingAsrSessionBase {
public:
    std::string family() const override { return "asr_test"; }
    VoiceTaskKind task_kind() const override { return VoiceTaskKind::Asr; }
    RunMode run_mode() const override { return mode; }
    void prepare(const SessionPreparationRequest &) override {}
    RunMode mode = RunMode::Streaming;
    int resets = 0;
    bool cumulative = false;
    bool provisional = false;
    bool flush_tail = false;
    bool final_metadata = false;

private:
    std::vector<WordTimestamp> decoded_words;
    void begin_stream(const TaskRequest & request) override {
        if (request.options.count("fail")) throw std::runtime_error("invalid request");
    }
    void reset_stream_state() override { ++resets; decoded_words.clear(); }
    void consume_audio(const AudioChunk & chunk) override {
        if (chunk.sample_rate != 16000) throw std::runtime_error("invalid sample rate");
        for (size_t i = 0; i < chunk.samples.size(); ++i) {
            if (provisional) {
                publish_hypothesis({i == 0 ? "I scream" : "Ice cream", "en"});
                continue;
            }
            WordTimestamp word;
            word.word = "word";
            word.span.start_sample = static_cast<int64_t>(decoded_words.size());
            word.span.end_sample = word.span.start_sample + 1;
            decoded_words.push_back(word);
            if (cumulative) {
                const auto text = committed_transcript().text + "word ";
                publish_transcript({text, "en"}, decoded_words);
            } else {
                append_transcript({"word ", "en"}, {word}, true);
            }
        }
    }
    void complete_stream(TaskResult & result) override {
        if (flush_tail) append_transcript({"tail", "en"});
        if (final_metadata) {
            // A decoder may know the last word's end only after the final flush.
            result.word_timestamps.back().span.end_sample = 42;
            AsrStreamMetadata metadata;
            metadata.is_final = true;
            publish_metadata(std::move(metadata));
        }
    }
};

template <typename F> void rejected(F call) {
    bool threw = false;
    try { call(); } catch (const std::runtime_error &) { threw = true; }
    require(threw, "invalid lifecycle call must fail");
}
}

int main(int argc, char ** argv) {
    std::unordered_map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--log") {
            engine::debug::configure_logging({true, std::nullopt});
        } else {
            require(i + 1 < argc, "missing option value");
            args[key] = argv[++i];
        }
    }
    Session session;
    AudioChunk chunk{16000, 1, 0, {0.1f}};
    rejected([&] { session.process_audio_chunk(chunk); });
    rejected([&] { session.finalize(); });
    session.start_stream({});
    require(session.process_audio_chunk(chunk).partial_text->text == "word ", "returned delta");
    require(session.finish_stream().text_output->text == "word ", "final result");
    rejected([&] { session.process_audio_chunk(chunk); });
    rejected([&] { session.finalize(); });

    std::vector<StreamEvent> events;
    session.set_stream_event_sink([&](const StreamEvent & event) { events.push_back(event); });
    session.start_stream({});
    require(!session.process_audio_chunk(chunk).partial_text, "callback must not also return delta");
    require(events.size() == 1 && events[0].partial_text->text == "word ", "exactly one callback");
    require(events[0].partial_text_snapshot->text == "word ", "snapshot preserved");
    AudioChunk empty{16000, 1, 1, {}};
    session.process_audio_chunk(empty);
    require(events.size() == 1, "empty chunks must not produce callback noise");
    auto invalid = chunk;
    invalid.sample_rate = 0;
    rejected([&] { session.process_audio_chunk(invalid); });
    session.process_audio_chunk(chunk);
    require(session.finalize().text_output->text == "word word ", "retry after validation error");

    session.start_stream({});
    session.process_audio_chunk(chunk);
    session.reset();
    rejected([&] { session.finalize(); });
    TaskRequest invalid_request;
    invalid_request.options["fail"] = "true";
    rejected([&] { session.start_stream(invalid_request); });
    rejected([&] { session.process_audio_chunk(chunk); });
    session.set_stream_event_sink(nullptr);
    session.start_stream({});
    session.process_audio_chunk(chunk);
    require(session.finalize().text_output->text == "word ", "request state must not leak");

    AudioChunk multiple{16000, 1, 0, {0.1f, 0.2f}};
    for (bool cumulative : {false, true}) {
        session.cumulative = cumulative;
        session.start_stream({});
        auto event = session.process_audio_chunk(multiple);
        require(event.partial_text->text == "word word ", "all feed deltas must survive");
        require(event.word_timestamps.size() == 2, "timestamp updates lost or duplicated");
        require(session.finalize().word_timestamps.size() == 2, "final timestamp aggregation");
    }
    session.provisional = true;
    session.start_stream({});
    auto hypothesis = session.process_audio_chunk(multiple);
    require(!hypothesis.partial_text, "hypothesis must not commit text");
    require(hypothesis.partial_text_snapshot->text == "Ice cream", "latest hypothesis replaces earlier one");
    require(session.finalize().text_output->text.empty(), "provisional text must not enter final transcript");
    session.provisional = false;
    session.cumulative = false;
    session.flush_tail = true;
    events.clear();
    session.set_stream_event_sink([&](const StreamEvent & e) { events.push_back(e); });
    session.start_stream({});
    require(!session.process_audio_chunk(multiple).partial_text, "multi-update feed must not replay callbacks");
    require(events.size() == 2, "callback must receive each update immediately");
    require(session.finalize().text_output->text == "word word tail", "flush must enter final transcript");
    require(events.size() == 3 && events.back().partial_text->text == "tail", "flush delivery exactly once");
    session.final_metadata = true;
    events.clear();
    session.start_stream({});
    session.process_audio_chunk(chunk);
    auto final_result = session.finalize();
    require(final_result.text_output->text == "word tail", "final metadata must preserve flushed text");
    require(final_result.word_timestamps.back().span.end_sample == 42, "final timestamp override lost");
    require(events.size() == 3 && events.back().is_final, "final metadata delivered once");
    require(!events.back().partial_text, "metadata must not replay committed text");
    session.mode = RunMode::Offline;
    rejected([&] { session.start_stream({}); });

    // Optional real-model coverage: one feed crosses several recognition windows.
    // Callback and return-only clients must receive the same committed text.
    if (args.count("--model")) {
        auto registry = make_default_registry();
        ModelLoadRequest load;
        load.model_path = args.at("--model");
        load.family_hint = args.at("--family");
        auto model = registry.load(load);
        SessionOptions options;
        options.backend.type = engine::core::BackendType::Cuda;
        options.backend.threads = 4;
        auto task = model->create_task_session({VoiceTaskKind::Asr, RunMode::Streaming}, options);
        auto * stream = dynamic_cast<IStreamingVoiceTaskSession *>(task.get());
        require(stream != nullptr, "streaming session required");
        const auto wav = engine::audio::read_wav_f32(std::filesystem::path(args.at("--audio")));
        TaskRequest request;
        request.audio_input = AudioBuffer{wav.sample_rate, wav.channels, wav.samples};
        if (args.count("--audio-chunk-duration-sec")) {
            request.options["audio_chunk_duration_sec"] = args.at("--audio-chunk-duration-sec");
        }
        stream->prepare(build_preparation_request(request));
        const AudioChunk input{wav.sample_rate, wav.channels, 0, wav.samples};
        std::string expected_delta, expected_text;
        for (int pass = 0; pass != 3; ++pass) {
            std::string delta;
            if (pass == 1) {
                stream->set_stream_event_sink([&](const StreamEvent & e) {
                    if (e.partial_text) delta += e.partial_text->text;
                });
            } else {
                stream->set_stream_event_sink(nullptr);
            }
            stream->start_stream(request);
            auto event = stream->process_audio_chunk(input);
            if (pass == 1) require(!event.partial_text, "callback duplicated as return value");
            if (event.partial_text) delta += event.partial_text->text;
            if (pass == 0) expected_delta = delta;
            else require(delta == expected_delta, "callback and returned deltas differ");
            require(!delta.empty(), "multi-window input must emit text before finalization");
            auto result = stream->finalize();
            require(result.text_output.has_value(), "missing final transcript");
            if (pass == 0) expected_text = result.text_output->text;
            else require(result.text_output->text == expected_text, "repeated final transcript differs");
            require(expected_text.find(expected_delta) == 0, "deltas lost or duplicated before finalization");
        }
    }
    return 0;
}
