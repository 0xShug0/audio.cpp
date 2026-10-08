#include "engine/framework/runtime/streaming_audio.h"
#include "engine/framework/runtime/streaming_tts_session.h"
#include "test_assert.h"

#include <iostream>
#include <stdexcept>
#include <vector>

using namespace engine::runtime;
using engine::test::require_eq;

class TestTtsSession final : public StreamingTtsSessionBase {
public:
    RunMode mode = RunMode::Streaming;
    bool fail = false;
    int resets = 0;
    StreamingAudioController<float> audio;

    std::string family() const override { return "test_tts"; }
    VoiceTaskKind task_kind() const override { return VoiceTaskKind::Tts; }
    RunMode run_mode() const override { return mode; }
    void prepare(const SessionPreparationRequest &) override {}

private:
    TaskResult generate_stream(const TaskRequest &) override {
        StreamingAudioConfig config;
        config.frames_per_chunk = 3;
        audio.begin(config, [](const StreamingAudioWindow<float> & w) {
            return AudioBuffer{24000, 1, {w.data, w.data + w.new_frames}};
        }, stream_event_sink());
        audio.push({0, 1, 2, 3, 4, 5, 6, 7});
        if (fail) throw std::runtime_error("generation failed");
        TaskResult result;
        result.audio_output = audio.finish();
        return result;
    }
    void reset_stream_state() override {
        ++resets;
        audio.reset();
    }
};

int main() {
    try {
        StreamingAudioController<int32_t> controller;
        StreamingAudioConfig config;
        config.policy = StreamingAudioChunkPolicy::Fixed;
        config.frames_per_chunk = 3;
        config.frame_width = 2;
        config.left_context_frames = 2;
        config.right_context_frames = 1;
        std::vector<float> published;
        std::vector<size_t> counts;
        size_t expected_start = 0;
        auto decode = [&](const StreamingAudioWindow<int32_t> &w) {
            require_eq(w.first_frame, expected_start, "generated offset excludes reference");
            require_eq(w.left_frames, size_t{2}, "left context retained");
            require_eq(w.right_frames, w.final ? size_t{0} : size_t{1}, "right lookahead");
            if (w.first_frame == 0) require_eq(w.data[0], int32_t{-2}, "reference suffix");
            AudioBuffer out{24000, 1, {}};
            for (size_t i = 0; i < w.new_frames; ++i) {
                out.samples.push_back(static_cast<float>(w.data[(w.left_frames + i) * 2]));
            }
            counts.push_back(w.new_frames);
            expected_start += w.new_frames;
            return out;
        };
        controller.begin(config, decode, [&](const StreamEvent &event) {
            published.insert(published.end(), event.audio_output->samples.begin(), event.audio_output->samples.end());
        }, {-3, -3, -2, -2, -1, -1});
        controller.push({0, 0, 1, 1, 2, 2});
        require_eq(counts.size(), size_t{0}, "wait for lookahead");
        controller.push({3, 3, 4, 4, 5, 5, 6, 6});
        auto audio = controller.finish();
        require_eq(counts == std::vector<size_t>{3, 3, 1}, true, "fixed chunks and final remainder");
        require_eq(audio.samples == std::vector<float>{0, 1, 2, 3, 4, 5, 6}, true, "no missing or repeated frames");
        require_eq(audio.samples == published, true, "final matches deltas");

        StreamingAudioController<float> floats;
        config.frame_width = 1;
        config.left_context_frames = config.right_context_frames = 0;
        config.frames_per_chunk = 1;
        int resets = 0;
        auto float_decode = [](const StreamingAudioWindow<float> &w) {
            return AudioBuffer{24000, 1, std::vector<float>(w.data, w.data + w.new_frames)};
        };
        for (int request = 0; request < 2; ++request) {
            floats.begin(config, float_decode, {}, {}, [&] { ++resets; },
                         [] { return AudioBuffer{24000, 1, {0.5F}}; });
            floats.push({0.1F, 0.2F});
            require_eq(floats.finish().samples == std::vector<float>{0.1F, 0.2F, 0.5F}, true,
                       "reset state and drain decoder tail");
        }
        require_eq(resets, 2, "decoder reset per request");
        floats.begin(config, float_decode, [](const StreamEvent &) { throw std::runtime_error("sink failure"); });
        bool failed = false;
        try { floats.push({0.1F}); } catch (const std::runtime_error &) { failed = true; }
        require_eq(failed, true, "sink errors propagate");
        floats.begin(config, float_decode);
        floats.push({0.3F});
        require_eq(floats.finish().samples == std::vector<float>{0.3F}, true, "reuse after error");
        TestTtsSession session;
        std::vector<size_t> chunks;
        std::vector<float> pcm;
        session.set_stream_event_sink([&](const StreamEvent & event) {
            chunks.push_back(event.audio_output->samples.size());
            pcm.insert(pcm.end(), event.audio_output->samples.begin(), event.audio_output->samples.end());
        });
        require_eq(session.streaming_policy().input == StreamingInputKind::None, true, "text input policy");
        for (int i = 0; i < 2; ++i) {
            chunks.clear();
            pcm.clear();
            session.start_stream({});
            auto result = session.finish_stream();
            require_eq(chunks == std::vector<size_t>{1, 2, 3, 2}, true, "default grows to non-power-of-two cap");
            require_eq(result.audio_output->samples == pcm, true, "base lifecycle final equals deltas");
            require_eq(pcm == std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7}, true, "repeat has no stale frames");
        }
        require_eq(session.resets, 4, "start and finalize reset request state");
        bool missing = false;
        try { session.finalize(); } catch (const std::runtime_error &) { missing = true; }
        require_eq(missing, true, "cannot finalize twice");
        session.fail = true;
        failed = false;
        try { session.start_stream({}); } catch (const std::runtime_error &) { failed = true; }
        require_eq(failed, true, "generation errors propagate");
        missing = false;
        try { session.finalize(); } catch (const std::runtime_error &) { missing = true; }
        require_eq(missing, true, "failed generation has no final result");
        session.fail = false;
        session.start_stream({});
        require_eq(session.finalize().audio_output->samples.size(), size_t{8}, "reuse after generation failure");
        session.set_stream_event_sink([](const StreamEvent &) { throw std::runtime_error("sink failure"); });
        failed = false;
        try { session.start_stream({}); } catch (const std::runtime_error &) { failed = true; }
        require_eq(failed, true, "base propagates sink failure");
        session.set_stream_event_sink({});
        session.start_stream({});
        require_eq(session.finalize().audio_output->samples.size(), size_t{8}, "reuse after sink failure");
        bool rejected = false;
        try { session.process_audio_chunk({}); } catch (const std::runtime_error &) { rejected = true; }
        require_eq(rejected, true, "incoming audio rejected");
        session.mode = RunMode::Offline;
        rejected = false;
        try { session.start_stream({}); } catch (const std::runtime_error &) { rejected = true; }
        require_eq(rejected, true, "offline session cannot start stream");
        std::cout << "streaming_audio_test: fixed/grow controller and TTS session lifecycle passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
