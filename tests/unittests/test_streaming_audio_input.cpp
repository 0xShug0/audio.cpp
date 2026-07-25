#include "../../app/streaming/pcm_source.h"
#include "../../app/streaming/streaming.h"

#include "test_assert.h"

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using engine::test::require;
using engine::test::require_eq;

// Records what the streaming driver pushes, so a live source and a materialized buffer can be
// compared chunk for chunk.
class RecordingSession : public engine::runtime::IStreamingVoiceTaskSession {
public:
    struct Chunk {
        int sample_rate = 0;
        int channels = 0;
        int64_t start_sample = 0;
        std::vector<float> samples;
    };

    explicit RecordingSession(int64_t chunk_samples) : chunk_samples_(chunk_samples) {}

    std::string family() const override { return "recording"; }
    engine::runtime::VoiceTaskKind task_kind() const override {
        return engine::runtime::VoiceTaskKind::Asr;
    }
    engine::runtime::RunMode run_mode() const override {
        return engine::runtime::RunMode::Streaming;
    }
    void prepare(const engine::runtime::SessionPreparationRequest &) override { prepared_ = true; }

    engine::runtime::StreamingPolicy streaming_policy() const override {
        engine::runtime::StreamingPolicy policy;
        policy.input = engine::runtime::StreamingInputKind::AudioChunks;
        policy.output = engine::runtime::StreamingOutputKind::FinalResult;
        policy.preferred_audio_chunk_samples = chunk_samples_;
        return policy;
    }
    void start_stream(const engine::runtime::TaskRequest &) override {
        started_ = true;
        chunks.clear();
    }
    void reset() override { chunks.clear(); }
    engine::runtime::StreamEvent process_audio_chunk(const engine::runtime::AudioChunk & chunk) override {
        chunks.push_back(Chunk{chunk.sample_rate, chunk.channels, chunk.start_sample, chunk.samples});
        return {};
    }
    engine::runtime::TaskResult finalize() override {
        finalized_ = true;
        engine::runtime::TaskResult result;
        result.text_output = engine::runtime::Transcript{"done", ""};
        return result;
    }

    bool prepared() const { return prepared_; }
    bool started() const { return started_; }
    bool finalized() const { return finalized_; }

    std::vector<Chunk> chunks;

private:
    int64_t chunk_samples_ = 0;
    bool prepared_ = false;
    bool started_ = false;
    bool finalized_ = false;
};

std::string s16le_bytes(const std::vector<int16_t> & values) {
    std::string bytes;
    bytes.reserve(values.size() * 2);
    for (const int16_t value : values) {
        const auto raw = static_cast<uint16_t>(value);
        bytes.push_back(static_cast<char>(raw & 0xFF));
        bytes.push_back(static_cast<char>((raw >> 8) & 0xFF));
    }
    return bytes;
}

engine::runtime::TaskRequest audio_request(engine::runtime::AudioBuffer audio) {
    engine::runtime::TaskRequest request;
    request.audio_input = std::move(audio);
    return request;
}

std::vector<float> flatten(const std::vector<RecordingSession::Chunk> & chunks) {
    std::vector<float> all;
    for (const auto & chunk : chunks) {
        all.insert(all.end(), chunk.samples.begin(), chunk.samples.end());
    }
    return all;
}

// The whole point of the source abstraction: a live stdin-style source and a fully materialized
// buffer must drive the session identically, so transcripts cannot diverge between the two.
void test_stdin_source_matches_buffer_source() {
    std::vector<int16_t> pcm(2500);
    for (size_t i = 0; i < pcm.size(); ++i) {
        pcm[i] = static_cast<int16_t>((i * 37) % 30000 - 15000);
    }

    engine::runtime::AudioBuffer buffer;
    buffer.sample_rate = 16000;
    buffer.channels = 1;
    buffer.samples.reserve(pcm.size());
    for (const int16_t value : pcm) {
        buffer.samples.push_back(static_cast<float>(value) / 32768.0F);
    }

    RecordingSession buffered(512);
    minitts::app::run_streaming_task(buffered, audio_request(buffer), nullptr);

    RecordingSession streamed(512);
    std::istringstream input(s16le_bytes(pcm));
    const auto stream = minitts::app::make_pcm_chunk_stream(
        input,
        minitts::app::AudioStreamFormat{16000, 1},
        minitts::app::PcmSampleFormat::S16LE);
    minitts::app::run_streaming_task(streamed, audio_request(buffer), nullptr, stream);

    require_eq(streamed.chunks.size(), buffered.chunks.size(), "chunk count");
    for (size_t i = 0; i < buffered.chunks.size(); ++i) {
        const auto & expected = buffered.chunks[i];
        const auto & actual = streamed.chunks[i];
        const std::string label = "chunk " + std::to_string(i);
        require_eq(actual.sample_rate, expected.sample_rate, label + " sample rate");
        require_eq(actual.channels, expected.channels, label + " channels");
        require_eq(actual.start_sample, expected.start_sample, label + " start sample");
        require_eq(actual.samples.size(), expected.samples.size(), label + " size");
        for (size_t j = 0; j < expected.samples.size(); ++j) {
            require(
                actual.samples[j] == expected.samples[j],
                label + " sample " + std::to_string(j) + " differs");
        }
    }
    require(streamed.started(), "stream was not started");
    require(streamed.finalized(), "stream was not finalized");
}

// The driver must honour the session's preferred chunk size and drain the source to EOF,
// including a final short chunk.
void test_chunk_sizing_and_drain() {
    const int64_t chunk_samples = 400;
    std::vector<int16_t> pcm(1000);  // 2 full chunks + a 200-sample tail
    for (size_t i = 0; i < pcm.size(); ++i) {
        pcm[i] = static_cast<int16_t>(i);
    }

    RecordingSession session(chunk_samples);
    std::istringstream input(s16le_bytes(pcm));
    const auto stream = minitts::app::make_pcm_chunk_stream(
        input,
        minitts::app::AudioStreamFormat{16000, 1},
        minitts::app::PcmSampleFormat::S16LE);

    engine::runtime::AudioBuffer contract;
    contract.sample_rate = 16000;
    contract.channels = 1;
    minitts::app::run_streaming_task(session, audio_request(contract), nullptr, stream);

    require_eq(session.chunks.size(), size_t{3}, "chunk count");
    require_eq(session.chunks[0].samples.size(), size_t{400}, "first chunk size");
    require_eq(session.chunks[1].samples.size(), size_t{400}, "second chunk size");
    require_eq(session.chunks[2].samples.size(), size_t{200}, "tail chunk size");
    require_eq(session.chunks[0].start_sample, int64_t{0}, "first start sample");
    require_eq(session.chunks[1].start_sample, int64_t{400}, "second start sample");
    require_eq(session.chunks[2].start_sample, int64_t{800}, "tail start sample");
    require_eq(flatten(session.chunks).size(), pcm.size(), "total samples delivered");
}

// Decoding is explicitly little-endian so a stream captured on one machine reads back the same
// on any host, and matches the scaling the WAV reader uses.
void test_sample_format_decoding() {
    const std::string s16 = s16le_bytes({0, 32767, -32768, 1234});
    std::istringstream s16_input(s16);
    std::vector<float> decoded;
    auto s16_stream = minitts::app::make_pcm_chunk_stream(
        s16_input,
        minitts::app::AudioStreamFormat{16000, 1},
        minitts::app::PcmSampleFormat::S16LE);
    s16_stream.read(4, decoded);
    require_eq(decoded.size(), size_t{4}, "s16le sample count");
    require(decoded[0] == 0.0F, "s16le zero");
    require(decoded[1] == 32767.0F / 32768.0F, "s16le positive full scale");
    require(decoded[2] == -1.0F, "s16le negative full scale");
    require(decoded[3] == 1234.0F / 32768.0F, "s16le midrange");

    // 1.0f and -2.0f little-endian.
    const std::string f32("\x00\x00\x80\x3F\x00\x00\x00\xC0", 8);
    std::istringstream f32_input(f32);
    auto f32_stream = minitts::app::make_pcm_chunk_stream(
        f32_input,
        minitts::app::AudioStreamFormat{16000, 1},
        minitts::app::PcmSampleFormat::F32LE);
    f32_stream.read(2, decoded);
    require_eq(decoded.size(), size_t{2}, "f32le sample count");
    require(decoded[0] == 1.0F, "f32le 1.0");
    require(decoded[1] == -2.0F, "f32le -2.0");
}

// A trailing partial frame cannot be interpreted, so chunks must always hold whole frames -
// sessions reject a chunk whose sample count is not divisible by the channel count.
void test_partial_trailing_frame_is_dropped() {
    const std::string bytes = s16le_bytes({1, 2, 3});  // 1.5 stereo frames
    std::istringstream input(bytes);
    auto stream = minitts::app::make_pcm_chunk_stream(
        input,
        minitts::app::AudioStreamFormat{16000, 2},
        minitts::app::PcmSampleFormat::S16LE);
    std::vector<float> samples;
    const bool more = stream.read(8, samples);
    require(!more, "short read should report end of stream");
    require_eq(samples.size(), size_t{2}, "partial trailing frame must be dropped");
}

void test_empty_stream_produces_no_chunks() {
    RecordingSession session(256);
    std::istringstream input("");
    const auto stream = minitts::app::make_pcm_chunk_stream(
        input,
        minitts::app::AudioStreamFormat{16000, 1},
        minitts::app::PcmSampleFormat::S16LE);

    engine::runtime::AudioBuffer contract;
    contract.sample_rate = 16000;
    contract.channels = 1;
    minitts::app::run_streaming_task(session, audio_request(contract), nullptr, stream);

    require(session.chunks.empty(), "empty stream must not produce chunks");
    require(session.finalized(), "empty stream must still finalize");
}

void test_format_parsing_and_validation() {
    require(
        minitts::app::parse_pcm_sample_format("s16le") == minitts::app::PcmSampleFormat::S16LE,
        "s16le parse");
    require(
        minitts::app::parse_pcm_sample_format("f32le") == minitts::app::PcmSampleFormat::F32LE,
        "f32le parse");
    require_eq(minitts::app::pcm_sample_format_bytes(minitts::app::PcmSampleFormat::S16LE), 2, "s16le width");
    require_eq(minitts::app::pcm_sample_format_bytes(minitts::app::PcmSampleFormat::F32LE), 4, "f32le width");

    bool rejected = false;
    try {
        (void)minitts::app::parse_pcm_sample_format("s24le");
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, "unsupported sample format must be rejected");

    std::istringstream input("");
    bool bad_format_rejected = false;
    try {
        (void)minitts::app::make_pcm_chunk_stream(
            input,
            minitts::app::AudioStreamFormat{0, 1},
            minitts::app::PcmSampleFormat::S16LE);
    } catch (const std::exception &) {
        bad_format_rejected = true;
    }
    require(bad_format_rejected, "non-positive sample rate must be rejected");
}

}  // namespace

int main() {
    try {
        test_stdin_source_matches_buffer_source();
        test_chunk_sizing_and_drain();
        test_sample_format_decoding();
        test_partial_trailing_frame_is_dropped();
        test_empty_stream_produces_no_chunks();
        test_format_parsing_and_validation();
    } catch (const std::exception & error) {
        std::cerr << "streaming_audio_input_test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "streaming_audio_input_test passed\n";
    return 0;
}
