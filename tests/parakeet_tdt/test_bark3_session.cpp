#include "engine/community_models/parakeet_tdt/bark3.h"
#include "test_assert.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/errors.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace rt = engine::runtime;
namespace bark = engine::community_models::parakeet_tdt;

namespace {
size_t allocation_probe(bool bark_policy) {
    auto context = std::unique_ptr<ggml_context, decltype(&ggml_free)>(
        ggml_init({4 * 1024 * 1024, nullptr, true}), ggml_free);
    engine::test::require(context != nullptr, "probe context");
    auto * input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 128, 128);
    ggml_set_input(input);
    auto * result = input;
    for (int i = 0; i < 32; ++i) {
        auto * scaled = ggml_scale(context.get(), result, 2.f);
        auto * cropped = ggml_view_2d(context.get(), scaled, scaled->ne[0] - 1, 128, scaled->nb[1], sizeof(float));
        result = ggml_cont(context.get(), cropped);
    }
    ggml_set_output(result);
    auto * graph = ggml_new_graph_custom(context.get(), 1024, false);
    ggml_build_forward_expand(graph, result);
    const auto cpu = rt::GraphOptimizationBackend::Cpu;
    auto options = bark_policy ? bark::bark3_encoder_graph_options(cpu)
                               : rt::graph_optimization_options_for_backend(cpu);
    options.enabled = true;  // Exercise allocation planning even with a caller opt-out env var.
    rt::optimize_graph(*graph, options);
    auto backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(
        engine::core::init_backend({engine::core::BackendType::Cpu, 0, 1}), ggml_backend_free);
    auto allocator = std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>(
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get())), ggml_gallocr_free);
    size_t bytes = 0;
    ggml_gallocr_reserve_n_size(allocator.get(), graph, nullptr, nullptr, &bytes);
    if (bark_policy) {
        engine::test::require(ggml_gallocr_reserve(allocator.get(), graph), "reserve view-lifetime probe");
        engine::test::require(ggml_gallocr_alloc_graph(allocator.get(), graph), "allocate view-lifetime probe");
        std::vector<float> values(128 * 128);
        for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(i + 1);
        ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
        engine::test::require(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "compute probe");
        std::vector<float> actual(96 * 128);
        ggml_backend_tensor_get(result, actual.data(), 0, actual.size() * sizeof(float));
        for (int row = 0; row < 128; ++row)
            for (int col = 0; col < 96; ++col)
                engine::test::require_eq(actual[row * 96 + col], std::ldexp(values[row * 128 + col + 32], 32), "reused buffers retain correct values");
    }
    return bytes;
}
void allocation_test() {
    const auto old_bytes = allocation_probe(false);
    const auto fixed_bytes = allocation_probe(true);
    engine::test::require(fixed_bytes < old_bytes / 4, "view lifetimes bound temporary allocation across layers");
}
rt::WordTimestamp word(int64_t begin, int64_t end) {
    rt::WordTimestamp w; w.word = "hello"; w.span.start_sample = begin; w.span.end_sample = end; return w;
}
template<class F> void rejects(F operation) {
    bool rejected = false;
    try { operation(); } catch (const std::exception &) { rejected = true; }
    engine::test::require(rejected, "invalid operation must be rejected");
}
class Offline final : public rt::IOfflineVoiceTaskSession {
public:
    rt::SessionPreparationRequest preparation;
    rt::AudioBuffer input;
    bool fail = false;
    std::string family() const override { return "parakeet_tdt"; }
    rt::VoiceTaskKind task_kind() const override { return rt::VoiceTaskKind::Asr; }
    rt::RunMode run_mode() const override { return rt::RunMode::Offline; }
    void prepare(const rt::SessionPreparationRequest & r) override { preparation = r; }
    rt::TaskResult run(const rt::TaskRequest & r) override {
        input = *r.audio_input;
        if (std::exchange(fail, false)) throw std::runtime_error("injected failure");
        rt::TaskResult out; out.word_timestamps = {word(3000, 5000), word(6000, 999999)}; return out;
    }
};
class Streaming final : public rt::IStreamingVoiceTaskSession {
public:
    int64_t received = 0;
    bool fail = false;
    rt::StreamEventCallback sink;
    std::string family() const override { return "parakeet_tdt"; }
    rt::VoiceTaskKind task_kind() const override { return rt::VoiceTaskKind::Asr; }
    rt::RunMode run_mode() const override { return rt::RunMode::Streaming; }
    void prepare(const rt::SessionPreparationRequest &) override {}
    void reset() override { received = 0; }
    void set_stream_event_sink(rt::StreamEventCallback callback) override { sink = std::move(callback); }
    rt::StreamEvent process_audio_chunk(const rt::AudioChunk & chunk) override {
        engine::test::require_eq(chunk.start_sample, received, "padded chunks remain contiguous");
        if (received == 0) {
            engine::test::require(chunk.samples.size() >= 4000, "first chunk has prefix");
            engine::test::require(std::all_of(chunk.samples.begin(), chunk.samples.begin() + 4000, [](float v) { return v == 0; }), "prefix is silence");
        }
        received += static_cast<int64_t>(chunk.samples.size());
        if (std::exchange(fail, false)) throw std::runtime_error("injected failure");
        rt::StreamEvent event; event.word_timestamps = {word(4000, received)};
        if (sink) sink(event);
        return event;
    }
    rt::TaskResult finalize() override { rt::TaskResult out; out.word_timestamps = {word(4000, received)}; return out; }
};
void offline_test() {
    auto implementation = std::make_unique<Offline>(); auto * raw = implementation.get();
    auto session = bark::wrap_bark3_offline(std::move(implementation));
    rt::TaskRequest request; request.audio_input = rt::AudioBuffer{16000, 1, std::vector<float>(16000, .2f)};
    rejects([&] { session->run(request); });
    rt::SessionPreparationRequest preparation; preparation.audio = rt::AudioPreparationContract{16000, 1, 16000};
    session->prepare(preparation);
    engine::test::require_eq(raw->preparation.audio->max_input_samples, int64_t(20000), "reserve includes prefix");
    auto result = session->run(request);
    engine::test::require_eq(raw->input.samples.size(), size_t(20000), "pad once");
    engine::test::require_eq(request.audio_input->samples.size(), size_t(16000), "caller input remains intact");
    engine::test::require_eq(result.word_timestamps[0].span.start_sample, int64_t(0), "clamp negative time");
    engine::test::require_eq(result.word_timestamps[0].span.end_sample, int64_t(1000), "remove exact 250 ms");
    engine::test::require_eq(result.word_timestamps[1].span.end_sample, int64_t(16000), "clamp to original end");
    raw->fail = true; rejects([&] { session->run(request); }); session->run(request);
    engine::test::require_eq(raw->input.samples.size(), size_t(20000), "failure/reuse does not accumulate padding");
    auto vad = request; vad.options["audio_chunk_mode"] = "vad";
    session->run(vad);
    engine::test::require_eq(raw->input.samples.size(), size_t(20000), "offline VAD dispatch receives one recording prefix");
    session->run(request);
    auto bad = request; bad.audio_input->samples.clear(); rejects([&] { session->run(bad); });
    bad = request; bad.audio_input->channels = 3; rejects([&] { session->run(bad); });
    for (const auto * duration : {"0", "-1", "nan", "inf"}) {
        bad = request; bad.options["audio_chunk_duration_sec"] = duration;
        bool invalid = false;
        try { session->run(bad); } catch (const rt::InvalidRequestError &) { invalid = true; }
        engine::test::require(invalid, "invalid VAD duration is a client error");
        session->run(request);
    }
    auto stereo = request; stereo.audio_input = rt::AudioBuffer{8000, 2, std::vector<float>(4000, .2f)};
    session->run(stereo);
    engine::test::require_eq(raw->input.sample_rate, 16000, "resample before padding");
    engine::test::require_eq(raw->input.channels, 1, "downmix before padding");
    engine::test::require_eq(raw->input.samples.size(), size_t(8000), "prefix is 4000 target samples");
    preparation.audio->max_input_samples = std::numeric_limits<int64_t>::max();
    rejects([&] { session->prepare(preparation); }); rejects([&] { session->run(request); });
}
void streaming_test() {
    auto implementation = std::make_unique<Streaming>(); auto * raw = implementation.get();
    auto session = bark::wrap_bark3_streaming(std::move(implementation));
    session->prepare({});
    int64_t callback_end = -1;
    session->set_stream_event_sink([&](const rt::StreamEvent & event) { callback_end = event.word_timestamps[0].span.end_sample; });
    rt::TaskRequest vad; vad.options["audio_chunk_mode"] = "vad";
    rejects([&] { session->start_stream(vad); });
    session->start_stream({});
    rejects([&] { session->prepare({}); });
    auto event = session->process_audio_chunk({16000, 1, 0, std::vector<float>(100, .2f)});
    engine::test::require_eq(event.word_timestamps[0].span.end_sample, int64_t(100), "return timestamp shifted once");
    engine::test::require_eq(callback_end, int64_t(100), "callback timestamp shifted once");
    rejects([&] { session->process_audio_chunk({16000, 1, 99, {.2f}}); });
    session->process_audio_chunk({16000, 1, 100, std::vector<float>(50, .2f)});
    engine::test::require_eq(raw->received, int64_t(4150), "one prefix across chunks");
    engine::test::require_eq(session->finalize().word_timestamps[0].span.end_sample, int64_t(150), "final timestamps on original clock");
    rejects([&] { session->finalize(); });
    session->start_stream({}); rejects([&] { session->finalize(); });
    session->start_stream({}); raw->fail = true;
    rejects([&] { session->process_audio_chunk({16000, 1, 0, {.2f}}); });
    rejects([&] { session->process_audio_chunk({16000, 1, 1, {.2f}}); });
    session->start_stream({}); session->process_audio_chunk({16000, 1, 0, {.2f}}); session->finalize();
    engine::test::require_eq(raw->received, int64_t(4001), "failed stream reset recovers");
}
}  // namespace
int main() {
    try { allocation_test(); offline_test(); streaming_test(); std::cout << "PASS Bark allocation lifetimes, prefix, sample clocks, repeated requests, resampling, streaming and failure recovery\n"; }
    catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
