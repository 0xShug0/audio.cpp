#include "engine/community_models/parakeet_tdt/hotwords.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace j = engine::io::json;
using namespace engine::runtime;
using namespace engine::community_models::parakeet_tdt;
j::Value words_json(const std::vector<WordTimestamp> & words) {
    j::Value::Array out;
    for (const auto & w : words) out.push_back(j::Value::make_object({
        {"word",j::Value::make_string(w.word)}, {"start_sample",j::Value::make_number(w.span.start_sample)},
        {"end_sample",j::Value::make_number(w.span.end_sample)}}));
    return j::Value::make_array(std::move(out));
}
int main(int argc, char ** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: phonon_features_probe input.json");
        const auto input = j::parse_file(argv[1]);
        if (const auto * queries = input.find("queries")) {
            std::vector<std::string> vocab;
            for (const auto & v : input.require("vocabulary").as_array()) vocab.push_back(v.as_string());
            PhononHotwordAutomaton a(parse_phonon_hotwords(j::stringify(input.require("hotwords"))), vocab,
                static_cast<int32_t>(input.require("blank").as_i64()), input.require("strength").as_f32());
            j::Value::Array rows;
            for (const auto & query : queries->as_array()) {
                int32_t state = 0;
                for (const auto & t : query.require("prefix").as_array()) state = a.step(state, static_cast<int32_t>(t.as_i64()));
                j::Value::Array bonuses;
                for (size_t t = 0; t < vocab.size(); ++t) bonuses.push_back(j::Value::make_number(a.bonus(state, static_cast<int32_t>(t))));
                rows.push_back(j::Value::make_object({{"state",j::Value::make_number(state)}, {"bonus",j::Value::make_array(std::move(bonuses))}}));
            }
            std::cout << j::stringify(j::Value::make_object({{"states",j::Value::make_number(a.states())}, {"rows",j::Value::make_array(std::move(rows))}}));
            return 0;
        }
        auto registry = make_default_registry();
        ModelLoadRequest load; load.family_hint = "parakeet_tdt"; load.model_path = input.require("model").as_string();
        auto model = registry.load(load);
        SessionOptions options; options.backend.threads = 8;
        const auto backend = input.require("backend").as_string();
        options.backend.type = backend == "cuda" ? engine::core::BackendType::Cuda :
            backend == "vulkan" ? engine::core::BackendType::Vulkan : engine::core::BackendType::Cpu;
        if (const auto * device = input.find("device")) options.backend.device = static_cast<int>(device->as_i64());
        const bool streaming = input.require("streaming").as_bool();
        auto session = model->create_task_session({VoiceTaskKind::Asr, streaming ? RunMode::Streaming : RunMode::Offline}, options);
        const auto wav = engine::audio::read_wav_f32(std::filesystem::path(input.require("audio").as_string()));
        TaskRequest request; request.audio_input = AudioBuffer{wav.sample_rate, wav.channels, wav.samples};
        if (const auto * terms = input.find("hotwords")) request.options["hotwords"] = j::stringify(*terms);
        if (const auto * strength = input.find("strength")) request.options["hotword_lambda"] = j::stringify(*strength);
        if (const auto * cap = input.find("max_tokens")) request.options["max_tokens"] = j::stringify(*cap);
        session->prepare(build_preparation_request(request));
        auto start = std::chrono::steady_clock::now();
        j::Value::Array events;
        TaskResult result;
        if (streaming) {
            auto * stream = dynamic_cast<IStreamingVoiceTaskSession *>(session.get());
            stream->set_stream_event_sink([&](const StreamEvent & e) {
                j::Value::Object row;
                if (e.partial_text) row["delta"] = j::Value::make_string(e.partial_text->text);
                if (e.partial_text_snapshot) row["snapshot"] = j::Value::make_string(e.partial_text_snapshot->text);
                row["words"] = words_json(e.word_timestamps);
                events.push_back(j::Value::make_object(std::move(row)));
            });
            stream->start_stream(request);
            size_t chunk = 800;
            if (const auto * size = input.find("chunk_samples")) chunk = static_cast<size_t>(size->as_i64());
            if (!chunk) throw std::runtime_error("chunk size must be positive");
            for (size_t offset = 0; offset < wav.samples.size(); offset += chunk) {
                auto end = std::min(offset + chunk, wav.samples.size());
                stream->process_audio_chunk({wav.sample_rate, wav.channels, static_cast<int64_t>(offset),
                    std::vector<float>(wav.samples.begin() + offset, wav.samples.begin() + end)});
            }
            result = stream->finalize();
        } else result = dynamic_cast<IOfflineVoiceTaskSession *>(session.get())->run(request);
        if (const auto * lifecycle = input.find("lifecycle"); streaming && lifecycle && lifecycle->as_bool()) {
            auto * stream = dynamic_cast<IStreamingVoiceTaskSession *>(session.get());
            const auto expected = result.text_output->text;
            const auto expected_words = j::stringify(words_json(result.word_timestamps));
            auto rejected = [](auto operation) {
                bool failed = false;
                try { operation(); } catch (const std::exception &) { failed = true; }
                if (!failed) throw std::runtime_error("expected lifecycle rejection");
            };
            rejected([&] { stream->finalize(); });
            rejected([&] { stream->process_audio_chunk({16000, 1, 0, {0.f}}); });
            auto replay = [&](size_t chunk) {
                for (size_t offset = 0; offset < wav.samples.size(); offset += chunk) {
                    const auto end = std::min(offset + chunk, wav.samples.size());
                    stream->process_audio_chunk({wav.sample_rate, wav.channels, static_cast<int64_t>(offset),
                        std::vector<float>(wav.samples.begin() + offset, wav.samples.begin() + end)});
                }
                const auto repeated = stream->finalize();
                if (repeated.text_output->text != expected || j::stringify(words_json(repeated.word_timestamps)) != expected_words)
                    throw std::runtime_error("stream chunk/reuse result changed");
            };
            stream->set_stream_event_sink({});
            for (auto chunk : {size_t{13}, size_t{2049}, wav.samples.size()}) {
                stream->start_stream(request);
                rejected([&] { stream->process_audio_chunk({16000, 1, 1, {0.f}}); });
                replay(chunk); // Bad ordering is validated before mutation.
            }
            stream->start_stream(request);
            rejected([&] { stream->process_audio_chunk({16000, 1, 0, std::vector<float>(800, std::numeric_limits<float>::quiet_NaN())}); });
            rejected([&] { stream->finalize(); });
            stream->start_stream(request); replay(800);
            stream->set_stream_event_sink([](const StreamEvent &) { throw std::runtime_error("injected sink failure"); });
            stream->start_stream(request);
            rejected([&] { replay(800); });
            stream->set_stream_event_sink({});
            stream->start_stream(request); replay(800);
            // Idle audio must remain bounded, yield no hallucinated segments,
            // and permit immediate reuse after finalization.
            stream->start_stream(request);
            stream->process_audio_chunk({16000, 1, 0, std::vector<float>(33 * 16000, 0.f)});
            const auto silent = stream->finalize();
            if (!silent.text_output->text.empty() || !silent.word_timestamps.empty()) throw std::runtime_error("silent stream produced words");
            stream->start_stream(request); replay(800);
        }
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << j::stringify(j::Value::make_object({{"text",j::Value::make_string(result.text_output->text)},
            {"words",words_json(result.word_timestamps)}, {"events",j::Value::make_array(std::move(events))}, {"seconds",j::Value::make_number(elapsed)}})) << '\n';
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
