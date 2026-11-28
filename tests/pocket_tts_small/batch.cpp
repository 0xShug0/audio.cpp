#include "app/cli/args.h"
#include "app/streaming/streaming.h"
#include "engine/models/pocket_tts_small/session.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/json.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>

int main(int argc, char ** argv) {
    using namespace engine;
    try {
        const auto model = minitts::cli::find_arg(argc, argv, "--model").value();
        const auto requests_file = minitts::cli::find_arg(argc, argv, "--requests").value();
        const auto out = std::filesystem::path(minitts::cli::find_arg(argc, argv, "--out-dir").value());
        const auto log = minitts::cli::find_arg(argc, argv, "--log-file").value();
        const auto backend = minitts::cli::find_arg(argc, argv, "--backend").value_or("cuda");
        const int batch_size = minitts::cli::parse_int_arg(argc, argv, "--batch-size", 4);
        const int threads = minitts::cli::parse_int_arg(argc, argv, "--threads", 8);
        const int seed = minitts::cli::parse_int_arg(argc, argv, "--seed", 42);
        const auto mode = minitts::cli::find_arg(argc, argv, "--mode").value_or("offline");
        if (mode != "offline" && mode != "streaming") throw std::runtime_error("Expected offline or streaming mode");
        const bool streaming = mode == "streaming";
        const bool sequential = minitts::cli::has_arg(argc, argv, "--sequential");
        const auto session_options = minitts::cli::find_arg(argc, argv, "--session-options");
        if (!minitts::cli::has_arg(argc, argv, "--log") || batch_size < 1 || threads < 1)
            throw std::runtime_error("Require --log and positive batch size/threads");
        minitts::cli::require_known_args(argc, argv);
        debug::configure_logging({true, log});
        std::filesystem::create_directories(out);
        const auto fixture = io::json::parse_file(requests_file);
        std::vector<runtime::TaskRequest> requests;
        for (const auto & item : fixture.as_array()) {
            runtime::TaskRequest request;
            request.text_input = runtime::Transcript{item.require("text").as_string(), "en"};
            runtime::VoiceReference speaker;
            speaker.cached_voice_id = item.require("voice_id").as_string();
            request.voice = runtime::VoiceCondition{speaker, std::nullopt};
            if (const auto options = item.find("options")) {
                for (const auto & [key, value] : options->as_object())
                    request.options[key] = value.is_string() ? value.as_string() : io::json::stringify(value);
            }
            requests.push_back(std::move(request));
        }
        if (requests.empty()) throw std::runtime_error("Require at least one request");
        runtime::ModelRegistry registry;
        registry.register_loader(models::pocket_tts_small::make_pocket_tts_small_loader());
        runtime::ModelLoadRequest load;
        load.model_path = model;
        load.family_hint = "pocket_tts_small";
        auto loaded = registry.load(load);
        runtime::SessionOptions options;
        options.backend = {minitts::cli::parse_backend(backend), 0, threads};
        if (session_options) {
            const auto parsed = io::json::parse(*session_options);
            for (const auto & [key, value] : parsed.as_object())
                options.options[key] = value.is_string() ? value.as_string() : io::json::stringify(value);
        }
        auto owned = loaded->create_task_session({runtime::VoiceTaskKind::Tts,
            streaming ? runtime::RunMode::Streaming : runtime::RunMode::Offline}, options);
        auto * single = dynamic_cast<runtime::IOfflineVoiceTaskSession *>(owned.get());
        auto * batched = dynamic_cast<runtime::IBatchedOfflineVoiceTaskSession *>(owned.get());
        auto * stream = dynamic_cast<runtime::IStreamingVoiceTaskSession *>(owned.get());
        if (streaming ? stream == nullptr : (!single || !batched)) throw std::runtime_error("Missing requested session interfaces");
        runtime::SessionPreparationRequest preparation;
        preparation.voice = requests.front().voice;
        preparation.options["seed"] = std::to_string(seed);
        owned->prepare(preparation);
        double wall_ms = 0, seconds = 0;
        size_t completed = 0;
        audio::WavWriteOptions wav_options;
        wav_options.format = audio::WavSampleFormat::Float32;
        std::string prepared_voice = *requests.front().voice->speaker->cached_voice_id;
        auto prepared_options = preparation.options;
        for (int pass = 0; pass < 2; ++pass) {
            debug::trace_log_scalar("pocket_tts_small.test.measured_pass", pass);
            for (size_t start = 0; start < requests.size(); start += (sequential || streaming) ? 1 : batch_size) {
                const auto end = std::min(requests.size(), start + ((sequential || streaming) ? 1 : batch_size));
                std::vector<runtime::TaskRequest> group(requests.begin() + start, requests.begin() + end);
                const auto started = std::chrono::steady_clock::now();
                std::vector<runtime::TaskResult> results;
                std::optional<double> ttfa_ms;
                size_t audio_events = 0;
                if (sequential || streaming) {
                    auto row_options = group.front().options;
                    row_options.try_emplace("seed", std::to_string(seed));
                    const auto & voice = *group.front().voice->speaker->cached_voice_id;
                    if (voice != prepared_voice || row_options != prepared_options) {
                        preparation.voice = group.front().voice;
                        preparation.options = row_options;
                        owned->prepare(preparation);
                        prepared_voice = voice;
                        prepared_options = std::move(row_options);
                    }
                    if (streaming) {
                        runtime::AudioBuffer emitted{24000, 1, {}};
                        auto result = minitts::app::run_streaming_task(*stream, group.front(),
                            [&](const runtime::StreamEvent & event) {
                                const bool has_audio = (event.audio_output && !event.audio_output->samples.empty()) ||
                                    std::any_of(event.named_audio_outputs.begin(), event.named_audio_outputs.end(),
                                        [](const auto & output) { return !output.audio.samples.empty(); });
                                if (!has_audio) return;
                                if (!ttfa_ms) ttfa_ms = debug::elapsed_ms(started);
                                ++audio_events;
                                if (event.audio_output && !event.audio_output->samples.empty())
                                    runtime::append_audio_buffer(emitted, *event.audio_output);
                                for (const auto & output : event.named_audio_outputs)
                                    if (!output.audio.samples.empty()) runtime::append_audio_buffer(emitted, output.audio);
                            });
                        if (!ttfa_ms || !result.audio_output || emitted.samples != result.audio_output->samples)
                            throw std::runtime_error("Streaming events differ from the final waveform or contain no audio");
                        if (result.audio_output->samples.size() > 1920 && audio_events < 2)
                            throw std::runtime_error("Streaming did not emit incremental audio");
                        results.push_back(std::move(result));
                    } else {
                        results.push_back(single->run(group.front()));
                    }
                } else {
                    results = batched->run_batch(group);
                }
                const auto elapsed = debug::elapsed_ms(started);
                if (results.size() != group.size()) throw std::runtime_error("Batch result count differs");
                debug::trace_log_scalar("pocket_tts_small.test.request_index", static_cast<int64_t>(start));
                if (ttfa_ms) {
                    debug::timing_log_scalar("session.ttfa_ms", *ttfa_ms);
                    debug::timing_log_scalar("pocket_tts_small.test.request_wall_ms", elapsed);
                    debug::trace_log_scalar("pocket_tts_small.test.audio_events", static_cast<int64_t>(audio_events));
                }
                if (pass == 0) continue;
                wall_ms += elapsed;
                for (size_t row = 0; row < results.size(); ++row) {
                    if (!results[row].audio_output) throw std::runtime_error("Missing batch waveform");
                    const auto & wav = *results[row].audio_output;
                    if (wav.samples.empty() || wav.sample_rate != 24000) throw std::runtime_error("Invalid batch waveform shape");
                    for (const float value : wav.samples) if (!std::isfinite(value)) throw std::runtime_error("Nonfinite batch waveform");
                    seconds += double(wav.samples.size()) / wav.sample_rate;
                    audio::write_wav(out / (std::to_string(start + row) + ".wav"), wav.sample_rate, wav.channels,
                        wav.samples, wav_options);
                    ++completed;
                }
            }
        }
        if (!sequential && !streaming) {
            const auto result = single->run(requests.front());
            if (!result.audio_output || result.audio_output->samples.empty())
                throw std::runtime_error("Single request after batching produced no audio");
            const auto & wav = *result.audio_output;
            for (float value : wav.samples) if (!std::isfinite(value))
                throw std::runtime_error("Single request after batching produced nonfinite audio");
            std::filesystem::create_directories(out / "single-after-batch");
            audio::write_wav(out / "single-after-batch" / "0.wav", wav.sample_rate, wav.channels,
                wav.samples, wav_options);
            debug::trace_log_scalar("pocket_tts_small.test.single_after_batch", 1);
        }
        debug::timing_log_scalar("session.wall_ms", wall_ms);
        debug::trace_log_scalar("pocket_tts_small.test.audio_seconds", seconds);
        debug::trace_log_scalar("pocket_tts_small.test.rtf", wall_ms / (1000 * seconds));
        std::cout << "requests=" << completed << " seconds=" << seconds << " wall_ms=" << wall_ms
                  << " rtf=" << wall_ms / (1000 * seconds) << '\n';
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
