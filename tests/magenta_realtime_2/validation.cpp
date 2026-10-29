#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/core/host_memory.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/registry.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

int main(int argc, char ** argv) {
    try {
        if (argc != 8 || std::string(argv[6]) != "--log") throw std::runtime_error(
            "Usage: magenta_validation MODEL.gguf cpu|cuda|vulkan offline|streaming CASES.json OUTPUT_DIR --log FILE");
        engine::debug::configure_logging({true, argv[7]});
        namespace rt = engine::runtime;
        namespace json = engine::io::json;
        auto registry = rt::make_default_registry();
        rt::ModelLoadRequest load;
        load.model_path = argv[1];
        load.family_hint = "magenta_realtime_2";
        const auto model = registry.load(load);
        rt::SessionOptions options;
        const std::string backend = argv[2], mode = argv[3];
        if (backend == "cuda") options.backend.type = engine::core::BackendType::Cuda;
        else if (backend == "vulkan") options.backend.type = engine::core::BackendType::Vulkan;
        else if (backend != "cpu") throw std::runtime_error("Unknown backend");
        options.backend.threads = 8;
        if (mode != "offline" && mode != "streaming") throw std::runtime_error("Unknown mode");
        const bool streaming = mode == "streaming";
        auto base = model->create_task_session({rt::VoiceTaskKind::AudioGeneration,
            streaming ? rt::RunMode::Streaming : rt::RunMode::Offline}, options);
        base->prepare({});
        const auto output_directory = std::filesystem::path(argv[5]);
        std::filesystem::create_directories(output_directory);
        std::ofstream metrics(output_directory / "metrics.csv");
        metrics << "case,audio_sec,wall_ms,rtf,events\n";
        const auto cases = json::parse_file(argv[4]);
        std::map<std::pair<std::string, std::map<std::string, std::string>>, std::vector<float>> outputs;
        for (const auto & item : cases.as_array()) {
            const auto id = json::require_string(item, "id"), text = json::require_string(item, "text");
            rt::TaskRequest request;
            request.text_input = rt::Transcript{text, "auto"};
            request.options = {{"duration_sec", json::stringify(item.require("duration_sec"))},
                               {"seed", json::stringify(item.require("seed"))}};
            if (const auto * additions = item.find("options")) {
                for (const auto & [key, value] : additions->as_object()) request.options[key] = value.is_string() ? value.as_string() : json::stringify(value);
            }
            std::vector<float> events;
            size_t event_count = 0;
            const auto start = std::chrono::steady_clock::now();
            rt::TaskResult result;
            if (streaming) {
                auto & session = dynamic_cast<rt::IStreamingVoiceTaskSession &>(*base);
                session.set_stream_event_sink([&](const rt::StreamEvent & event) {
                    if (event.audio_output) {
                        events.insert(events.end(), event.audio_output->samples.begin(), event.audio_output->samples.end());
                        ++event_count;
                    }
                });
                session.start_stream(request);
                result = session.finish_stream();
            } else result = dynamic_cast<rt::IOfflineVoiceTaskSession &>(*base).run(request);
            const double wall_ms = engine::debug::elapsed_ms(start);
            if (!result.audio_output || result.audio_output->channels != 2 || result.audio_output->sample_rate != 48000)
                throw std::runtime_error("Invalid Magenta output format");
            const auto & samples = result.audio_output->samples;
            const double audio_sec = static_cast<double>(samples.size()) / 96000;
            double energy = 0;
            for (float sample : samples) {
                if (!std::isfinite(sample)) throw std::runtime_error("Non-finite generated waveform");
                energy += static_cast<double>(sample) * sample;
            }
            if (energy / samples.size() < 1e-8) throw std::runtime_error("Generated waveform is silent");
            if (streaming && (events != samples || event_count == 0)) throw std::runtime_error("Streaming events differ from final waveform");
            engine::audio::WavWriteOptions wav_options;
            wav_options.format = engine::audio::WavSampleFormat::Float32;
            engine::audio::write_wav(output_directory / (id + ".wav"), 48000, 2, samples, wav_options);
            if (id != "warmup") {
                const auto key = std::make_pair(text, std::map<std::string, std::string>{request.options.begin(), request.options.end()});
                const auto previous = outputs.find(key);
                if (previous != outputs.end() && previous->second != samples) throw std::runtime_error("Repeated request waveform changed");
                outputs[key] = samples;
            }
            metrics << id << ',' << audio_sec << ',' << wall_ms << ',' << wall_ms / (audio_sec * 1000) << ',' << event_count << '\n';
            metrics.flush();
            std::cout << id << " session.wall_ms=" << wall_ms << " audio.duration_sec=" << audio_sec << " RTF=" << wall_ms / (1000 * audio_sec) << '\n';
        }
        const auto memory = engine::core::process_memory_peak();
        engine::debug::timing_log_scalar("process.peak_rss_mb", static_cast<double>(memory.rss_bytes) / (1024 * 1024));
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
