#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/registry.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>

int main(int argc, char ** argv) {
    try {
        if (argc != 8 || std::string(argv[6]) != "--log")
            throw std::runtime_error("Usage: mossformer2_validation MODEL BACKEND sep|s2s CASES.json OUTPUT_DIR --log FILE");
        namespace rt = engine::runtime;
        namespace json = engine::io::json;
        namespace audio = engine::audio;
        engine::debug::configure_logging({true, argv[7]});
        auto registry = rt::make_default_registry();
        rt::ModelLoadRequest load;
        load.model_path = argv[1];
        load.family_hint = "mossformer2";
        auto model = registry.load(load);
        rt::SessionOptions options;
        const std::string backend = argv[2], task = argv[3];
        if (backend == "cuda") options.backend.type = engine::core::BackendType::Cuda;
        else if (backend == "vulkan") options.backend.type = engine::core::BackendType::Vulkan;
        else if (backend != "cpu") throw std::runtime_error("Unknown validation backend");
        options.backend.threads = 8;
        if (task != "sep" && task != "s2s") throw std::runtime_error("Unknown validation task");
        auto base = model->create_task_session({task == "sep" ? rt::VoiceTaskKind::SourceSeparation : rt::VoiceTaskKind::SpeechToSpeech,
            rt::RunMode::Offline}, options);
        base->prepare({});
        auto & session = dynamic_cast<rt::IOfflineVoiceTaskSession &>(*base);
        const std::filesystem::path cases_path = argv[4], output = argv[5];
        std::filesystem::create_directories(output);
        std::ofstream metrics(output / "metrics.csv");
        metrics << "case,audio_sec,wall_ms,rtf\n";
        std::map<std::string, std::vector<float>> previous;
        const auto cases = json::parse_file(cases_path);
        for (const auto & item : cases.as_array()) {
            const auto id = json::require_string(item, "id");
            const auto input_path = cases_path.parent_path() / json::require_string(item, "audio");
            const auto input = audio::read_wav_f32(input_path);
            rt::TaskRequest request;
            request.audio_input = rt::AudioBuffer{input.sample_rate, input.channels, input.samples};
            if (const auto * settings = item.find("options"))
                for (const auto & [key, value] : settings->as_object())
                    request.options[key] = value.is_string() ? value.as_string() : json::stringify(value);
            const auto start = std::chrono::steady_clock::now();
            auto result = session.run(request);
            const auto wall = engine::debug::elapsed_ms(start);
            std::vector<rt::NamedAudioBuffer> tracks = std::move(result.named_audio_outputs);
            if (result.audio_output) tracks.push_back({"restored", std::move(*result.audio_output), {}});
            if (tracks.empty()) throw std::runtime_error("No model output");
            double duration = 0;
            for (const auto & track : tracks) {
                const auto & buffer = track.audio;
                if (buffer.channels != 1 || buffer.samples.empty()) throw std::runtime_error("Invalid model output format");
                double energy = 0;
                for (auto sample : buffer.samples) {
                    if (!std::isfinite(sample)) throw std::runtime_error("Non-finite model output");
                    energy += double(sample) * sample;
                }
                const bool expect_input = json::optional_bool(item, "expect_input_equal", false);
                if (expect_input && (buffer.sample_rate != input.sample_rate || buffer.channels != input.channels ||
                                     buffer.samples != input.samples))
                    throw std::runtime_error("Expected unchanged input audio");
                if (energy == 0 && !expect_input) throw std::runtime_error("Silent model output");
                if (json::optional_bool(item, "require_signal_at_end", false) && buffer.samples.back() == 0)
                    throw std::runtime_error("Model output lost the final signal sample");
                duration = static_cast<double>(buffer.samples.size()) / buffer.sample_rate;
                const auto key = input_path.string() + json::stringify(item.find("options") ? *item.find("options") : json::Value::make_object({})) + track.id;
                const auto found = previous.find(key);
                if (found != previous.end() && found->second != buffer.samples) {
                    if (task == "sep" || found->second.size() != buffer.samples.size())
                        throw std::runtime_error("Repeated request output changed");
                    // SE's upstream frontend dithers every call; require numerical
                    // stability rather than identical random samples for s2s.
                    double dot = 0, previous_energy = 0;
                    for (size_t i = 0; i < buffer.samples.size(); ++i) {
                        dot += double(found->second[i]) * buffer.samples[i];
                        previous_energy += double(found->second[i]) * found->second[i];
                    }
                    if (previous_energy == 0 || energy == 0 || dot / std::sqrt(previous_energy * energy) < 0.99999)
                        throw std::runtime_error("Repeated request waveform similarity regressed");
                }
                previous[key] = buffer.samples;
                audio::WavWriteOptions format;
                format.format = audio::WavSampleFormat::Float32;
                audio::write_wav(output / (id + (task == "sep" ? "_" + track.id : "") + ".wav"),
                    buffer.sample_rate, buffer.channels, buffer.samples, format);
            }
            metrics << id << ',' << duration << ',' << wall << ',' << wall / (1000 * duration) << '\n';
            metrics.flush();
            std::cout << id << " session.wall_ms=" << wall << " audio.duration_sec=" << duration
                      << " RTF=" << wall / (1000 * duration) << '\n';
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
