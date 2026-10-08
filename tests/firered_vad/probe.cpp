#include "engine/models/firered_vad/runtime.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/debug/trace.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>

using namespace engine;

void write_floats(const std::string & path, const std::vector<float> & values) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
    if (!file) throw std::runtime_error("Cannot write " + path);
}

std::vector<float> read_floats(const std::string & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() % sizeof(float)) throw std::runtime_error("Cannot read " + path);
    std::vector<float> values(static_cast<size_t>(file.tellg()) / sizeof(float));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(values.data()), values.size() * sizeof(float));
    if (!file) throw std::runtime_error("Cannot read " + path);
    return values;
}

template<class F> double median_ms(F fn) {
    for (int i = 0; i < 5; ++i) fn();
    std::vector<double> times;
    for (int i = 0; i < 21; ++i) {
        auto started = std::chrono::steady_clock::now();
        fn();
        times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

int main(int argc, char ** argv) {
    try {
        const bool log = argc > 1 && std::string(argv[argc - 1]) == "--log";
        if (log) --argc;
        debug::configure_logging({log});
        if (argc != 7 && argc != 8) throw std::runtime_error("Usage: probe MODEL AUDIO cpu|metal|cuda|vulkan THREADS OUTPUT_PREFIX REFERENCE_PREFIX [MAX_SPEECH_FRAMES] [--log]");
        auto bundle = model_spec::load_resource_bundle_for_family(argv[1], "firered_vad");
        core::BackendConfig backend;
        const std::string name = argv[3];
        if (name == "cpu") backend.type = core::BackendType::Cpu;
        else if (name == "metal") backend.type = core::BackendType::Metal;
        else if (name == "cuda") backend.type = core::BackendType::Cuda;
        else if (name == "vulkan") backend.type = core::BackendType::Vulkan;
        else throw std::runtime_error("Unsupported backend: " + name);
        backend.threads = std::stoi(argv[4]);
        core::ExecutionContext execution(backend);
        models::firered_vad::FireRedDfsmnRuntime model(bundle.open_tensor_source("weights"), execution);
        auto wav = audio::read_wav_f32(std::filesystem::path(argv[2]));
        runtime::AudioBuffer input{wav.sample_rate, wav.channels, std::move(wav.samples)};
        const std::string output = argv[5], reference = argv[6];
        models::firered_vad::DetectionOptions options;
        if (model.causal()) { options.threshold = 0.5; options.min_speech_frames = 8; }
        if (argc == 8) options.max_speech_frames = std::stoi(argv[7]);
        const auto cold_start = std::chrono::steady_clock::now();
        auto result = model.detect(input, options);
        const double cold_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cold_start).count();
        auto features = model.extract_features(input);
        write_floats(output + ".features.f32", features);
        write_floats(output + ".probs.f32", model.infer_features(features));
        auto reference_features = read_floats(reference + ".features.f32");
        if (reference_features.size() != features.size()) throw std::runtime_error("Frontend frame count mismatch");
        write_floats(output + ".reference_features_probs.f32", model.infer_features(reference_features));
        io::json::Value::Array segments;
        for (const auto & segment : result.speech_segments)
            segments.push_back(io::json::Value::make_array({io::json::Value::make_number(segment.span.start_sample / 16000.0), io::json::Value::make_number(segment.span.end_sample / 16000.0)}));
        std::ofstream(output + ".segments.json") << io::json::stringify(io::json::Value::make_array(std::move(segments))) << '\n';
        if (model.causal()) {
            model.reset(options);
            std::vector<float> chunked;
            for (size_t start = 0; start < reference_features.size(); start += 16 * 80) {
                const size_t end = std::min(reference_features.size(), start + 16 * 80);
                auto values = model.infer_features({reference_features.begin() + start, reference_features.begin() + end}, true);
                chunked.insert(chunked.end(), values.begin(), values.end());
            }
            write_floats(output + ".chunked_probs.f32", chunked);
            const auto full = model.infer_features(reference_features);
            for (size_t chunk_frames : {3, 4, 15, 16, 17, 31, 32, 33, 63, 64, 65, 255, 256, 257}) {
                model.reset(options);
                for (size_t start = 0; start < reference_features.size(); start += chunk_frames * 80) {
                    const size_t end = std::min(reference_features.size(), start + chunk_frames * 80);
                    const auto values = model.infer_features({reference_features.begin() + start, reference_features.begin() + end}, true);
                    for (size_t frame = 0; frame < values.size(); ++frame)
                        if (std::abs(values[frame] - full[start / 80 + frame]) > 0.002f)
                            throw std::runtime_error("Streaming probability mismatch at convolution block boundary");
                }
            }
            std::cout << "block_boundary_parity=passed\n";
            model.reset(options);
            const auto stream_start = std::chrono::steady_clock::now();
            double first_event_ms = -1, worst_chunk_ms = 0;
            size_t first_event_samples = 0;
            for (size_t start = 0; start < input.samples.size(); start += 2560) {
                const size_t end = std::min(input.samples.size(), start + 2560);
                const auto chunk_start = std::chrono::steady_clock::now();
                auto event = model.process({16000, 1, static_cast<int64_t>(start), {input.samples.begin() + start, input.samples.begin() + end}});
                worst_chunk_ms = std::max(worst_chunk_ms, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - chunk_start).count());
                if (first_event_ms < 0 && !event.voice_activity.empty()) {
                    first_event_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stream_start).count();
                    first_event_samples = end;
                }
            }
            auto streamed = model.finalize();
            if (streamed.speech_segments.size() != result.speech_segments.size()) throw std::runtime_error("Streaming segment count mismatch");
            for (size_t i = 0; i < streamed.speech_segments.size(); ++i)
                if (streamed.speech_segments[i].span.start_sample != result.speech_segments[i].span.start_sample ||
                    streamed.speech_segments[i].span.end_sample != result.speech_segments[i].span.end_sample)
                    throw std::runtime_error("Streaming timestamp mismatch");
            model.reset(options);
            const std::array<size_t, 6> packets{1, 399, 160, 640, 2560, 7777};
            for (size_t start = 0, packet = 0; start < input.samples.size(); ++packet) {
                const size_t end = std::min(input.samples.size(), start + packets[packet % packets.size()]);
                model.process({16000, 1, static_cast<int64_t>(start), {input.samples.begin() + start, input.samples.begin() + end}});
                start = end;
            }
            streamed = model.finalize();
            if (streamed.speech_segments.size() != result.speech_segments.size()) throw std::runtime_error("Irregular streaming segment count mismatch");
            for (size_t i = 0; i < streamed.speech_segments.size(); ++i)
                if (streamed.speech_segments[i].span.start_sample != result.speech_segments[i].span.start_sample ||
                    streamed.speech_segments[i].span.end_sample != result.speech_segments[i].span.end_sample)
                    throw std::runtime_error("Irregular streaming timestamp mismatch");
            std::cout << "streaming_parity=passed first_event_compute_ms=" << first_event_ms
                      << " first_event_input_sec=" << first_event_samples / 16000.0
                      << " worst_chunk_ms=" << worst_chunk_ms << '\n';
        }
        if (!model.detect({16000, 1, {}}, options).speech_segments.empty() ||
            !model.detect({16000, 1, std::vector<float>(399, 0.1f)}, options).speech_segments.empty() ||
            !model.detect({16000, 1, std::vector<float>(16000, 0.0f)}, options).speech_segments.empty())
            throw std::runtime_error("Empty, short or silent audio produced speech");
        bool rejected = false;
        try { model.extract_features({8000, 1, {0.1f}}); } catch (const std::runtime_error &) { rejected = true; }
        if (!rejected) throw std::runtime_error("Wrong sample rate was accepted");
        rejected = false;
        try { model.extract_features({16000, 1, {std::numeric_limits<float>::quiet_NaN()}}); } catch (const std::runtime_error &) { rejected = true; }
        if (!rejected) throw std::runtime_error("Non-finite audio was accepted");
        std::cout << "input_validation=passed\n";
        const double frontend = median_ms([&] { model.extract_features(input); });
        const double network = median_ms([&] { model.infer_features(features); });
        const double total = median_ms([&] { model.detect(input, options); });
        debug::timing_log_scalar("session.wall_ms", total);
        std::cout << "frames=" << features.size() / 80 << " backend=" << argv[3] << " threads=" << backend.threads
                  << " cold_ms=" << cold_ms << " frontend_ms=" << frontend << " network_ms=" << network << " total_ms=" << total
                  << " rtf=" << total * 16.0 / input.samples.size() << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
