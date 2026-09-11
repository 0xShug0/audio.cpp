#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr const char * kDefaultWarmupText = "The warmup voice sounds steady and natural today.";
constexpr const char * kDefaultReferenceText =
    "Some call me nature. Others call me Mother Nature. I've been here for over 4.5 billion years. "
    "22,500 times longer than you.";

std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

std::vector<std::string> arg_values(int argc, char ** argv, const std::string & name) {
    std::vector<std::string> values;
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            values.push_back(argv[i + 1]);
        }
    }
    return values;
}

int int_arg(int argc, char ** argv, const std::string & name, int fallback) {
    return std::stoi(arg_value(argc, argv, name, std::to_string(fallback)));
}

float float_arg(int argc, char ** argv, const std::string & name, float fallback) {
    return std::stof(arg_value(argc, argv, name, std::to_string(fallback)));
}

bool bool_arg(int argc, char ** argv, const std::string & name, bool fallback) {
    const std::string value = arg_value(argc, argv, name, fallback ? "true" : "false");
    if (value == "1" || value == "true" || value == "yes") {
        return true;
    }
    if (value == "0" || value == "false" || value == "no") {
        return false;
    }
    throw std::runtime_error("invalid boolean value for " + name + ": " + value);
}

engine::core::BackendType parse_backend(const std::string & value) {
    if (value == "cpu") {
        return engine::core::BackendType::Cpu;
    }
    if (value == "cuda") {
        return engine::core::BackendType::Cuda;
    }
    if (value == "vulkan") {
        return engine::core::BackendType::Vulkan;
    }
    throw std::runtime_error("unsupported Qwen3 TTS warmbench backend: " + value);
}

std::string summary_json(const engine::runtime::TaskResult & result, double wall_ms) {
    if (!result.audio_output.has_value()) {
        throw std::runtime_error("Qwen3 TTS warmbench expected audio_output");
    }
    const auto & audio = *result.audio_output;
    double sum = 0.0;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    float min_value = audio.samples.empty() ? 0.0F : audio.samples.front();
    float max_value = audio.samples.empty() ? 0.0F : audio.samples.front();
    for (const float sample : audio.samples) {
        sum += sample;
        sum_abs += std::abs(sample);
        sum_sq += static_cast<double>(sample) * static_cast<double>(sample);
        min_value = std::min(min_value, sample);
        max_value = std::max(max_value, sample);
    }
    const double count = static_cast<double>(std::max<size_t>(1, audio.samples.size()));
    std::ostringstream out;
    out << std::setprecision(10);
    out << "{";
    out << "\"family\":\"qwen3_tts\",";
    out << "\"sample_rate\":" << audio.sample_rate << ",";
    out << "\"channels\":" << audio.channels << ",";
    out << "\"samples\":" << audio.samples.size() << ",";
    out << "\"duration_sec\":" << (audio.sample_rate > 0 ? static_cast<double>(audio.samples.size()) / audio.sample_rate : 0.0) << ",";
    out << "\"sum\":" << sum << ",";
    out << "\"mean_abs\":" << (sum_abs / count) << ",";
    out << "\"rms\":" << std::sqrt(sum_sq / count) << ",";
    out << "\"min\":" << min_value << ",";
    out << "\"max\":" << max_value << ",";
    out << "\"synthesize_wall_ms\":" << wall_ms;
    out << "}";
    return out.str();
}

void set_process_env(const char * key, const std::string & value) {
#if defined(_WIN32)
    if (_putenv_s(key, value.c_str()) != 0) {
        throw std::runtime_error("failed to set environment variable: " + std::string(key));
    }
#else
    if (setenv(key, value.c_str(), 1) != 0) {
        throw std::runtime_error("failed to set environment variable: " + std::string(key));
    }
#endif
}

engine::runtime::TaskRequest make_request(
    const std::string & text,
    const std::string & language,
    const engine::runtime::AudioBuffer * reference_audio,
    const std::string & reference_text,
    const std::string & voice_design_instruct,
    const std::string & speaker,
    int max_new_tokens,
    int seed,
    bool do_sample,
    bool subtalker_do_sample,
    int top_k,
    float top_p,
    float temperature,
    int subtalker_top_k,
    float subtalker_top_p,
    float subtalker_temperature) {
    engine::runtime::TaskRequest request;
    request.text_input = engine::runtime::Transcript{text, language};
    if (reference_audio != nullptr) {
        request.voice = engine::runtime::VoiceCondition{};
        request.voice->speaker = engine::runtime::VoiceReference{};
        request.voice->speaker->audio = *reference_audio;
        request.options["reference_text"] = reference_text;
    }
    if (!voice_design_instruct.empty()) {
        request.options["instruct"] = voice_design_instruct;
    }
    if (!speaker.empty()) {
        request.options["speaker"] = speaker;
    }
    request.options["max_new_tokens"] = std::to_string(max_new_tokens);
    request.options["seed"] = std::to_string(seed);
    request.options["do_sample"] = do_sample ? "true" : "false";
    request.options["subtalker_dosample"] = subtalker_do_sample ? "true" : "false";
    request.options["top_k"] = std::to_string(top_k);
    request.options["top_p"] = std::to_string(top_p);
    request.options["temperature"] = std::to_string(temperature);
    request.options["subtalker_top_k"] = std::to_string(subtalker_top_k);
    request.options["subtalker_top_p"] = std::to_string(subtalker_top_p);
    request.options["subtalker_temperature"] = std::to_string(subtalker_temperature);
    return request;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto model_path = std::filesystem::path(arg_value(argc, argv, "--model", "models/Qwen3-TTS-12Hz-0.6B-Base"));
        const auto backend_name = arg_value(argc, argv, "--backend", "cpu");
        const int device = int_arg(argc, argv, "--device", 0);
        const int threads = int_arg(argc, argv, "--threads", 4);
        const int warmup = int_arg(argc, argv, "--warmup", 1);
        const int iterations = int_arg(argc, argv, "--iterations", 1);
        const int max_new_tokens = int_arg(argc, argv, "--max-new-tokens", 512);
        const int seed = int_arg(argc, argv, "--seed", 1234);
        const bool do_sample = bool_arg(argc, argv, "--do-sample", true);
        const bool subtalker_do_sample = bool_arg(argc, argv, "--subtalker-do-sample", true);
        const int top_k = int_arg(argc, argv, "--top-k", 50);
        const float top_p = float_arg(argc, argv, "--top-p", 1.0F);
        const float temperature = float_arg(argc, argv, "--temperature", 0.9F);
        const int subtalker_top_k = int_arg(argc, argv, "--subtalker-top-k", 50);
        const float subtalker_top_p = float_arg(argc, argv, "--subtalker-top-p", 1.0F);
        const float subtalker_temperature = float_arg(argc, argv, "--subtalker-temperature", 0.9F);
        const std::string task_name = arg_value(argc, argv, "--task", "tts");
        const std::string language = arg_value(argc, argv, "--language", "Auto");
        const std::string speaker = arg_value(argc, argv, "--speaker", "");
        std::string warmup_speaker = arg_value(argc, argv, "--warmup-speaker", "");
        std::vector<std::string> request_speakers = arg_values(argc, argv, "--request-speaker");
        std::string voice_design_instruct = arg_value(argc, argv, "--instruct", "");
        if (voice_design_instruct.empty()) {
            voice_design_instruct = arg_value(argc, argv, "--voice-design-instruct", "");
        }
        std::string warmup_instruct = arg_value(argc, argv, "--warmup-instruct", "");
        std::vector<std::string> request_instructs = arg_values(argc, argv, "--request-instruct");
        std::vector<std::string> session_option_args = arg_values(argc, argv, "--session-option");
        const std::string reference_text = arg_value(argc, argv, "--reference-text", kDefaultReferenceText);
        const auto reference_audio_path = std::filesystem::path(arg_value(argc, argv, "--clone-audio", "resources/sample.wav"));
        const auto audio_out_path = std::filesystem::path(arg_value(argc, argv, "--audio-out", "qwen3_tts_cpp_audio.wav"));
        const auto audio_out_dir = std::filesystem::path(arg_value(argc, argv, "--audio-out-dir", ""));
        const auto timing_path = std::filesystem::path(arg_value(argc, argv, "--timing-file", "qwen3_tts_cpp_timing.log"));
        std::string warmup_text = arg_value(argc, argv, "--warmup-text", kDefaultWarmupText);
        std::vector<std::string> texts = arg_values(argc, argv, "--text");
        if (texts.empty()) {
            texts.push_back("The benchmark request should produce clear speech for comparison.");
        }

        if (!timing_path.parent_path().empty()) {
            std::filesystem::create_directories(timing_path.parent_path());
        }
        set_process_env("ENGINE_TRACE_ENABLED", "0");
        set_process_env("ENGINE_TIMING_ENABLED", "1");
        set_process_env("ENGINE_TIMING_FILE", timing_path.string());
        engine::debug::configure_logging(engine::debug::LoggingConfig{true, timing_path.string()});

        const bool voice_design = task_name == "voice_design";
        const bool custom_voice = task_name == "custom_voice";
        if (!voice_design && !custom_voice && task_name != "tts") {
            throw std::runtime_error("unsupported Qwen3 warmbench task: " + task_name);
        }
        if (warmup_speaker.empty()) {
            warmup_speaker = speaker;
        }
        if (warmup_instruct.empty()) {
            warmup_instruct = voice_design_instruct;
        }
        if ((voice_design || custom_voice) && !request_instructs.empty() && request_instructs.size() < texts.size()) {
            throw std::runtime_error(
                "need at least one --request-instruct per measured request for Qwen3 task");
        }
        if (custom_voice && speaker.empty()) {
            throw std::runtime_error("Qwen3 custom voice warmbench requires --speaker");
        }
        if (custom_voice && !request_speakers.empty() && request_speakers.size() < texts.size()) {
            throw std::runtime_error(
                "need at least one --request-speaker per measured request for Qwen3 custom voice");
        }
        std::optional<engine::runtime::AudioBuffer> reference_audio = std::nullopt;
        if (!voice_design && !custom_voice) {
            const auto ref_wav = engine::audio::read_wav_f32(reference_audio_path);
            reference_audio = engine::runtime::AudioBuffer{
                ref_wav.sample_rate,
                ref_wav.channels,
                ref_wav.samples,
            };
        }

        engine::runtime::ModelLoadRequest load_request;
        load_request.model_path = model_path;
        load_request.family_hint = "qwen3_tts";
        auto registry = engine::runtime::make_default_registry();
        auto model = registry.load(load_request);

        engine::runtime::SessionOptions session_options;
        session_options.backend.type = parse_backend(backend_name);
        session_options.backend.device = device;
        session_options.backend.threads = threads;
        for (const auto & option : session_option_args) {
            const auto sep = option.find('=');
            if (sep == std::string::npos || sep == 0) {
                throw std::runtime_error("invalid Qwen3 warmbench --session-option: " + option);
            }
            session_options.options[option.substr(0, sep)] = option.substr(sep + 1);
        }

        const engine::runtime::TaskSpec task{
            voice_design ? engine::runtime::VoiceTaskKind::VoiceDesign : engine::runtime::VoiceTaskKind::Tts,
            engine::runtime::RunMode::Offline,
        };
        auto session_base = model->create_task_session(task, session_options);
        auto * session = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session_base.get());
        if (session == nullptr) {
            throw std::runtime_error("loaded Qwen3 session is not an offline TTS session");
        }

        const auto warmup_request = make_request(
            warmup_text,
            language,
            reference_audio.has_value() ? &*reference_audio : nullptr,
            reference_text,
            warmup_instruct,
            warmup_speaker,
            max_new_tokens,
            seed,
            do_sample,
            subtalker_do_sample,
            top_k,
            top_p,
            temperature,
            subtalker_top_k,
            subtalker_top_p,
            subtalker_temperature);
        std::vector<engine::runtime::TaskRequest> requests;
        requests.reserve(texts.size());
        for (size_t request_index = 0; request_index < texts.size(); ++request_index) {
            const std::string & text = texts[request_index];
            const std::string & instruct =
                request_instructs.empty() ? voice_design_instruct : request_instructs[request_index];
            const std::string & request_speaker =
                request_speakers.empty() ? speaker : request_speakers[request_index];
            requests.push_back(make_request(
                text,
                language,
                reference_audio.has_value() ? &*reference_audio : nullptr,
                reference_text,
                instruct,
                request_speaker,
                max_new_tokens,
                seed,
                do_sample,
                subtalker_do_sample,
                top_k,
                top_p,
                temperature,
                subtalker_top_k,
                subtalker_top_p,
                subtalker_temperature));
        }

        session_base->prepare(engine::runtime::build_preparation_request(warmup_request));
        std::vector<engine::runtime::TaskResult> warmup_results;
        warmup_results.reserve(static_cast<size_t>(std::max(0, warmup)));
        for (int i = 0; i < warmup; ++i) {
            warmup_results.push_back(session->run(warmup_request));
        }

        std::vector<engine::runtime::TaskResult> last_results(requests.size());
        std::vector<double> wall_sums(requests.size(), 0.0);
        std::vector<double> last_wall_ms(requests.size(), 0.0);
        for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
            for (int i = 0; i < iterations; ++i) {
                const auto started = std::chrono::steady_clock::now();
                last_results[request_index] = session->run(requests[request_index]);
                const auto ended = std::chrono::steady_clock::now();
                last_wall_ms[request_index] = std::chrono::duration<double, std::milli>(ended - started).count();
                wall_sums[request_index] += last_wall_ms[request_index];
            }
        }

        for (size_t warmup_index = 0; warmup_index < warmup_results.size(); ++warmup_index) {
            std::cout << "warmup_text[" << warmup_index << "]=" << warmup_text << "\n";
            std::cout << "warmup_summary_json[" << warmup_index << "]="
                      << summary_json(warmup_results[warmup_index], 0.0) << "\n";
        }
        for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
            std::cout << "text[" << request_index << "]=" << texts[request_index] << "\n";
            std::cout << "summary_json[" << request_index << "]="
                      << summary_json(last_results[request_index], last_wall_ms[request_index]) << "\n";
        }
        std::cout << "timing_out=" << timing_path.string() << "\n";
        if (!audio_out_dir.empty()) {
            std::filesystem::create_directories(audio_out_dir);
            for (size_t request_index = 0; request_index < last_results.size(); ++request_index) {
                const auto & audio = *last_results[request_index].audio_output;
                const auto request_audio_out = audio_out_dir / ("request_" + std::to_string(request_index) + ".wav");
                engine::audio::write_pcm16_wav(request_audio_out, audio.sample_rate, audio.channels, audio.samples);
                std::cout << "audio_out[" << request_index << "]=" << request_audio_out.string() << "\n";
            }
        }
        const auto & audio = *last_results.back().audio_output;
        engine::audio::write_pcm16_wav(audio_out_path, audio.sample_rate, audio.channels, audio.samples);
        std::cout << "audio_out=" << audio_out_path.string() << "\n";
        for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
            std::cout << "average[" << request_index << "]\n";
            std::cout << "qwen3_tts.synthesize_wall_ms="
                      << (wall_sums[request_index] / static_cast<double>(std::max(1, iterations))) << "\n";
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "qwen3_tts_warm_bench failed: " << error.what() << "\n";
        return 1;
    }
}
