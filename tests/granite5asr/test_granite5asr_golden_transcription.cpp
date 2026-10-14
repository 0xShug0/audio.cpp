#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/text.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef ENGINE_REPO_ROOT
#define ENGINE_REPO_ROOT "."
#endif

namespace {

constexpr int kExitPass = 0;
constexpr int kExitFail = 1;
constexpr int kExitSkip = 125;

const char * kExpectedText = "concord returned to its place amidst the tents";

std::filesystem::path repo_path(const std::string & relative) {
    return std::filesystem::path(ENGINE_REPO_ROOT) / relative;
}

std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

engine::core::BackendType parse_backend(const std::string & value) {
    if (value == "cpu") return engine::core::BackendType::Cpu;
    if (value == "cuda") return engine::core::BackendType::Cuda;
    if (value == "metal") return engine::core::BackendType::Metal;
    if (value == "vulkan") return engine::core::BackendType::Vulkan;
    if (value == "best") return engine::core::BackendType::BestAvailable;
    throw std::runtime_error("unsupported backend: " + value);
}

std::string normalize_text(std::string text) {
    std::string out;
    out.reserve(text.size());
    for (char ch : text) {
        if (std::isalnum(static_cast<unsigned char>(ch)) || std::isspace(static_cast<unsigned char>(ch))) {
            out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        }
    }
    return engine::io::trim_ascii_whitespace(std::move(out));
}

// Words must be non-empty, ordered without overlap, inside the buffer, and
// join to the normalized transcript.
bool check_words(
    const std::string & label,
    const engine::runtime::TaskResult & result,
    int64_t audio_samples) {
    if (!result.text_output.has_value()) {
        std::cerr << "FAIL[" << label << "]: no text output\n";
        return false;
    }
    if (result.word_timestamps.empty()) {
        std::cerr << "FAIL[" << label << "]: expected word timestamps\n";
        return false;
    }
    int64_t previous_end = 0;
    std::string joined;
    for (const auto & word : result.word_timestamps) {
        if (word.span.start_sample < 0 ||
            word.span.end_sample > audio_samples ||
            word.span.end_sample <= word.span.start_sample) {
            std::cerr << "FAIL[" << label << "]: word span outside buffer: " << word.word
                      << " [" << word.span.start_sample << ", " << word.span.end_sample
                      << ") samples=" << audio_samples << "\n";
            return false;
        }
        if (word.span.start_sample < previous_end) {
            std::cerr << "FAIL[" << label << "]: word overlaps the previous word: " << word.word
                      << " starts at " << word.span.start_sample
                      << " before " << previous_end << "\n";
            return false;
        }
        previous_end = word.span.end_sample;
        const auto trimmed = engine::io::trim_ascii_whitespace(word.word);
        if (trimmed.empty()) {
            std::cerr << "FAIL[" << label << "]: empty word timestamp\n";
            return false;
        }
        if (!joined.empty()) {
            joined.push_back(' ');
        }
        joined += trimmed;
    }
    const auto transcript = normalize_text(result.text_output->text);
    if (normalize_text(joined) != transcript) {
        std::cerr << "FAIL[" << label << "]: joined words do not match transcript\n";
        std::cerr << "Joined normalized:     " << normalize_text(joined) << "\n";
        std::cerr << "Transcript normalized: " << transcript << "\n";
        return false;
    }
    const auto & first = result.word_timestamps.front();
    const auto & last = result.word_timestamps.back();
    std::cout << "[" << label << "] transcript: " << result.text_output->text << "\n";
    std::cout << "[" << label << "] words=" << result.word_timestamps.size()
              << " first=" << first.word << " [" << first.span.start_sample << ", " << first.span.end_sample << ")"
              << " last=" << last.word << " [" << last.span.start_sample << ", " << last.span.end_sample << ")\n";
    return true;
}

double seconds(int64_t sample, int sample_rate) {
    return static_cast<double>(sample) / static_cast<double>(sample_rate);
}

// The same speech at another input rate must land at the same time in seconds,
// within one 80 ms CTC frame, and in samples of that input buffer.
bool check_input_rate(
    engine::runtime::IOfflineVoiceTaskSession & session,
    const engine::runtime::AudioBuffer & reference_audio,
    const engine::runtime::TaskResult & reference,
    int sample_rate,
    const std::string & chunk_seconds) {
    engine::runtime::AudioBuffer audio;
    audio.sample_rate = sample_rate;
    audio.channels = 1;
    audio.samples = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        reference_audio.samples, reference_audio.sample_rate, reference_audio.channels, sample_rate);
    const int64_t audio_samples = static_cast<int64_t>(audio.samples.size());
    const std::string label = std::to_string(sample_rate / 1000) + "khz";

    engine::runtime::TaskRequest request;
    request.audio_input = audio;
    const auto result = session.run(request);
    if (!check_words(label, result, audio_samples)) {
        return false;
    }
    constexpr double kFrameSeconds = 0.08;
    const auto & expected_first = reference.word_timestamps.front();
    const auto & expected_last = reference.word_timestamps.back();
    const auto & first = result.word_timestamps.front();
    const auto & last = result.word_timestamps.back();
    const auto near = [&](int64_t actual, int64_t expected) {
        return std::abs(seconds(actual, sample_rate) - seconds(expected, reference_audio.sample_rate)) <= kFrameSeconds;
    };
    if (first.word != expected_first.word || last.word != expected_last.word ||
        !near(first.span.start_sample, expected_first.span.start_sample) ||
        !near(last.span.start_sample, expected_last.span.start_sample) ||
        !near(last.span.end_sample, expected_last.span.end_sample)) {
        std::cerr << "FAIL[" << label << "]: word times do not match the reference clip\n"
                  << "  expected last=" << expected_last.word << " ["
                  << seconds(expected_last.span.start_sample, reference_audio.sample_rate) << "s, "
                  << seconds(expected_last.span.end_sample, reference_audio.sample_rate) << "s)\n"
                  << "  actual last=" << last.word << " ["
                  << seconds(last.span.start_sample, sample_rate) << "s, "
                  << seconds(last.span.end_sample, sample_rate) << "s)\n";
        return false;
    }

    engine::runtime::TaskRequest chunked_request = request;
    chunked_request.options["audio_chunk_mode"] = "fixed";
    chunked_request.options["audio_chunk_duration_sec"] = chunk_seconds;
    return check_words(label + "_fixed_" + chunk_seconds + "s", session.run(chunked_request), audio_samples);
}

}  // namespace

int main(int argc, char ** argv) {
    const std::filesystem::path model_path = arg_value(
        argc, argv, "--model", repo_path("granite5asr").string());
    const std::filesystem::path audio_path = arg_value(
        argc, argv, "--audio", repo_path("assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0000.wav").string());
    const std::string weight_type = arg_value(argc, argv, "--weight-type", "");
    const std::string backend_name = arg_value(argc, argv, "--backend", "cpu");
    const std::string chunk_seconds = arg_value(argc, argv, "--chunk-seconds", "1");

    const bool model_available =
        engine::io::is_existing_file(model_path) ||
        engine::io::is_existing_file(model_path / "config.json");
    if (!model_available || !engine::io::is_existing_file(audio_path)) {
        std::fprintf(
            stderr,
            "SKIP: test_granite5asr_golden_transcription requires model weights at '%s' "
            "and test audio at '%s'.\n",
            model_path.string().c_str(),
            audio_path.string().c_str());
        return kExitSkip;
    }

    try {
        auto registry = engine::runtime::make_default_registry();
        engine::runtime::ModelLoadRequest load_request;
        load_request.model_path = model_path;
        load_request.family_hint = "granite5asr";
        auto model = registry.load(load_request);

        engine::runtime::SessionOptions session_options;
        session_options.backend.type = parse_backend(backend_name);
        if (!weight_type.empty()) {
            session_options.options["granite5asr.weight_type"] = weight_type;
        }

        auto session = model->create_task_session(
            {engine::runtime::VoiceTaskKind::Asr, engine::runtime::RunMode::Offline},
            session_options);
        auto * offline_session = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session.get());
        if (!offline_session) {
            std::cerr << "FAIL: session is not an IOfflineVoiceTaskSession\n";
            return kExitFail;
        }

        const auto wav_data = engine::audio::read_wav_f32(audio_path);
        engine::runtime::AudioBuffer audio;
        audio.sample_rate = wav_data.sample_rate;
        audio.channels = wav_data.channels;
        audio.samples = wav_data.samples;
        const int64_t audio_samples = static_cast<int64_t>(
            audio.samples.size() / static_cast<size_t>(std::max(1, audio.channels)));
        std::cout << "Backend: " << backend_name
                  << " sample_rate=" << audio.sample_rate
                  << " samples=" << audio_samples << "\n";

        const auto prep = engine::runtime::build_preparation_request(audio);
        offline_session->prepare(prep);

        engine::runtime::TaskRequest request;
        request.audio_input = audio;
        const auto result = offline_session->run(request);

        if (!result.text_output.has_value()) {
            std::cerr << "FAIL: Granite 5 ASR produced no text output\n";
            return kExitFail;
        }

        const std::string actual_raw = result.text_output->text;
        const std::string actual = normalize_text(actual_raw);
        const std::string expected = normalize_text(kExpectedText);

        std::cout << "Raw transcript:      " << actual_raw << "\n";
        std::cout << "Normalized actual:   " << actual << "\n";
        std::cout << "Normalized expected: " << expected << "\n";

        if (actual != expected) {
            std::cerr << "FAIL: transcript mismatch!\n";
            return kExitFail;
        }
        if (!check_words("offline", result, audio_samples)) {
            return kExitFail;
        }

        // Short fixed chunks put several chunk boundaries inside the clip.
        engine::runtime::TaskRequest chunked_request = request;
        chunked_request.options["audio_chunk_mode"] = "fixed";
        chunked_request.options["audio_chunk_duration_sec"] = chunk_seconds;
        const auto chunked = offline_session->run(chunked_request);
        if (!check_words("fixed_" + chunk_seconds + "s", chunked, audio_samples)) {
            return kExitFail;
        }

        for (const int sample_rate : {48000, 8000}) {
            if (!check_input_rate(*offline_session, audio, result, sample_rate, chunk_seconds)) {
                return kExitFail;
            }
        }

        if (audio.sample_rate == 16000 && audio.channels == 1) {
            auto streaming_session = model->create_task_session(
                {engine::runtime::VoiceTaskKind::Asr, engine::runtime::RunMode::Streaming},
                session_options);
            auto * streaming = dynamic_cast<engine::runtime::IStreamingVoiceTaskSession *>(streaming_session.get());
            if (!streaming) {
                std::cerr << "FAIL: session is not an IStreamingVoiceTaskSession\n";
                return kExitFail;
            }
            streaming->prepare(prep);
            streaming->start_stream(engine::runtime::TaskRequest{});
            constexpr size_t kStreamChunk = 512;
            for (size_t offset = 0; offset < audio.samples.size(); offset += kStreamChunk) {
                const size_t end = std::min(audio.samples.size(), offset + kStreamChunk);
                engine::runtime::AudioChunk chunk;
                chunk.sample_rate = audio.sample_rate;
                chunk.channels = audio.channels;
                chunk.start_sample = static_cast<int64_t>(offset);
                chunk.samples.assign(
                    audio.samples.begin() + static_cast<std::ptrdiff_t>(offset),
                    audio.samples.begin() + static_cast<std::ptrdiff_t>(end));
                (void)streaming->process_audio_chunk(chunk);
            }
            const auto streamed = streaming->finish_stream();
            if (!streamed.text_output.has_value() || normalize_text(streamed.text_output->text) != expected) {
                std::cerr << "FAIL: streaming transcript mismatch\n";
                return kExitFail;
            }
            if (!check_words("streaming", streamed, audio_samples)) {
                return kExitFail;
            }
        }

        std::cout << "PASS: Granite 5 ASR golden transcription verified successfully.\n";
        return kExitPass;
    } catch (const std::exception & ex) {
        std::cerr << "FAIL: exception: " << ex.what() << "\n";
        return kExitFail;
    }
}
