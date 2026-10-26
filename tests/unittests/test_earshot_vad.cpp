#include "engine/framework/audio/chunking.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/registry.h"
#include "engine/community_models/earshot_vad/runtime.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef ENGINE_REPO_ROOT
#define ENGINE_REPO_ROOT "."
#endif

#ifndef ENGINE_TEST_ASSET_ROOT
#define ENGINE_TEST_ASSET_ROOT "tests/unittests/assets"
#endif

namespace {

namespace ev = engine::community_models::earshot_vad;
namespace rt = engine::runtime;

std::filesystem::path repo_path(const std::string & relative) {
    return std::filesystem::path(ENGINE_REPO_ROOT) / relative;
}

std::filesystem::path model_dir() {
    return repo_path("assets/framework/models/earshot_vad");
}

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::shared_ptr<const ev::EarshotWeights> weights() {
    return ev::load_earshot_weights_cached(ev::resolve_earshot_assets(model_dir()).checkpoint_path);
}

std::vector<float> load_wav(const std::string & relative) {
    return engine::audio::read_wav_f32_as_mono_linear_resampled(repo_path(relative), ev::kSampleRate);
}

struct LoadedSession {
    std::unique_ptr<rt::ILoadedVoiceModel> model;
    std::unique_ptr<rt::IVoiceTaskSession> session;
};

LoadedSession open_session(rt::RunMode mode) {
    LoadedSession out;
    out.model = rt::make_default_registry().load(model_dir());
    require(out.model->metadata().family == "earshot_vad", "registry did not pick earshot_vad");
    out.session = out.model->create_task_session({rt::VoiceTaskKind::Vad, mode}, {});
    rt::SessionPreparationRequest prepare;
    out.session->prepare(prepare);
    return out;
}

std::vector<rt::SpeechSegment> run_offline(const rt::AudioBuffer & audio) {
    auto loaded = open_session(rt::RunMode::Offline);
    auto & offline = dynamic_cast<rt::IOfflineVoiceTaskSession &>(*loaded.session);
    rt::TaskRequest request;
    request.audio_input = audio;
    return offline.run(request).speech_segments;
}

std::vector<rt::SpeechSegment> run_streaming(
    const std::vector<float> & audio, size_t chunk, size_t * end_events, size_t * start_events = nullptr) {
    auto loaded = open_session(rt::RunMode::Streaming);
    auto & streaming = dynamic_cast<rt::IStreamingVoiceTaskSession &>(*loaded.session);
    streaming.start_stream({});
    size_t ends = 0;
    size_t starts = 0;
    for (size_t offset = 0; offset < audio.size(); offset += chunk) {
        rt::AudioChunk piece;
        piece.sample_rate = ev::kSampleRate;
        piece.channels = 1;
        piece.start_sample = static_cast<int64_t>(offset);
        piece.samples.assign(audio.begin() + offset, audio.begin() + std::min(audio.size(), offset + chunk));
        for (const auto & event : streaming.process_audio_chunk(piece).voice_activity) {
            if (event.kind == rt::VoiceActivityEvent::Kind::SpeechStart) {
                ++starts;
            }
            if (event.kind == rt::VoiceActivityEvent::Kind::SpeechEnd) {
                require(event.segment.has_value(), "SpeechEnd without segment");
                ++ends;
            }
        }
    }
    if (end_events != nullptr) {
        *end_events = ends;
    }
    if (start_events != nullptr) {
        *start_events = starts;
    }
    return streaming.finish_stream().speech_segments;
}

void test_upstream_parity() {
    const auto reference = engine::io::json::parse_jsonc_file(
        std::filesystem::path(ENGINE_TEST_ASSET_ROOT) / "earshot_vad" / "upstream_reference.json");
    const auto w = weights();
    for (const auto & file : reference.find("files")->as_array()) {
        const auto wav = file.find("wav")->as_string();
        const auto audio = load_wav(wav);
        require(static_cast<int64_t>(audio.size()) == file.find("num_samples")->as_i64(), wav + ": sample count");
        const auto & expected = file.find("scores")->as_array();
        const auto scores = ev::score_earshot_frames(w, audio.data(), static_cast<int64_t>(audio.size()), 1);
        require(scores.size() == expected.size(), wav + ": frame count");
        double max_diff = 0.0;
        for (size_t i = 0; i < scores.size(); ++i) {
            max_diff = std::max(max_diff, std::abs(static_cast<double>(scores[i]) - expected[i].as_number()));
        }
        // FFT order, logf and mel sum order differ from upstream; observed max ~1e-4.
        require(max_diff <= 5e-4, wav + ": max score diff " + std::to_string(max_diff));

        // Threads and the frame-by-frame detector run the same kernels: identical bits.
        const auto threaded = ev::score_earshot_frames(w, audio.data(), static_cast<int64_t>(audio.size()), 4);
        require(threaded == scores, wav + ": threaded scoring differs from one thread");
        ev::EarshotDetector detector(w);
        for (size_t f = 0; f < scores.size(); ++f) {
            const float s = detector.predict(audio.data() + static_cast<int64_t>(f) * ev::kFrameSize);
            require(s == scores[f], wav + ": detector differs from batched scoring");
        }

        ev::EarshotSegmenterConfig upstream;
        upstream.keep_trailing_segment = false;
        const auto segments = ev::segment_earshot_scores(scores, upstream);
        const auto & expected_segments = file.find("segments")->as_array();
        require(segments.size() == expected_segments.size(), wav + ": segment count");
        for (size_t i = 0; i < segments.size(); ++i) {
            // Upstream segments.rs skips its frame counter increment on every frame that
            // returns a segment, so segment i is reported i frames early there.
            const auto lag = static_cast<int64_t>(i) * ev::kFrameSize;
            const auto & pair = expected_segments[i].as_array();
            require(segments[i].start_frame * ev::kFrameSize == pair[0].as_i64() + lag &&
                        segments[i].end_frame * ev::kFrameSize == pair[1].as_i64() + lag,
                    wav + ": segment " + std::to_string(i) + " differs from upstream");
        }
    }
}

void test_trailing_segment_kept() {
    // This clip ends mid-speech; upstream drops that segment, audio.cpp keeps it.
    const auto audio = load_wav("assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0000.wav");
    const auto segments = run_offline({ev::kSampleRate, 1, audio});
    require(segments.size() == 1, "expected one trailing speech segment");
    require(segments[0].span.end_sample == static_cast<int64_t>(audio.size()), "trailing segment must reach the end");
    require(segments[0].span.start_sample > 0 && segments[0].confidence > 0.5f, "trailing segment looks wrong");
}

void test_streaming_matches_offline() {
    const auto audio = load_wav("tests/parakeet_tdt/assets/2086-149220-0033.wav");
    const auto offline = run_offline({ev::kSampleRate, 1, audio});
    require(offline.size() >= 2, "expected several segments");
    for (const size_t chunk : {size_t{1}, size_t{100}, size_t{256}, size_t{4000}}) {
        size_t ends = 0;
        const auto streamed = run_streaming(audio, chunk, &ends);
        require(streamed.size() == offline.size(), "streaming segment count, chunk " + std::to_string(chunk));
        for (size_t i = 0; i < streamed.size(); ++i) {
            require(streamed[i].span.start_sample == offline[i].span.start_sample &&
                        streamed[i].span.end_sample == offline[i].span.end_sample,
                    "streaming segment differs, chunk " + std::to_string(chunk));
        }
        require(ends + 1 >= streamed.size(), "missing SpeechEnd events");
    }
    // Streaming takes mono only (the driver counts interleaved samples).
    {
        auto loaded = open_session(rt::RunMode::Streaming);
        auto & streaming = dynamic_cast<rt::IStreamingVoiceTaskSession &>(*loaded.session);
        streaming.start_stream({});
        bool threw = false;
        try {
            (void)streaming.process_audio_chunk({ev::kSampleRate, 2, 0, std::vector<float>(512, 0.0f)});
        } catch (const std::runtime_error &) {
            threw = true;
        }
        require(threw, "stereo streaming chunk accepted");
    }
    // Every segment in the final result was announced by a SpeechStart event, including a
    // trailing segment closed by finalize() right at the minimum duration.
    for (int64_t frames = 30; frames <= 50; ++frames) {
        const std::vector<float> cut(audio.begin(), audio.begin() + frames * ev::kFrameSize);
        size_t starts = 0;
        const auto streamed = run_streaming(cut, 256, nullptr, &starts);
        require(starts == streamed.size(),
                "segment without SpeechStart event at " + std::to_string(frames) + " frames");
    }
    // A second run on a fresh stream must be identical.
    require(run_streaming(audio, 333, nullptr).size() == offline.size(), "stream restart is not deterministic");
}

void test_silence_and_noise() {
    std::vector<float> silence(ev::kSampleRate * 5, 0.0f);
    require(run_offline({ev::kSampleRate, 1, silence}).empty(), "silence produced segments");
    std::mt19937 rng(7);
    std::normal_distribution<float> noise(0.0f, 0.01f);
    std::vector<float> hiss(ev::kSampleRate * 5);
    for (auto & sample : hiss) {
        sample = noise(rng);
    }
    require(run_offline({ev::kSampleRate, 1, hiss}).empty(), "white noise produced segments");
    require(run_offline({ev::kSampleRate, 1, std::vector<float>(100, 0.0f)}).empty(), "short input produced segments");
}

void test_stereo_resampled_input() {
    const auto audio = load_wav("tests/parakeet_tdt/assets/2086-149220-0033.wav");
    const auto reference = run_offline({ev::kSampleRate, 1, audio});
    // 48 kHz interleaved stereo via linear upsampling.
    const auto upsampled = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(audio, 16000, 1, 48000);
    std::vector<float> stereo(upsampled.size() * 2);
    for (size_t i = 0; i < upsampled.size(); ++i) {
        stereo[2 * i] = upsampled[i];
        stereo[2 * i + 1] = upsampled[i];
    }
    const auto segments = run_offline({48000, 2, stereo});
    require(segments.size() == reference.size(), "48 kHz stereo segment count");
    for (size_t i = 0; i < segments.size(); ++i) {
        const auto tolerance = 2 * ev::kFrameSize * 3;
        require(std::abs(segments[i].span.start_sample - reference[i].span.start_sample * 3) <= tolerance &&
                    std::abs(segments[i].span.end_sample - reference[i].span.end_sample * 3) <= tolerance,
                "48 kHz stereo spans are not mapped back to input samples");
    }
}

void test_options() {
    const auto config = ev::earshot_config_from_options({
        {"threshold", "0.7"},
        {"neg_threshold", "0.3"},
        {"speech_pad_ms", "0"},
        {"min_speech_duration_ms", "250"},
        {"min_silence_duration_ms", "100"},
        {"earshot_vad.keep_trailing_segment", "false"},
    });
    require(config.threshold == 0.7f && config.neg_threshold == 0.3f, "threshold options");
    require(config.preroll_frames == 0 && config.min_duration_frames == 16 && config.end_delay_frames == 7, "ms options");
    require(!config.keep_trailing_segment, "keep_trailing_segment option");
    bool threw = false;
    try {
        (void)ev::earshot_config_from_options({{"threshold", "1.5"}});
    } catch (const std::runtime_error &) {
        threw = true;
    }
    require(threw, "out-of-range threshold accepted");
    // Extreme settings must still give well-formed, ordered spans.
    const auto audio = load_wav("tests/parakeet_tdt/assets/2086-149220-0033.wav");
    const auto w = weights();
    const auto scores = ev::score_earshot_frames(w, audio.data(), static_cast<int64_t>(audio.size()), 1);
    const auto loose = ev::earshot_config_from_options({{"speech_pad_ms", "0"}, {"min_speech_duration_ms", "0"},
                                                         {"start_delay_ms", "0"}, {"min_silence_duration_ms", "0"}});
    const auto segments = ev::segment_earshot_scores(scores, loose);
    require(!segments.empty(), "loose options produced no segments");
    for (const auto & segment : segments) {
        require(segment.end_frame > segment.start_frame, "empty segment with loose options");
        require(segment.end_frame <= static_cast<int64_t>(scores.size()), "segment past the end");
    }
    const auto defaults = ev::earshot_config_from_options({});
    require(defaults.preroll_frames == 5 && defaults.min_duration_frames == 6 && defaults.start_delay_frames == 2 &&
                defaults.end_delay_frames == 4,
            "defaults must match upstream SegmenterOptions");
}

void test_loader_detection() {
    const auto registry = rt::make_default_registry();
    rt::ModelLoadRequest silero;
    silero.model_path = repo_path("assets/framework/models/silero_vad");
    require(registry.load(silero)->metadata().family == "silero_vad", "silero dir must still load as silero_vad");
    rt::ModelLoadRequest by_alias;
    by_alias.model_path = model_dir();
    by_alias.family_hint = "earshot";
    require(registry.load(by_alias)->metadata().family == "earshot_vad", "earshot alias");
}

void test_vad_chunk_planner() {
    // Hosts that load their VAD through the registry (e.g. gigaam_asr) feed it to the chunk planner.
    auto model = rt::make_default_registry().load(model_dir() / ev::kWeightFile);
    require(model->metadata().family == "earshot_vad", "registry load of the earshot weight file");
    auto session = model->create_task_session({rt::VoiceTaskKind::Vad, rt::RunMode::Offline}, {});
    auto & offline = dynamic_cast<rt::IOfflineVoiceTaskSession &>(*session);
    const auto audio = load_wav("tests/parakeet_tdt/assets/2086-149220-0033.wav");
    const auto chunks = engine::audio::plan_vad_audio_chunks(
        {ev::kSampleRate, 1, audio}, offline, engine::audio::VadAudioChunkOptions{30 * ev::kSampleRate, 8000, 4000});
    require(!chunks.empty(), "VAD chunk planner produced no chunks");
}

}  // namespace

int main() {
    std::cout << "earshot_vad kernels: " << ev::earshot_kernel_name() << "\n";
    const std::vector<std::pair<const char *, void (*)()>> tests = {
        {"upstream_parity", test_upstream_parity},
        {"trailing_segment_kept", test_trailing_segment_kept},
        {"streaming_matches_offline", test_streaming_matches_offline},
        {"silence_and_noise", test_silence_and_noise},
        {"stereo_resampled_input", test_stereo_resampled_input},
        {"options", test_options},
        {"loader_detection", test_loader_detection},
        {"vad_chunk_planner", test_vad_chunk_planner},
    };
    int failures = 0;
    for (const auto & [name, test] : tests) {
        try {
            test();
            std::cout << "[PASS] " << name << "\n";
        } catch (const std::exception & error) {
            std::cerr << "[FAIL] " << name << ": " << error.what() << "\n";
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
