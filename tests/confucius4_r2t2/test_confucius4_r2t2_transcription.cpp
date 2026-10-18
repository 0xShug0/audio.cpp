#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"
#include "engine/community_models/confucius4_r2t2/assets.h"
#include "engine/community_models/confucius4_r2t2/tokenizer_text.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef ENGINE_REPO_ROOT
#define ENGINE_REPO_ROOT "."
#endif

namespace {

constexpr int kExitPass = 0;
constexpr int kExitFail = 1;
constexpr int kExitSkip = 125;

// Golden output from the macOS MPS reference implementation
// (tests/confucius4_r2t2/golden_sample16k.json, produced by make_golden.py).
const char * kExpectedOffline =
    "Some call me nature, others call me mother nature. I've been here for over 4.5 billion years, 22,500 times longer than you.";
const char * kExpectedStreamFinal =
    "Some call me nature, others call me mother nature. I've been here for over 4.5 billion years, 22,500 times longer than you.";

constexpr int64_t kStreamingChunkMs = 320;
constexpr int64_t kStreamingMaxNewTokens = 32;

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

engine::runtime::AudioBuffer read_audio(const std::filesystem::path & path) {
    const auto wav = engine::audio::read_wav_f32(path);
    engine::runtime::AudioBuffer audio;
    audio.sample_rate = wav.sample_rate;
    audio.channels = wav.channels;
    audio.samples = wav.samples;
    return audio;
}

std::string run_offline(
    engine::runtime::ILoadedVoiceModel & model,
    const engine::runtime::AudioBuffer & audio,
    const engine::runtime::SessionOptions & options) {
    auto session = model.create_task_session(
        engine::runtime::TaskSpec{engine::runtime::VoiceTaskKind::Asr, engine::runtime::RunMode::Offline},
        options);
    auto * offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session.get());
    if (offline == nullptr) {
        throw std::runtime_error("R2T2 session is not an offline session");
    }
    offline->prepare(engine::runtime::build_preparation_request(audio));
    engine::runtime::TaskRequest request;
    request.audio_input = audio;
    const auto result = offline->run(request);
    if (!result.text_output.has_value()) {
        throw std::runtime_error("offline run produced no text");
    }
    return result.text_output->text;
}

std::string run_streaming(
    engine::runtime::ILoadedVoiceModel & model,
    const engine::runtime::AudioBuffer & audio,
    const engine::runtime::SessionOptions & options) {
    auto session = model.create_task_session(
        engine::runtime::TaskSpec{engine::runtime::VoiceTaskKind::Asr, engine::runtime::RunMode::Streaming},
        options);
    auto * streaming = dynamic_cast<engine::runtime::IStreamingVoiceTaskSession *>(session.get());
    if (streaming == nullptr) {
        throw std::runtime_error("R2T2 session is not a streaming session");
    }
    engine::runtime::TaskRequest request;
    request.audio_input = audio;
    streaming->prepare(engine::runtime::build_preparation_request(request));

    std::string committed;
    streaming->set_stream_event_sink([&](const engine::runtime::StreamEvent & event) {
        if (event.partial_text.has_value()) {
            committed += event.partial_text->text;
        }
    });
    request.options["language"] = "Auto";
    streaming->start_stream(request);

    const int64_t chunk_frames = std::max<int64_t>(
        1,
        static_cast<int64_t>(audio.sample_rate) * kStreamingChunkMs / 1000);
    const int64_t frames = static_cast<int64_t>(audio.samples.size() / static_cast<size_t>(audio.channels));
    for (int64_t start = 0; start < frames; start += chunk_frames) {
        const int64_t take = std::min<int64_t>(chunk_frames, frames - start);
        engine::runtime::AudioChunk chunk;
        chunk.sample_rate = audio.sample_rate;
        chunk.channels = audio.channels;
        chunk.start_sample = start;
        const auto begin = audio.samples.begin() + static_cast<std::ptrdiff_t>(start * audio.channels);
        chunk.samples.assign(begin, begin + static_cast<std::ptrdiff_t>(take * audio.channels));
        streaming->process_audio_chunk(chunk);
    }
    const auto result = streaming->finish_stream();
    if (!result.text_output.has_value()) {
        throw std::runtime_error("streaming run produced no text");
    }
    std::cout << "Committed stream:    " << committed << "\n";
    // finish_stream() returns the final tail separately; the emitted deltas
    // must form a nonempty transcript prefix, never consume metadata offsets.
    if (committed.empty() || std::string(kExpectedStreamFinal).compare(0, committed.size(), committed) != 0) {
        throw std::runtime_error("committed deltas are not a prefix of the expected transcript: " + committed);
    }
    return result.text_output->text;
}

// Debug aid: print token ids for a text argument so the family tokenizer can be
// diffed against the reference Hugging Face tokenizer.
int encode_probe(const std::filesystem::path & model_path, const std::string & text) {
    auto assets = engine::community_models::confucius4_r2t2::load_confucius4_r2t2_assets(model_path, "confucius4_r2t2");
    engine::community_models::confucius4_r2t2::R2T2ASRTextTokenizer tokenizer(assets);
    const auto ids = tokenizer.encode(text);
    std::cout << "count=" << ids.size() << "\nids=";
    for (size_t i = 0; i < ids.size(); ++i) {
        std::cout << (i == 0 ? "" : ",") << ids[i];
    }
    std::cout << "\n";
    return kExitPass;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::filesystem::path model_path = arg_value(argc, argv, "--model", repo_path("models/Confucius4-R2T2").string());
    const auto encode_text = arg_value(argc, argv, "--encode", "");
    if (!encode_text.empty()) {
        return encode_probe(model_path, encode_text);
    }
    const std::filesystem::path audio_path = arg_value(argc, argv, "--audio", repo_path("assets/resources/sample_16k.wav").string());
    const std::string backend_name = arg_value(argc, argv, "--backend", "best");

    const bool model_available =
        engine::io::is_existing_file(model_path) ||
        engine::io::is_existing_file(model_path / "config.json");
    if (!model_available || !engine::io::is_existing_file(audio_path)) {
        std::fprintf(
            stderr,
            "SKIP: test_confucius4_r2t2_transcription requires model weights at '%s' and audio at '%s'.\n",
            model_path.string().c_str(),
            audio_path.string().c_str());
        return kExitSkip;
    }

    try {
        auto registry = engine::runtime::make_default_registry();
        engine::runtime::ModelLoadRequest load_request;
        load_request.model_path = model_path;
        load_request.family_hint = "confucius4_r2t2";
        auto model = registry.load(load_request);

        engine::runtime::SessionOptions options;
        options.backend.type = parse_backend(backend_name);
        options.backend.threads = 8;
        options.options["confucius4_r2t2.chunk_size_ms"] = std::to_string(kStreamingChunkMs);
        options.options["confucius4_r2t2.max_tokens"] = std::to_string(kStreamingMaxNewTokens);

        const auto audio = read_audio(audio_path);

        const std::string offline = run_offline(*model, audio, options);
        std::cout << "Offline transcript:  " << offline << "\n";
        if (offline != kExpectedOffline) {
            std::cerr << "FAIL: offline transcript mismatch\n"
                      << "  expected: " << kExpectedOffline << "\n"
                      << "  actual:   " << offline << "\n";
            return kExitFail;
        }

        const std::string stream_final = run_streaming(*model, audio, options);
        std::cout << "Streaming final:     " << stream_final << "\n";
        if (stream_final != kExpectedStreamFinal) {
            std::cerr << "FAIL: streaming transcript mismatch\n"
                      << "  expected: " << kExpectedStreamFinal << "\n"
                      << "  actual:   " << stream_final << "\n";
            return kExitFail;
        }

        std::cout << "PASS: Confucius4-R2T2 offline and streaming transcripts match the MPS golden.\n";
        return kExitPass;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return kExitFail;
    }
}
