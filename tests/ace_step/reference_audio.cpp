#include "engine/framework/debug/trace.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/models/ace_step/pre_dit.h"
#include "engine/models/ace_step/request_parser.h"
#include "engine/models/ace_step/session.h"
#include "../unittests/test_assert.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>

using namespace engine;
using namespace engine::models::ace_step;
using engine::test::require;

int main(int argc, char ** argv) {
    try {
        require(argc == 2 || argc == 6, "usage: ace_step_reference_audio_test --log [MODEL BACKEND REFERENCE OUTPUT_DIR]");
        require(std::string(argv[1]) == "--log", "--log is required");
        debug::configure_logging({true, std::nullopt});
        runtime::TaskRequest request;
        request.text_input = runtime::Transcript{"Warm acoustic folk song with a clear singing voice and gentle guitar", "en"};
        request.options = {{"seed", "42"}, {"enable_thinking", "false"}, {"use_cot_metas", "false"},
                           {"use_cot_caption", "false"}, {"use_cot_language", "false"},
                           {"lyrics", "[verse] The morning light is shining through the trees"},
                           {"duration_sec", "8"}, {"num_inference_steps", "8"}};
        require(!ace_step_parse_request(request).reference, "absent reference must stay absent");
        runtime::AudioBuffer short_audio;
        short_audio.sample_rate = 24000;
        short_audio.channels = 1;
        short_audio.samples.assign(24000, 0.25F);
        request.audio_input = short_audio;
        request.audio_input->samples.assign(24000, -0.2F);
        request.voice.emplace().speaker.emplace().audio = short_audio;
        require(!ace_step_parse_request(request).reference, "TTS voice reference must not supply ACE-Step conditioning");
        request.voice.reset();
        const auto temporary = std::filesystem::temp_directory_path() /
            ("ace-step-reference-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(temporary);
        const auto reference_path = temporary / "reference.wav";
        audio::write_pcm16_wav(reference_path, short_audio.sample_rate, short_audio.channels, short_audio.samples);
        request.options["reference_audio"] = reference_path.string();
        for (const auto * route : {"text2music", "cover", "cover-nofsq", "repaint", "extract", "lego", "complete"}) {
            request.options["route"] = route;
            const auto parsed = ace_step_parse_request(request);
            require(parsed.reference && parsed.reference->audio, "reference must survive route selection");
            require(parsed.reference->audio->samples == short_audio.samples, "reference data changed in parser");
            require(parsed.source_audio && parsed.source_audio->samples == request.audio_input->samples, "source lost in parser");
        }
        request.options["reference_audio"] = "";
        bool empty_path_rejected = false;
        try { (void) ace_step_parse_request(request); }
        catch (const std::runtime_error &) { empty_path_rejected = true; }
        require(empty_path_rejected, "empty reference path was accepted");
        request.options["reference_audio"] = (temporary / "missing.wav").string();
        bool missing_rejected = false;
        try { (void) ace_step_parse_request(request); }
        catch (const std::runtime_error &) { missing_rejected = true; }
        require(missing_rejected, "missing reference file was accepted");
        request.options.erase("reference_audio");
        auto short_digest = ace_step_prepare_reference_audio(short_audio, 42);
        require(short_digest.sample_rate == 48000 && short_digest.channels == 2, "reference format");
        require(short_digest.samples.size() == 30 * 48000 * 2, "reference length");
        // Sinc resampling has boundary transients; away from the edges a
        // constant mono input must remain the same on both stereo channels.
        require(std::abs(short_digest.samples[4800] - 0.25F) < 1e-4F, "resampling constant signal");
        for (size_t i = 0; i < short_digest.samples.size(); i += 2) {
            require(short_digest.samples[i] == short_digest.samples[i + 1], "mono channel duplication");
            if (i % (48000 * 2) == 4800) {
                require(std::abs(short_digest.samples[i] - 0.25F) < 1e-4F, "short reference must loop rather than zero-pad");
            }
        }
        runtime::AudioBuffer ramp;
        ramp.sample_rate = 48000;
        ramp.channels = 2;
        constexpr size_t frames = 45 * 48000;
        ramp.samples.resize(frames * 2);
        for (size_t i = 0; i < frames; ++i) {
            ramp.samples[2 * i] = float(i + 1) / float(frames + 1);
            ramp.samples[2 * i + 1] = -ramp.samples[2 * i];
        }
        const auto digest = ace_step_prepare_reference_audio(ramp, 42);
        require(digest.samples == ace_step_prepare_reference_audio(ramp, 42).samples, "repeatable digest");
        require(digest.samples != ace_step_prepare_reference_audio(ramp, 43).samples, "seed must select windows");
        for (size_t part = 0; part < 3; ++part) {
            const size_t dst = part * 10 * 48000 * 2;
            const size_t start = static_cast<size_t>(std::llround(double(digest.samples[dst]) * (frames + 1))) - 1;
            require(start >= part * frames / 3 && start + 10 * 48000 <= (part + 1) * frames / 3, "window outside its third");
            require(std::equal(digest.samples.begin() + dst, digest.samples.begin() + dst + 10 * 48000 * 2,
                               ramp.samples.begin() + start * 2), "window samples differ from source");
        }
        for (bool empty : {false, true}) {
            auto invalid = short_audio;
            invalid.samples.assign(empty ? 0 : 24000, 0.0F);
            bool rejected = false;
            try { (void) ace_step_prepare_reference_audio(invalid, 42); }
            catch (const std::runtime_error &) { rejected = true; }
            require(rejected, "invalid reference was silently accepted");
        }
        std::cout << "PASS: parser routes, independent source, resampling, repetition, window boundaries, seed, invalid inputs\n";
        if (argc == 6) {
            auto assets = load_ace_step_assets(argv[2]);
            runtime::SessionOptions options;
            const std::string backend = argv[3];
            require(backend == "cuda" || backend == "vulkan" || backend == "cpu", "unsupported test backend");
            options.backend.type = backend == "cuda" ? core::BackendType::Cuda
                : backend == "vulkan" ? core::BackendType::Vulkan : core::BackendType::Cpu;
            options.backend.threads = 8;
            AceStepSession session({runtime::VoiceTaskKind::AudioGeneration, runtime::RunMode::Offline}, options, assets);
            session.prepare({});
            auto reversed = audio::read_wav_f32(std::filesystem::path(argv[4]));
            std::reverse(reversed.samples.begin(), reversed.samples.end());
            const auto reversed_path = temporary / "reversed.wav";
            audio::write_pcm16_wav(reversed_path, reversed.sample_rate, reversed.channels, reversed.samples);
            request.audio_input.reset();
            request.options["route"] = "text2music";
            std::vector<float> none, first, changed;
            const std::filesystem::path output = argv[5];
            std::filesystem::create_directories(output);
            // No reference -> A -> B -> A -> no reference. Reuse one session.
            for (int i = 0; i < 5; ++i) {
                request.options.erase("reference_audio");
                if (i > 0 && i < 4) {
                    request.options["reference_audio"] = i == 2 ? reversed_path.string() : argv[4];
                }
                const auto result = session.run(request);
                require(result.audio_output && !result.audio_output->samples.empty(), "empty generated audio");
                const auto & samples = result.audio_output->samples;
                require(std::all_of(samples.begin(), samples.end(), [](float x) { return std::isfinite(x); }), "nonfinite generated audio");
                if (i == 0) none = samples;
                if (i == 1) { first = samples; require(first != none, "reference is ignored"); }
                if (i == 2) { changed = samples; require(changed != first, "changed reference is ignored"); }
                if (i == 3) require(samples == first, "reference A changed after B");
                if (i == 4) require(samples == none, "no-reference output changed after references");
                audio::write_pcm16_wav(output / (std::to_string(i) + ".wav"), result.audio_output->sample_rate, result.audio_output->channels, samples);
            }
            std::cout << "PASS: one-session none/A/B/A/none lifecycle, changed references and exact repeats\n";
        }
        std::filesystem::remove_all(temporary);
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
