#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/registry.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace rt = engine::runtime;

namespace {
void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

rt::VoiceCondition reference(const char * path) {
    auto wav = engine::audio::read_wav_f32(std::filesystem::path(path));
    rt::VoiceCondition voice;
    voice.speaker = rt::VoiceReference{};
    voice.speaker->audio = rt::AudioBuffer{wav.sample_rate, wav.channels, std::move(wav.samples)};
    return voice;
}

template<class F> void rejected(F && operation, const char * message) {
    bool failed = false;
    try { operation(); } catch (const std::runtime_error &) { failed = true; }
    require(failed, message);
}
}

int main(int argc, char ** argv) {
    try {
        if (argc < 6 || argc > 8) throw std::runtime_error(
            "usage: kugelaudio_cloning_session_probe MODEL OUTPUT_DIR REF1.wav REF2.wav cpu|cuda|vulkan [ENCODER|-] [SPEC|embedded]");
        engine::debug::configure_logging({true, std::nullopt});
        auto registry = rt::make_default_registry();
        rt::ModelLoadRequest load;
        load.model_path = argv[1];
        load.family_hint = "kugelaudio";
        load.model_spec_override = argc > 7 ? argv[7] : "embedded";
        auto model = registry.load(load);
        rt::SessionOptions options;
        options.backend.threads = 8;
        const std::string backend = argv[5];
        if (backend == "cuda") options.backend.type = engine::core::BackendType::Cuda;
        else if (backend == "vulkan") options.backend.type = engine::core::BackendType::Vulkan;
        else if (backend != "cpu") throw std::runtime_error("unsupported backend");
        if (argc > 6 && std::string(argv[6]) != "-")
            options.options["kugelaudio.reference_encoder_path"] = argv[6];
        const auto first_voice = reference(argv[3]), second_voice = reference(argv[4]);
        const auto dir = std::filesystem::path(argv[2]);
        std::filesystem::create_directories(dir);
        rt::TaskRequest request;
        request.text_input = rt::Transcript{"Hello, this is a test of the cloned voice.", "en"};
        request.options = {{"voice_id", "english_male"}, {"seed", "1234"}, {"max_tokens", "64"}};
        auto save = [&](const rt::TaskResult & result, const char * name) {
            require(result.audio_output.has_value(), "missing output audio");
            const auto & audio = *result.audio_output;
            require(audio.sample_rate == 24000 && audio.channels == 1 && audio.samples.size() > 2400,
                    "incorrect output format or duration");
            double energy = 0;
            for (float value : audio.samples) {
                require(std::isfinite(value), "non-finite output");
                energy += double(value) * value;
            }
            require(energy / audio.samples.size() > 1e-8, "silent output");
            engine::audio::WavWriteOptions wav;
            wav.format = engine::audio::WavSampleFormat::Float32;
            engine::audio::write_wav(dir / (std::string(name) + ".wav"), 24000, 1, audio.samples, wav);
            return audio.samples;
        };
        std::vector<float> clone;
        {
            auto base = model->create_task_session({rt::VoiceTaskKind::Tts, rt::RunMode::Offline}, options);
            auto * session = dynamic_cast<rt::IOfflineVoiceTaskSession *>(base.get());
            require(session != nullptr, "missing offline interface");
            base->prepare({});
            const auto preset = save(session->run(request), "preset-before");
            rt::SessionPreparationRequest preparation;
            preparation.voice = first_voice;
            const auto started = std::chrono::steady_clock::now();
            base->prepare(preparation);
            std::cout << "reference_prepare_seconds=" << std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count() << std::endl;
            clone = save(session->run(request), "clone-reader");
            require(clone == save(session->run(request), "clone-repeat"), "cached seeded clone differs");
            auto invalid = preparation;
            invalid.voice->speaker->audio->samples.assign(2400, 0);
            rejected([&] { base->prepare(invalid); }, "silent reference was accepted");
            require(clone == save(session->run(request), "clone-after-error"), "failed preparation changed voice");
            auto long_form = request;
            long_form.text_input->text += " The next sentence should use the same speaker.";
            long_form.options["text_chunk_size"] = "50";
            require(save(session->run(long_form), "clone-long-form").size() > clone.size(), "long-form clone was incomplete");
            auto changed = request;
            changed.voice = second_voice;
            require(clone != save(session->run(changed), "clone-other"), "changed reference reused cached voice");
            changed.voice = rt::VoiceCondition{};
            changed.voice->speaker = rt::VoiceReference{};
            changed.voice->speaker->cached_voice_id = "english_male";
            require(preset == save(session->run(changed), "preset-after"), "cloning altered preset output");
            base->prepare({});
            require(preset == save(session->run(request), "preset-cleared"), "preparation did not clear voice");
            std::cout << "PASS: preset preservation, cached repeat, reference change, failed preparation recovery, long-form chunks, clear" << std::endl;
        }
        {
            auto base = model->create_task_session({rt::VoiceTaskKind::VoiceCloning, rt::RunMode::Offline}, options);
            auto * session = dynamic_cast<rt::IOfflineVoiceTaskSession *>(base.get());
            require(session != nullptr, "missing cloning interface");
            base->prepare({});
            rejected([&] { session->run(request); }, "clone without reference was accepted");
            request.voice = first_voice;
            require(clone == save(session->run(request), "clone-task"), "clone and TTS conditioning differ");
            std::cout << "PASS: clone task and missing-reference recovery" << std::endl;
        }
        {
            auto base = model->create_task_session({rt::VoiceTaskKind::VoiceCloning, rt::RunMode::Streaming}, options);
            auto * session = dynamic_cast<rt::IStreamingVoiceTaskSession *>(base.get());
            require(session != nullptr, "missing streaming interface");
            base->prepare(rt::build_preparation_request(request));
            std::vector<float> chunks;
            size_t events = 0;
            session->set_stream_event_sink([&](const rt::StreamEvent & event) {
                if (!event.audio_output) return;
                ++events;
                chunks.insert(chunks.end(), event.audio_output->samples.begin(), event.audio_output->samples.end());
            });
            session->start_stream(request);
            const auto stream = save(session->finish_stream(), "clone-stream");
            require(events > 1 && chunks == stream && stream.size() == clone.size(), "streamed chunks differ from final audio");
            // Offline synthesis alone can peak-normalize the decoder output.
            float peak = 0;
            for (float value : stream) peak = std::max(peak, std::abs(value));
            const float gain = peak > 1 ? 0.95F / peak : 1.0F;
            for (size_t i = 0; i < stream.size(); ++i)
                require(std::abs(stream[i] * gain - clone[i]) < 1e-6F, "streaming changed generated speech");
            std::cout << "PASS: streaming clone, events=" << events << std::endl;
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
