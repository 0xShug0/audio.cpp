#include "engine/framework/runtime/registry.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/io/json.h"
#include "engine/framework/debug/trace.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>

int main(int argc, char ** argv) {
    try {
        if (argc < 3) throw std::runtime_error("usage: kitten_tts2_session_probe MODEL OUTPUT_DIR [THREADS] [WEIGHT_TYPE] [REFERENCE_MODEL_DIR] [cpu|cuda]");
        if (std::filesystem::path(argv[1]).extension() == ".gguf") {
            // Check stored precision, not just the runtime's upcast tensor types.
            // A misspelled namespace in converter overrides silently quantizes
            // these tensors and can otherwise escape an end-to-end smoke test.
            namespace assets = engine::assets;
            auto source = assets::open_tensor_source(argv[1]);
            const std::pair<const char *, assets::TensorStorageType> expected[] = {
                {"language_model/model.embed_tokens.weight", assets::TensorStorageType::F16},
                {"language_model/spk_proj.0.weight", assets::TensorStorageType::BF16},
                {"s3gen/flow.encoder_proj.weight", assets::TensorStorageType::F32},
                {"speaker/embedding.weight", assets::TensorStorageType::F32},
            };
            for (const auto & [name, type] : expected)
                if (assets::tensor_storage_type_for_dtype(source->require_metadata(name).dtype) != type)
                    throw std::runtime_error(std::string("incorrect packaged precision: ") + name);
            std::cout << "PASS: packaged tensor precision" << std::endl;
        }
        engine::debug::configure_logging({true,std::nullopt});
        engine::runtime::ModelLoadRequest load;
        load.model_path = argv[1];
        load.family_hint = "kitten_tts2";
        auto registry = engine::runtime::make_default_registry();
        auto model = registry.load(load);
        engine::runtime::SessionOptions options;
        options.backend.type = engine::core::BackendType::Cpu;
        if (argc > 6) {
            const std::string backend = argv[6];
            if (backend == "cuda") options.backend.type = engine::core::BackendType::Cuda;
            else if (backend != "cpu") throw std::runtime_error("probe backend must be cpu or cuda");
        }
        options.backend.threads = argc > 3 ? std::stoi(argv[3]) : 8;
        if (argc > 4) options.options["kitten_tts2.weight_type"] = argv[4];
        const engine::runtime::TaskSpec task{engine::runtime::VoiceTaskKind::Tts, engine::runtime::RunMode::Offline};
        auto base = model->create_task_session(task, options);
        auto * session = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(base.get());
        if (!session) throw std::runtime_error("expected offline session");
        base->prepare({});
        std::filesystem::create_directories(argv[2]);
        engine::runtime::TaskRequest request;
        request.text_input = engine::runtime::Transcript{"Hello there. This is a test of Kitten speech.", "en"};
        request.options = {{"seed", "1234"}, {"voice_id", "Bruno"}};
        std::vector<float> first;
        for (int i = 0; i < 4; ++i) {
            if (i == 2) request.options["voice_id"] = "Bella";
            if (i == 3) {
                request.options["voice_id"] = "Bruno";
                request.options["text_chunk_size"] = "100";
                request.text_input->text = "The morning sun fills the room with light. A small bird sings outside the window. "
                    "Today we are testing long form speech, with each sentence joining the next. "
                    "The same model and decoder stay loaded for all of these requests.";
            }
            const auto started = std::chrono::steady_clock::now();
            auto result = session->run(request);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            if (!result.audio_output || result.audio_output->sample_rate != 24000 || result.audio_output->channels != 1)
                throw std::runtime_error("incorrect audio format");
            const auto & audio = *result.audio_output;
            if (audio.samples.size() < 2400) throw std::runtime_error("audio is too short");
            double energy = 0;
            for (float value : audio.samples) {
                if (!std::isfinite(value)) throw std::runtime_error("non-finite audio");
                energy += value * value;
            }
            if (energy / audio.samples.size() < 1e-8) throw std::runtime_error("silent audio");
            const auto file = std::filesystem::path(argv[2]) / ("request_" + std::to_string(i) + ".wav");
            engine::audio::WavWriteOptions wav_options;
            wav_options.format = engine::audio::WavSampleFormat::Float32;
            engine::audio::write_wav(file, audio.sample_rate, audio.channels, audio.samples, wav_options);
            if (const auto * trace = std::getenv("AUDIOCPP_KITTEN_TTS2_TRACE_DIR"); trace && *trace) {
                const auto trace_output = std::filesystem::path(argv[2]) / ("request_" + std::to_string(i) + "_trace");
                std::filesystem::create_directories(trace_output);
                for (const auto * name : {"prompt.json", "prefill_logits.f32", "codes.json"})
                    std::filesystem::copy_file(std::filesystem::path(trace) / name, trace_output / name,
                        std::filesystem::copy_options::overwrite_existing);
            }
            if (i == 0) first = audio.samples;
            if (i == 1 && first != audio.samples) throw std::runtime_error("same-session seeded repeat differs");
            if (i == 2 && first == audio.samples) throw std::runtime_error("voice switch did not change output");
            std::cout << "request=" << i << " seconds=" << seconds << " audio_sec=" << audio.samples.size()/24000.0
                << " rtf=" << seconds/(audio.samples.size()/24000.0) << std::endl;
        }
        auto invalid = request;
        invalid.options["voice_id"] = "NoSuchVoice";
        bool rejected = false;
        try { session->run(invalid); } catch (const std::runtime_error &) { rejected = true; }
        if (!rejected) throw std::runtime_error("unknown voice was not rejected");
        std::cout << "PASS: seeded repeat, voice switch, long-form chunks, and unknown-voice error" << std::endl;
        if (argc > 5) {
            const auto root = std::filesystem::path(argv[5]);
            const auto voices = engine::io::json::parse_file(root/"voices/voices.json");
            auto reference = [&](const char * name) {
                const auto & item = voices.require(name);
                const auto wav = engine::audio::read_wav_f32(root/"voices"/item.require("reference").as_string());
                request.audio_input = engine::runtime::AudioBuffer{wav.sample_rate,wav.channels,wav.samples};
                request.options["reference_text"] = item.require("transcript").as_string();
            };
            request.text_input->text = "Hello there. This is a test of native voice cloning.";
            request.options = {{"seed","1234"}};
            reference("Bruno");
            auto invalid = request;
            invalid.options.erase("reference_text");
            rejected = false;
            try { session->run(invalid); } catch (const std::runtime_error &) { rejected = true; }
            if (!rejected) throw std::runtime_error("missing reference transcript was not rejected");
            invalid = request;
            invalid.audio_input->samples.resize(100);
            rejected = false;
            try { session->run(invalid); } catch (const std::runtime_error &) { rejected = true; }
            if (!rejected) throw std::runtime_error("short reference was not rejected");
            std::vector<float> cloned;
            for (int i = 0; i < 3; ++i) {
                if (i == 2) reference("Bella");
                const auto started = std::chrono::steady_clock::now();
                auto result = session->run(request);
                if (!result.audio_output || result.audio_output->samples.size() < 2400)
                    throw std::runtime_error("cloning returned no audio");
                const auto & audio = *result.audio_output;
                double energy = 0;
                for (float value : audio.samples) {
                    if (!std::isfinite(value)) throw std::runtime_error("cloning produced non-finite audio");
                    energy += value*value;
                }
                if (energy/audio.samples.size() < 1e-8) throw std::runtime_error("cloning produced silence");
                if (i == 0) cloned = audio.samples;
                if (i == 1 && cloned != audio.samples) throw std::runtime_error("cloned voice seeded repeat differs");
                if (i == 2 && cloned == audio.samples) throw std::runtime_error("changed reference reused old voice");
                engine::audio::WavWriteOptions wav_options;
                wav_options.format = engine::audio::WavSampleFormat::Float32;
                engine::audio::write_wav(std::filesystem::path(argv[2])/("clone_"+std::to_string(i)+".wav"),
                    audio.sample_rate, audio.channels, audio.samples, wav_options);
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
                std::cout << "clone=" << i << " seconds=" << seconds << " audio_sec=" << audio.samples.size()/24000.0
                    << " rtf=" << seconds/(audio.samples.size()/24000.0) << std::endl;
            }
            std::cout << "PASS: native cloning, cached repeat, changed reference, missing transcript and short clip errors" << std::endl;
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
