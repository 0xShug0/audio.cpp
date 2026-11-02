#include "engine/framework/runtime/registry.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/audio/wav_reader.h"
#include <iostream>

using namespace engine;

int main(int argc, char ** argv) {
    try {
        if (argc != 6)
            throw std::runtime_error("Usage: flashhead_session_test MODEL PORTRAIT AUDIO BACKEND LOG");
        debug::configure_logging({true, argv[5]});
        auto registry = runtime::make_default_registry();
        runtime::ModelLoadRequest load;
        load.model_path = argv[1];
        load.family_hint = "soulx_flashhead";
        auto model = registry.load(load);
        runtime::SessionOptions options;
        const std::string backend = argv[4];
        if (backend == "cuda")
            options.backend.type = core::BackendType::Cuda;
        else if (backend == "vulkan")
            options.backend.type = core::BackendType::Vulkan;
        else if (backend == "cpu")
            options.backend.type = core::BackendType::Cpu;
        else
            throw std::runtime_error("Unknown backend");
        options.backend.threads = 8;
        auto base =
            model->create_task_session({runtime::VoiceTaskKind::AudioGeneration, runtime::RunMode::Streaming}, options);
        auto * session = dynamic_cast<runtime::IStreamingVoiceTaskSession *>(base.get());
        if (!session)
            throw std::runtime_error("Missing streaming session");
        base->prepare({});
        runtime::TaskRequest request;
        request.options = {{"reference_image_file", argv[2]}, {"height", "256"}, {"width", "256"}, {"seed", "42"}};
        const auto wav = audio::read_wav_f32(std::filesystem::path(argv[3]));
        if (wav.sample_rate != 16000 || wav.channels != 1 || wav.samples.size() < 16000)
            throw std::runtime_error("Test requires at least one second of mono 16 kHz audio");
        const auto run = [&](size_t count, size_t block, bool callback) {
            std::vector<runtime::VoiceArtifact> output;
            session->set_stream_event_sink(callback ? runtime::StreamEventCallback([&](const auto & event) {
                output.insert(output.end(), event.output_artifacts.begin(), event.output_artifacts.end());
            })
                                                    : runtime::StreamEventCallback{});
            session->start_stream(request);
            for (size_t begin = 0; begin < count; begin += block) {
                const auto end = std::min(count, begin + block);
                auto event = session->process_audio_chunk(
                    {16000, 1, static_cast<int64_t>(begin), {wav.samples.begin() + begin, wav.samples.begin() + end}});
                output.insert(output.end(), event.output_artifacts.begin(), event.output_artifacts.end());
            }
            auto result = session->finalize();
            output.insert(output.end(), result.output_artifacts.begin(), result.output_artifacts.end());
            if (output.size() != (count + 15359) / 15360)
                throw std::runtime_error("Missing or duplicate video chunks");
            const auto height = std::stoll(request.options.at("height"));
            const auto width = std::stoll(request.options.at("width"));
            for (const auto & item : output)
                if (item.payload.size() != static_cast<size_t>(24 * height * width * 3))
                    throw std::runtime_error("Invalid RGB chunk size");
            return output;
        };
        const auto equal = [](const auto & a, const auto & b) {
            if (a.size() != b.size())
                throw std::runtime_error("Output chunk count changed");
            for (size_t i = 0; i < a.size(); ++i)
                if (a[i].payload != b[i].payload)
                    throw std::runtime_error("Repeated/fragmented request changed output");
        };
        const auto first = run(16000, 15360, true);
        equal(first, run(16000, 997, false));
        (void)run(4000, 4000, true);
        equal(first, run(16000, 16000, true));
        request.options["height"] = request.options["width"] = "512";
        (void)run(16000, 16000, true);
        request.options["height"] = request.options["width"] = "256";
        equal(first, run(16000, 16000, true));
        session->set_stream_event_sink({});
        auto * synchronous = dynamic_cast<runtime::IOfflineVoiceTaskSession *>(base.get());
        if (!synchronous)
            throw std::runtime_error("Missing synchronous run interface");
        request.audio_input = runtime::AudioBuffer{16000, 1, {wav.samples.begin(), wav.samples.begin() + 16000}};
        equal(first, synchronous->run(request).output_artifacts);
        equal(first, run(16000, 997, false));
        session->start_stream(request);
        bool empty_rejected = false;
        try {
            (void)session->finalize();
        } catch (const std::exception &) {
            empty_rejected = true;
        }
        if (!empty_rejected)
            throw std::runtime_error("Empty stream accepted");
        base.reset();
        auto offline_base =
            model->create_task_session({runtime::VoiceTaskKind::AudioGeneration, runtime::RunMode::Offline}, options);
        auto * offline = dynamic_cast<runtime::IOfflineVoiceTaskSession *>(offline_base.get());
        if (!offline)
            throw std::runtime_error("Missing offline session");
        offline_base->prepare({});
        request.audio_input = runtime::AudioBuffer{16000, 1, {wav.samples.begin(), wav.samples.begin() + 16000}};
        equal(first, offline->run(request).output_artifacts);
        request.options["seed"] = "43";
        const auto changed = offline->run(request).output_artifacts;
        if (changed.front().payload == first.front().payload)
            throw std::runtime_error("Seed did not affect generation");
        std::cout << "PASS: repeated requests, long-short-long, shape switching, fragmented streaming, padded final chunk, callback and "
                     "pull delivery, offline equivalence, seed control, empty-stream rejection\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
