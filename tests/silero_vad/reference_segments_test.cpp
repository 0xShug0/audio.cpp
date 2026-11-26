// The bundled Silero VAD weights are the official snakers4/silero-vad v6.2 model: offline VAD with
// default options on assets/resources/sample_16k.wav returns the segments silero-vad 6.2's
// get_speech_timestamps(model, sampling_rate=16000) gives (default parameters, TorchScript, CPU),
// each boundary within one 512-sample window to allow for platform float differences. The weights
// bundled before tools/gguf/silero_vad_weights_prepare.py missed by three windows.
//
// Usage: silero_vad_reference_segments_test <silero_vad model directory> <sample_16k.wav>

#include "engine/framework/audio/conversion.h"
#include "engine/framework/runtime/registry.h"
#include "../unittests/test_assert.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace rt = engine::runtime;

namespace {

struct Segment {
    int64_t start;
    int64_t end;
};

// silero-vad 6.2.3, get_speech_timestamps(wav, model, sampling_rate=16000), v6.2 TorchScript model.
const std::vector<Segment> kReference = {{8224, 33760}, {47136, 81376}, {94240, 156128}, {170528, 222176}};
constexpr int64_t kToleranceSamples = 512;  // one Silero window at 16 kHz

std::string describe(const std::vector<rt::SpeechSegment> & segments) {
    std::string out;
    for (const auto & s : segments) {
        out += " [" + std::to_string(s.span.start_sample) + ", " + std::to_string(s.span.end_sample) + "]";
    }
    return out;
}

}  // namespace

int main(int argc, char ** argv) try {
    using engine::test::require;
    require(argc == 3, "usage: silero_vad_reference_segments_test <silero_vad model directory> <sample_16k.wav>");

    rt::ModelLoadRequest load;
    load.model_path = argv[1];
    const auto model = rt::load_vad_model(load);
    const auto task_session =
        model->create_task_session(rt::TaskSpec{rt::VoiceTaskKind::Vad, rt::RunMode::Offline}, rt::SessionOptions{});
    auto * session = dynamic_cast<rt::IOfflineVoiceTaskSession *>(task_session.get());
    require(session != nullptr, "Silero VAD has no offline session");

    const auto wav = engine::audio::read_wav_f32(std::filesystem::path(argv[2]));
    require(wav.sample_rate == 16000 && wav.channels == 1, "sample_16k.wav must be 16 kHz mono");
    rt::TaskRequest request;
    request.audio_input = rt::AudioBuffer{wav.sample_rate, wav.channels, wav.samples};
    session->prepare(rt::build_preparation_request(request));
    const auto result = session->run(request);

    const auto & segments = result.speech_segments;
    const std::string got = "segments:" + describe(segments);
    require(segments.size() == kReference.size(),
            "expected " + std::to_string(kReference.size()) + " segments as silero-vad 6.2, got " + got);
    for (size_t i = 0; i < segments.size(); ++i) {
        const int64_t start_diff = std::llabs(segments[i].span.start_sample - kReference[i].start);
        const int64_t end_diff = std::llabs(segments[i].span.end_sample - kReference[i].end);
        require(start_diff <= kToleranceSamples && end_diff <= kToleranceSamples,
                "segment " + std::to_string(i) + " is off silero-vad 6.2's [" + std::to_string(kReference[i].start) + ", " +
                    std::to_string(kReference[i].end) + "] by more than one window; " + got);
    }
    std::cout << "Silero VAD matches silero-vad 6.2 on sample_16k.wav:" << describe(segments) << "\n";
    return 0;
} catch (const std::exception & e) {
    std::cerr << e.what() << '\n';
    return 1;
}
