// Silero VAD offline runs accept any sample rate: the input is resampled to 16 kHz and the
// segments come back in samples of the input. For 24, 44.1 and 48 kHz copies of a 16 kHz
// clip, every segment must lie inside the input, and speech / non-speech must agree with the
// clip's own segmentation on at least 95% of its duration. (Resampling twice moves the
// speech probabilities slightly, so a borderline burst may appear or vanish: segment counts
// are not compared.)
//
// Usage: silero_vad_resample_test <silero_vad model directory> <16 kHz speech wav>

#include "engine/framework/audio/conversion.h"
#include "engine/framework/runtime/registry.h"
#include "../unittests/test_assert.h"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace rt = engine::runtime;

namespace {

constexpr int kGridHz = 100;  // compare segmentations on a 10 ms grid

std::vector<rt::TimeSpan> run_vad(rt::IOfflineVoiceTaskSession & session, std::vector<float> mono, int sample_rate) {
    rt::TaskRequest request;
    request.audio_input = rt::AudioBuffer{sample_rate, 1, std::move(mono)};
    session.prepare(rt::build_preparation_request(request));
    const auto result = session.run(request);
    std::vector<rt::TimeSpan> spans;
    for (const auto & segment : result.speech_segments) {
        spans.push_back(segment.span);
    }
    return spans;
}

// Speech flag for each 10 ms cell whose centre lies inside a segment.
std::vector<bool> speech_mask(const std::vector<rt::TimeSpan> & spans, int sample_rate, size_t cells) {
    std::vector<bool> mask(cells, false);
    for (size_t cell = 0; cell < cells; ++cell) {
        const double centre = (static_cast<double>(cell) + 0.5) / kGridHz * sample_rate;
        for (const auto & span : spans) {
            if (centre >= static_cast<double>(span.start_sample) && centre < static_cast<double>(span.end_sample)) {
                mask[cell] = true;
                break;
            }
        }
    }
    return mask;
}

std::string describe(const std::vector<rt::TimeSpan> & spans, int sample_rate) {
    std::string out;
    for (const auto & span : spans) {
        out += " [" + std::to_string(static_cast<double>(span.start_sample) / sample_rate) + ", " +
               std::to_string(static_cast<double>(span.end_sample) / sample_rate) + "]";
    }
    return out;
}

}  // namespace

int main(int argc, char ** argv) try {
    using engine::test::require;
    require(argc == 3, "usage: silero_vad_resample_test <silero_vad model directory> <16 kHz speech wav>");

    rt::ModelLoadRequest load;
    load.model_path = argv[1];
    const auto model = rt::load_vad_model(load);
    const auto task_session =
        model->create_task_session(rt::TaskSpec{rt::VoiceTaskKind::Vad, rt::RunMode::Offline}, rt::SessionOptions{});
    auto * session = dynamic_cast<rt::IOfflineVoiceTaskSession *>(task_session.get());
    require(session != nullptr, "Silero VAD has no offline session");

    const auto wav = engine::audio::read_wav_f32(std::filesystem::path(argv[2]));
    require(wav.sample_rate == 16000, "the reference clip must be 16 kHz");
    const auto mono = wav.channels == 1 ? wav.samples
                                        : engine::audio::mixdown_interleaved_to_mono_average(wav.samples, wav.channels);
    const auto reference = run_vad(*session, mono, 16000);
    require(!reference.empty(), "no speech found in the reference clip");
    const size_t cells = mono.size() * kGridHz / 16000;
    const auto reference_mask = speech_mask(reference, 16000, cells);

    for (const int rate : {24000, 44100, 48000}) {
        const std::string label = std::to_string(rate) + " Hz";
        auto resampled = engine::audio::resample_mono_torchaudio_sinc_hann(mono, 16000, rate);
        const auto frames = static_cast<int64_t>(resampled.size());
        const auto spans = run_vad(*session, std::move(resampled), rate);
        const std::string segments =
            label + " segments (s):" + describe(spans, rate) + "; 16000 Hz segments (s):" + describe(reference, 16000);
        require(!spans.empty(), "no speech found at " + segments);
        for (const auto & span : spans) {
            require(span.start_sample >= 0 && span.start_sample < span.end_sample && span.end_sample <= frames,
                    "a segment outside the input at " + segments);
        }
        const auto mask = speech_mask(spans, rate, cells);
        size_t agree = 0;
        for (size_t cell = 0; cell < cells; ++cell) {
            agree += mask[cell] == reference_mask[cell] ? 1 : 0;
        }
        const double agreement = static_cast<double>(agree) / static_cast<double>(cells);
        require(agreement >= 0.95, "speech agrees on " + std::to_string(100.0 * agreement) + "% at " + segments);
        std::cout << label << ": " << spans.size() << " segments, speech agrees on " << 100.0 * agreement
                  << "% of the clip\n";
    }
    std::cout << "Silero VAD resampling tests passed (16000 Hz: " << reference.size() << " segments)\n";
    return 0;
} catch (const std::exception & e) {
    std::cerr << e.what() << '\n';
    return 1;
}
