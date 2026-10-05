#include "engine/community_models/lfm2_audio/detokenizer.h"
#include "engine/community_models/lfm2_audio/tts.h"
#include "engine/framework/audio/fft.h"
#include "lfm2_audio_test_package.h"
#include "test_assert.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace lfm2 = engine::community_models::lfm2_audio;
using engine::test::require;
using lfm2_audio_test::require_throws_with;

constexpr double kPi = 3.14159265358979323846;

void test_voices() {
    require(lfm2::lfm2_tts_voices("en") == std::vector<std::string>{"us_male", "us_female", "uk_male", "uk_female"}, "English voices");
    require(lfm2::lfm2_tts_voices("ja").empty(), "the Japanese checkpoint lists no voices");
    require(lfm2::lfm2_tts_system_prompt("en", "uk_female") == "Perform TTS. Use the UK female voice.", "UK female prompt");
    require(lfm2::lfm2_tts_system_prompt("en", "") == "Perform TTS. Use the US male voice.", "the first voice is the default");
    require(lfm2::lfm2_tts_system_prompt("ja", "") == "Perform TTS in japanese.", "Japanese prompt");

    require_throws_with([] { (void)lfm2::lfm2_tts_system_prompt("en", "US male"); }, "unknown LFM2-Audio voice", "voice ids, not descriptions");
    require_throws_with([] { (void)lfm2::lfm2_tts_system_prompt("ja", "us_male"); }, "has one voice", "Japanese takes no voice");
    require_throws_with([] { (void)lfm2::lfm2_tts_system_prompt("de", ""); }, "no TTS prompt", "an unknown language");
}

// The detokenizer's ISTFT inverts an STFT of the zero-padded signal with the
// same window, since overlap-add divides by the window envelope. Magnitudes
// far above 100 (log-magnitude > 4.6), which the framework's Vocos ISTFT would
// clamp, must come through.
void test_istft_inverts_stft() {
    const int64_t n_fft = 64;
    const int64_t hop = 16;
    const int64_t rows = 40;
    const int64_t bins = n_fft / 2 + 1;
    const int64_t pad = (n_fft - hop) / 2;

    std::vector<float> window(static_cast<size_t>(n_fft));
    for (int64_t i = 0; i < n_fft; ++i) {
        window[static_cast<size_t>(i)] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(i) / static_cast<double>(n_fft)));
    }

    lfm2_audio_test::Random random(5);
    std::vector<float> signal = random.uniform(static_cast<size_t>(rows * hop), 1.0f);
    for (auto & sample : signal) {
        sample *= 400.0f;
    }

    std::vector<float> padded(static_cast<size_t>((rows - 1) * hop + n_fft), 0.0f);
    std::copy(signal.begin(), signal.end(), padded.begin() + pad);

    std::vector<float> framed(static_cast<size_t>(rows * n_fft));
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t i = 0; i < n_fft; ++i) {
            framed[static_cast<size_t>(row * n_fft + i)] = padded[static_cast<size_t>(row * hop + i)] * window[static_cast<size_t>(i)];
        }
    }

    std::vector<std::complex<float>> spectrum(static_cast<size_t>(rows * bins));
    engine::audio::real_fft_forward(
        {static_cast<size_t>(rows), static_cast<size_t>(n_fft)},
        {static_cast<std::ptrdiff_t>(n_fft * sizeof(float)), static_cast<std::ptrdiff_t>(sizeof(float))},
        {static_cast<std::ptrdiff_t>(bins * sizeof(std::complex<float>)), static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))},
        1,
        framed.data(),
        spectrum.data());

    std::vector<float> log_magnitude_phase(static_cast<size_t>(rows * 2 * bins));
    float largest = 0.0f;
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t bin = 0; bin < bins; ++bin) {
            const auto value = spectrum[static_cast<size_t>(row * bins + bin)];
            log_magnitude_phase[static_cast<size_t>(row * 2 * bins + bin)] = std::log(std::abs(value));
            log_magnitude_phase[static_cast<size_t>(row * 2 * bins + bins + bin)] = std::arg(value);
            largest = std::max(largest, std::log(std::abs(value)));
        }
    }

    require(largest > 5.0f, "the test spectrum reaches magnitudes above 100");
    const auto out = lfm2::lfm2_audio_istft(log_magnitude_phase, rows, window, hop);
    require(out.size() == signal.size(), "ISTFT gives hop samples per row");

    double worst = 0.0;
    for (size_t i = 0; i < out.size(); ++i) {
        worst = std::max(worst, std::fabs(static_cast<double>(out[i]) - signal[i]));
    }

    require(worst < 1e-3 * 400.0, "ISTFT reconstructs the signal, worst error " + std::to_string(worst));

    require_throws_with([&] { (void)lfm2::lfm2_audio_istft(log_magnitude_phase, rows + 1, window, hop); }, "does not match the window",
        "a spectrum of the wrong size");

    // Streamed in uneven pieces, the rows give exactly the one-call samples.
    for (const int64_t piece : {int64_t{1}, int64_t{3}, int64_t{7}}) {
        lfm2::Lfm2StreamingIstft stream(window, hop);
        std::vector<float> streamed;
        for (int64_t row = 0; row < rows; row += piece) {
            const int64_t count = std::min(piece, rows - row);
            const std::vector<float> part(log_magnitude_phase.begin() + row * 2 * bins, log_magnitude_phase.begin() + (row + count) * 2 * bins);
            const auto samples = stream.push(part, count);
            // A sample is final once no later window reaches it.
            require(static_cast<int64_t>(streamed.size() + samples.size()) == std::max<int64_t>(0, (row + count) * hop - pad),
                "streamed samples after " + std::to_string(row + count) + " rows");
            streamed.insert(streamed.end(), samples.begin(), samples.end());
        }

        const auto rest = stream.finish();
        streamed.insert(streamed.end(), rest.begin(), rest.end());
        require(streamed == out, "ISTFT in pieces of " + std::to_string(piece) + " rows gives the one-call samples");
        // finish() emitted the samples a later row would add to.
        require_throws_with([&] { (void)stream.push(log_magnitude_phase, rows); }, "no rows after finish", "rows after finish()");
    }
}

// A frame speaks unless a codebook picked end-of-audio: the first, which ends
// the audio, or another, which liquid-audio's demo skips.
void test_speaking_frames() {
    const int32_t end = 2048;
    require(lfm2::lfm2_speaks({1, 2, 3, 4, 5, 6, 7, 2047}, end), "a frame of codes speaks");
    require(!lfm2::lfm2_speaks({end, end, end, end, end, end, end, end}, end), "the frame that ends the audio");
    require(!lfm2::lfm2_speaks({end, 2, 3, 4, 5, 6, 7, 8}, end), "end-of-audio first");
    require(!lfm2::lfm2_speaks({1, 2, 3, end, 5, 6, 7, 8}, end), "end-of-audio for another codebook");
}

}  // namespace

int main() {
    try {
        test_voices();
        test_istft_inverts_stft();
        test_speaking_frames();
        std::cout << "lfm2_audio_tts_test: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "lfm2_audio_tts_test: " << error.what() << '\n';
        return 1;
    }
}
