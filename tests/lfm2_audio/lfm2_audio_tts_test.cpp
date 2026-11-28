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
#include <limits>
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

std::vector<int32_t> picks(lfm2::Lfm2CodeSampler & sampler, const std::vector<float> & logits, size_t count) {
    std::vector<int32_t> out;
    for (size_t i = 0; i < count; ++i) {
        auto values = logits;
        out.push_back(sampler.pick(values));
    }

    return out;
}

// S2S text tokens follow liquid-audio's _sample_text_token: greedy by
// default, at temperature 0 and at top-k 1; otherwise the logits are divided
// by the temperature, those below the k-th are dropped (ties with it kept, as
// torch.topk's threshold keeps them) and a token is drawn.
void test_text_sampling() {
    lfm2_audio_test::Random random(11);
    for (const auto & sampling : {lfm2::Lfm2TextSampling{}, lfm2::Lfm2TextSampling{0.0f, 50}, lfm2::Lfm2TextSampling{0.7f, 1}}) {
        lfm2::Lfm2CodeSampler sampler(sampling, 7);
        const std::string label = "temperature " + std::to_string(sampling.temperature) + ", top-k " + std::to_string(sampling.top_k);
        for (int i = 0; i < 200; ++i) {
            auto logits = random.uniform(65536, 10.0f);
            const int32_t expected = lfm2::lfm2_greedy(logits);
            require(sampler.pick(logits) == expected, label + " is greedy");
        }

        std::vector<float> broken = {1.0f, std::nanf(""), 0.0f};
        require_throws_with([&] { (void)sampler.pick(broken); }, "non-finite logits", label + " with a NaN logit");
    }

    // Sampled text rejects them too, where HfSampler would draw from the
    // finite rest.
    for (const auto & sampling : {lfm2::Lfm2TextSampling{0.7f, 50}, lfm2::Lfm2TextSampling{1.0f, 0}}) {
        lfm2::Lfm2CodeSampler sampler(sampling, 7);
        const std::string label = "temperature " + std::to_string(sampling.temperature) + ", top-k " + std::to_string(sampling.top_k);
        for (const float value : {std::nanf(""), std::numeric_limits<float>::infinity()}) {
            std::vector<float> broken = {1.0f, value, 0.0f};
            require_throws_with([&] { (void)sampler.pick(broken); }, "non-finite logits", label + " with a logit of " + std::to_string(value));
        }
    }

    constexpr size_t kDraws = 20000;
    std::vector<size_t> counts(4, 0);
    lfm2::Lfm2CodeSampler top_2(lfm2::Lfm2TextSampling{1.0f, 2}, 7);
    for (const int32_t token : picks(top_2, {5.0f, 4.0f, 4.0f, 1.0f}, kDraws)) {
        ++counts.at(static_cast<size_t>(token));
    }

    require(counts[0] > 0 && counts[1] > 0 && counts[2] > 0 && counts[3] == 0, "top-k 2 keeps the tie with the second logit and drops the rest");

    // p(1) = 3 / (1 + 3) at temperature 1, and sqrt(3) / (1 + sqrt(3)) at 2;
    // within 4 standard deviations.
    for (const float temperature : {1.0f, 2.0f}) {
        lfm2::Lfm2CodeSampler sampler(lfm2::Lfm2TextSampling{temperature, 0}, 7);
        const auto drawn = picks(sampler, {0.0f, static_cast<float>(std::log(3.0))}, kDraws);
        const double share = static_cast<double>(std::count(drawn.begin(), drawn.end(), 1)) / kDraws;
        const double expected = std::pow(3.0, 1.0 / temperature) / (1.0 + std::pow(3.0, 1.0 / temperature));
        require(std::fabs(share - expected) < 4.0 * std::sqrt(expected * (1.0 - expected) / kDraws),
            "temperature " + std::to_string(temperature) + " draws token 1 with p " + std::to_string(share) + ", expected " +
                std::to_string(expected));
    }

    // Text draws come from a stream of their own: they leave the audio codes'
    // draws of the same seed as they were, and differ from them.
    const std::vector<float> flat(64, 0.0f);
    for (const uint64_t seed : {uint64_t{0}, uint64_t{1}, uint64_t{1234}}) {
        lfm2::Lfm2CodeSampler alone(lfm2::Lfm2AudioSampling{1.0f, 0, seed});
        lfm2::Lfm2CodeSampler audio(lfm2::Lfm2AudioSampling{1.0f, 0, seed});
        lfm2::Lfm2CodeSampler text(lfm2::Lfm2TextSampling{1.0f, 0}, seed);
        lfm2::Lfm2CodeSampler text_again(lfm2::Lfm2TextSampling{1.0f, 0}, seed);
        lfm2::Lfm2CodeSampler text_next(lfm2::Lfm2TextSampling{1.0f, 0}, seed + 1);
        std::vector<int32_t> interleaved;
        std::vector<int32_t> text_picks;
        for (int i = 0; i < 1000; ++i) {
            text_picks.push_back(picks(text, flat, 1).front());
            interleaved.push_back(picks(audio, flat, 1).front());
        }

        const std::string label = "seed " + std::to_string(seed);
        const auto audio_alone = picks(alone, flat, 1000);
        require(interleaved == audio_alone, label + ": text draws in between leave the audio draws as they were");
        require(text_picks != audio_alone, label + ": text draws differ from the audio draws");
        require(picks(text_again, flat, 1000) == text_picks, label + ": text draws repeat");
        require(picks(text_next, flat, 1000) != text_picks, label + ": the next seed draws other text");
    }
}

}  // namespace

int main() {
    try {
        test_voices();
        test_istft_inverts_stft();
        test_speaking_frames();
        test_text_sampling();
        std::cout << "lfm2_audio_tts_test: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "lfm2_audio_tts_test: " << error.what() << '\n';
        return 1;
    }
}
