#include "engine/community_models/lfm2_audio/audio_encoder.h"
#include "lfm2_audio_test_package.h"
#include "test_assert.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using engine::community_models::lfm2_audio::Lfm2AudioFeatureExtractor;
using engine::community_models::lfm2_audio::Lfm2AudioFeatures;
using engine::test::require;
using engine::test::require_close;
using engine::test::require_eq;

constexpr int64_t kMels = 128;
constexpr double kPi = 3.14159265358979323846;

// One second at 16 kHz: three tones, silence, then a quiet tone, so the
// features cover loud, empty and near-floor frames.
std::vector<float> test_signal() {
    std::vector<float> samples(16000);
    for (size_t i = 0; i < samples.size(); ++i) {
        const double t = static_cast<double>(i) / 16000.0;
        double value = 0.0;
        if (i < 6400) {
            value = 0.3 * std::sin(2.0 * kPi * 300.0 * t) + 0.2 * std::sin(2.0 * kPi * 1200.0 * t) +
                    0.1 * std::sin(2.0 * kPi * 3500.0 * t);
        } else if (i >= 9600) {
            value = 1e-3 * std::sin(2.0 * kPi * 800.0 * t);
        }

        samples[i] = static_cast<float>(value);
    }

    return samples;
}

float at(const Lfm2AudioFeatures & features, int64_t bin, int64_t frame) {
    return features.values.at(static_cast<size_t>(bin * features.frames + frame));
}

void test_matches_reference() {
    const auto features = Lfm2AudioFeatureExtractor(kMels, 1).extract(test_signal());
    require_eq(features.n_mels, kMels, "n_mels");
    require_eq(features.frames, 101, "frames");
    require_eq(features.values.size(), static_cast<size_t>(kMels * 101), "values");

    // chat.audio_in from liquid-audio 1.3.0's ChatState.add_audio on the same
    // signal, LFM2.5-Audio-1.5B revision c362a0625dfe45aa588dce5f0ada28a7e5707628.
    struct Expected {
        int64_t bin;
        int64_t frame;
        float value;
    };

    const Expected expected[] = {
        {8, 0, 3.610352f},
        {8, 20, 0.923038f},
        {24, 50, -0.637137f},
        {24, 60, -0.047616f},
        {36, 70, 0.428759f},
        {60, 30, 0.228873f},
        {100, 20, -0.091306f},
        {127, 90, -0.214723f},
    };

    for (const auto & e : expected) {
        require_close(at(features, e.bin, e.frame), e.value, 1e-4f,
            "features[" + std::to_string(e.bin) + "][" + std::to_string(e.frame) + "]");
    }
}

// Noise just above the 2^-24 log floor, so every bin varies by about 0.01.
// There, normalizing with std + 1e-5 (NeMo) instead of sqrt(var + 1e-5)
// moves the values by 0.1 to 1, and log(max(x, guard)) instead of
// log(x + guard) moves them further.
void test_near_log_floor() {
    lfm2_audio_test::Random random(3);
    std::vector<float> samples(16000);
    for (auto & sample : samples) {
        sample = random.uniform(3e-5f);
    }
    const auto features = Lfm2AudioFeatureExtractor(kMels, 1).extract(samples);

    // liquid-audio 1.3.0's ChatState.add_audio on the same samples. Bins with a
    // std of 0.01-0.012, where sqrt(var + 1e-5) moves these by 0.11 to 0.24 and
    // float rounding in the log stays well under the tolerance.
    struct Expected {
        int64_t bin;
        int64_t frame;
        float value;
    };
    const Expected expected[] = {
        {67, 65, 5.791136f},
        {68, 18, 4.063290f},
        {69, 81, 4.607155f},
        {70, 81, 4.619960f},
        {71, 51, 3.896761f},
        {74, 37, 3.126717f},
    };

    for (const auto & e : expected) {
        require_close(at(features, e.bin, e.frame), e.value, 2e-3f,
            "near-floor features[" + std::to_string(e.bin) + "][" + std::to_string(e.frame) + "]");
    }
}

// Digital silence: every frame of every bin is the same, so the features are
// exactly zero. With float32 statistics the rounding in the mean, divided by
// the 1e-5 std, came out near +-1, and the model read words into it.
void test_digital_silence() {
    for (const size_t seconds : {1, 5, 30}) {
        const auto features = Lfm2AudioFeatureExtractor(kMels, 1).extract(std::vector<float>(16000 * seconds, 0.0f));
        const bool zero = std::all_of(features.values.begin(), features.values.end(), [](float v) { return v == 0.0f; });
        require(zero, std::to_string(seconds) + " s of zeros must give zero features");
    }
}

void test_normalization() {
    const auto features = Lfm2AudioFeatureExtractor(kMels, 1).extract(test_signal());
    const int64_t valid = features.frames - 1;
    for (int64_t bin = 0; bin < kMels; ++bin) {
        double sum = 0.0;
        for (int64_t frame = 0; frame < valid; ++frame) {
            sum += at(features, bin, frame);
        }

        const double mean = sum / static_cast<double>(valid);
        double squares = 0.0;
        for (int64_t frame = 0; frame < valid; ++frame) {
            const double d = at(features, bin, frame) - mean;
            squares += d * d;
        }

        const double stddev = std::sqrt(squares / static_cast<double>(valid - 1));
        const auto label = "bin " + std::to_string(bin);
        require_close(static_cast<float>(mean), 0.0f, 1e-4f, label + " mean");
        require_close(static_cast<float>(stddev), 1.0f, 1e-4f, label + " stddev");
        require_eq(at(features, bin, valid), 0.0f, label + " last frame");
    }
}

void test_frame_count() {
    const Lfm2AudioFeatureExtractor extractor(kMels, 1);
    require_eq(extractor.extract(std::vector<float>(160, 0.1f)).frames, 2, "160 samples");
    require_eq(extractor.extract(std::vector<float>(479, 0.1f)).frames, 3, "479 samples");
    bool threw = false;
    try {
        (void)extractor.extract(std::vector<float>(159, 0.1f));
    } catch (const std::runtime_error &) {
        threw = true;
    }

    require(threw, "less than one hop of audio must be rejected");
}

void test_threads_do_not_change_features() {
    const auto samples = test_signal();
    const auto one = Lfm2AudioFeatureExtractor(kMels, 1).extract(samples);
    const auto four = Lfm2AudioFeatureExtractor(kMels, 4).extract(samples);
    require(one.values == four.values, "features differ between 1 and 4 threads");
}

}  // namespace

int main() {
    try {
        test_matches_reference();
        test_near_log_floor();
        test_digital_silence();
        test_normalization();
        test_frame_count();
        test_threads_do_not_change_features();
        std::cout << "lfm2_audio_features_test: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "lfm2_audio_features_test: " << error.what() << '\n';
        return 1;
    }
}
