#include "engine/community_models/whistle_asr/frontend.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::community_models::whistle_asr {
namespace {

constexpr int64_t kRate = 16000;
constexpr size_t kMaximumSamples = 30 * kRate;
constexpr int64_t kFft = 512;
constexpr int64_t kWindow = 400;
constexpr int64_t kHop = 160;
constexpr int64_t kBins = kFft / 2 + 1;
constexpr int64_t kMels = 80;
constexpr size_t kGainBlock = 320;

audio::NemoMelFrontend make_frontend(std::vector<float> filterbank) {
    if (filterbank.size() != static_cast<size_t>(kBins * kMels) ||
        !std::all_of(filterbank.begin(), filterbank.end(), [](float value) {
            return std::isfinite(value) && value >= 0.0f;
        })) {
        throw std::invalid_argument("Whistle requires a 257-by-80 nonnegative mel filterbank");
    }
    // The extracted Whistle filterbank is bin-major; the shared frontend wants [n_mels, bins].
    audio::AudioTensor bank;
    bank.shape = {kMels, kBins};
    bank.values.resize(filterbank.size());
    for (int64_t bin = 0; bin < kBins; ++bin) {
        for (int64_t mel = 0; mel < kMels; ++mel) {
            bank.values[static_cast<size_t>(mel * kBins + bin)] =
                filterbank[static_cast<size_t>(bin * kMels + mel)];
        }
    }
    audio::NemoMelFrontendConfig config;
    config.sample_rate = kRate;
    config.n_mels = kMels;
    config.stft = {kFft, kHop, kWindow, true, audio::STFTPadMode::Constant, audio::STFTFamily::Default};
    config.input_rate = audio::MelInputRate::RequireMatch;
    config.window = audio::MelWindow::SymmetricPrecise;
    config.mel_bank = audio::MelBank::FromArgument;
    config.mel_path = audio::MelPath::ComplexPowerLn;
    config.log_zero_guard = 0x1p-24f;
    config.norm = audio::MelNorm::PerBinF64;
    config.layout = audio::MelLayout::TimeMajor;
    config.pad_basis = audio::PadBasis::ValidFrames;
    return audio::NemoMelFrontend(std::move(config), {}, std::move(bank));
}

// Whistle scales the waveform so the 99th-percentile block RMS lands at 0.1.
double percentile_gain(const std::vector<float> & samples) {
    const size_t blocks = samples.size() / kGainBlock;
    if (blocks == 0) {
        return 1.0;
    }
    std::vector<double> rms(blocks);
    for (size_t block = 0; block < blocks; ++block) {
        double energy = 0.0;
        for (size_t index = 0; index < kGainBlock; ++index) {
            const double value = samples[block * kGainBlock + index];
            energy += value * value;
        }
        rms[block] = std::sqrt(energy / static_cast<double>(kGainBlock));
    }
    const size_t percentile = static_cast<size_t>((blocks - 1) * 0.99);
    std::nth_element(rms.begin(), rms.begin() + static_cast<std::ptrdiff_t>(percentile), rms.end());
    return rms[percentile] > 1.0e-6 ? 0.1 / rms[percentile] : 1.0;
}

}  // namespace

WhistleFrontend::WhistleFrontend(std::vector<float> filterbank)
    : frontend_(make_frontend(std::move(filterbank))) {}

MelFeatures WhistleFrontend::extract(const std::vector<float> & samples) const {
    if (samples.size() > kMaximumSamples ||
        !std::all_of(samples.begin(), samples.end(), [](float sample) { return std::isfinite(sample); })) {
        throw std::invalid_argument("Whistle requires finite 16 kHz mono PCM no longer than 30 seconds");
    }
    const size_t frames = samples.size() / static_cast<size_t>(kHop);
    if (frames < 4) {
        return {};
    }
    const double gain = percentile_gain(samples);
    std::vector<float> scaled(samples.size());
    for (size_t index = 0; index < samples.size(); ++index) {
        scaled[index] = static_cast<float>(samples[index] * gain);
    }
    audio::NemoMelRunConfig run;
    run.center = true;
    run.valid_frame_rule = audio::ValidFrameRule::FloorHops;
    auto features = frontend_.extract_mono(std::move(scaled), run);
    if (features.frames != static_cast<int64_t>(frames)) {
        throw std::runtime_error("Whistle frontend produced an unexpected frame count");
    }
    MelFeatures result;
    result.frames = frames;
    result.values = std::move(features.values);
    return result;
}

}  // namespace engine::community_models::whistle_asr
