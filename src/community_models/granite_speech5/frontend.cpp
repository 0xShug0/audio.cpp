#include "engine/community_models/granite_speech5/frontend.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::community_models::granite_speech5 {
namespace {

audio::NemoMelFrontend make_frontend(const std::shared_ptr<const GraniteSpeech5Assets> & assets) {
    if (assets == nullptr) {
        throw std::runtime_error("granite_speech5 frontend requires valid assets");
    }
    const auto & source = assets->config.frontend;
    audio::NemoMelFrontendConfig config;
    config.sample_rate = source.sample_rate;
    config.n_mels = source.n_mels;
    config.stft = {source.n_fft, source.hop_length, source.win_length, true,
                   audio::STFTPadMode::Reflect, audio::STFTFamily::Default};
    config.window = audio::MelWindow::PeriodicF64;
    config.mel_bank = audio::MelBank::HTK;
    config.mel_path = audio::MelPath::DensePowerLog10;
    config.log_zero_guard = 1e-10f;
    config.max_log_drop = 8.0f;
    config.log_scale = 0.25f;
    config.log_bias = 1.0f;
    config.add_deltas = false;
    config.stack_frames = 1;
    config.layout = audio::MelLayout::TimeMajor;
    return audio::NemoMelFrontend(std::move(config));
}

void validate_audio_buffer(const runtime::AudioBuffer & audio) {
    if (audio.channels <= 0) {
        throw std::runtime_error("granite_speech5 frontend requires a positive channel count");
    }
    if (audio.sample_rate <= 0) {
        throw std::runtime_error("granite_speech5 frontend requires a positive sample rate");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("granite_speech5 frontend interleaved samples do not match the channel count");
    }
    for (float sample : audio.samples) {
        if (!std::isfinite(sample)) {
            throw std::runtime_error("granite_speech5 frontend requires finite amplitudes");
        }
    }
}

}  // namespace

GraniteSpeech5Frontend::GraniteSpeech5Frontend(
    std::shared_ptr<const GraniteSpeech5Assets> assets)
    : frontend_(make_frontend(assets)) {}

GraniteSpeech5FrontendFeatures GraniteSpeech5Frontend::extract(
    const runtime::AudioBuffer & audio) const {
    validate_audio_buffer(audio);

    GraniteSpeech5FrontendFeatures features;
    if (audio.samples.empty()) {
        return features;
    }

    // Downmix + resample to 16 kHz mono through the framework path. All
    // lengths below derive from this post-resample count.
    const std::vector<float> mono = frontend_.prepare_audio(audio.samples, audio.sample_rate, audio.channels);
    const auto post_resample_samples = static_cast<int64_t>(mono.size());
    if (post_resample_samples == 0) {
        return features;
    }
    if (granite_speech5_clip_too_short(post_resample_samples)) {
        throw std::runtime_error(
            "granite_speech5 clip too short after resample; Task D must merge tails of <=256 samples");
    }
    features.post_resample_samples = post_resample_samples;
    features.lengths = granite_speech5_lengths_for_samples(post_resample_samples);

    audio::NemoMelRunConfig run;
    run.center = true;
    run.valid_frame_rule = audio::ValidFrameRule::StftFrames;
    const auto mel = frontend_.extract_mono(mono, run);
    if (mel.frames != features.lengths.mel_frames) {
        throw std::runtime_error(
            "granite_speech5 frontend mel frame count does not match the length contract");
    }
    if (mel.feature_size != 80) {
        throw std::runtime_error("granite_speech5 frontend mel width is not 80");
    }
    features.mel = mel.values;
    features.mel_frames = mel.frames;

    // Explicit pairwise stacking: row t concatenates raw frames 2t and 2t+1.
    // A trailing odd frame is dropped, matching the reference restack.
    features.encoder_steps = features.lengths.encoder_steps;
    features.stacked.assign(
        static_cast<size_t>(features.encoder_steps * kGraniteSpeech5StackWidth), 0.0f);
    for (int64_t step = 0; step < features.encoder_steps; ++step) {
        for (int64_t half = 0; half < 2; ++half) {
            const int64_t frame = 2 * step + half;
            const float * src = features.mel.data() + static_cast<size_t>(frame * 80);
            float * dst = features.stacked.data() +
                static_cast<size_t>(step * kGraniteSpeech5StackWidth + half * 80);
            for (int64_t m = 0; m < 80; ++m) {
                dst[m] = src[m];
            }
        }
    }
    return features;
}

}  // namespace engine::community_models::granite_speech5
