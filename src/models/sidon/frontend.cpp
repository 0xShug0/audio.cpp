#include "engine/models/sidon/frontend.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/kaldi_fbank.h"
#include "engine/framework/audio/resampling.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <stdexcept>

namespace engine::models::sidon {

std::vector<float> prepare_audio(const runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty())
        throw std::runtime_error("Sidon requires non-empty audio with a valid sample rate and channel count");
    float peak = 0;
    for (float value : audio.samples) peak = std::max(peak, std::abs(value));
    if (peak == 0) throw std::runtime_error("Sidon peak normalization requires non-silent input");
    auto normalized = audio.samples;
    for (float & value : normalized) value = float(0.9 * (double(value) / peak));
    auto mono = audio::mixdown_interleaved_to_mono_average(normalized, audio.channels);

    // The released demo applies torchaudio's 50 Hz high-pass biquad before resampling.
    const float omega = 2.0f * 3.14159265358979323846f * 50.0f / audio.sample_rate;
    const float cosine = std::cos(omega);
    const float alpha = std::sin(omega) / (2.0f * 0.707f);
    const float a0 = 1.0f + alpha;
    const float b0 = (1.0f + cosine) / 2.0f / a0;
    const float b1 = -(1.0f + cosine) / a0;
    const float a1 = -2.0f * cosine / a0;
    const float a2 = (1.0f - alpha) / a0;
    float x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (float & sample : mono) {
        const float x = sample;
        const float y = b0 * x + b1 * x1 + b0 * x2 - a1 * y1 - a2 * y2;
        x2 = x1;
        x1 = x;
        y2 = y1;
        y1 = y;
        sample = std::clamp(y, -1.0f, 1.0f);
    }
    if (audio.sample_rate != 16000)
        mono = audio::resample_mono_torchaudio_sinc_hann(mono, audio.sample_rate, 16000);
    mono.resize(mono.size() + 24000, 0.0f);
    return mono;
}

modules::Wav2Vec2BertEncoderInput extract_features(const std::vector<float> & padded_audio, int threads) {
    audio::KaldiFbankOptions options;
    options.window_type = audio::KaldiFbankWindowType::Povey;
    options.lfr_m = 1;
    options.lfr_n = 1;
    options.upscale_samples = true;
    if (padded_audio.size() < 560) throw std::runtime_error("Sidon frontend needs at least two mel frames");
    const int frames = int((padded_audio.size() - 400) / 160 + 1);
    const int workers = std::clamp(threads, 1, std::max(1, frames / 128));
    audio::KaldiFbankFeatures features;
    if (workers == 1) {
        features = audio::extract_kaldi_fbank(padded_audio, options);
    } else {
        std::vector<std::future<audio::KaldiFbankFeatures>> pending;
        for (int worker = 0; worker < workers; ++worker) {
            const int begin = frames * worker / workers;
            const int end = frames * (worker + 1) / workers;
            pending.push_back(std::async(std::launch::async, [&, begin, end] {
                // Each frame is independent; retain the final window's full 400 samples.
                std::vector<float> part(padded_audio.begin() + begin * 160,
                    padded_audio.begin() + (end - 1) * 160 + 400);
                return audio::extract_kaldi_fbank(part, options);
            }));
        }
        features.frames = frames;
        features.feature_dim = 80;
        features.values.reserve(size_t(frames) * 80);
        for (auto & task : pending) {
            auto part = task.get();
            features.values.insert(features.values.end(), part.values.begin(), part.values.end());
        }
    }
    if (features.frames < 2) throw std::runtime_error("Sidon frontend needs at least two mel frames");
    for (int channel = 0; channel < 80; ++channel) {
        double mean = 0;
        for (int t = 0; t < features.frames; ++t) mean += features.values[t * 80 + channel];
        mean /= features.frames;
        double variance = 0;
        for (int t = 0; t < features.frames; ++t) {
            const double centered = features.values[t * 80 + channel] - mean;
            variance += centered * centered;
        }
        const double stddev = std::sqrt(variance / (features.frames - 1) + 1e-7);
        for (int t = 0; t < features.frames; ++t)
            features.values[t * 80 + channel] = float((features.values[t * 80 + channel] - mean) / stddev);
    }
    if (features.frames % 2) {
        features.values.resize(features.values.size() + 80, 1.0f);
        ++features.frames;
    }
    modules::Wav2Vec2BertEncoderInput output;
    output.frames = features.frames / 2;
    output.dims = 160;
    output.values = std::move(features.values);
    output.attention_mask.assign(output.frames, 1);
    return output;
}

}  // namespace engine::models::sidon
