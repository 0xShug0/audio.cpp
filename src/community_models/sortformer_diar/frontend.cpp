#include "engine/community_models/sortformer_diar/frontend.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/waveform_ops.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::community_models::sortformer_diar {
namespace {

SortformerV2FeatureBatch compute_mono_features(
    std::vector<float> mono,
    const SortformerV2Assets & assets,
    int64_t sample_rate,
    int64_t threads,
    bool centered,
    bool cap_to_input_hops,
    int64_t expected_frames) {
    const auto & config = assets.feature_config;
    mono = audio::apply_preemphasis(std::move(mono), config.preemphasis);

    const audio::STFTConfig stft_config{
        config.n_fft,
        config.hop_length,
        config.win_length,
        centered,
        // NeMo/transcribe use constant padding for Sortformer. Mid-stream
        // windows already contain their left STFT pre-roll and therefore use
        // center=false; only the initial window needs centered padding.
        audio::STFTPadMode::Constant,
    };
    audio::AudioTensor mel;
    if (assets.mel_filterbank.empty()) {
        mel = audio::LogMelSpectrogram().compute(
            mono,
            1,
            static_cast<int64_t>(mono.size()),
            sample_rate,
            stft_config,
            config.num_mel_bins,
            static_cast<size_t>(std::max<int64_t>(1, threads)));
    } else {
        const audio::AudioTensor filterbank{
            assets.mel_filterbank,
            {config.num_mel_bins, config.n_fft / 2 + 1},
        };
        const auto & window = audio::get_cached_stft_window(stft_config);
        mel = audio::LogMelSpectrogram().compute(
            mono,
            window,
            1,
            static_cast<int64_t>(mono.size()),
            stft_config,
            filterbank,
            static_cast<size_t>(std::max<int64_t>(1, threads)));
    }
    const int64_t raw_frames = mel.shape[2];
    const int64_t hop_limited_frames =
        (static_cast<int64_t>(mono.size()) + config.hop_length - 1) / config.hop_length;
    int64_t valid_frames = cap_to_input_hops
        ? std::min<int64_t>(raw_frames, hop_limited_frames)
        : raw_frames;
    if (expected_frames >= 0) {
        if (raw_frames < expected_frames) {
            throw std::runtime_error("Sortformer v2 streaming frontend produced too few mel frames");
        }
        valid_frames = expected_frames;
    }

    SortformerV2FeatureBatch batch;
    batch.frames = ((valid_frames + 15) / 16) * 16;
    batch.valid_frames = valid_frames;
    batch.time_major.resize(
        static_cast<size_t>(batch.frames * config.num_mel_bins), 0.0f);
    for (int64_t t = 0; t < valid_frames; ++t) {
        for (int64_t m = 0; m < config.num_mel_bins; ++m) {
            batch.time_major[static_cast<size_t>(t * config.num_mel_bins + m)] =
                mel.values[static_cast<size_t>(m * raw_frames + t)];
        }
    }
    return batch;
}

}  // namespace

SortformerV2FeatureBatch compute_sortformer_v2_features(
    const runtime::AudioBuffer & audio,
    const SortformerV2Assets & assets,
    int64_t threads) {
    if (audio.sample_rate != assets.feature_config.sample_rate) {
        throw std::runtime_error("Sortformer v2 currently requires 16 kHz input audio");
    }
    auto mono = audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
    return compute_mono_features(
        std::move(mono), assets, audio.sample_rate, threads,
        /*centered=*/true, /*cap_to_input_hops=*/true, /*expected_frames=*/-1);
}

SortformerV2FeatureBatch compute_sortformer_v2_stream_features(
    const std::vector<float> & mono_samples,
    const SortformerV2Assets & assets,
    int64_t threads,
    bool initial_window,
    int64_t expected_frames) {
    return compute_mono_features(
        std::vector<float>(mono_samples), assets,
        assets.feature_config.sample_rate, threads,
        /*centered=*/initial_window, /*cap_to_input_hops=*/initial_window,
        expected_frames);
}

}  // namespace engine::community_models::sortformer_diar
