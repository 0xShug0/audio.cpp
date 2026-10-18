#include "engine/community_models/sortformer_diar/frontend.h"

#include "engine/framework/audio/nemo_mel_frontend.h"

#include <algorithm>
#include <utility>

namespace engine::community_models::sortformer_diar {

audio::NemoMelFrontend make_sortformer_v2_frontend(const SortformerV2Assets & assets) {
    const auto & source = assets.feature_config;
    audio::NemoMelFrontendConfig config;
    config.sample_rate = source.sample_rate;
    config.n_mels = source.num_mel_bins;
    config.stft = {source.n_fft, source.hop_length, source.win_length,
                   true, audio::STFTPadMode::Constant, audio::STFTFamily::Default};
    config.input_rate = audio::MelInputRate::RequireMatch;
    config.preemphasis = source.preemphasis;
    config.mel_path = audio::MelPath::LogMelSpectrogram;
    config.frame_multiple = 16;
    config.pad_basis = audio::PadBasis::ValidFrames;
    if (assets.mel_filterbank.empty()) {
        return audio::NemoMelFrontend(std::move(config));
    }
    config.mel_bank = audio::MelBank::FromArgument;
    return audio::NemoMelFrontend(
        std::move(config), {},
        audio::AudioTensor{assets.mel_filterbank, {source.num_mel_bins, source.n_fft / 2 + 1}});
}

SortformerV2FeatureBatch compute_sortformer_v2_features(
    const runtime::AudioBuffer & audio,
    const SortformerV2Assets & assets,
    int64_t threads) {
    auto features = assets.frontend->extract_audio(
        audio.samples, audio.sample_rate, audio.channels,
        {true, audio::ValidFrameRule::CeilHops},
        static_cast<size_t>(std::max<int64_t>(1, threads)));
    SortformerV2FeatureBatch batch;
    batch.frames = features.frames;
    batch.valid_frames = features.valid_frames;
    batch.time_major = std::move(features.values);
    return batch;
}

SortformerV2FeatureBatch compute_sortformer_v2_stream_features(
    const std::vector<float> & mono_samples,
    const SortformerV2Assets & assets,
    int64_t threads,
    bool initial_window,
    int64_t expected_frames) {
    auto features = assets.frontend->extract_mono(
        mono_samples,
        {initial_window,
         initial_window ? audio::ValidFrameRule::CeilHops : audio::ValidFrameRule::StftFrames,
         expected_frames},
        static_cast<size_t>(std::max<int64_t>(1, threads)));
    SortformerV2FeatureBatch batch;
    batch.frames = features.frames;
    batch.valid_frames = features.valid_frames;
    batch.time_major = std::move(features.values);
    return batch;
}

}  // namespace engine::community_models::sortformer_diar
