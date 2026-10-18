#include "engine/models/sortformer_diar/frontend.h"

#include "engine/framework/audio/nemo_mel_frontend.h"

#include <algorithm>
#include <utility>

namespace engine::models::sortformer_diar {

audio::NemoMelFrontend make_sortformer_frontend(const SortformerAssets & assets) {
    const auto & source = assets.feature_config;
    audio::NemoMelFrontendConfig config;
    config.sample_rate = source.sample_rate;
    config.n_mels = source.num_mel_bins;
    config.stft = {source.n_fft, source.hop_length, source.win_length,
                   true, audio::STFTPadMode::Constant, audio::STFTFamily::Default};
    config.input_rate = audio::MelInputRate::RequireMatch;
    config.wave_scale = audio::WaveScale::DivideByMaxPlusEps;
    config.preemphasis = source.preemphasis;
    config.mel_path = audio::MelPath::LogMelSpectrogram;
    config.norm = audio::MelNorm::PerBinF32;
    config.frame_multiple = 16;
    config.pad_basis = audio::PadBasis::ValidFrames;
    return audio::NemoMelFrontend(std::move(config));
}

SortformerFeatureBatch compute_sortformer_features(
    const runtime::AudioBuffer & audio,
    const SortformerAssets & assets,
    int64_t threads,
    SortformerRunTimings * timings) {
    const auto mel = assets.frontend->extract_audio(
        audio.samples, audio.sample_rate, audio.channels,
        {true, audio::ValidFrameRule::FloorHops},
        static_cast<size_t>(std::max<int64_t>(1, threads)));
    if (timings != nullptr) {
        timings->log_mel_ms += mel.mel_ms;
        timings->feature_normalizer_ms += mel.normalize_ms;
    }
    SortformerFeatureBatch batch;
    batch.frames = mel.frames;
    batch.valid_frames = mel.valid_frames;
    batch.time_major = mel.values;
    return batch;
}

}  // namespace engine::models::sortformer_diar
