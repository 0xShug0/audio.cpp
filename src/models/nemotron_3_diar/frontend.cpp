#include "engine/models/nemotron_3_diar/frontend.h"
#include "engine/models/nemotron_3_diar/streaming.h"

#include <algorithm>
#include <stdexcept>

namespace engine::models::nemotron_3_diar {

audio::NemoMelFrontend make_frontend(const Assets & assets) {
    const auto & source = assets.feature_config;
    audio::NemoMelFrontendConfig config;
    config.sample_rate = source.sample_rate;
    config.n_mels = source.num_mel_bins;
    config.stft = {source.n_fft, source.hop_length, source.win_length,
                   true, audio::STFTPadMode::Constant, audio::STFTFamily::Default};
    config.window = audio::MelWindow::SymmetricPrecise;
    config.input_rate = audio::MelInputRate::RequireMatch;
    config.preemphasis = source.preemphasis;
    config.mel_path = audio::MelPath::LogMelSpectrogram;
    config.frame_multiple = 1;
    config.pad_basis = audio::PadBasis::ValidFrames;
    config.mel_bank = audio::MelBank::FromArgument;
    return audio::NemoMelFrontend(
        std::move(config), {},
        audio::AudioTensor{assets.mel_filterbank, {source.num_mel_bins, source.n_fft / 2 + 1}});
}

FeatureBatch compute_features(
    const std::vector<runtime::AudioBuffer> & audio,
    const Assets & assets,
    int64_t threads) {
    if (audio.empty()) {
        throw std::runtime_error("Nemotron 3 diarization batch must not be empty");
    }
    struct Row {
        int64_t frames = 0;
        std::vector<float> values;
    };
    std::vector<Row> rows;
    rows.reserve(audio.size());
    int64_t maximum_frames = 0;
    for (const auto & item : audio) {
        auto features = assets.frontend->extract_audio(
            item.samples, item.sample_rate, item.channels,
            {true, audio::ValidFrameRule::CeilHops},
            static_cast<size_t>(std::max<int64_t>(1, threads)));
        maximum_frames = std::max(maximum_frames, features.valid_frames);
        rows.push_back({features.valid_frames, std::move(features.values)});
    }
    const int64_t factor = assets.model_config.encoder.subsampling_factor;
    const int64_t mel_bins = assets.feature_config.num_mel_bins;
    FeatureBatch result;
    result.batch = static_cast<int64_t>(audio.size());
    result.encoder_frames = (maximum_frames + factor - 1) / factor;
    result.feature_frames = result.encoder_frames * factor;
    result.stacked.assign(static_cast<size_t>(
        result.batch * result.encoder_frames * factor * mel_bins), 0.0F);
    result.valid_feature_frames.reserve(rows.size());
    result.valid_encoder_frames.reserve(rows.size());
    for (size_t batch = 0; batch < rows.size(); ++batch) {
        const auto & row = rows[batch];
        result.valid_feature_frames.push_back(row.frames);
        result.valid_encoder_frames.push_back((row.frames + factor - 1) / factor);
        const size_t destination = batch * static_cast<size_t>(result.encoder_frames * factor * mel_bins);
        const size_t count = std::min(
            row.values.size(), static_cast<size_t>(row.frames * mel_bins));
        std::copy_n(row.values.begin(), count, result.stacked.begin() + static_cast<std::ptrdiff_t>(destination));
    }
    return result;
}

FeatureBatch compute_stream_features(
    const std::vector<StreamWindow> & windows,
    const Assets & assets,
    int64_t threads) {
    if (windows.empty()) throw std::runtime_error("Nemotron 3 diarization stream window batch is empty");
    struct Row {
        int64_t frames = 0;
        std::vector<float> values;
    };
    std::vector<Row> rows;
    rows.reserve(windows.size());
    int64_t maximum_frames = 0;
    for (const auto & window : windows) {
        auto features = assets.frontend->extract_mono(
            window.mono_samples,
            {window.mel_start == 0,
             window.mel_start == 0 ? audio::ValidFrameRule::CeilHops : audio::ValidFrameRule::StftFrames,
             window.mel_frames},
            static_cast<size_t>(std::max<int64_t>(1, threads)));
        if (features.valid_frames != window.mel_frames) {
            throw std::runtime_error("Nemotron 3 diarization streaming frontend geometry mismatch");
        }
        maximum_frames = std::max(maximum_frames, features.valid_frames);
        rows.push_back({features.valid_frames, std::move(features.values)});
    }
    const int64_t factor = assets.model_config.encoder.subsampling_factor;
    const int64_t mel_bins = assets.feature_config.num_mel_bins;
    FeatureBatch result;
    result.batch = static_cast<int64_t>(windows.size());
    result.encoder_frames = (maximum_frames + factor - 1) / factor;
    result.feature_frames = result.encoder_frames * factor;
    result.stacked.assign(static_cast<size_t>(
        result.batch * result.encoder_frames * factor * mel_bins), 0.0F);
    result.valid_feature_frames.reserve(rows.size());
    result.valid_encoder_frames.reserve(rows.size());
    for (size_t batch = 0; batch < rows.size(); ++batch) {
        result.valid_feature_frames.push_back(rows[batch].frames);
        result.valid_encoder_frames.push_back((rows[batch].frames + factor - 1) / factor);
        const size_t destination = batch * static_cast<size_t>(result.encoder_frames * factor * mel_bins);
        std::copy_n(
            rows[batch].values.begin(),
            std::min(rows[batch].values.size(), static_cast<size_t>(rows[batch].frames * mel_bins)),
            result.stacked.begin() + static_cast<std::ptrdiff_t>(destination));
    }
    return result;
}

}  // namespace engine::models::nemotron_3_diar
