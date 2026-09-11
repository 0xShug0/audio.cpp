#include "engine/models/muscriptor/frontend.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::muscriptor {
namespace {

using Clock = std::chrono::steady_clock;

engine::audio::AudioTensor load_filterbank(const MuScriptorAssets & assets) {
    const auto & config = assets.config;
    auto values = assets.model_weights->require_f32(
        "condition_provider.conditioners.self_wav.mel_spec_transform.mel_scale.fb",
        {config.n_fft / 2 + 1, config.n_mels});
    engine::audio::AudioTensor dense;
    dense.shape = {config.n_mels, config.n_fft / 2 + 1};
    dense.values.assign(static_cast<size_t>(config.n_mels * (config.n_fft / 2 + 1)), 0.0F);
    for (int64_t f = 0; f < config.n_fft / 2 + 1; ++f) {
        for (int64_t m = 0; m < config.n_mels; ++m) {
            dense.values[static_cast<size_t>(m * (config.n_fft / 2 + 1) + f)] =
                values[static_cast<size_t>(f * config.n_mels + m)];
        }
    }
    return dense;
}

engine::audio::STFTConfig stft_config(const MuScriptorConfig & config) {
    engine::audio::STFTConfig out;
    out.n_fft = config.n_fft;
    out.hop_length = config.hop_length;
    out.win_length = config.n_fft;
    out.center = true;
    out.pad_mode = engine::audio::STFTPadMode::Reflect;
    out.family = engine::audio::STFTFamily::Kokoro;
    return out;
}

std::vector<float> log_mel_from_sparse_complex(
    const std::vector<float> & complex_stft,
    int64_t batch_index,
    int64_t batch,
    int64_t freq_bins,
    int64_t frames,
    int64_t n_mels,
    const engine::audio::SparseMelFilterbank & filterbank) {
    if (filterbank.dense.shape.size() != 2 || filterbank.dense.shape[0] != n_mels || filterbank.dense.shape[1] != freq_bins) {
        throw std::runtime_error("MuScriptor sparse filterbank shape mismatch");
    }
    if (static_cast<int64_t>(complex_stft.size()) != batch * freq_bins * frames * 2 ||
        batch_index < 0 || batch_index >= batch) {
        throw std::runtime_error("MuScriptor complex STFT tensor shape mismatch");
    }
    std::vector<float> log_mel(static_cast<size_t>(frames * n_mels), 0.0F);
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) if(frames * n_mels >= 4096)
#endif
    for (int64_t t = 0; t < frames; ++t) {
        for (int64_t m = 0; m < n_mels; ++m) {
            long double sum = 0.0L;
            const int64_t start = filterbank.starts[static_cast<size_t>(m)];
            const int64_t end = filterbank.ends[static_cast<size_t>(m)];
            for (int64_t f = start; f < end; ++f) {
                const size_t complex_base = static_cast<size_t>((((batch_index * freq_bins + f) * frames) + t) * 2);
                const float re = complex_stft[complex_base];
                const float im = complex_stft[complex_base + 1];
                const float mag = std::sqrt(re * re + im * im);
                sum += static_cast<long double>(filterbank.dense.values[static_cast<size_t>(m * freq_bins + f)]) *
                       static_cast<long double>(mag);
            }
            log_mel[static_cast<size_t>(t * n_mels + m)] = std::log(static_cast<float>(sum) + 1.0e-6F);
        }
    }
    return log_mel;
}

}  // namespace

MuScriptorFrontend::MuScriptorFrontend(std::shared_ptr<const MuScriptorAssets> assets, size_t threads)
    : assets_(std::move(assets)),
      threads_(threads) {
    if (assets_ == nullptr) {
        throw std::runtime_error("MuScriptor frontend requires assets");
    }
    filterbank_ = engine::audio::MelFilterbank().prepare_sparse(load_filterbank(*assets_));
}

std::vector<MuScriptorAudioChunk> MuScriptorFrontend::extract_chunks(const runtime::AudioBuffer & audio) const {
    const auto total_start = Clock::now();
    if (audio.sample_rate <= 0 || audio.channels <= 0) {
        throw std::runtime_error("MuScriptor audio input requires positive sample_rate and channels");
    }
    const auto & config = assets_->config;
    const auto resample_start = Clock::now();
    auto mono = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio.samples,
        audio.sample_rate,
        audio.channels,
        config.sample_rate);
    engine::debug::timing_log_scalar("muscriptor.frontend.resample_ms", engine::debug::elapsed_ms(resample_start));
    if (mono.empty()) {
        throw std::runtime_error("MuScriptor audio input is empty");
    }
    const int64_t segment_samples = static_cast<int64_t>(config.sample_rate) * config.segment_seconds;
    const int64_t chunks = (static_cast<int64_t>(mono.size()) + segment_samples - 1) / segment_samples;
    const auto segment_start = Clock::now();
    std::vector<float> segments(static_cast<size_t>(chunks * segment_samples), 0.0F);
    for (int64_t chunk = 0; chunk < chunks; ++chunk) {
        const int64_t start = chunk * segment_samples;
        const int64_t copy = std::min<int64_t>(segment_samples, static_cast<int64_t>(mono.size()) - start);
        std::copy_n(
            mono.begin() + static_cast<std::ptrdiff_t>(start),
            static_cast<size_t>(copy),
            segments.begin() + static_cast<std::ptrdiff_t>(chunk * segment_samples));
    }
    engine::debug::timing_log_scalar("muscriptor.frontend.segment_ms", engine::debug::elapsed_ms(segment_start));
    std::vector<MuScriptorAudioChunk> out;
    out.reserve(static_cast<size_t>(chunks));
    const auto cfg = stft_config(config);
    const auto & window = engine::audio::get_cached_stft_window(cfg);
    const auto stft_start = Clock::now();
    const auto complex_stft = engine::audio::STFT().compute_complex(segments, window, chunks, segment_samples, cfg, threads_);
    engine::debug::timing_log_scalar("muscriptor.frontend.stft_ms", engine::debug::elapsed_ms(stft_start));
    const int64_t freq_bins = complex_stft.shape.at(1);
    const int64_t frames = complex_stft.shape.at(2);
    const auto mel_start = Clock::now();
    for (int64_t chunk = 0; chunk < chunks; ++chunk) {
        MuScriptorAudioChunk audio_chunk;
        audio_chunk.log_mel =
            log_mel_from_sparse_complex(complex_stft.values, chunk, chunks, freq_bins, frames, config.n_mels, filterbank_);
        audio_chunk.mask.resize(static_cast<size_t>(frames), 0);
        const double valid_frames = static_cast<double>(segment_samples) / static_cast<double>(config.hop_length);
        for (int64_t t = 0; t < frames; ++t) {
            audio_chunk.mask[static_cast<size_t>(t)] = static_cast<double>(t) < valid_frames ? 1 : 0;
        }
        out.push_back(std::move(audio_chunk));
    }
    engine::debug::timing_log_scalar("muscriptor.frontend.mel_mask_ms", engine::debug::elapsed_ms(mel_start));
    engine::debug::timing_log_scalar("muscriptor.frontend.total_ms", engine::debug::elapsed_ms(total_start, Clock::now()));
    engine::debug::trace_log_scalar("muscriptor.frontend.chunks", chunks);
    return out;
}

}  // namespace engine::models::muscriptor
