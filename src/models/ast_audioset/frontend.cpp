#include "engine/models/ast_audioset/frontend.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/kaldi_fbank.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace engine::models::ast_audioset {

std::vector<float> extract_features(
    const runtime::AudioBuffer & audio,
    const Config & config) {
    const auto started = std::chrono::steady_clock::now();
    if (audio.samples.empty() || audio.sample_rate <= 0 || audio.channels <= 0) {
        throw std::runtime_error("AST requires non-empty audio");
    }
    const auto mono = engine::audio::convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
        audio.samples,
        audio.sample_rate,
        audio.channels,
        config.sample_rate,
        engine::audio::torchaudio_sinc_hann_float32_options(),
        engine::audio::MonoMixAccumulation::Float32);
    engine::audio::KaldiFbankOptions options;
    options.sample_rate = config.sample_rate;
    options.num_mels = static_cast<int>(config.num_mel_bins);
    options.frame_length_ms = 25.0f;
    options.frame_shift_ms = 10.0f;
    options.window_type = engine::audio::KaldiFbankWindowType::Hanning;
    options.lfr_m = 1;
    options.lfr_n = 1;
    options.preemphasis = 0.97f;
    options.low_frequency = 20.0f;
    options.high_frequency = 0.0f;
    options.remove_dc_offset = true;
    options.upscale_samples = false;
    options.sparse_filterbank = true;
    const auto fbank = engine::audio::extract_kaldi_fbank(mono, options);

    const float denominator = 2.0f * config.feature_std;
    const float padding_value = -config.feature_mean / denominator;
    std::vector<float> output(
        static_cast<size_t>(config.max_length * config.num_mel_bins),
        padding_value);
    const int64_t frames = std::min<int64_t>(fbank.frames, config.max_length);
    for (int64_t frame = 0; frame < frames; ++frame) {
        for (int64_t mel = 0; mel < config.num_mel_bins; ++mel) {
            output[static_cast<size_t>(mel * config.max_length + frame)] =
                (fbank.values[static_cast<size_t>(frame * config.num_mel_bins + mel)] -
                 config.feature_mean) /
                denominator;
        }
    }
    debug::timing_log_scalar("ast_audioset.frontend_ms", debug::elapsed_ms(started));
    return output;
}

}  // namespace engine::models::ast_audioset
