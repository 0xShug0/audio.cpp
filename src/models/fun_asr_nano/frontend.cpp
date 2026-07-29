#include "engine/models/fun_asr_nano/frontend.h"

#include "engine/framework/audio/kaldi_fbank.h"

#include <stdexcept>
#include <utility>

namespace engine::models::fun_asr_nano {

FunAsrNanoFrontend::FunAsrNanoFrontend(FunAsrNanoFrontendConfig config)
    : config_(std::move(config)) {
  if (config_.sample_rate <= 0 || config_.feature_size <= 0 ||
      config_.frame_length_ms <= 0 || config_.frame_shift_ms <= 0 ||
      config_.lfr_m <= 0 || config_.lfr_n <= 0) {
    throw std::runtime_error(
        "Fun-ASR-Nano frontend configuration must be positive");
  }
}

FunAsrNanoAudioFeatures
FunAsrNanoFrontend::extract(const std::vector<float> &audio,
                            int sample_rate) const {
  if (sample_rate != config_.sample_rate) {
    throw std::runtime_error(
        "Fun-ASR-Nano frontend requires 16 kHz input audio");
  }
  engine::audio::KaldiFbankOptions options;
  options.sample_rate = config_.sample_rate;
  options.num_mels = static_cast<int>(config_.feature_size);
  options.frame_length_ms = static_cast<float>(config_.frame_length_ms);
  options.frame_shift_ms = static_cast<float>(config_.frame_shift_ms);
  options.lfr_m = static_cast<int>(config_.lfr_m);
  options.lfr_n = static_cast<int>(config_.lfr_n);
  options.preemphasis = config_.preemphasis;
  const auto features = engine::audio::extract_kaldi_fbank(audio, options);

  FunAsrNanoAudioFeatures output;
  output.values = features.values;
  output.frames = features.frames;
  output.feature_dim = features.feature_dim;
  output.valid_frames = features.frames;
  return output;
}

} // namespace engine::models::fun_asr_nano
