#include "engine/framework/audio/mel_spectrogram_frontend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using engine::audio::MelLogPrecision;
using engine::audio::MelFilterbankNormalization;
using engine::audio::MelHannWindow;
using engine::audio::MelOutputLayout;
using engine::audio::MelValueTransform;
using engine::audio::MelSpectrumMode;
using engine::audio::MelFilterbankProjection;
using engine::audio::MelSpectrogramFeatures;
using engine::audio::MelSpectrogramFrontendConfig;

MelSpectrogramFeatures
legacy_projection(const std::vector<float> &planar, int64_t channels,
                  const MelSpectrogramFrontendConfig &config) {
  const int64_t samples = static_cast<int64_t>(planar.size()) / channels;
  const auto window_family = config.window == MelHannWindow::Periodic
      ? engine::audio::STFTFamily::Kokoro
      : engine::audio::STFTFamily::Default;
  const engine::audio::STFTConfig stft_config{
      config.n_fft,  config.hop_length, config.win_length,
      config.stft_center, config.stft_pad_mode, window_family,
  };
  const auto &window = engine::audio::get_cached_stft_window(stft_config);
  const auto magnitude = engine::audio::STFT().compute_magnitude(
      planar, window, channels, samples, stft_config, 1);
  const int64_t freq_bins = magnitude.shape[1];
  const int64_t stft_frames = magnitude.shape[2];
  const int64_t frames =
      config.max_frames > 0
          ? std::min(stft_frames - config.drop_last_frames, config.max_frames)
          : stft_frames - config.drop_last_frames;
  const auto filterbank = engine::audio::MelFilterbank().build({
      config.sample_rate,
      config.n_fft,
      config.n_mels,
      config.mel_fmin,
      config.mel_fmax,
      config.filterbank_normalization == MelFilterbankNormalization::Slaney,
  });
  const auto sparse = engine::audio::MelFilterbank().prepare_sparse(filterbank);

  MelSpectrogramFeatures out;
  out.channels = channels;
  out.frames = ((frames + config.frame_multiple - 1) / config.frame_multiple) *
               config.frame_multiple;
  out.mel_bins = config.n_mels;
  out.values.assign(static_cast<size_t>(channels * out.frames * config.n_mels),
                    config.frame_pad_value);

  for (int64_t channel = 0; channel < channels; ++channel) {
    for (int64_t mel = 0; mel < config.n_mels; ++mel) {
      for (int64_t frame = 0; frame < frames; ++frame) {
        float sum = 0.0f;
        double double_sum = 0.0;
        long double wide_sum = 0.0;
        const int64_t start = config.filterbank_projection == MelFilterbankProjection::SparseF32
                                  ? sparse.starts[static_cast<size_t>(mel)]
                                  : 0;
        const int64_t end = config.filterbank_projection == MelFilterbankProjection::SparseF32
                                ? sparse.ends[static_cast<size_t>(mel)]
                                : freq_bins;
        for (int64_t freq = start; freq < end; ++freq) {
          float value = magnitude.values[static_cast<size_t>(
              ((channel * freq_bins + freq) * stft_frames) + frame)];
          if (config.magnitude_epsilon > 0.0f) {
            value = std::sqrt(value * value + config.magnitude_epsilon);
          }
          if (config.spectrum_mode == MelSpectrumMode::PowerBeforeProjection) {
            value *= value;
          }
          const float weight =
              filterbank.values[static_cast<size_t>(mel * freq_bins + freq)];
          if (config.filterbank_projection == MelFilterbankProjection::DenseLongDouble) {
            wide_sum += static_cast<long double>(weight) *
                        static_cast<long double>(value);
          } else if (config.filterbank_projection == MelFilterbankProjection::DenseF64) {
            double_sum +=
                static_cast<double>(weight) * static_cast<double>(value);
          } else if (config.spectrum_mode == MelSpectrumMode::PowerDuringProjection) {
            sum += weight * value * value;
          } else {
            sum += weight * value;
          }
        }
        if (config.filterbank_projection == MelFilterbankProjection::DenseLongDouble) {
          sum = static_cast<float>(wide_sum);
        }
        const size_t output_index =
            config.layout == MelOutputLayout::FeatureMajor
                ? static_cast<size_t>(
                      ((channel * config.n_mels + mel) * out.frames) + frame)
                : static_cast<size_t>(
                      ((channel * out.frames + frame) * config.n_mels) + mel);
        if (config.filterbank_projection == MelFilterbankProjection::DenseF64) {
          out.values[output_index] =
              config.value_transform == MelValueTransform::None
                  ? static_cast<float>(double_sum)
              : config.value_transform == MelValueTransform::Log10
                  ? static_cast<float>(
                        std::log10(std::max(double_sum, config.log_floor)))
                  : static_cast<float>(
                        std::log(std::max(double_sum, config.log_floor)));
        } else {
          out.values[output_index] =
              config.value_transform == MelValueTransform::None ? sum
              : config.value_transform == MelValueTransform::Log10
                  ? config.log_precision == MelLogPrecision::F64
                        ? static_cast<float>(std::log10(std::max(
                              static_cast<double>(sum), config.log_floor)))
                        : std::log10(std::max(
                              sum, static_cast<float>(config.log_floor)))
              : config.log_precision == MelLogPrecision::F64
                  ? static_cast<float>(std::log(
                        std::max(static_cast<double>(sum), config.log_floor)))
                  : std::log(
                        std::max(sum, static_cast<float>(config.log_floor)));
        }
      }
    }
  }
  if (config.log_dynamic_range > 0.0f) {
    const float peak = *std::max_element(out.values.begin(), out.values.end());
    const float floor = peak - config.log_dynamic_range;
    for (float &value : out.values) {
      value = (std::max(value, floor) + config.log_shift) / config.log_divisor;
    }
  }
  return out;
}

std::vector<float> make_planar_waveform(int64_t channels, int64_t samples) {
  constexpr double kPi = 3.14159265358979323846;
  std::vector<float> waveform(static_cast<size_t>(channels * samples));
  for (int64_t channel = 0; channel < channels; ++channel) {
    for (int64_t sample = 0; sample < samples; ++sample) {
      const double time = static_cast<double>(sample) / 16000.0;
      waveform[static_cast<size_t>(channel * samples + sample)] =
          static_cast<float>(
              0.4 * std::sin(2.0 * kPi * (180.0 + 37.0 * channel) * time) +
              0.13 * std::cos(2.0 * kPi * 713.0 * time));
    }
  }
  return waveform;
}

void require_bitwise_equal(const MelSpectrogramFeatures &actual,
                           const MelSpectrogramFeatures &expected,
                           const std::string &label) {
  if (actual.channels != expected.channels ||
      actual.frames != expected.frames || actual.mel_bins != expected.mel_bins ||
      actual.values.size() != expected.values.size()) {
    throw std::runtime_error(label + " shape mismatch");
  }
  if (std::memcmp(actual.values.data(), expected.values.data(),
                  actual.values.size() * sizeof(float)) != 0) {
    for (size_t index = 0; index < actual.values.size(); ++index) {
      if (std::memcmp(&actual.values[index], &expected.values[index],
                      sizeof(float)) != 0) {
        throw std::runtime_error(label + " value mismatch at index " +
                                 std::to_string(index));
      }
    }
  }
}

void test_optimized_projection_matches_legacy_loop() {
  MelSpectrogramFrontendConfig base;
  base.sample_rate = 16000;
  base.n_fft = 64;
  base.hop_length = 16;
  base.win_length = 64;
  base.n_mels = 12;
  base.mel_fmin = 20.0f;
  base.mel_fmax = 7600.0f;
  base.stft_center = true;
  base.waveform_padding = engine::audio::MelWaveformPadding::None;
  base.frame_multiple = 4;
  base.frame_pad_value = 17.0f;
  const auto waveform = make_planar_waveform(2, 2053);

  std::vector<MelSpectrogramFrontendConfig> cases;
  cases.push_back(base);

  auto config = base;
  config.filterbank_projection = MelFilterbankProjection::SparseF32;
  config.magnitude_epsilon = 1.0e-9f;
  cases.push_back(config);

  config = base;
  config.spectrum_mode = MelSpectrumMode::PowerBeforeProjection;
  config.value_transform = MelValueTransform::Log10;
  config.layout = MelOutputLayout::TimeMajor;
  cases.push_back(config);

  config = base;
  config.filterbank_projection = MelFilterbankProjection::SparseF32;
  config.spectrum_mode = MelSpectrumMode::PowerDuringProjection;
  config.value_transform = MelValueTransform::Log10;
  config.log_dynamic_range = 8.0f;
  config.log_shift = 4.0f;
  config.log_divisor = 4.0f;
  cases.push_back(config);

  config = base;
  config.filterbank_projection = MelFilterbankProjection::DenseF64;
  config.magnitude_epsilon = 1.0e-9f;
  config.value_transform = MelValueTransform::Log10;
  config.layout = MelOutputLayout::TimeMajor;
  cases.push_back(config);

  config = base;
  config.filterbank_projection = MelFilterbankProjection::DenseLongDouble;
  config.log_precision = MelLogPrecision::F64;
  config.layout = MelOutputLayout::TimeMajor;
  cases.push_back(config);

  config = base;
  config.filterbank_projection = MelFilterbankProjection::DenseLongDouble;
  config.spectrum_mode = MelSpectrumMode::PowerBeforeProjection;
  config.value_transform = MelValueTransform::None;
  cases.push_back(config);

  for (size_t index = 0; index < cases.size(); ++index) {
    const auto expected = legacy_projection(waveform, 2, cases[index]);
    const auto actual = engine::audio::MelSpectrogramFrontend(cases[index])
                            .extract_planar(waveform, 2, 1);
    require_bitwise_equal(actual, expected,
                          "mel projection case " + std::to_string(index));
  }
}

} // namespace

int main() {
  try {
    test_optimized_projection_matches_legacy_loop();
    std::cout << "mel spectrogram frontend tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "mel spectrogram frontend test failed: " << error.what()
              << '\n';
    return 1;
  }
}
