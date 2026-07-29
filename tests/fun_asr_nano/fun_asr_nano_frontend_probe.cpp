#include "engine/framework/audio/kaldi_fbank.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/io/json.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kSampleRate = 16000;
constexpr float kAbsoluteTolerance = 2.0e-4F;
constexpr float kRelativeTolerance = 2.0e-4F;
constexpr const char *kTransformersCommit =
    "48e7f65fb274172e15aa88875d780c67c37606c7";

template <typename Callable>
void expect_runtime_error(Callable &&callable, const std::string &context) {
  try {
    callable();
  } catch (const std::runtime_error &) {
    return;
  }
  throw std::runtime_error(context + " did not reject invalid options");
}

std::vector<float> load_f32_slice(const std::filesystem::path &data_path,
                                  const engine::io::json::Value &descriptor) {
  const int64_t signed_offset = descriptor.require("offset_f32").as_i64();
  const int64_t signed_count = descriptor.require("count").as_i64();
  if (signed_offset < 0 || signed_count < 0 ||
      static_cast<uint64_t>(signed_count) >
          std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("invalid frontend reference data slice");
  }
  const auto offset = static_cast<uint64_t>(signed_offset);
  const auto count = static_cast<uint64_t>(signed_count);
  std::ifstream input(data_path, std::ios::binary);
  input.seekg(static_cast<std::streamoff>(offset * 4));
  if (!input) {
    throw std::runtime_error("could not seek in frontend reference data");
  }

  std::vector<float> values(static_cast<size_t>(count));
  for (float &value : values) {
    uint8_t bytes[4]{};
    input.read(reinterpret_cast<char *>(bytes), 4);
    if (!input) {
      throw std::runtime_error("truncated frontend reference data");
    }
    const uint32_t bits = static_cast<uint32_t>(bytes[0]) |
                          (static_cast<uint32_t>(bytes[1]) << 8) |
                          (static_cast<uint32_t>(bytes[2]) << 16) |
                          (static_cast<uint32_t>(bytes[3]) << 24);
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(value));
  }
  return values;
}

std::vector<float> downmix(const engine::audio::WavData &wav) {
  if (wav.sample_rate != kSampleRate || wav.channels <= 0) {
    throw std::runtime_error(
        "sample WAV must be 16 kHz with at least one channel");
  }
  if (wav.samples.size() % static_cast<size_t>(wav.channels) != 0) {
    throw std::runtime_error("sample WAV has an incomplete interleaved frame");
  }
  const size_t frames = wav.samples.size() / static_cast<size_t>(wav.channels);
  std::vector<float> mono(frames, 0.0F);
  for (size_t frame = 0; frame < frames; ++frame) {
    double sum = 0.0;
    for (int channel = 0; channel < wav.channels; ++channel) {
      sum += wav.samples[frame * static_cast<size_t>(wav.channels) +
                         static_cast<size_t>(channel)];
    }
    mono[frame] = static_cast<float>(sum / static_cast<double>(wav.channels));
  }
  return mono;
}

std::vector<float> make_fixture(const std::string &name,
                                const engine::io::json::Value &expected,
                                const std::filesystem::path &data_path,
                                const std::filesystem::path &sample_wav) {
  if (name == "sample_16k") {
    return downmix(engine::audio::read_wav_f32(sample_wav));
  }
  if (const auto *waveform = expected.find("waveform")) {
    return load_f32_slice(data_path, *waveform);
  }
  throw std::runtime_error(
      "synthetic frontend fixture is missing its reference waveform: " + name);
}

void compare_fixture(const std::string &name,
                     const engine::io::json::Value &expected,
                     const std::filesystem::path &data_path,
                     const std::filesystem::path &sample_wav) {
  engine::audio::KaldiFbankOptions options;
  options.sample_rate = kSampleRate;
  options.num_mels = 80;
  options.frame_length_ms = 25.0F;
  options.frame_shift_ms = 10.0F;
  options.lfr_m = 7;
  options.lfr_n = 6;
  options.upscale_samples = false;
  options.apply_cmvn = false;

  const auto audio = make_fixture(name, expected, data_path, sample_wav);
  const auto actual = engine::audio::extract_kaldi_fbank(audio, options);
  const int expected_frames =
      static_cast<int>(expected.require("frames").as_i64());
  const int expected_width =
      static_cast<int>(expected.require("width").as_i64());
  const auto expected_values =
      load_f32_slice(data_path, expected.require("features"));

  if (actual.frames != expected_frames ||
      actual.feature_dim != expected_width) {
    throw std::runtime_error(name + " shape mismatch: expected " +
                             std::to_string(expected_frames) + "x" +
                             std::to_string(expected_width) + ", got " +
                             std::to_string(actual.frames) + "x" +
                             std::to_string(actual.feature_dim));
  }
  if (actual.values.size() != expected_values.size()) {
    throw std::runtime_error(name + " flattened feature count mismatch");
  }

  float max_absolute_error = 0.0F;
  float max_relative_error = 0.0F;
  size_t worst_index = 0;
  for (size_t index = 0; index < actual.values.size(); ++index) {
    const float absolute_error =
        std::abs(actual.values[index] - expected_values[index]);
    const float relative_error =
        absolute_error / std::max(std::abs(expected_values[index]), 1.0e-12F);
    if (absolute_error > max_absolute_error) {
      max_absolute_error = absolute_error;
      worst_index = index;
    }
    max_relative_error = std::max(max_relative_error, relative_error);
    const float limit = kAbsoluteTolerance +
                        kRelativeTolerance * std::abs(expected_values[index]);
    if (absolute_error > limit) {
      throw std::runtime_error(
          name + " parity mismatch at index " + std::to_string(index) +
          ": expected " + std::to_string(expected_values[index]) + ", got " +
          std::to_string(actual.values[index]) + ", absolute error " +
          std::to_string(absolute_error) + ", limit " + std::to_string(limit));
    }
  }

  std::cout << name << ": " << actual.frames << "x" << actual.feature_dim
            << ", max_abs=" << max_absolute_error
            << ", max_rel=" << max_relative_error
            << ", worst_index=" << worst_index << '\n';
}

void check_frontend_contracts() {
  engine::audio::KaldiFbankOptions options;
  const auto short_result = engine::audio::extract_kaldi_fbank(
      std::vector<float>(399, 0.0F), options);
  if (short_result.frames != 0 || short_result.feature_dim != 560 ||
      !short_result.values.empty()) {
    throw std::runtime_error(
        "short audio must return an empty, dimensioned feature result");
  }

  auto invalid = options;
  invalid.lfr_n = 0;
  expect_runtime_error(
      [&] {
        static_cast<void>(engine::audio::extract_kaldi_fbank({}, invalid));
      },
      "zero LFR stride");

  invalid = options;
  invalid.low_frequency = 9000.0F;
  expect_runtime_error(
      [&] {
        static_cast<void>(engine::audio::extract_kaldi_fbank({}, invalid));
      },
      "frequency bounds");

  invalid = options;
  invalid.preemphasis = std::numeric_limits<float>::quiet_NaN();
  expect_runtime_error(
      [&] {
        static_cast<void>(engine::audio::extract_kaldi_fbank({}, invalid));
      },
      "non-finite preemphasis");

  invalid = options;
  invalid.low_frequency = std::numeric_limits<float>::quiet_NaN();
  expect_runtime_error(
      [&] {
        static_cast<void>(engine::audio::extract_kaldi_fbank({}, invalid));
      },
      "non-finite frequency bounds");

  std::vector<float> impulse(400, 0.0F);
  impulse[200] = 1.0F;
  const auto baseline = engine::audio::extract_kaldi_fbank(impulse, options);
  options.apply_cmvn = true;
  options.cmvn_shift.assign(560, 1.0F);
  options.cmvn_scale.assign(560, 2.0F);
  const auto normalized = engine::audio::extract_kaldi_fbank(impulse, options);
  if (normalized.frames != baseline.frames ||
      normalized.values.size() != baseline.values.size()) {
    throw std::runtime_error("CMVN changed the frontend shape");
  }
  for (size_t index = 0; index < baseline.values.size(); ++index) {
    const float expected = (baseline.values[index] + 1.0F) * 2.0F;
    if (std::abs(normalized.values[index] - expected) > 1.0e-6F) {
      throw std::runtime_error("CMVN did not apply (feature + shift) * scale");
    }
  }
}

} // namespace

int main(int argc, char **argv) try {
  if (argc != 4) {
    std::cerr << "usage: fun_asr_nano_frontend_probe <reference.json> "
                 "<reference.bin> <sample.wav>\n";
    return 2;
  }

  const auto reference =
      engine::io::json::parse_file(std::filesystem::path(argv[1]));
  if (reference.require("schema_version").as_i64() != 2 ||
      reference.require("reference").as_string() !=
          "transformers.FunAsrNanoFeatureExtractor" ||
      reference.require("transformers_commit").as_string() !=
          kTransformersCommit ||
      reference.require("data_format").as_string() != "little-endian-float32" ||
      reference.require("sample_rate").as_i64() != kSampleRate ||
      reference.require("feature_size").as_i64() != 80 ||
      reference.require("lfr_m").as_i64() != 7 ||
      reference.require("lfr_n").as_i64() != 6) {
    throw std::runtime_error("unexpected frontend reference metadata");
  }

  const auto &fixtures = reference.require("fixtures");
  const std::filesystem::path data_path(argv[2]);
  const std::filesystem::path sample_wav(argv[3]);
  check_frontend_contracts();
  for (const std::string name :
       {"silence", "impulse", "sine_440hz", "sample_16k"}) {
    compare_fixture(name, fixtures.require(name), data_path, sample_wav);
  }
  return 0;
} catch (const std::exception &error) {
  std::cerr << "fun_asr_nano_frontend_probe failed: " << error.what() << '\n';
  return 1;
}
