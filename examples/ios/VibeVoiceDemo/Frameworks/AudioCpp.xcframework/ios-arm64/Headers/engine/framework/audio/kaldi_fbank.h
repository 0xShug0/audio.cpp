#pragma once

#include <vector>

namespace engine::audio {

struct KaldiFbankOptions {
  int sample_rate = 16000;
  int num_mels = 80;
  float frame_length_ms = 25.0F;
  float frame_shift_ms = 10.0F;
  int lfr_m = 7;
  int lfr_n = 6;
  float preemphasis = 0.97F;
  float low_frequency = 20.0F;
  float high_frequency = 0.0F;
  bool remove_dc_offset = true;
  bool upscale_samples = false;
  bool apply_cmvn = false;
  std::vector<float> cmvn_shift;
  std::vector<float> cmvn_scale;
};

struct KaldiFbankFeatures {
  int frames = 0;
  int feature_dim = 0;
  std::vector<float> values;
};

// Extracts snip-edges Kaldi log-mel features followed by centered LFR stacking.
KaldiFbankFeatures extract_kaldi_fbank(const std::vector<float> &audio,
                                       const KaldiFbankOptions &options = {});

} // namespace engine::audio
