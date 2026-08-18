#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::f5_tts {

// ---- inference configuration ----
struct F5SynthesisRequest {
    std::string text;              // text to speak (dialect token prepended internally)
    std::string dialect = "UNK";   // UNK MSA SAU UAE ALG IRQ EGY MAR OMN TUN LEV SDN LBY
    std::vector<float> ref_audio;  // mono float samples, ANY sample rate (resampled to 24k)
    int ref_sample_rate = 24000;
    std::string ref_text;          // transcript of ref_audio
    float speed = 1.0F;
    int steps = 32;
    float cfg_strength = 2.0F;
    float sway_sampling_coef = -1.0F;
    uint32_t seed = 0;
    bool fixed_seed = false;
    int threads = 0;  // 0 = hardware concurrency
};

struct F5SynthesisResult {
    std::vector<float> audio;  // 24 kHz mono
    int64_t sample_rate = 24000;
    double generation_seconds = 0.0;  // wall time
};

// Full Habibi/F5 inference: text + ref audio -> waveform.
F5SynthesisResult f5_synthesize(
    const std::string & model_path,
    const std::string & vocos_path,
    const F5SynthesisRequest & request);

#ifdef F5_MEL_TEST
// test hook: log-mel frontend for parity tests
std::vector<float> f5_test_mel(const std::vector<float> & wav);
std::vector<float> f5_test_vocos(const std::string & vocos_path, const std::vector<float> & mel);
#endif

}  // namespace engine::models::f5_tts
