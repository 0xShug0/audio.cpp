#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::minimax_music3 {

// Checkpoint contract constants of the released MiniMax-Music3 weights. They are fixed by
// the reference inference recipe rather than carried in configs.
struct MiniMaxMusic3Contract {
    static constexpr int32_t kAudioEndTokenId = 151670;
    static constexpr int32_t kAudioCfgTokenId = 151654;
    static constexpr int32_t kAudioCodeOffset = 151675;
    static constexpr int32_t kSemanticVocabSize = 16384;
    static constexpr int64_t kMaxPromptTokens = 5000;
    static constexpr int64_t kMaxAudioFrames = 9000;
    static constexpr float kArCfgScale = 1.5F;
    static constexpr int32_t kArCfgTopK = 50;
    static constexpr int32_t kArSamplingTopK = 50;
    static constexpr int64_t kChunkFrames = 200;
    static constexpr int64_t kChunkHop = 100;
    static constexpr int64_t kOverlapLatentLength = 172;
    static constexpr int64_t kCropLeftLatent = 86;
    static constexpr int64_t kCropRightLatent = 344 - 86;
    static constexpr float kFrameRate = 25.0F;
};

struct MiniMaxMusic3GenerateRequest {
    std::string caption;
    std::string lyrics;
    float audio_duration = 60.0F;
    int64_t num_inference_steps = 30;
    float guidance_scale = 1.7F;
    uint32_t seed = 0;
};

struct MiniMaxMusic3GenerateResult {
    int sample_rate = 44100;
    int channels = 2;
    std::vector<float> samples;  // interleaved stereo in [-1, 1]
};

}  // namespace engine::models::minimax_music3
