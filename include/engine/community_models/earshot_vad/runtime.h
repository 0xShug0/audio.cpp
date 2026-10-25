#pragma once

#include "engine/framework/runtime/session.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::community_models::earshot_vad {

// Native port of Earshot (https://github.com/pykeio/earshot, MIT OR Apache-2.0).
inline constexpr int kSampleRate = 16000;
inline constexpr int64_t kFrameSize = 256;  // 16 ms
inline constexpr int64_t kContextFrames = 3;
inline constexpr int64_t kWindowSize = kFrameSize * kContextFrames;
inline constexpr int64_t kFftSize = 1024;
inline constexpr int64_t kBins = kFftSize / 2 + 1;
inline constexpr int64_t kMels = 40;
inline constexpr const char * kWeightFile = "earshot_vad.safetensors";

struct EarshotWeights {
    std::array<float, kMels> norm_weight{};
    std::array<float, 9> layer1_kernel{};
    std::array<float, 16> layer1_weight{};
    std::array<float, 16> layer1_bias{};
    std::array<float, 48> layer2_kernel{};
    std::array<float, 256> layer2_weight{};
    std::array<float, 16> layer2_bias{};
    std::array<float, 48> layer3_kernel{};
    std::array<float, 256> layer3_weight{};
    // [in, out] copies of layer{2,3}_weight so the pointwise loops vectorize over outputs.
    std::array<float, 256> layer2_weight_t{};
    std::array<float, 256> layer3_weight_t{};
    std::vector<int16_t> rnn1_weight;    // [128, 80], Q9
    std::vector<int16_t> rnn2_weight;    // [128, 64], Q9
    std::array<int16_t, 128> output_weight{};  // Q9
    std::array<float, kWindowSize> hann_window{};
    std::array<int32_t, kMels> mel_start{};
    std::array<int32_t, kMels> mel_length{};
    std::array<int32_t, kMels> mel_offset{};
    std::vector<float> mel_coeffs;
};

struct EarshotAssetPaths {
    std::filesystem::path model_root;
    std::filesystem::path checkpoint_path;
};

EarshotAssetPaths resolve_earshot_assets(const std::filesystem::path & model_path);
std::shared_ptr<const EarshotWeights> load_earshot_weights_cached(const std::filesystem::path & checkpoint_path);

/** Segmenter settings in 16 ms frames. Defaults match upstream SegmenterOptions. */
struct EarshotSegmenterConfig {
    float threshold = 0.6f;          // attack threshold
    float neg_threshold = 0.5f;      // decay threshold
    float smoothing_factor = 0.4f;   // EMA factor on raw scores, [0, 1)
    int preroll_frames = 5;
    int min_duration_frames = 6;
    int start_delay_frames = 2;
    int end_delay_frames = 4;
    // Upstream drops a segment still open at the end of the audio; keep it by default.
    bool keep_trailing_segment = true;
};

EarshotSegmenterConfig earshot_config_from_options(
    const std::unordered_map<std::string, std::string> & options,
    const EarshotSegmenterConfig * fallback = nullptr);

/** Streaming per-frame detector; mirrors upstream earshot::Detector. */
class EarshotDetector {
public:
    explicit EarshotDetector(std::shared_ptr<const EarshotWeights> weights);
    ~EarshotDetector();
    EarshotDetector(EarshotDetector &&) noexcept;
    EarshotDetector & operator=(EarshotDetector &&) noexcept;

    void reset();
    /** Score one frame of kFrameSize samples in [-1, 1]. */
    float predict(const float * frame);

private:
    struct State;
    std::shared_ptr<const EarshotWeights> weights_;
    std::unique_ptr<State> state_;
};

/**
 * Score every complete 256-sample frame of 16 kHz mono audio in [-1, 1].
 * Equivalent to feeding the frames to a fresh EarshotDetector, with the
 * feature front end batched and spread over `threads`.
 */
std::vector<float> score_earshot_frames(
    const std::shared_ptr<const EarshotWeights> & weights,
    const float * samples,
    int64_t sample_count,
    int threads);

struct EarshotFrameSegment {
    int64_t start_frame = 0;
    int64_t end_frame = 0;
    float confidence = 0.0f;
};

/** Port of upstream segments.rs, fed one raw frame score at a time. */
class EarshotSegmenter {
public:
    struct Update {
        std::optional<int64_t> started_frame;  // segment became reportable; start frame
        std::optional<EarshotFrameSegment> finished;
    };

    explicit EarshotSegmenter(EarshotSegmenterConfig config);
    Update push(float raw_score);
    /** Flush at end of input; returns the open segment when keep_trailing_segment is set. */
    std::optional<EarshotFrameSegment> finish();
    int64_t frames() const noexcept { return n_frames_; }
    bool started() const noexcept { return start_reported_; }
    const EarshotSegmenterConfig & config() const noexcept { return config_; }

private:
    enum class State { Silence, PendingStart, Voice };
    EarshotSegmenterConfig config_;
    State state_ = State::Silence;
    float smoothed_ = 0.0f;
    int64_t n_frames_ = 0;
    int voice_frames_ = 0;
    int silence_frames_ = 0;
    int64_t start_ = 0;
    bool start_reported_ = false;
    double score_sum_ = 0.0;
    int64_t score_count_ = 0;
};

std::vector<EarshotFrameSegment> segment_earshot_scores(
    const std::vector<float> & scores,
    const EarshotSegmenterConfig & config);

}  // namespace engine::community_models::earshot_vad
