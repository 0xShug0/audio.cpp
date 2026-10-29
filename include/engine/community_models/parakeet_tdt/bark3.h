#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/framework/runtime/graph_optimizer.h"
#include "engine/community_models/parakeet_tdt/frontend.h"

#include <memory>
#include <string_view>
#include <vector>

namespace engine::community_models::parakeet_tdt {

// Model-local adaptation of the publisher's 250 ms input prefix. The ordinary
// Parakeet and Phonon sessions keep their existing preprocessing and clocks.
inline constexpr int64_t kBark3InputPrefixSamples = 4000;
inline bool is_bark3_variant(std::string_view variant) {
    return variant == "underdog_bark3_pro" || variant == "underdog_bark3_plus";
}
runtime::GraphOptimizationOptions bark3_encoder_graph_options(runtime::GraphOptimizationBackend backend);

// Cache only unnormalized, context-independent log-mel frames. Each window
// still recomputes its boundary frames and its whole-window normalization;
// no encoder/attention state is reused.
// Windows advance along one immutable waveform. Reset before another stream.
class BarkStreamingFrontend {
public:
    explicit BarkStreamingFrontend(std::shared_ptr<const ParakeetTDTAssets> assets);
    ParakeetFrontendFeatures extract(const runtime::AudioBuffer & window, int64_t start_sample);
    void reset();

private:
    engine::audio::NemoMelFrontend raw_frontend_;
    int64_t sample_rate_;
    int64_t hop_;
    int64_t half_fft_;
    int64_t feature_dim_;
    int64_t cached_start_ = 0;
    int64_t cached_samples_ = 0;
    ParakeetFrontendFeatures cached_;
};

void remove_bark3_prefix_timestamps(std::vector<runtime::WordTimestamp> & words, int64_t original_samples);
std::unique_ptr<runtime::IOfflineVoiceTaskSession> wrap_bark3_offline(
    std::unique_ptr<runtime::IOfflineVoiceTaskSession> session);
std::unique_ptr<runtime::IStreamingVoiceTaskSession> wrap_bark3_streaming(
    std::unique_ptr<runtime::IStreamingVoiceTaskSession> session);

}  // namespace engine::community_models::parakeet_tdt
