#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/framework/runtime/graph_optimizer.h"

#include <memory>
#include <vector>

namespace engine::community_models::parakeet_tdt {

// Model-local adaptation of the publisher's 250 ms input prefix. The ordinary
// Parakeet and Phonon sessions keep their existing preprocessing and clocks.
inline constexpr int64_t kBark3InputPrefixSamples = 4000;
runtime::GraphOptimizationOptions bark3_encoder_graph_options(runtime::GraphOptimizationBackend backend);
void remove_bark3_prefix_timestamps(std::vector<runtime::WordTimestamp> & words, int64_t original_samples);
std::unique_ptr<runtime::IOfflineVoiceTaskSession> wrap_bark3_offline(
    std::unique_ptr<runtime::IOfflineVoiceTaskSession> session);
std::unique_ptr<runtime::IStreamingVoiceTaskSession> wrap_bark3_streaming(
    std::unique_ptr<runtime::IStreamingVoiceTaskSession> session);

}  // namespace engine::community_models::parakeet_tdt
