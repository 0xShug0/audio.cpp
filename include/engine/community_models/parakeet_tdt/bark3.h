#pragma once

#include "engine/framework/runtime/session.h"

#include <memory>

namespace engine::community_models::parakeet_tdt {

// Model-local adaptation of the publisher's 250 ms input prefix. The ordinary
// Parakeet and Phonon sessions keep their existing preprocessing and clocks.
inline constexpr int64_t kBark3InputPrefixSamples = 4000;
std::unique_ptr<runtime::IOfflineVoiceTaskSession> wrap_bark3_offline(
    std::unique_ptr<runtime::IOfflineVoiceTaskSession> session);
std::unique_ptr<runtime::IStreamingVoiceTaskSession> wrap_bark3_streaming(
    std::unique_ptr<runtime::IStreamingVoiceTaskSession> session);

}  // namespace engine::community_models::parakeet_tdt
