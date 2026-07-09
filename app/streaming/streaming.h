#pragma once

#include "engine/framework/runtime/session.h"

#include <functional>

namespace minitts::app {

using StreamEventSink = std::function<void(const engine::runtime::StreamEvent &)>;

engine::runtime::TaskResult run_streaming_task(
    engine::runtime::IStreamingVoiceTaskSession & session,
    const engine::runtime::TaskRequest & request,
    const StreamEventSink & sink);

}  // namespace minitts::app
