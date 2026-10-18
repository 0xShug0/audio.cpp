#pragma once

#include "engine/framework/io/json.h"
#include "engine/framework/runtime/errors.h"

#include <string>

namespace minitts::server {

// /v1/tasks/stream answers with one JSON body once the run is over, unless the
// request's top-level "stream_format" is "sse": then each event goes out as a
// server-sent event as the model produces it. Absent or null keeps the JSON
// body. Any other value is turned away, so that a misspelt one is not answered
// with a body the client did not ask for.
inline bool task_stream_sse_requested(const engine::io::json::Value & body) {
    const auto * value = body.find("stream_format");
    if (value == nullptr || value->is_null()) {
        return false;
    }
    if (value->is_string() && value->as_string() == "sse") {
        return true;
    }
    throw engine::runtime::InvalidRequestError("task stream stream_format must be sse");
}

// The data of the SSE lines, the same in both runtimes: each event as the
// object the JSON response puts in "events", then the object it puts under
// "result".
inline std::string task_stream_event_line(const std::string & event_json) {
    return "{\"type\":\"task.stream.event\",\"event\":" + event_json + "}";
}

inline std::string task_stream_done_line(const std::string & result_json) {
    return "{\"type\":\"task.stream.done\",\"result\":" + result_json + "}";
}

} // namespace minitts::server
