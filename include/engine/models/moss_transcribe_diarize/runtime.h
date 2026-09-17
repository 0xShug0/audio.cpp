#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"

#include <memory>
#include <string>

namespace engine::models::moss_transcribe_diarize {

class TranscribeRuntime {
public:
    TranscribeRuntime(const assets::ResourceBundle & resources, core::ExecutionContext & execution);
    ~TranscribeRuntime();
    std::string transcribe(const runtime::AudioBuffer & audio, const std::string & instruction, int64_t max_tokens);
    void start(const runtime::AudioBuffer & audio, const std::string & instruction, int64_t max_tokens);
    std::optional<std::string> next_text();
    void reset();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::moss_transcribe_diarize
