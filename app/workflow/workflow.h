#pragma once

#include "file_sink.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/registry.h"

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>

namespace minitts::app {

struct WorkflowRunOptions {
    std::filesystem::path workflow_path;
    std::filesystem::path output_dir;
    engine::core::BackendConfig backend;
    std::optional<std::filesystem::path> final_audio_out;
    std::unordered_map<std::string, std::string> load_options;
    std::unordered_map<std::string, std::string> session_options;
    std::unordered_map<std::string, std::string> workflow_inputs;
    std::optional<std::filesystem::path> model_spec_override;
    std::string audio_converter = "ffmpeg";
};

void run_json_workflow(
    const engine::runtime::ModelRegistry & registry,
    const WorkflowRunOptions & options);

}  // namespace minitts::app
