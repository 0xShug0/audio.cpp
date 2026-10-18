#pragma once

#include "../workflow/execution.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace minitts::cli {

bool has_batch_input(int argc, char ** argv);
minitts::app::AppBatchRequest build_batch_request_from_cli(
    int argc,
    char ** argv,
    const engine::runtime::TaskRequest & base_request,
    const std::string & audio_role);

}  // namespace minitts::cli
