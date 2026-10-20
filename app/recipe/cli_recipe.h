#pragma once

#include "recipe.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/registry.h"

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>

namespace minitts::app {

struct CliRecipeOptions {
    std::unordered_map<std::string, std::filesystem::path> model_paths;
    std::unordered_map<std::string, std::string> model_families;
    std::unordered_map<std::string, std::string> load_options;
    std::unordered_map<std::string, std::string> session_options;
    std::optional<std::filesystem::path> model_spec_override;
    engine::core::BackendConfig backend;
};

class CliRecipeModelExecutor final : public IRecipeModelExecutor {
public:
    CliRecipeModelExecutor(engine::runtime::ModelRegistry & registry, CliRecipeOptions options);
    ~CliRecipeModelExecutor() override;

    engine::runtime::TaskResult run(
        std::string_view role,
        const engine::runtime::TaskSpec & task,
        const engine::runtime::TaskRequest & request) override;

private:
    struct ModelEntry;

    engine::runtime::ModelRegistry & registry_;
    CliRecipeOptions options_;
    std::unordered_map<std::string, std::unique_ptr<ModelEntry>> models_;
};

}  // namespace minitts::app
