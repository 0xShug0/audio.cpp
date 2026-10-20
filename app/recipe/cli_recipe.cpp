#include "cli_recipe.h"

#include "engine/framework/core/load_progress.h"

#include <stdexcept>
#include <utility>

namespace minitts::app {

struct CliRecipeModelExecutor::ModelEntry {
    engine::runtime::TaskSpec task;
    std::unique_ptr<engine::runtime::ILoadedVoiceModel> model;
    std::unique_ptr<engine::runtime::IVoiceTaskSession> session;
    engine::runtime::IOfflineVoiceTaskSession * offline = nullptr;
};

CliRecipeModelExecutor::CliRecipeModelExecutor(
    engine::runtime::ModelRegistry & registry,
    CliRecipeOptions options)
    : registry_(registry), options_(std::move(options)) {}

CliRecipeModelExecutor::~CliRecipeModelExecutor() = default;

engine::runtime::TaskResult CliRecipeModelExecutor::run(
    std::string_view role_view,
    const engine::runtime::TaskSpec & task,
    const engine::runtime::TaskRequest & request) {
    const std::string role(role_view);
    auto found = models_.find(role);
    if (found == models_.end()) {
        const auto path = options_.model_paths.find(role);
        const auto family = options_.model_families.find(role);
        if (path == options_.model_paths.end() || family == options_.model_families.end()) {
            throw std::runtime_error(
                "recipe role '" + role + "' requires both --recipe-model " + role +
                "=<path> and --recipe-family " + role + "=<family>");
        }

        engine::runtime::ModelLoadRequest load_request;
        load_request.model_path = path->second;
        load_request.model_spec_override = options_.model_spec_override;
        load_request.family_hint = family->second;
        load_request.options = options_for_recipe_role(options_.load_options, role, {role}, false);

        engine::runtime::SessionOptions session_options;
        session_options.backend = options_.backend;
        session_options.options = options_for_recipe_role(options_.session_options, role, {role}, false);

        auto entry = std::make_unique<ModelEntry>();
        entry->task = task;
        entry->model = registry_.load(load_request);
        entry->session = entry->model->create_task_session(task, session_options);
        engine::runtime::finish_model_load_trace();
        entry->offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(entry->session.get());
        if (!entry->offline) {
            throw std::runtime_error("recipe role '" + role + "' requires offline execution");
        }
        found = models_.emplace(role, std::move(entry)).first;
    }
    if (found->second->task.task != task.task || found->second->task.mode != task.mode) {
        throw std::runtime_error("recipe role '" + role + "' was reused with a different task");
    }
    found->second->session->prepare(engine::runtime::build_preparation_request(request));
    return found->second->offline->run(request);
}

}  // namespace minitts::app
