#pragma once

#include "engine/framework/runtime/session.h"

#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace minitts::app {

struct RecipeModelRole {
    std::string name;
    engine::runtime::TaskSpec task;
};

using RecipeEventSink = std::function<void(const engine::runtime::StreamEvent &)>;

class IRecipeModelExecutor {
public:
    virtual ~IRecipeModelExecutor() = default;
    virtual engine::runtime::TaskResult run(
        std::string_view role,
        const engine::runtime::TaskSpec & task,
        const engine::runtime::TaskRequest & request) = 0;
};

class IRecipe {
public:
    virtual ~IRecipe() = default;
    virtual std::string id() const = 0;
    virtual std::string description() const = 0;
    virtual std::vector<RecipeModelRole> roles() const = 0;
    virtual engine::runtime::TaskResult run(
        IRecipeModelExecutor & executor,
        const engine::runtime::TaskRequest & request,
        const RecipeEventSink & event_sink = {}) const = 0;
};

using RecipeFactory = std::function<std::unique_ptr<IRecipe>()>;

class RecipePluginRegistration {
public:
    explicit RecipePluginRegistration(RecipeFactory factory);
};

std::unordered_map<std::string, std::string> options_for_recipe_role(
    const std::unordered_map<std::string, std::string> & options,
    std::string_view role,
    std::initializer_list<std::string_view> recipe_roles,
    bool include_shared);

class RecipeRegistry {
public:
    void add(std::unique_ptr<IRecipe> recipe);
    const IRecipe & require(std::string_view id) const;
    std::vector<std::string> ids() const;

private:
    std::unordered_map<std::string, std::unique_ptr<IRecipe>> recipes_;
};

RecipeRegistry make_recipe_registry();

}  // namespace minitts::app
