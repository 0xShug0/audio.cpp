#include "recipe.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace minitts::app {
namespace {

std::vector<RecipeFactory> & recipe_factories() {
    static std::vector<RecipeFactory> factories;
    return factories;
}

}  // namespace

RecipePluginRegistration::RecipePluginRegistration(RecipeFactory factory) {
    if (!factory) {
        throw std::runtime_error("cannot register an empty recipe factory");
    }
    recipe_factories().push_back(std::move(factory));
}

std::unordered_map<std::string, std::string> options_for_recipe_role(
    const std::unordered_map<std::string, std::string> & options,
    std::string_view role,
    std::initializer_list<std::string_view> recipe_roles,
    bool include_shared) {
    std::unordered_map<std::string, std::string> selected;
    const std::string prefix = std::string(role) + ".";
    for (const auto & [key, value] : options) {
        if (key.rfind(prefix, 0) == 0) {
            continue;
        }
        bool belongs_to_another_role = false;
        for (const auto candidate : recipe_roles) {
            const std::string candidate_prefix = std::string(candidate) + ".";
            if (key.rfind(candidate_prefix, 0) == 0) {
                belongs_to_another_role = true;
                break;
            }
        }
        if (include_shared && !belongs_to_another_role) {
            selected.emplace(key, value);
        }
    }
    for (const auto & [key, value] : options) {
        if (key.rfind(prefix, 0) == 0) {
            selected.insert_or_assign(key.substr(prefix.size()), value);
        }
    }
    return selected;
}

void RecipeRegistry::add(std::unique_ptr<IRecipe> recipe) {
    if (!recipe) {
        throw std::runtime_error("cannot register a null recipe");
    }
    const auto id = recipe->id();
    if (id.empty()) {
        throw std::runtime_error("recipe id must not be empty");
    }
    if (!recipes_.emplace(id, std::move(recipe)).second) {
        throw std::runtime_error("duplicate recipe id: " + id);
    }
}

const IRecipe & RecipeRegistry::require(std::string_view id) const {
    const auto found = recipes_.find(std::string(id));
    if (found == recipes_.end()) {
        throw std::runtime_error("unknown recipe: " + std::string(id));
    }
    return *found->second;
}

std::vector<std::string> RecipeRegistry::ids() const {
    std::vector<std::string> ids;
    ids.reserve(recipes_.size());
    for (const auto & [id, _] : recipes_) {
        ids.push_back(id);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

RecipeRegistry make_recipe_registry() {
    RecipeRegistry registry;
    for (const auto & factory : recipe_factories()) {
        registry.add(factory());
    }
    return registry;
}

}  // namespace minitts::app
