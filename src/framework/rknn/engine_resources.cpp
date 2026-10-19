#include "engine/framework/rknn/engine_resources.h"

#include "engine/framework/io/json.h"
#include "engine/framework/rknn/engine.h"

#include <utility>

namespace engine::rknn {

std::vector<runtime::ModelEngineRequirement> exec_plan_engine_requirements(
    const std::string & exec_plan_value,
    const std::filesystem::path & model_root) {
    std::vector<runtime::ModelEngineRequirement> out;
    io::json::Value plan;
    try {
        plan = io::json::parse(exec_plan_value);
    } catch (const std::exception &) {
        // A malformed plan is the session's error to raise, on load, with the component vocabulary
        // it owns. A resource question about it has nothing to say.
        return out;
    }
    if (!plan.is_object()) {
        return out;
    }
    for (const auto & [_, component] : plan.as_object()) {
        if (!component.is_object()) {
            continue;
        }
        const auto & members = component.as_object();
        const auto artifacts = members.find("artifacts");
        if (artifacts == members.end() || !artifacts->second.is_object()) {
            continue;
        }
        for (const auto & [_, artifact] : artifacts->second.as_object()) {
            if (!artifact.is_string()) {
                continue;
            }
            // A ladder travels in one artifact value, comma-separated (see the CosyVoice3 flow
            // sidecar), so every name in the value is an engine the deployment asked for.
            const std::string & declared = artifact.as_string();
            size_t start = 0;
            while (start <= declared.size()) {
                const auto comma = declared.find(',', start);
                const auto stop = comma == std::string::npos ? declared.size() : comma;
                if (stop > start) {
                    runtime::ModelEngineRequirement requirement;
                    requirement.path = std::filesystem::path(resolve_engine_path(
                        declared.substr(start, stop - start), model_root));
                    requirement.resident_mib =
                        engine_memory_mib(requirement.path.string(), "model resources");
                    out.push_back(std::move(requirement));
                }
                if (comma == std::string::npos) {
                    break;
                }
                start = comma + 1;
            }
        }
    }
    return out;
}

}  // namespace engine::rknn
