#include "engine/framework/model_spec/package.h"
#include "engine/models/dettato/runtime.h"

namespace engine::models::dettato {
std::shared_ptr<const DettatoAssets> load_dettato_assets(const std::filesystem::path &path) {
    auto out = std::make_shared<DettatoAssets>();
    out->resources = model_spec::load_resource_bundle_for_family(path, "dettato");
    out->config = io::json::parse_file(out->resources.require_file("config"));
    const std::array<const char *, 4> names{"transcriber", "evidence", "grouper", "drums"};
    for (size_t i = 0; i < names.size(); ++i)
        out->weights[i] = out->resources.open_tensor_source(names[i]);
    return out;
}
} // namespace engine::models::dettato
