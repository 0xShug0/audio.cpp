#include "engine/models/chatterbox_turbo/assets.h"

#include "components/dot_prefix_source.h"

#include "engine/framework/model_spec/package.h"

#include <stdexcept>

namespace engine::models::chatterbox_turbo {

std::shared_ptr<const ChatterboxTurboAssets> load_chatterbox_turbo_assets(const std::filesystem::path & model_path) {
    auto out = std::make_shared<ChatterboxTurboAssets>();
    out->resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("chatterbox_turbo"));
    out->t3_gguf_path = out->resources.require_file("t3_gguf_path");
    if (out->t3_gguf_path.extension() != ".gguf") {
        throw std::runtime_error("Chatterbox Turbo requires a GGUF model path (point --model at the T3 GGUF file directly)");
    }
    // The turbo GGUF's tensor names are flat-dotted ("t3.blk.0...", "conds.gen...") with no "/"
    // namespace separator, so the framework's own prefix-stripping (which expects "/") can't be
    // used here -- see components/dot_prefix_source.h.
    const auto t3_raw = engine::assets::open_tensor_source(out->t3_gguf_path);
    out->t3_turbo_weights = make_dot_prefixed_tensor_source(t3_raw, "t3");
    out->builtin_conditionals_turbo = make_dot_prefixed_tensor_source(t3_raw, "conds");
    return out;
}

}  // namespace engine::models::chatterbox_turbo
