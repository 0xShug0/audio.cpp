#include "engine/community_models/xtts_v2/loader.h"
#include "engine/community_models/xtts_v2/assets.h"
#include "engine/community_models/xtts_v2/session.h"
#include "engine/framework/runtime/spec_backed_model.h"

namespace engine::models::xtts_v2 {
namespace {

// XTTS packages contain two GGUFs; identify the pair rather than treating an
// arbitrary model directory as a standalone GGUF source.
class XttSv2Loader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override { return "xtts_v2"; }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint && *request.family_hint != family()) return false;
        return find_xtts_v2_gpt_file(request.model_path).has_value();
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_xtts_v2_assets(request.model_path);
        const auto contract = runtime::require_model_contract(family());
        runtime::ModelInspection out;
        out.model_root = assets->resources.model_root();
        out.metadata = contract->metadata;
        out.capabilities = contract->capabilities;
        out.cli = contract->cli;
        out.discovered_weights = {{"gpt", assets->gpt_gguf_path}, {"vocoder", assets->vocoder_gguf_path}};
        return out;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        auto assets = load_xtts_v2_assets(request.model_path);
        runtime::SpecBackedVoiceModelConfig<XttSv2Assets> config;
        config.family = family();
        config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
            std::shared_ptr<const XttSv2Assets> assets,
            std::shared_ptr<const engine::model_spec::ModelContract> contract) {
            return std::make_unique<XttSv2TtsSession>(task, std::move(assets), options, std::move(contract));
        };
        return std::make_unique<runtime::SpecBackedLoadedVoiceModel<XttSv2Assets>>(
            std::move(config), runtime::require_model_contract(family()), std::move(assets));
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        return runtime::require_model_contract(family())->capabilities;
    }
};
}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_xtts_v2_loader() {
    return std::make_shared<XttSv2Loader>();
}
}  // namespace engine::models::xtts_v2
