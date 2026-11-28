#include "engine/community_models/zonos3/loader.h"

#include "engine/community_models/zonos3/session.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <stdexcept>
#include <utility>

namespace engine::models::zonos3 {

std::shared_ptr<runtime::IVoiceModelLoader> make_zonos3_loader() {
    runtime::SpecBackedVoiceModelConfig<Zonos3Assets> config;
    config.family = "zonos3";
    config.load_assets = load_zonos3_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                               std::shared_ptr<const Zonos3Assets> assets,
                               std::shared_ptr<const engine::model_spec::ModelContract> contract)
        -> std::unique_ptr<runtime::IVoiceTaskSession> {
        runtime::validate_spec_backed_session_options(options, *contract, "zonos3", "ZONOS3");
        return std::make_unique<Zonos3Session>(task, options, std::move(assets), std::move(contract));
    };
    // The package directory holds several GGUFs, so the model path names the backbone; its companions resolve next
    // to it.
    config.package_load = [](const assets::ModelPackage & package) {
        for (const auto & file : package.files) {
            if (file.filename() == "backbone-ptq.gguf") {
                return runtime::PackageLoadTarget{"zonos3", package.root / file, {}, {}};
            }
        }
        throw std::runtime_error("ZONOS3 package has no backbone-ptq.gguf");
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::zonos3
