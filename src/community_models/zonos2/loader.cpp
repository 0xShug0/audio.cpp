#include "engine/community_models/zonos2/loader.h"

#include "engine/community_models/zonos2/session.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <stdexcept>
#include <utility>

namespace engine::models::zonos2 {

std::shared_ptr<runtime::IVoiceModelLoader> make_zonos2_loader() {
    runtime::SpecBackedVoiceModelConfig<Zonos2Assets> config;
    config.family = "zonos2";
    config.load_assets = load_zonos2_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                               std::shared_ptr<const Zonos2Assets> assets,
                               std::shared_ptr<const engine::model_spec::ModelContract> contract)
        -> std::unique_ptr<runtime::IVoiceTaskSession> {
        runtime::validate_spec_backed_session_options(options, *contract, "zonos2", "ZONOS2");
        return std::make_unique<Zonos2Session>(task, options, std::move(assets), std::move(contract));
    };
    // The package directory holds several GGUFs, so the model path names the backbone; dac.gguf and spk-encoder.gguf
    // resolve next to it.
    config.package_load = [](const assets::ModelPackage & package) {
        for (const auto & file : package.files) {
            const auto name = file.filename().string();
            if (name.size() > 12 && name.rfind("zonos2-", 0) == 0 && name.substr(name.size() - 5) == ".gguf") {
                return runtime::PackageLoadTarget{"zonos2", package.root / file, {}, {}};
            }
        }
        throw std::runtime_error("ZONOS2 package has no zonos2-*.gguf backbone");
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::zonos2
