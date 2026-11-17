#include "engine/models/soulx_singer/routing.h"
#include "engine/models/soulx_singer/session.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/model_routing.h"
#include "engine/framework/runtime/spec_backed_model.h"

namespace engine::models::soulx_singer {
namespace {
void load_svc_assets(ModelAssets & assets) {
    assets.whisper = assets.resources.open_tensor_source("whisper_weights");
    assets.rmvpe = assets.resources.open_tensor_source("rmvpe_weights");
}

void load_svs_assets(ModelAssets & assets) {
    const auto phones = assets.resources.parse_json("phoneset");
    for (const auto & phone : phones.as_array())
        assets.phones.emplace(phone.as_string(), static_cast<int32_t>(assets.phones.size()));
}

ConditioningComponents create_svc_conditioning(const ModelAssets & assets, core::ExecutionContext & execution) {
    ConditioningComponents components;
    components.whisper = std::make_unique<modules::WhisperFrontendComponent>(
        modules::WhisperFrontendComponent::load_openai_layout(
            assets.whisper, execution.config(), {80, 1500, 512, 8, 6, 1e-5f},
            {"soulx_singer.whisper", 16ull << 20, 32ull << 20}));
    components.pitch = std::make_unique<modules::RmvpePitchExtractorComponent>(
        assets.rmvpe, execution.config(), assets::TensorStorageType::F32,
        modules::RmvpePitchExtractorConfig{"soulx_singer.rmvpe", 16ull << 20});
    return components;
}

ConditioningComponents create_svs_conditioning(const ModelAssets & assets, core::ExecutionContext & execution) {
    ConditioningComponents components;
    components.score = std::make_unique<ScoreConditioner>(*assets.model, execution);
    return components;
}

const SingerImplementation svc{runtime::VoiceTaskKind::Svc, load_svc_assets, create_svc_conditioning, make_svc_session};
const SingerImplementation svs{runtime::VoiceTaskKind::AudioGeneration, load_svs_assets, create_svs_conditioning, make_svs_session};
} // namespace

// Task/component registrations. Extend routes here; shared execution does not interpret variant labels.
const runtime::ModelRouteEntry<runtime::VoiceTaskKind, SingerImplementation> entries[] = {
    {svc.task, &svc},
    {svs.task, &svs},
};
const runtime::ModelRouteTable<runtime::VoiceTaskKind, SingerImplementation> table("SoulX-Singer task", entries);

const SingerImplementation & implementation_for_task(runtime::VoiceTaskKind task) {
    return table.require(task);
}

std::shared_ptr<runtime::IVoiceModelLoader> make_soulx_singer_loader() {
    runtime::SpecBackedVoiceModelConfig<ModelAssets> config;
    config.family = "soulx_singer";
    config.load_assets = load_assets;
    config.create_session = [](const auto & task, const auto & options, auto assets, auto contract) {
        if (task.task != assets->impl->task || task.mode != runtime::RunMode::Offline)
            throw runtime::InvalidRequestError("SoulX-Singer checkpoint does not support this task or mode");
        const auto factory = assets->impl->create_session;
        return factory(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}
} // namespace engine::models::soulx_singer
