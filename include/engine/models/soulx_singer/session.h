#pragma once
#include "engine/models/soulx_singer/runtime.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/model_spec/metadata.h"
namespace engine::models::soulx_singer {
std::shared_ptr<runtime::IVoiceModelLoader> make_soulx_singer_loader();
std::unique_ptr<runtime::IVoiceTaskSession> make_svc_session(
    const runtime::TaskSpec &, const runtime::SessionOptions &,
    std::shared_ptr<const ModelAssets>, std::shared_ptr<const model_spec::ModelContract>);
std::unique_ptr<runtime::IVoiceTaskSession> make_svs_session(
    const runtime::TaskSpec &, const runtime::SessionOptions &,
    std::shared_ptr<const ModelAssets>, std::shared_ptr<const model_spec::ModelContract>);
} // namespace engine::models::soulx_singer
