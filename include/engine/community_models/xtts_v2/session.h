#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/community_models/xtts_v2/assets.h"

#include <memory>

namespace engine::models::xtts_v2 {

class XttSv2GptEngine;
class XttSv2Vocoder;

class XttSv2TtsSession final : public runtime::IOfflineVoiceTaskSession {
public:
    XttSv2TtsSession(std::shared_ptr<const XttSv2Assets> assets, const runtime::SessionOptions& session_options);
    ~XttSv2TtsSession() override;

    XttSv2TtsSession(const XttSv2TtsSession &) = delete;
    XttSv2TtsSession & operator=(const XttSv2TtsSession &) = delete;

    std::string family() const override { return "xtts_v2"; }
    runtime::VoiceTaskKind task_kind() const override { return runtime::VoiceTaskKind::Tts; }
    runtime::RunMode run_mode() const override { return runtime::RunMode::Offline; }
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::xtts_v2

