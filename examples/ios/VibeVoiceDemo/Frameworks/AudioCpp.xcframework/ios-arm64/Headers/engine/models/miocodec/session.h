#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/framework/runtime/session_base.h"

#include <memory>

namespace engine::modules {
class WavlmEncoderComponent;
}

namespace engine::models::miocodec {

struct MioCodecAssets;
struct MioCodecWeights;
class MioCodecContentEncoderRuntime;
class MioCodecGlobalEncoderRuntime;
class MioCodecGlobalReferenceEncoder;
class MioCodecWaveDecoderRuntime;
class MioCodecWaveformReconstructor;
class MioCodecSslFeatureExtractor;

class MioCodecSession final
    : public runtime::RuntimeSessionBase,
      public runtime::IOfflineVoiceTaskSession {
public:
    MioCodecSession(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options,
        std::shared_ptr<const MioCodecAssets> assets);
    ~MioCodecSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

    const std::shared_ptr<const MioCodecWeights> & codec_weights() const noexcept;
    const engine::modules::WavlmEncoderComponent & wavlm() const noexcept;

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const MioCodecAssets> assets_;
    std::shared_ptr<const MioCodecWeights> codec_weights_;
    std::unique_ptr<MioCodecSslFeatureExtractor> local_ssl_;
    std::unique_ptr<MioCodecGlobalReferenceEncoder> global_reference_;
    std::unique_ptr<MioCodecContentEncoderRuntime> content_encoder_;
    std::unique_ptr<MioCodecGlobalEncoderRuntime> global_encoder_;
    std::unique_ptr<MioCodecWaveDecoderRuntime> wave_decoder_;
    std::unique_ptr<MioCodecWaveformReconstructor> waveform_reconstructor_;
};

}  // namespace engine::models::miocodec
