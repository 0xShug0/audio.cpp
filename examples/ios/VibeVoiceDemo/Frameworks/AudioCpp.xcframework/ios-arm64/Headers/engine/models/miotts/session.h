#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session.h"
#include "engine/framework/runtime/session_base.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace engine::models::miocodec {
struct MioCodecAssets;
struct MioCodecWeights;
class MioCodecGlobalEncoderRuntime;
class MioCodecGlobalReferenceEncoder;
class MioCodecWaveDecoderRuntime;
class MioCodecWaveformReconstructor;
}

namespace engine::models::miotts {

struct MioTTSAssets;
class MioTTSCausalLMRuntime;
class MioTTSTokenizer;

class MioTTSSession final
    : public runtime::RuntimeSessionBase,
      public runtime::IOfflineVoiceTaskSession {
public:
    MioTTSSession(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options,
        std::shared_ptr<const MioTTSAssets> assets);
    ~MioTTSSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::IOfflineVoiceTaskSession & best_of_n_asr_session();

    runtime::TaskSpec task_;
    std::shared_ptr<const MioTTSAssets> assets_;
    int64_t text_chunk_size_ = 180;
    bool best_of_n_enabled_ = false;
    int best_of_n_default_ = 1;
    int best_of_n_max_ = 8;
    std::string best_of_n_language_ = "auto";
    std::filesystem::path best_of_n_asr_model_path_;
    std::shared_ptr<const miocodec::MioCodecAssets> codec_assets_;
    std::shared_ptr<const miocodec::MioCodecWeights> codec_weights_;
    std::unique_ptr<MioTTSTokenizer> tokenizer_;
    std::unique_ptr<MioTTSCausalLMRuntime> language_model_;
    std::unique_ptr<runtime::ILoadedVoiceModel> best_of_n_asr_model_;
    std::unique_ptr<runtime::IVoiceTaskSession> best_of_n_asr_session_;
    bool best_of_n_asr_prepared_ = false;
    std::unique_ptr<miocodec::MioCodecGlobalEncoderRuntime> global_encoder_;
    std::unique_ptr<miocodec::MioCodecGlobalReferenceEncoder> global_reference_;
    std::unique_ptr<miocodec::MioCodecWaveDecoderRuntime> wave_decoder_;
    std::unique_ptr<miocodec::MioCodecWaveformReconstructor> waveform_reconstructor_;
};

}  // namespace engine::models::miotts
