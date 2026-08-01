#pragma once

#include "engine/community_models/glm_tts/assets.h"
#include "engine/community_models/glm_tts/frontend.h"
#include "engine/community_models/glm_tts/flow.h"
#include "engine/community_models/glm_tts/llama.h"
#include "engine/community_models/glm_tts/speech_tokenizer.h"
#include "engine/community_models/glm_tts/tokenizer_text.h"
#include "engine/framework/modules/speech_encoders/campplus_encoder.h"
#include "engine/framework/modules/vocoders/hift_vocoder.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/session_base.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace engine::models::glm_tts {

std::shared_ptr<runtime::IVoiceModelLoader> make_glm_tts_loader();

class GlmTTSSession final : public runtime::RuntimeSessionBase,
                            public runtime::IOfflineVoiceTaskSession {
public:
    GlmTTSSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const GlmTTSAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~GlmTTSSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(
        const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(
        const runtime::TaskRequest & request) override;

private:
    struct ReferenceCacheKey {
        int sample_rate = 0;
        int channels = 0;
        uint64_t sample_count = 0;
        uint64_t sample_hash = 0;
    };

    struct ReferenceCacheKeyEqual {
        bool operator()(
            const ReferenceCacheKey & lhs,
            const ReferenceCacheKey & rhs) const;
    };

    struct ReferenceCacheEntry {
        std::vector<int32_t> speech_tokens;
        GlmTTSMelFeatures prompt_mel;
        std::vector<float> speaker_embedding;
    };

    GlmTTSSpeechTokenizer & speech_tokenizer();
    GlmTTSLlamaRuntime & llama();
    GlmTTSFlowRuntime & flow();
    modules::CampplusEncoderComponent & campplus();
    modules::HiftVocoderComponent & hift();
    const ReferenceCacheEntry & resolve_reference(
        const runtime::AudioBuffer & audio);

    runtime::TaskSpec task_;
    std::shared_ptr<const GlmTTSAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    GlmTTSTextTokenizer text_tokenizer_;
    assets::TensorStorageType weight_storage_type_ =
        assets::TensorStorageType::Native;
    bool mem_saver_ = false;
    bool aggressive_mem_saver_ = false;
    runtime::CacheSlots<
        ReferenceCacheKey,
        ReferenceCacheEntry,
        ReferenceCacheKeyEqual> reference_cache_;
    std::optional<ReferenceCacheEntry> uncached_reference_;
    std::unique_ptr<GlmTTSSpeechTokenizer> speech_tokenizer_;
    std::unique_ptr<GlmTTSLlamaRuntime> llama_;
    std::unique_ptr<GlmTTSFlowRuntime> flow_;
    std::unique_ptr<modules::CampplusEncoderComponent> campplus_;
    std::unique_ptr<modules::HiftVocoderComponent> hift_;
};

}  // namespace engine::models::glm_tts
