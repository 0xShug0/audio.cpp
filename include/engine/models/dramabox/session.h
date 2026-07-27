#pragma once

#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/dramabox/audio_vae.h"
#include "engine/models/dramabox/assets.h"
#include "engine/models/dramabox/dit.h"
#include "engine/models/dramabox/gemma3_encoder.h"
#include "engine/models/dramabox/gemma_tokenizer.h"
#include "engine/models/dramabox/prompt_connector.h"
#include "engine/models/dramabox/vocoder.h"

#include <memory>
#include <optional>
#include <cstdint>
#include <string>

namespace engine::models::dramabox {

class DramaBoxSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    DramaBoxSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const DramaBoxAssets> assets);
    ~DramaBoxSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::AudioBuffer generate_audio(const DramaBoxRequest & parsed, const runtime::TaskRequest & request);
    DramaBoxConditioningEncoding prompt_conditioning(const std::string & prompt);
    DramaBoxConditioningEncoding prompt_conditioning_for_guidance(
        const std::string & prompt,
        const std::string & negative_prompt,
        bool cfg_enabled);
    DramaBoxEncodedReferenceLatents encode_reference_latents(
        const DramaBoxRequest & parsed,
        const runtime::TaskRequest & request);

    struct ReferenceCacheKey {
        std::string voice_ref;
        std::optional<uint64_t> inline_audio_key;
        float ref_duration = 0.0F;
    };

    struct ReferenceCacheKeyEqual {
        bool operator()(const ReferenceCacheKey & lhs, const ReferenceCacheKey & rhs) const noexcept {
            return lhs.voice_ref == rhs.voice_ref &&
                lhs.inline_audio_key == rhs.inline_audio_key &&
                lhs.ref_duration == rhs.ref_duration;
        }
    };

    struct PromptCacheKey {
        std::string prompt;
        int64_t tokens = 0;
    };

    struct PromptCacheKeyEqual {
        bool operator()(const PromptCacheKey & lhs, const PromptCacheKey & rhs) const noexcept {
            return lhs.prompt == rhs.prompt && lhs.tokens == rhs.tokens;
        }
    };

    runtime::TaskSpec task_;
    std::shared_ptr<const DramaBoxAssets> assets_;
    DramaBoxPerfMode perf_mode_ = DramaBoxPerfMode::Exact;
    std::unique_ptr<DramaBoxGemmaTokenizer> tokenizer_;
    std::unique_ptr<DramaBoxGemma3PromptRuntime> gemma_prompt_;
    std::unique_ptr<DramaBoxPromptConnectorRuntime> prompt_connector_;
    std::unique_ptr<DramaBoxDitRuntime> dit_;
    std::unique_ptr<DramaBoxAudioVaeEncoderRuntime> audio_encoder_;
    std::unique_ptr<DramaBoxAudioVaeDecoderRuntime> audio_decoder_;
    std::unique_ptr<DramaBoxVocoderRuntime> vocoder_;
    runtime::CacheSlots<PromptCacheKey, DramaBoxConditioningEncoding, PromptCacheKeyEqual> prompt_conditioning_cache_;
    runtime::CacheSlots<PromptCacheKey, DramaBoxConditioningEncoding, PromptCacheKeyEqual> negative_conditioning_cache_;
    runtime::CacheSlots<ReferenceCacheKey, DramaBoxEncodedReferenceLatents, ReferenceCacheKeyEqual> reference_latents_;
};

}  // namespace engine::models::dramabox
