#pragma once

#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/community_models/vietneu_tts/assets.h"
#include "engine/community_models/vietneu_tts/prompt_tts_voice_clone.h"
#include "engine/community_models/vietneu_tts/speaker_encoder.h"
#include "engine/community_models/vietneu_tts/talker.h"
#include "engine/community_models/vietneu_tts/tokenizer_speech_decoder.h"
#include "engine/community_models/vietneu_tts/tokenizer_speech_encoder.h"
#include "engine/community_models/vietneu_tts/tokenizer_text.h"

#include "engine/models/moss/shared/audio_tokenizer_decoder.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace engine::models::vietneu_tts {

class VietneuTTSSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    VietneuTTSSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const VietneuTTSAssets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    struct VoicePromptCacheKey {
        std::string reference_text;
        Qwen3VoiceCloneMode mode = Qwen3VoiceCloneMode::Icl;
        int sample_rate = 0;
        int channels = 0;
        uint64_t sample_count = 0;
        uint64_t sample_hash = 0;
    };

    struct VoicePromptCacheKeyEqual {
        bool operator()(const VoicePromptCacheKey & lhs, const VoicePromptCacheKey & rhs) const noexcept;
    };

    struct VoicePromptCacheEntry {
        Qwen3VoiceClonePrompt prompt;
    };

    VietneuTTSRequest make_request(const runtime::TaskRequest & request) const;
    const Qwen3VoiceClonePrompt & resolve_voice_prompt(
        const Qwen3VoiceCloneInput & input,
        const VietneuTTSVoiceClonePromptBuilder & prompt_builder);

    runtime::TaskSpec task_;
    std::shared_ptr<const VietneuTTSAssets> assets_;
    size_t talker_graph_arena_bytes_ = 256ull * 1024ull * 1024ull;
    size_t speech_encoder_graph_arena_bytes_ = 32ull * 1024ull * 1024ull;
    size_t speech_decoder_graph_arena_bytes_ = 32ull * 1024ull * 1024ull;
    size_t speaker_encoder_graph_arena_bytes_ = 32ull * 1024ull * 1024ull;
    // No-alloc GGML context capacities for reusable constant tensor descriptors.
    // Generous on purpose; ConstantTensorCache fits them to the host when it has
    // to, so a small machine is not asked to reserve what it does not have.
    size_t talker_constant_context_bytes_ = 4ull * 1024ull * 1024ull * 1024ull;
    size_t code_predictor_constant_context_bytes_ = 1536ull * 1024ull * 1024ull;
    size_t speech_decoder_constant_context_bytes_ = 1536ull * 1024ull * 1024ull;
    engine::assets::TensorStorageType talker_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType speech_encoder_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType speech_decoder_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType conv_weight_storage_type_ = engine::assets::TensorStorageType::F32;
    bool mem_saver_ = false;
    Qwen3TextTokenizer text_tokenizer_;
    VietneuTalker talker_;
    std::shared_ptr<const VietneuTalkerWeightsRuntime> talker_weights_;
    std::shared_ptr<VietneuTalkerStepRuntime> talker_step_;
    core::ExecutionContext voice_prompt_context_;
    std::unique_ptr<engine::models::moss::MossAudioTokenizerDecoder> moss_speech_decoder_;
    std::unique_ptr<Qwen3SpeechTokenizerEncoderRuntime> speech_encoder_;
    std::unique_ptr<VietneuSpeakerEncoderRuntime> speaker_encoder_;
    runtime::CacheSlots<VoicePromptCacheKey, VoicePromptCacheEntry, VoicePromptCacheKeyEqual> voice_prompt_cache_;
    std::optional<VoicePromptCacheEntry> uncached_voice_prompt_;
};

}  // namespace engine::models::vietneu_tts
