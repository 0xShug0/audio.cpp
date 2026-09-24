#pragma once

#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/community_models/vieneu_v3_turbo/assets.h"
#include "engine/community_models/vieneu_v3_turbo/prompt_tts_voice_clone.h"
#include "engine/community_models/vieneu_v3_turbo/text_frontend.h"
#include "engine/community_models/vieneu_v3_turbo/speaker_encoder.h"
#include "engine/community_models/vieneu_v3_turbo/talker.h"
#include "engine/community_models/vieneu_v3_turbo/tokenizer_speech_decoder.h"
#include "engine/community_models/vieneu_v3_turbo/tokenizer_speech_encoder.h"
#include "engine/community_models/vieneu_v3_turbo/tokenizer_text.h"

#include "engine/framework/codecs/moss_audio_tokenizer_codec_runtime.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace engine::models::vieneu_v3_turbo {

class VieNeuTTSSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    VieNeuTTSSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const VieNeuTTSAssets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    // Packaged voices carry no reference audio, so the audio fields alone are equal for
    // every one of them; the codes and the speaker embedding have to be part of the key.
    struct VoicePromptCacheKey {
        std::string reference_text;
        Qwen3VoiceCloneMode mode = Qwen3VoiceCloneMode::Icl;
        int sample_rate = 0;
        int channels = 0;
        uint64_t sample_count = 0;
        uint64_t sample_hash = 0;
        uint64_t reference_codes_count = 0;
        uint64_t reference_codes_hash = 0;
        uint64_t speaker_embedding_count = 0;
        uint64_t speaker_embedding_hash = 0;
    };

    struct VoicePromptCacheKeyEqual {
        bool operator()(const VoicePromptCacheKey & lhs, const VoicePromptCacheKey & rhs) const noexcept;
    };

    struct VoicePromptCacheEntry {
        Qwen3VoiceClonePrompt prompt;
    };

    static VoicePromptCacheKey voice_prompt_cache_key(const Qwen3VoiceCloneInput & input);
    VieNeuTTSRequest make_request(const runtime::TaskRequest & request) const;
    std::optional<Qwen3VoiceCloneInput> make_voice_clone_input(
        const runtime::TaskRequest & request) const;
    const Qwen3VoiceClonePrompt & resolve_voice_prompt(
        const Qwen3VoiceCloneInput & input,
        const VieNeuTTSVoiceClonePromptBuilder & prompt_builder);

    runtime::TaskSpec task_;
    std::shared_ptr<const VieNeuTTSAssets> assets_;
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
    VieNeuTalker talker_;
    std::shared_ptr<const VieNeuTalkerWeightsRuntime> talker_weights_;
    std::shared_ptr<VieNeuTalkerStepRuntime> talker_step_;
    core::ExecutionContext voice_prompt_context_;
    std::unique_ptr<engine::codecs::MossAudioTokenizerCodecRuntime> moss_speech_decoder_;
    std::unique_ptr<VieNeuSpeakerEncoderRuntime> speaker_encoder_;
    /// Present only when the session was given a sea-g2p library; then `--text`
    /// may be raw text instead of phonemes.
    std::unique_ptr<TextFrontend> text_frontend_;
    runtime::CacheSlots<VoicePromptCacheKey, VoicePromptCacheEntry, VoicePromptCacheKeyEqual> voice_prompt_cache_;
    std::optional<VoicePromptCacheEntry> uncached_voice_prompt_;
};

}  // namespace engine::models::vieneu_v3_turbo
