#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/omnivoice/assets.h"
#include "engine/models/omnivoice/audio_tokenizer.h"
#include "engine/models/omnivoice/generator.h"
#include "engine/models/omnivoice/postprocess.h"
#include "engine/models/omnivoice/prompt_builder.h"
#include "engine/models/omnivoice/tokenizer_text.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace engine::models::omnivoice {

class OmniVoiceSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    OmniVoiceSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const OmniVoiceAssets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    struct SessionDefaults {
        std::optional<runtime::Transcript> text = std::nullopt;
        std::optional<runtime::AudioBuffer> reference_audio = std::nullopt;
        std::optional<std::string> reference_text = std::nullopt;
        std::optional<std::string> instruct = std::nullopt;
        std::unordered_map<std::string, std::string> options;
    };

    struct ReferencePromptCacheEntry {
        bool preprocess_prompt = true;
        bool reference_text_provided = false;
        int sample_rate = 0;
        int channels = 0;
        uint64_t sample_count = 0;
        uint64_t sample_hash = 0;
        OmniVoiceAudioTokens tokens;
    };

    OmniVoiceRequest make_request(const runtime::TaskRequest & request) const;
    std::optional<runtime::AudioBuffer> resolve_reference_audio(const runtime::TaskRequest & request) const;
    std::optional<std::string> resolve_reference_text(const runtime::TaskRequest & request) const;
    std::optional<std::string> resolve_instruct(const runtime::TaskRequest & request) const;
    std::unordered_map<std::string, std::string> merged_request_options(const runtime::TaskRequest & request) const;
    OmniVoiceAudioTokens resolve_reference_audio_tokens(
        const runtime::AudioBuffer & audio,
        bool preprocess_prompt,
        bool reference_text_provided);

    runtime::TaskSpec task_;
    std::shared_ptr<const OmniVoiceAssets> assets_;
    size_t audio_tokenizer_graph_arena_bytes_ = 128ull * 1024ull * 1024ull;
    size_t generator_prefill_graph_arena_bytes_ = 256ull * 1024ull * 1024ull;
    size_t generator_decode_graph_arena_bytes_ = 256ull * 1024ull * 1024ull;
    size_t audio_tokenizer_weight_context_bytes_ = 128ull * 1024ull * 1024ull;
    size_t generator_weight_context_bytes_ = 256ull * 1024ull * 1024ull;
    engine::assets::TensorStorageType audio_tokenizer_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType generator_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    bool mem_saver_ = false;
    OmniVoiceTextTokenizer tokenizer_;
    OmniVoiceAudioTokenizerRuntime audio_tokenizer_;
    OmniVoicePromptBuilder prompt_builder_;
    OmniVoiceGeneratorRuntime generator_;
    OmniVoicePostprocessor postprocessor_;
    SessionDefaults session_defaults_;
    std::optional<ReferencePromptCacheEntry> reference_prompt_cache_;
};

}  // namespace engine::models::omnivoice
