#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/codecs/mimi_codec_runtime.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/personaplex/assets.h"
#include "engine/models/personaplex/depformer.h"
#include "engine/models/personaplex/lm_runtime.h"
#include "engine/models/personaplex/request.h"

#include <array>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace engine::models::personaplex {

std::shared_ptr<runtime::IVoiceModelLoader> make_personaplex_loader();

class PersonaPlexSession final
    : public runtime::RuntimeSessionBase,
      public runtime::IOfflineVoiceTaskSession,
      public runtime::IStreamingVoiceTaskSession {
public:
    PersonaPlexSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const PersonaPlexAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~PersonaPlexSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;
    runtime::StreamingPolicy streaming_policy() const override;
    void start_stream(const runtime::TaskRequest & request) override;
    std::optional<runtime::StreamEvent> next_stream_event() override;
    void set_stream_event_sink(runtime::StreamEventCallback sink) override;
    runtime::TaskResult finish_stream() override;
    void reset() override;
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override;
    runtime::TaskResult finalize() override;

private:
    struct ConversationState;

    void ensure_runtime_graphs();
    std::unique_ptr<ConversationState> start_conversation(const PersonaPlexGenerationOptions & generation);
    std::optional<std::array<int32_t, 8>> run_user_frame(ConversationState & state, const int32_t * user_codes);
    runtime::StreamEvent process_stream_audio(bool flush);

    runtime::TaskSpec task_;
    std::shared_ptr<const PersonaPlexAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    runtime::StreamEventCallback stream_sink_;
    size_t graph_arena_bytes_ = 1024ull * 1024ull * 1024ull;
    size_t lm_weight_context_bytes_ = 64ull * 1024ull * 1024ull;
    size_t depformer_weight_context_bytes_ = 64ull * 1024ull * 1024ull;
    size_t mimi_weight_context_bytes_ = 64ull * 1024ull * 1024ull;
    assets::TensorStorageType weight_storage_type_ = assets::TensorStorageType::Native;
    std::shared_ptr<const PersonaPlexLMWeights> lm_weights_;
    std::shared_ptr<const PersonaPlexDepformerWeights> depformer_weights_;
    std::unique_ptr<engine::codecs::MimiCodecComponent> mimi_codec_;
    std::unique_ptr<PersonaPlexMainStepGraph> main_step_graph_;
    std::unique_ptr<PersonaPlexDepformerRuntime> depformer_runtime_;
    std::unique_ptr<engine::codecs::MimiEncoderRuntime> mimi_encoder_;
    std::unique_ptr<engine::codecs::MimiDecoderRuntime> mimi_decoder_;
    engine::sampling::HfSamplerScratch sampler_scratch_;
    std::mt19937 fallback_rng_;
    runtime::TaskRequest stream_request_;
    std::unique_ptr<ConversationState> stream_state_;
    bool stream_started_ = false;
};

}  // namespace engine::models::personaplex
