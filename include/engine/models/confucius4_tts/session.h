#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/confucius4_tts/assets.h"
#include "engine/models/confucius4_tts/audio_features.h"
#include "engine/models/confucius4_tts/request.h"
#include "engine/models/confucius4_tts/s2a.h"
#include "engine/models/confucius4_tts/semantic_encoder.h"
#include "engine/models/confucius4_tts/style_encoder.h"
#include "engine/models/confucius4_tts/t2s.h"
#include "engine/models/confucius4_tts/tokenizer_text.h"
#include "engine/models/confucius4_tts/vocoder.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::confucius4_tts {

std::shared_ptr<runtime::IVoiceModelLoader> make_confucius4_tts_loader();

struct ConfuciusReferenceIdentity {
    std::string id;
};

struct ConfuciusPreparedVoice {
    ConfuciusPreparedReferenceAudio audio;
    ConfuciusSemanticEmbedding semantic;
    ConfuciusStyleEmbedding style;
};

class ConfuciusSession final
    : public runtime::RuntimeSessionBase,
      public runtime::IOfflineVoiceTaskSession,
      public runtime::IStreamingVoiceTaskSession {
public:
    ConfuciusSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const ConfuciusAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~ConfuciusSession() override;

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
    struct ReferenceIdentityEqual {
        bool operator()(const ConfuciusReferenceIdentity & lhs, const ConfuciusReferenceIdentity & rhs) const noexcept;
    };
    ConfuciusPreparedVoice & resolve_voice(const ConfuciusRequest & request);
    ConfuciusPreparedVoice prepare_voice(const ConfuciusRequest & request);
    runtime::AudioBuffer synthesize_segment(
        const ConfuciusTextSegment & segment,
        const ConfuciusPreparedVoice & voice,
        const ConfuciusGenerationOptions & options,
        uint64_t & rng_offset_blocks);
    runtime::AudioBuffer synthesize(const ConfuciusRequest & request, const ConfuciusPreparedVoice & voice);

    runtime::TaskSpec task_;
    std::shared_ptr<const ConfuciusAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    ConfuciusTextTokenizer tokenizer_;
    std::unique_ptr<ConfuciusWav2Vec2BertRuntime> semantic_encoder_;
    std::unique_ptr<ConfuciusStyleEncoder> style_encoder_;
    std::unique_ptr<ConfuciusT2SRuntime> t2s_;
    std::unique_ptr<ConfuciusS2ARuntime> s2a_;
    std::unique_ptr<ConfuciusBigVganVocoder> vocoder_;
    runtime::CacheSlots<ConfuciusReferenceIdentity, ConfuciusPreparedVoice, ReferenceIdentityEqual> reference_cache_;
    std::optional<ConfuciusPreparedVoice> uncached_reference_;
    std::optional<ConfuciusRequest> prepared_defaults_;
    std::optional<ConfuciusRequest> streaming_request_;
    const ConfuciusPreparedVoice * streaming_voice_ = nullptr;
    std::vector<ConfuciusTextSegment> streaming_segments_;
    std::vector<runtime::AudioBuffer> streaming_chunks_;
    runtime::StreamEventCallback stream_sink_;
    size_t streaming_segment_index_ = 0;
    uint64_t streaming_rng_offset_blocks_ = 0;
    size_t graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
    size_t weight_context_bytes_ = 1024ull * 1024ull * 1024ull;
    engine::assets::TensorStorageType matmul_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType conv_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    bool mem_saver_ = false;
};

}  // namespace engine::models::confucius4_tts
