#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/session.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/text/chunking.h"
#include "engine/community_models/audio8_tts/assets.h"
#include "engine/community_models/audio8_tts/generator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::audio8_tts {

std::shared_ptr<runtime::IVoiceModelLoader> make_audio8_tts_loader();

class Audio8TtsSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession
    , public runtime::IStreamingVoiceTaskSession {
public:
    Audio8TtsSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const Audio8TtsAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~Audio8TtsSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

    // IStreamingVoiceTaskSession — omnivoice/session.h:22 pattern
    runtime::StreamingPolicy streaming_policy() const override;
    void start_stream(const runtime::TaskRequest & request) override;
    std::optional<runtime::StreamEvent> next_stream_event() override;
    void set_stream_event_sink(runtime::StreamEventCallback sink) override;
    runtime::TaskResult finish_stream() override;
    void reset() override;
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override;
    runtime::TaskResult finalize() override;

private:
    struct ReferenceCacheKey {
        std::string source_id;
        int sample_rate = 0;
        int channels = 0;
        uint64_t sample_count = 0;
        uint64_t sample_hash = 0;
    };

    struct ReferenceCacheKeyEqual {
        bool operator()(const ReferenceCacheKey & lhs, const ReferenceCacheKey & rhs) const;
    };

    struct ReferenceCacheEntry {
        Audio8TtsCodes codes;
    };

    Audio8TtsRequest make_request(const runtime::TaskRequest & request) const;
    const Audio8TtsCodes & resolve_reference_codes(const Audio8TtsReference & reference);
    // Streaming helpers — omnivoice/session.cpp:744/797 pattern
    std::vector<std::string> plan_text_chunks(
        const Audio8TtsRequest & request,
        const Audio8TtsPrompt & prompt) const;
    void initialize_streaming_request(const runtime::TaskRequest & request);
    runtime::AudioBuffer synthesize_stream_chunk(size_t chunk_index);

    runtime::TaskSpec task_;
    std::shared_ptr<const Audio8TtsAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::unique_ptr<Audio8TtsGenerator> generator_;
    std::optional<Audio8TtsRequest> defaults_;
    runtime::CacheSlots<ReferenceCacheKey, ReferenceCacheEntry, ReferenceCacheKeyEqual> reference_cache_;
    std::optional<ReferenceCacheEntry> uncached_reference_;

    // Streaming state — mirror omnivoice/session.h:97
    std::optional<Audio8TtsRequest> stream_request_;
    std::vector<std::string> stream_text_chunks_;
    std::vector<Audio8TtsCodes> stream_reference_codes_;
    std::optional<Audio8TtsConversationTurn> stream_previous_turn_;
    std::optional<std::string> stream_language_;
    runtime::AudioBuffer stream_merged_audio_;
    size_t stream_chunk_index_ = 0;
    bool stream_started_ = false;
    bool stream_has_reference_ = false;
};

}  // namespace engine::models::audio8_tts
