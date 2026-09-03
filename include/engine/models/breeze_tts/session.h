#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/breeze_tts/assets.h"
#include "engine/models/breeze_tts/speech_decoder.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace engine::models::breeze_tts {

class BreezeGeneratorRuntime;
struct BreezeGenerationRequest;

std::shared_ptr<engine::runtime::IVoiceModelLoader> make_breeze_tts_loader();

class BreezeTTSSession final
    : public engine::runtime::RuntimeSessionBase
    , public engine::runtime::IOfflineVoiceTaskSession
    , public engine::runtime::IStreamingVoiceTaskSession {
public:
    BreezeTTSSession(
        engine::runtime::TaskSpec task,
        engine::runtime::SessionOptions options,
        std::shared_ptr<const BreezeTTSAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~BreezeTTSSession() override;

    std::string family() const override;
    engine::runtime::VoiceTaskKind task_kind() const override;
    engine::runtime::RunMode run_mode() const override;
    void prepare(const engine::runtime::SessionPreparationRequest & request) override;
    engine::runtime::TaskResult run(const engine::runtime::TaskRequest & request) override;
    engine::runtime::StreamingPolicy streaming_policy() const override;
    void start_stream(const engine::runtime::TaskRequest & request) override;
    std::optional<engine::runtime::StreamEvent> next_stream_event() override;
    void set_stream_event_sink(engine::runtime::StreamEventCallback sink) override;
    engine::runtime::TaskResult finish_stream() override;
    void reset() override;
    engine::runtime::StreamEvent process_audio_chunk(const engine::runtime::AudioChunk & chunk) override;
    engine::runtime::TaskResult finalize() override;

private:
    struct ReferenceCacheKey {
        int sample_rate = 0;
        int channels = 0;
        uint64_t sample_count = 0;
        uint64_t sample_hash = 0;
    };

    struct ReferenceCacheKeyEqual {
        bool operator()(const ReferenceCacheKey & lhs, const ReferenceCacheKey & rhs) const noexcept;
    };

    struct ReferenceCacheEntry {
        BreezeSpeechCodes codes;
    };

    BreezeSpeechCodes resolve_reference_codes(const engine::runtime::AudioBuffer & audio);
    BreezeGenerationRequest build_generation_request(
        const engine::runtime::TaskRequest & request,
        const std::optional<BreezeSpeechCodes> & reference_codes,
        size_t chunk_index) const;

    engine::runtime::TaskSpec task_;
    std::shared_ptr<const BreezeTTSAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::unique_ptr<BreezeGeneratorRuntime> generator_;
    engine::runtime::CacheSlots<ReferenceCacheKey, ReferenceCacheEntry, ReferenceCacheKeyEqual> reference_cache_;
    std::optional<ReferenceCacheEntry> uncached_reference_;
    std::optional<engine::runtime::StreamEvent> next_subchunk_event();
    std::vector<engine::runtime::TaskRequest> stream_chunk_requests_;
    std::optional<BreezeSpeechCodes> stream_reference_codes_;
    engine::runtime::AudioBuffer stream_merged_audio_;
    size_t stream_chunk_index_ = 0;
    bool stream_started_ = false;
    bool stream_subchunk_ = false;
    size_t stream_frames_per_event_ = 32;
    int64_t stream_lookahead_margin_ = 12;
    bool stream_chunk_active_ = false;
    std::vector<int32_t> stream_codes_;
    int64_t stream_total_frames_ = 0;
    size_t stream_emitted_samples_ = 0;
    size_t stream_event_seq_ = 0;
};

}  // namespace engine::models::breeze_tts
