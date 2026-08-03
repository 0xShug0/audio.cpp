#pragma once

#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/higgs_audio_tts/assets.h"
#include "engine/models/higgs_audio_tts/ar.h"
#include "engine/models/higgs_audio_tts/codec.h"
#include "engine/models/higgs_audio_tts/generator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace engine::models::higgs_audio_tts {

class HiggsTTSSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    HiggsTTSSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const HiggsAssets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    struct ReferenceCacheEntry {
        HiggsCodecEncodeOutput codes;
    };

    struct ReferenceCacheKey {
        int sample_rate = 0;
        int channels = 0;
        uint64_t sample_count = 0;
        uint64_t sample_hash = 0;
        std::string reference_text;
    };

    struct ReferenceCacheKeyEqual {
        bool operator()(const ReferenceCacheKey & lhs, const ReferenceCacheKey & rhs) const noexcept {
            return lhs.sample_rate == rhs.sample_rate &&
                   lhs.channels == rhs.channels &&
                   lhs.sample_count == rhs.sample_count &&
                   lhs.sample_hash == rhs.sample_hash &&
                   lhs.reference_text == rhs.reference_text;
        }
    };

    HiggsGenerationRequest make_generation_request(
        const runtime::TaskRequest & request,
        const HiggsCodecEncodeOutput * resolved_reference_codes = nullptr);
    const HiggsCodecEncodeOutput & resolve_reference_codes(
        const runtime::AudioBuffer & audio,
        const std::string & reference_text);

    runtime::TaskSpec task_;
    std::shared_ptr<const HiggsAssets> assets_;
    size_t ar_weight_context_bytes_ = 4096ull * 1024ull * 1024ull;
    size_t codec_weight_context_bytes_ = 1536ull * 1024ull * 1024ull;
    size_t ar_decode_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
    size_t codec_decode_graph_arena_bytes_ = 128ull * 1024ull * 1024ull;
    size_t codec_encode_graph_arena_bytes_ = 256ull * 1024ull * 1024ull;
    assets::TensorStorageType ar_weight_storage_type_ = assets::TensorStorageType::Native;
    assets::TensorStorageType codec_weight_storage_type_ = assets::TensorStorageType::Native;
    std::shared_ptr<HiggsARRuntime> ar_;
    std::shared_ptr<HiggsCodecRuntime> codec_;
    std::unique_ptr<HiggsGenerator> generator_;
    runtime::CacheSlots<ReferenceCacheKey, ReferenceCacheEntry, ReferenceCacheKeyEqual> reference_cache_;
    std::optional<ReferenceCacheEntry> uncached_reference_;
};

}  // namespace engine::models::higgs_audio_tts
