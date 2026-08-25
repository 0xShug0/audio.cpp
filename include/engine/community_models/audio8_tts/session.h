#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/community_models/audio8_tts/assets.h"
#include "engine/community_models/audio8_tts/generator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace engine::models::audio8_tts {

std::shared_ptr<runtime::IVoiceModelLoader> make_audio8_tts_loader();

class Audio8TtsSession final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
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

    runtime::TaskSpec task_;
    std::shared_ptr<const Audio8TtsAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::unique_ptr<Audio8TtsGenerator> generator_;
    std::optional<Audio8TtsRequest> defaults_;
    runtime::CacheSlots<ReferenceCacheKey, ReferenceCacheEntry, ReferenceCacheKeyEqual> reference_cache_;
    std::optional<ReferenceCacheEntry> uncached_reference_;
};

}  // namespace engine::models::audio8_tts
