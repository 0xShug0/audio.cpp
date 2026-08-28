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

namespace engine::models::breeze_tts {

class BreezeGeneratorRuntime;

std::shared_ptr<engine::runtime::IVoiceModelLoader> make_breeze_tts_loader();

class BreezeTTSSession final
    : public engine::runtime::RuntimeSessionBase
    , public engine::runtime::IOfflineVoiceTaskSession {
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

    engine::runtime::TaskSpec task_;
    std::shared_ptr<const BreezeTTSAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::unique_ptr<BreezeGeneratorRuntime> generator_;
    engine::runtime::CacheSlots<ReferenceCacheKey, ReferenceCacheEntry, ReferenceCacheKeyEqual> reference_cache_;
    std::optional<ReferenceCacheEntry> uncached_reference_;
};

}  // namespace engine::models::breeze_tts
