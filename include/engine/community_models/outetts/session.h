#pragma once

#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/community_models/outetts/assets.h"
#include "engine/community_models/outetts/dac.h"
#include "engine/community_models/outetts/llama.h"
#include "engine/community_models/outetts/tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace engine::models::qwen3_forced_aligner {
class Qwen3ForcedAlignerSession;
}

namespace engine::models::outetts {

class OuteTTSSession final : public runtime::RuntimeSessionBase,
                             public runtime::IOfflineVoiceTaskSession {
public:
  OuteTTSSession(runtime::TaskSpec task, runtime::SessionOptions options,
                 std::shared_ptr<const OuteTTSAssets> assets);
  ~OuteTTSSession() override;
  std::string family() const override;
  runtime::VoiceTaskKind task_kind() const override;
  runtime::RunMode run_mode() const override;
  void prepare(const runtime::SessionPreparationRequest &request) override;
  runtime::TaskResult run(const runtime::TaskRequest &request) override;

private:
  struct ReferenceProfileCacheKey {
    uint64_t audio_hash = 0;
    int sample_rate = 0;
    int channels = 0;
    size_t sample_count = 0;
    std::string text;
    std::string language;
  };

  struct ReferenceProfileCacheKeyEqual {
    bool operator()(const ReferenceProfileCacheKey &lhs,
                    const ReferenceProfileCacheKey &rhs) const;
  };

  OuteTTSLlamaRuntime &llama(bool voice_cloning);
  OuteTTSVoiceProfile prepare_voice_profile(
      const runtime::AudioBuffer &audio, const std::string &reference_text,
      const std::string &language, bool &cache_hit);

  runtime::TaskSpec task_;
  std::shared_ptr<const OuteTTSAssets> assets_;
  OuteTTSTokenizer tokenizer_;
  std::unique_ptr<OuteTTSLlamaRuntime> llama_;
  std::optional<assets::TensorStorageType> llama_storage_type_;
  std::unique_ptr<
      engine::models::qwen3_forced_aligner::Qwen3ForcedAlignerSession>
      aligner_session_;
  std::shared_ptr<const engine::models::qwen3_asr::Qwen3ASRAssets>
      aligner_assets_;
  OuteTTSDacDecoder dac_;
  bool mem_saver_ = false;
  runtime::CacheSlots<ReferenceProfileCacheKey, OuteTTSVoiceProfile,
                      ReferenceProfileCacheKeyEqual>
      reference_profile_cache_;
  std::optional<OuteTTSVoiceProfile> voice_profile_;
};

} // namespace engine::models::outetts
