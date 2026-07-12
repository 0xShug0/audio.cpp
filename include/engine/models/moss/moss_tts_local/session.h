#pragma once

#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/moss/shared/audio_tokenizer_decoder.h"
#include "engine/models/moss/shared/audio_tokenizer_encoder.h"
#include "engine/models/moss/moss_tts_local/assets.h"
#include "engine/models/moss/moss_tts_local/backbone.h"
#include "engine/models/moss/moss_tts_local/depth_transformer.h"
#include "engine/models/moss/moss_tts_local/generator.h"
#include "engine/models/moss/moss_tts_local/tokenizer_text.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::moss_tts_local {

// Offline TTS session: renders text into a 48 kHz stereo waveform by chaining the verified
// pieces -- the text processor builds the generation prefix, the generator (Qwen3 backbone +
// depth transformer) emits RVQ codes frame by frame, and the codec decoder turns those codes
// into audio. When a speaker reference is supplied, the codec encoder turns it into RLFQ
// codes that seed a voice-clone prompt.
class MossTTSLocalSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    MossTTSLocalSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const MossTTSLocalAssets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    moss::MossAudioTokenizerEncoder & encoder();

    struct ReferenceAudioCacheKey {
        uint64_t hash = 0;
        int sample_rate = 0;
        int channels = 0;
        size_t sample_count = 0;
    };

    struct ReferenceAudioCacheKeyEqual {
        bool operator()(const ReferenceAudioCacheKey & lhs, const ReferenceAudioCacheKey & rhs) const;
    };

    struct ReferenceVoiceCacheEntry {
        std::vector<std::vector<int32_t>> codes;
    };

    runtime::TaskSpec task_;
    std::shared_ptr<const MossTTSLocalAssets> assets_;
    // Declared before the generator so the generator (which holds references to them) is
    // destroyed first.
    std::unique_ptr<MossBackboneRuntime> backbone_;
    std::unique_ptr<MossDepthTransformer> depth_;
    std::unique_ptr<MossTextProcessor> processor_;
    std::unique_ptr<moss::MossAudioTokenizerDecoder> codec_;
    std::unique_ptr<MossGenerator> generator_;
    // Lazily built the first time a speaker reference is provided (voice cloning).
    std::unique_ptr<core::ExecutionContext> reference_encoder_execution_context_;
    std::unique_ptr<moss::MossAudioTokenizerEncoder> encoder_;
    runtime::CacheSlots<ReferenceAudioCacheKey, ReferenceVoiceCacheEntry, ReferenceAudioCacheKeyEqual>
        reference_voice_cache_;
};

}  // namespace engine::models::moss_tts_local
