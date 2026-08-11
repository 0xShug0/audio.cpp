#pragma once

#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/index_tts2_5/assets.h"
#include "engine/models/index_tts2_5/audio_features.h"
#include "engine/models/index_tts2_5/gpt.h"
#include "engine/models/index_tts2_5/qwen_emotion.h"
#include "engine/models/index_tts2_5/request.h"
#include "engine/models/index_tts2_5/s2mel.h"
#include "engine/models/index_tts2_5/semantic_codec.h"
#include "engine/models/index_tts2_5/semantic_encoder.h"
#include "engine/models/index_tts2_5/style_encoder.h"
#include "engine/models/index_tts2_5/tokenizer_text.h"
#include "engine/models/index_tts2_5/vocoder.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::index_tts2_5 {

struct IndexTTS25AudioIdentity {
    int sample_rate = 0;
    int channels = 0;
    uint64_t sample_count = 0;
    uint64_t sample_hash = 0;
};

class IndexTTS25Session final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    IndexTTS25Session(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const IndexTTS25Assets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    struct SpeakerState {
        IndexTTS25AudioIdentity identity;
        IndexTTS25SemanticEmbedding semantic;
        IndexTTS25MelOutput reference_mel;
        IndexTTS25StyleEmbedding style;
        IndexTTS25S2MelSequence prompt_condition;
    };

    struct EmotionState {
        IndexTTS25AudioIdentity identity;
        IndexTTS25SemanticEmbedding semantic;
    };

    struct AudioIdentityEqual {
        bool operator()(
            const IndexTTS25AudioIdentity & lhs,
            const IndexTTS25AudioIdentity & rhs) const;
    };

    const SpeakerState & resolve_speaker_state(const runtime::AudioBuffer & audio);
    const EmotionState & resolve_emotion_state(const runtime::AudioBuffer & audio);
    std::vector<float> resolve_emotion_vector(
        const IndexTTS25Request & request,
        const SpeakerState & speaker,
        const EmotionState & emotion);
    runtime::AudioBuffer synthesize_segment(
        const std::vector<int32_t> & text_tokens,
        int32_t lang_id,
        size_t segment_index,
        const std::string & dump_dir,
        const SpeakerState & speaker,
        const EmotionState & emotion,
        const std::vector<float> & emotion_vector,
        const IndexTTS25GenerationOptions & options,
        uint32_t segment_seed);

    std::vector<float> explicit_emotion_matrix_vector(
        const std::vector<float> & emotion_weights,
        const IndexTTS25StyleEmbedding & style,
        bool use_random,
        uint32_t seed) const;

    runtime::TaskSpec task_;
    std::shared_ptr<const IndexTTS25Assets> assets_;
    size_t gpt_graph_arena_bytes_ = 2048ull * 1024ull * 1024ull;
    size_t s2mel_graph_arena_bytes_ = 2048ull * 1024ull * 1024ull;
    size_t reference_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
    size_t emotion_text_prefill_graph_arena_bytes_ = 2048ull * 1024ull * 1024ull;
    size_t emotion_text_decode_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
    size_t weight_context_bytes_ = 32ull * 1024ull * 1024ull;
    int64_t emotion_text_max_new_tokens_ = 256;
    engine::assets::TensorStorageType matmul_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType conv_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    bool mem_saver_ = false;

    IndexTTS25TextTokenizer tokenizer_;
    std::unique_ptr<IndexTTS25Wav2Vec2BertRuntime> semantic_encoder_;
    std::unique_ptr<IndexTTS25SemanticCodecRuntime> semantic_codec_;
    std::unique_ptr<IndexTTS25StyleEncoder> style_encoder_;
    std::unique_ptr<IndexTTS25GptRuntime> gpt_;
    std::unique_ptr<IndexTTS25S2MelRuntime> s2mel_;
    std::unique_ptr<IndexTTS25BigVganVocoder> vocoder_;
    std::unique_ptr<IndexTTS25QwenEmotionRuntime> qwen_emotion_;

    std::vector<float> speaker_matrix_;
    std::vector<float> emotion_matrix_;
    runtime::CacheSlots<IndexTTS25AudioIdentity, SpeakerState, AudioIdentityEqual> speaker_cache_;
    runtime::CacheSlots<IndexTTS25AudioIdentity, EmotionState, AudioIdentityEqual> emotion_cache_;
    runtime::CacheSlots<std::string, std::vector<float>> emotion_text_weights_cache_;
    std::optional<SpeakerState> uncached_speaker_state_;
    std::optional<EmotionState> uncached_emotion_state_;
};

}  // namespace engine::models::index_tts2_5
