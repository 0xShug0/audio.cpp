#pragma once

#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/index_tts2/assets.h"
#include "engine/models/index_tts2/audio_features.h"
#include "engine/models/index_tts2/gpt.h"
#include "engine/models/index_tts2/qwen_emotion.h"
#include "engine/models/index_tts2/request.h"
#include "engine/models/index_tts2/s2mel.h"
#include "engine/models/index_tts2/semantic_codec.h"
#include "engine/models/index_tts2/semantic_encoder.h"
#include "engine/models/index_tts2/style_encoder.h"
#include "engine/models/index_tts2/tokenizer_text.h"
#include "engine/models/index_tts2/vocoder.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::index_tts2 {

struct IndexTTS2AudioIdentity {
    int sample_rate = 0;
    int channels = 0;
    uint64_t sample_count = 0;
    uint64_t sample_hash = 0;
};

class IndexTTS2Session final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    IndexTTS2Session(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const IndexTTS2Assets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    struct SpeakerState {
        IndexTTS2AudioIdentity identity;
        IndexTTS2SemanticEmbedding semantic;
        IndexTTS2MelOutput reference_mel;
        IndexTTS2StyleEmbedding style;
        IndexTTS2S2MelSequence prompt_condition;
    };

    struct EmotionState {
        IndexTTS2AudioIdentity identity;
        IndexTTS2SemanticEmbedding semantic;
    };

    struct AudioIdentityEqual {
        bool operator()(
            const IndexTTS2AudioIdentity & lhs,
            const IndexTTS2AudioIdentity & rhs) const;
    };

    const SpeakerState & resolve_speaker_state(const runtime::AudioBuffer & audio);
    const EmotionState & resolve_emotion_state(const runtime::AudioBuffer & audio);
    std::vector<float> resolve_emotion_vector(
        const IndexTTS2Request & request,
        const SpeakerState & speaker,
        const EmotionState & emotion);
    runtime::AudioBuffer synthesize_segment(
        const std::vector<int32_t> & text_tokens,
        const SpeakerState & speaker,
        const EmotionState & emotion,
        const std::vector<float> & emotion_vector,
        const IndexTTS2GenerationOptions & options,
        uint32_t segment_seed);

    std::vector<float> explicit_emotion_matrix_vector(
        const std::vector<float> & emotion_weights,
        const IndexTTS2StyleEmbedding & style,
        bool use_random,
        uint32_t seed) const;

    runtime::TaskSpec task_;
    std::shared_ptr<const IndexTTS2Assets> assets_;
    size_t gpt_graph_arena_bytes_ = 2048ull * 1024ull * 1024ull;
    size_t s2mel_graph_arena_bytes_ = 2048ull * 1024ull * 1024ull;
    size_t reference_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
    size_t emotion_text_prefill_graph_arena_bytes_ = 2048ull * 1024ull * 1024ull;
    size_t emotion_text_decode_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
    size_t weight_context_bytes_ = 4ull * 1024ull * 1024ull * 1024ull;
    int64_t emotion_text_max_new_tokens_ = 256;
    engine::assets::TensorStorageType matmul_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType conv_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    bool mem_saver_ = false;

    IndexTTS2TextTokenizer tokenizer_;
    std::unique_ptr<IndexTTS2Wav2Vec2BertRuntime> semantic_encoder_;
    std::unique_ptr<IndexTTS2SemanticCodecRuntime> semantic_codec_;
    std::unique_ptr<IndexTTS2StyleEncoder> style_encoder_;
    std::unique_ptr<IndexTTS2GptRuntime> gpt_;
    std::unique_ptr<IndexTTS2S2MelRuntime> s2mel_;
    std::unique_ptr<IndexTTS2BigVganVocoder> vocoder_;
    std::unique_ptr<IndexTTS2QwenEmotionRuntime> qwen_emotion_;

    std::vector<float> speaker_matrix_;
    std::vector<float> emotion_matrix_;
    runtime::CacheSlots<IndexTTS2AudioIdentity, SpeakerState, AudioIdentityEqual> speaker_cache_;
    runtime::CacheSlots<IndexTTS2AudioIdentity, EmotionState, AudioIdentityEqual> emotion_cache_;
    runtime::CacheSlots<std::string, std::vector<float>> emotion_text_weights_cache_;
    std::optional<SpeakerState> uncached_speaker_state_;
    std::optional<EmotionState> uncached_emotion_state_;
};

}  // namespace engine::models::index_tts2
