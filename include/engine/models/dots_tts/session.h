#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/modules/speech_encoders/campplus_encoder.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/dots_tts/assets.h"
#include "engine/models/dots_tts/audio_vae.h"
#include "engine/models/dots_tts/flow.h"
#include "engine/models/dots_tts/latent.h"
#include "engine/models/dots_tts/llm.h"
#include "engine/models/dots_tts/patch_encoder.h"
#include "engine/models/dots_tts/request.h"
#include "engine/models/dots_tts/tokenizer.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::dots_tts {

std::shared_ptr<runtime::IVoiceModelLoader> make_dots_tts_loader();

class DotsSession final
    : public runtime::RuntimeSessionBase,
      public runtime::IOfflineVoiceTaskSession,
      public runtime::IStreamingVoiceTaskSession {
public:
    DotsSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const DotsAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~DotsSession() override;

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
    struct SegmentState;
    struct PromptConditioning;
    struct PromptFeatureCacheKey {
        int sample_rate = 0;
        int channels = 0;
        uint64_t sample_count = 0;
        uint64_t sample_hash = 0;
        bool has_duration = false;
        uint32_t duration_bits = 0;
    };

    struct PromptFeatureCacheKeyEqual {
        bool operator()(const PromptFeatureCacheKey & lhs, const PromptFeatureCacheKey & rhs) const noexcept;
    };

    struct PromptFeatureCacheEntry {
        std::vector<float> speaker_embedding;
        DotsEncoderLatents encoded_latents;
    };

    PromptConditioning prepare_prompt_conditioning(const DotsRequest & request);
    std::vector<DotsLatentMatrix> generate_latent_patches(
        const DotsRequest & request,
        const std::string & text,
        PromptConditioning & conditioning,
        const std::function<void(const DotsLatentMatrix &, int64_t)> & on_payload_patch = {});
    std::vector<DotsLatentMatrix> generate_edit_latent_patches(
        const DotsRequest & request,
        const std::function<void(const DotsLatentMatrix &, int64_t)> & on_payload_patch = {});
    runtime::AudioBuffer synthesize_segment(const DotsRequest & request, const std::string & text);
    runtime::AudioBuffer synthesize_edit(const DotsRequest & request);
    runtime::AudioBuffer synthesize_streaming_segment(
        const DotsRequest & request,
        const std::string & text,
        size_t segment_index,
        PromptConditioning & conditioning,
        DotsAudioVaeStreamState & stream_state,
        const runtime::StreamEventCallback & sink);
    runtime::AudioBuffer synthesize_chunked(
        const runtime::TaskRequest & request,
        const DotsRequest & parsed);
    void ensure_speaker_encoder_loaded();
    void ensure_audio_vae_loaded();
    void ensure_patch_encoder_loaded();
    void ensure_llm_loaded();
    void ensure_flow_loaded();
    void release_conditioning_phase_components();
    void release_generation_phase_components();
    void release_audio_phase_components();

    runtime::TaskSpec task_;
    std::shared_ptr<const DotsAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    DotsTokenizer tokenizer_;
    DotsLatentCodec latent_codec_;
    engine::modules::CampplusEncoderComponent speaker_encoder_;
    DotsAudioVaeComponent audio_vae_;
    DotsPatchEncoderComponent patch_encoder_;
    DotsLlmComponent llm_;
    DotsFlowComponent flow_;
    runtime::CacheSlots<PromptFeatureCacheKey, PromptFeatureCacheEntry, PromptFeatureCacheKeyEqual> prompt_feature_cache_;
    std::optional<DotsRequest> prepared_defaults_;
    std::optional<runtime::TaskResult> streaming_result_;
    runtime::StreamEventCallback stream_sink_;
    engine::assets::TensorStorageType weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType conv_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType speaker_encoder_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType vocoder_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType patch_encoder_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType llm_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType flow_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    bool mem_saver_ = false;
};

}  // namespace engine::models::dots_tts
