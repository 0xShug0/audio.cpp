#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/streaming_tts_session.h"
#include "engine/framework/codecs/moss_audio_tokenizer_codec_runtime.h"
#include "engine/models/moss/moss_tts_nano/assets.h"
#include "engine/models/moss/moss_tts_nano/generator.h"
#include "engine/models/moss/moss_tts_nano/global_transformer.h"
#include "engine/models/moss/moss_tts_nano/local_frame_decoder.h"
#include "engine/models/moss/moss_tts_nano/prompt_builder.h"
#include "engine/models/moss/moss_tts_nano/tokenizer_text.h"

#include <cstddef>
#include <memory>
#include <optional>

namespace engine::models::moss_tts_nano {

class MossTTSNanoSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession
    , public runtime::StreamingTtsSessionBase {
public:
    MossTTSNanoSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const MossTTSNanoAssets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskResult generate_stream(const runtime::TaskRequest & request) override;
    void reset_stream_state() override;
    runtime::TaskResult synthesize(const runtime::TaskRequest & request, bool streaming);
    MossTTSNanoAudioCodes encode_reference_audio(const runtime::AudioBuffer & audio, int64_t active_codebooks);
    runtime::AudioBuffer decode_generated_audio(
        const MossTTSNanoAudioCodes & codes, int64_t active_codebooks, bool streaming = false);
    MossTTSNanoRequest make_request(const runtime::TaskRequest & request) const;
    const MossTTSNanoAudioCodes & reference_codes_for_request(const MossTTSNanoRequest & request);

    runtime::TaskSpec task_;
    std::shared_ptr<const MossTTSNanoAssets> assets_;
    size_t global_prefill_graph_arena_bytes_ = 256ull * 1024ull * 1024ull;
    size_t global_decode_graph_arena_bytes_ = 128ull * 1024ull * 1024ull;
    size_t global_weight_context_bytes_ = 512ull * 1024ull * 1024ull;
    size_t local_frame_graph_arena_bytes_ = 64ull * 1024ull * 1024ull;
    size_t local_frame_weight_context_bytes_ = 128ull * 1024ull * 1024ull;
    size_t audio_tokenizer_encoder_graph_arena_bytes_ = 64ull * 1024ull * 1024ull;
    size_t audio_tokenizer_decoder_graph_arena_bytes_ = 64ull * 1024ull * 1024ull;
    size_t audio_tokenizer_weight_context_bytes_ = 128ull * 1024ull * 1024ull;
    engine::assets::TensorStorageType global_weight_storage_type_ = engine::assets::TensorStorageType::F32;
    engine::assets::TensorStorageType local_frame_weight_storage_type_ = engine::assets::TensorStorageType::F32;
    MossTTSNanoTextTokenizer text_tokenizer_;
    MossTTSNanoPromptBuilder prompt_builder_;
    MossTTSNanoGlobalTransformerRuntime global_transformer_;
    MossTTSNanoLocalFrameDecoderRuntime local_frame_decoder_;
    MossTTSNanoGenerator generator_;
    engine::codecs::MossAudioTokenizerCodecRuntime codec_;
    std::optional<runtime::AudioBuffer> prepared_prompt_audio_;
    std::optional<MossTTSNanoAudioCodes> prepared_reference_codes_;
};

/* Reference-voice cache bookkeeping.
 *
 * Encoding a reference runs the audio tokenizer encoder, so the codes are kept for as
 * long as the same voice is used again. These two rules are the whole cache: a
 * prepared reference only invalidates when the audio itself changes, and the codes are
 * reusable when they were produced for that audio and codebook count. They are free
 * functions so the rules can be tested without a model. */
void invalidate_reference_codes_if_changed(
    std::optional<runtime::AudioBuffer> & cached_audio,
    std::optional<MossTTSNanoAudioCodes> & cached_codes,
    const runtime::AudioBuffer & audio);

bool reference_codes_reusable(
    const std::optional<runtime::AudioBuffer> & cached_audio,
    const std::optional<MossTTSNanoAudioCodes> & cached_codes,
    const runtime::AudioBuffer & audio,
    int64_t active_codebooks);

}  // namespace engine::models::moss_tts_nano
