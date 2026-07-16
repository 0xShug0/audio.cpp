#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/moss/shared/audio_tokenizer_decoder.h"
#include "engine/models/moss/shared/audio_tokenizer_encoder.h"
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
    , public runtime::IOfflineVoiceTaskSession {
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
    moss::MossAudioTokenizerEncoder & encoder();
    MossTTSNanoAudioCodes encode_reference_audio(const runtime::AudioBuffer & audio, int64_t active_codebooks);
    runtime::AudioBuffer decode_generated_audio(const MossTTSNanoAudioCodes & codes, int64_t active_codebooks);
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
    moss::MossAudioTokenizerDecoder decoder_;
    std::unique_ptr<core::ExecutionContext> reference_encoder_execution_context_;
    std::unique_ptr<moss::MossAudioTokenizerEncoder> encoder_;
    std::optional<runtime::AudioBuffer> prepared_prompt_audio_;
    std::optional<MossTTSNanoAudioCodes> prepared_reference_codes_;
};

}  // namespace engine::models::moss_tts_nano
