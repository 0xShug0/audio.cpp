#pragma once

#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/model.h"
#include "engine/community_models/confucius4_r2t2/assets.h"
#include "engine/community_models/confucius4_r2t2/audio_encoder.h"
#include "engine/community_models/confucius4_r2t2/frontend_whisper.h"
#include "engine/community_models/confucius4_r2t2/thinker.h"
#include "engine/community_models/confucius4_r2t2/tokenizer_text.h"
#include "engine/community_models/confucius4_r2t2/types.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::confucius4_r2t2 {

/// Spec-backed loader factory (schema-v1 contract): the framework derives
/// metadata, capabilities and option validation from model_specs/confucius4_r2t2.json,
/// so this family ships no per-model loader.{h,cpp}.
std::shared_ptr<runtime::IVoiceModelLoader> make_confucius4_r2t2_loader();

/// Streaming decode configuration; defaults mirror
/// R2T2ASRModel.init_streaming_state() in the reference implementation.
struct R2T2ASRStreamConfig {
    double chunk_seconds = 0.32;
    int64_t unfixed_chunk_num = 2;
    int64_t unfixed_token_num = 5;
    bool rollback_punctuation = false;
    int64_t max_new_tokens = 32;
};

/// Confucius4-R2T2 streaming ASR session.
///
/// This family owns its full Qwen3-ASR-derived graph (audio tower, thinker,
/// tokenizer) plus two execution paths that the plain Qwen3-ASR family does not
/// have:
///
///  * offline transcription with R2T2 text parsing, and
///  * Longest Stable Prefix (LSP) streaming: every chunk re-decodes the whole
///    accumulated audio with a prompt carrying the previously recognized text
///    minus a small token rollback, and only the stable prefix of the result is
///    committed downstream (append-only).
class R2T2ASRSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession
    , public runtime::IStreamingVoiceTaskSession {
public:
    R2T2ASRSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const R2T2ASRAssets> assets);
    ~R2T2ASRSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;

    runtime::TaskResult run(const runtime::TaskRequest & request) override;

    runtime::StreamingPolicy streaming_policy() const override;
    void start_stream(const runtime::TaskRequest & request) override;
    void set_stream_event_sink(runtime::StreamEventCallback sink) override;
    void reset() override;
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override;
    runtime::TaskResult finish_stream() override;
    runtime::TaskResult finalize() override;

private:
    struct StreamOutcome {
        std::string text;
        std::string fixed_text;
    };

    R2T2ASRRequest make_request(const runtime::TaskRequest & request) const;
    R2T2ASRResult run_single(const R2T2ASRRequest & request);
    std::string generate_text(const R2T2ASRPrompt & prompt, const R2T2ASRAudioEmbeddings & embeddings);

    StreamOutcome decode_stream_chunk(bool final_flush);
    std::string build_stream_prefix(bool final_flush) const;
    std::string decode_rollback_prefix(const std::vector<int32_t> & ids, int64_t rollback) const;
    void publish_stream_delta(const std::string & fixed_text, runtime::StreamEvent & event);

    runtime::TaskSpec task_;
    std::shared_ptr<const R2T2ASRAssets> assets_;
    R2T2ASRStreamConfig stream_config_;
    size_t audio_encoder_graph_arena_bytes_ = 128ull * 1024ull * 1024ull;
    size_t thinker_prefill_graph_arena_bytes_ = 256ull * 1024ull * 1024ull;
    size_t thinker_decode_graph_arena_bytes_ = 256ull * 1024ull * 1024ull;
    size_t thinker_weight_context_bytes_ = 64ull * 1024ull * 1024ull;
    engine::assets::TensorStorageType audio_encoder_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType thinker_weight_storage_type_ = engine::assets::TensorStorageType::Native;

    R2T2ASRTextTokenizer tokenizer_;
    R2T2ASRWhisperFrontend frontend_;
    R2T2ASRAudioEncoderRuntime audio_encoder_;
    R2T2ASRThinkerRuntime thinker_;

    // Streaming state (mirrors ASRStreamingState in the reference code).
    runtime::TaskRequest streaming_request_;
    runtime::TaskResult streaming_result_;
    std::string prompt_raw_;
    std::string force_language_;
    std::string context_;
    std::string language_;
    std::string text_;
    std::string raw_decoded_;
    std::vector<float> buffer_;
    std::vector<float> audio_accum_;
    int64_t chunk_size_samples_ = 0;
    int64_t chunk_id_ = 0;
    size_t published_codepoints_ = 0;
    int stream_sample_rate_ = 0;
    int stream_channels_ = 1;
    runtime::StreamEventCallback stream_event_sink_;
    bool stream_started_ = false;
    std::chrono::steady_clock::time_point stream_wall_start_{};
};

}  // namespace engine::community_models::confucius4_r2t2
