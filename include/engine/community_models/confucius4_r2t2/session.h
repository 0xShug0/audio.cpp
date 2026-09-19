#pragma once

#include "engine/framework/runtime/session.h"
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
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::silero_vad {
class SileroRuntime;
struct SileroVADConfig;
}

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

/// VAD-driven endpointing for long streaming sessions (dictation / input
/// method). Defaults mirror the reference ws_server.py FireRed VAD intent
/// (0.4 speech threshold, 200 ms minimum silence, 50 ms onset pad, 20 s
/// maximum speech frame) expressed in Silero VAD terms. When enabled, a
/// speech end closes the current segment: the session runs the authoritative
/// final flush, publishes a VoiceActivityEvent::SpeechEnd carrying the
/// segment text, and re-opens a fresh LSP segment so audio towers never see
/// more than one segment worth of audio (far below the 1500-frame position
/// table). A bounded non-speech context window is retained between segments;
/// it also counts toward the segment cap.
struct R2T2ASREndpointingConfig {
    bool enabled = false;
    std::filesystem::path vad_model_path = "assets/framework/models/silero_vad";
    float threshold = 0.4f;
    int min_speech_ms = 100;
    int min_silence_ms = 200;
    int speech_pad_ms = 50;
    int gap_keep_ms = 2000;
    double max_segment_seconds = 20.0;
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

    // Endpointed streaming: the family embeds a Silero VAD runtime and splits
    // the incoming stream into speech segments. See R2T2ASREndpointingConfig.
    void ensure_vad_runtime();
    /// Steps the VAD over the chunk; returns true when the current segment
    /// must end (accepted speech end).
    void process_endpoint_frame(const runtime::AudioChunk & chunk, runtime::StreamEvent & event);
    bool feed_vad(const runtime::AudioChunk & chunk);
    /// Runs the authoritative final flush for the open segment, publishes the
    /// segment boundary event, and re-opens a fresh LSP segment.
    void flush_segment(runtime::StreamEvent & event, bool from_vad);
    void begin_new_segment();
    void append_stream_text(const std::string & segment_text);
    std::string joined_stream_text(const std::string & current_segment_text) const;
    int64_t to_stream_samples(int64_t vad_samples) const;
    const models::silero_vad::SileroVADConfig & vad_config() const;

    R2T2ASREndpointingConfig endpointing_;
    std::unique_ptr<models::silero_vad::SileroRuntime> vad_runtime_;
    std::unique_ptr<models::silero_vad::SileroVADConfig> vad_config_;
    // Session-global VAD bookkeeping: survives segment resets so spans stay
    // monotonic across the whole stream.
    std::vector<float> endpoint_input_;
    std::vector<float> vad_remainder_;
    int64_t vad_consumed_samples_ = 0;
    // Gap-context window (interleaved, stream format) retained before onset.
    std::vector<float> vad_seed_;
    bool in_speech_ = false;
    bool segment_has_audio_ = false;
    runtime::VoiceActivityEvent pending_speech_end_;
    int64_t segment_start_stream_sample_ = 0;
    int64_t segment_stream_frames_ = 0;
    int64_t max_segment_stream_frames_ = 0;
    int64_t stream_frames_consumed_ = 0;
    std::vector<std::pair<runtime::TimeSpan, std::string>> completed_segments_;
    int64_t segment_index_ = 0;

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
