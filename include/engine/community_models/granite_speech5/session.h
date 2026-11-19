#pragma once

// Offline ASR/translation session. Owns the frontend, encoder, projector and
// shared greedy decoder; graph/cache resources remain session-owned.
// Audio chunking is non-overlapping, merges undersized tails, and joins
// Japanese text without inserting spaces between adjacent Japanese characters.

#include "engine/community_models/granite_speech5/assets.h"
#include "engine/community_models/granite_speech5/decoder.h"
#include "engine/community_models/granite_speech5/encoder.h"
#include "engine/community_models/granite_speech5/frontend.h"
#include "engine/community_models/granite_speech5/projector.h"
#include "engine/community_models/granite_speech5/prompt.h"
#include "engine/framework/audio/chunking.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session.h"
#include "engine/framework/runtime/session_base.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace engine::community_models::granite_speech5 {

// Supported source-language hint codes. Empty = automatic
// recognition: no hint text, no language result.
inline constexpr const char * kGraniteSpeech5SupportedLanguages[] = {
    "en", "fr", "de", "es", "pt", "ja",
};

// Request-option defaults and bounds (mirror model_specs/granite_speech5.json).
inline constexpr int64_t kGraniteSpeech5DefaultMaxTokens = 512;
inline constexpr int64_t kGraniteSpeech5MinMaxTokens = 1;
inline constexpr int64_t kGraniteSpeech5MaxMaxTokens = 16383;
inline constexpr float kGraniteSpeech5DefaultChunkSeconds = 120.0F;
inline constexpr float kGraniteSpeech5MaxChunkSeconds = 120.0F;

// Resolved per-request options: language is the source hint ("" = auto);
// target_language is the translation target ("" = ASR; a normalized target
// code selects translation). keywords are the parsed text_input.text terms
// (comma-separated; empty = no biasing). Keywords travel ONLY in
// text_input.text on every transport (CLI --text, JSON text, multipart
// prompt|text): no separate keywords option or option_arrays channel exists,
// so there is no precedence to resolve — a single comma-separated channel
// keeps one parsing/validation path and matches the OpenAI-compat prompt
// convention on multipart. target_language travels ONLY in request
// options["target_language"] (CLI --request-option, JSON options): the
// single-file multipart shape has no options channel, so a target_language
// part there is rejected loudly (never silently transcribed as ASR).
struct GraniteSpeech5RequestOptions {
    std::string language;
    std::string target_language;
    std::vector<std::string> keywords;
    int64_t max_tokens = kGraniteSpeech5DefaultMaxTokens;
    audio::AudioChunkMode chunk_mode = audio::AudioChunkMode::Auto;
    float chunk_duration_sec = kGraniteSpeech5DefaultChunkSeconds;
};

// Validates the request options: language (explicit request option wins over
// text_input.language; trim + lowercase ASCII; blank = auto; unsupported
// rejects), target_language (options-only request key; blank/absent = ASR,
// else a normalized target code, unsupported rejects; never read from
// text_input), keywords (parsed from text_input.text: comma-separated,
// trimmed, empties dropped; whitespace-only = none), max_tokens (1..16383),
// chunk mode (auto/fixed/vad/none only), chunk duration (positive finite <=
// 120, ignored in none).
GraniteSpeech5RequestOptions granite_speech5_resolve_request_options(
    const runtime::TaskRequest & request);

// Fixed-mode span planner over post-resample 16 kHz mono samples: contiguous
// non-overlapping [start, end) spans of chunk_samples (last span shorter).
// Throws on non-positive inputs.
std::vector<runtime::TimeSpan> granite_speech5_plan_fixed_spans(
    int64_t total_samples, int64_t chunk_samples);

// Tail normalization shared by the fixed and VAD planners: spans with
// (0, kGraniteSpeech5MinSamples] samples merge into a neighbor (next when
// one exists, else previous); any span left over the cap is split into
// near-equal in-bound parts (the common case rebalances the last two).
// A lone tiny span throws for fixed/whole inputs (too short to transcribe);
// VAD callers drop it instead (pass vad_single_tiny_ok=true) so a lone VAD
// blip yields no spans (empty transcript) rather than an error.
std::vector<runtime::TimeSpan> granite_speech5_merge_tiny_tails(
    std::vector<runtime::TimeSpan> spans,
    int64_t chunk_samples,
    bool vad_single_tiny_ok = false);

// Join rule: trims the chunk, then appends with a single space separator
// only when both boundary code points are ASCII (latin word separation);
// Japanese/non-ASCII neighbors concatenate directly.
void granite_speech5_append_chunk_text(std::string & merged, std::string chunk_text);

class GraniteSpeech5Session final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    GraniteSpeech5Session(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const GraniteSpeech5Assets> assets,
        assets::TensorStorageType weight_storage);
    ~GraniteSpeech5Session() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    std::string transcribe_span(
        const runtime::AudioBuffer & clip,
        const GraniteSpeech5RequestOptions & options);
    runtime::IOfflineVoiceTaskSession & vad_session();

    runtime::TaskSpec task_;
    std::shared_ptr<const GraniteSpeech5Assets> assets_;
    GraniteSpeech5Frontend frontend_;
    GraniteSpeech5EncoderRuntime encoder_;
    GraniteSpeech5ProjectorRuntime projector_;
    GraniteSpeech5PromptBuilder prompt_builder_;
    std::vector<std::string> reserved_markers_;
    runtime::GreedyCausalDecoderRuntime decoder_;
    std::string vad_model_path_;
    std::unique_ptr<runtime::ILoadedVoiceModel> vad_model_;
    std::unique_ptr<runtime::IOfflineVoiceTaskSession> vad_session_;
    // Sessions run under server/CLI leases; serialize run() so concurrent
    // calls cannot interleave the shared decoder graphs.
    std::mutex run_mutex_;
};

class GraniteSpeech5LoadedModel final : public runtime::ILoadedVoiceModel {
public:
    GraniteSpeech5LoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const GraniteSpeech5Assets> assets,
        assets::TensorStorageType weight_storage);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const GraniteSpeech5Assets> assets_;
    assets::TensorStorageType weight_storage_;
};

std::unique_ptr<GraniteSpeech5LoadedModel> load_granite_speech5_model(
    const runtime::ModelLoadRequest & request);

std::shared_ptr<runtime::IVoiceModelLoader> make_granite_speech5_loader();

}  // namespace engine::community_models::granite_speech5
