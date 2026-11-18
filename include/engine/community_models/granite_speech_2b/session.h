#pragma once

// granite_speech_2b offline ASR session + loader (Task D): integrates the B1
// frontend/lengths, B2 encoder/projector, C1 prompt/injection, and C2 greedy
// decoder into a runnable offline ASR family.
//
// Option contract (spec v1, model_specs/granite_speech_2b.json):
//   request: language (optional source hint en/fr/de/es/pt/ja, blank = auto),
//     target_language (optional translation target en/fr/de/es/pt/ja/it/zh;
//     blank/absent = ASR; unsupported rejects), keywords via text_input.text
//     (optional comma-separated biasing terms; same list applied to every
//     chunk; blank = none), max_tokens (default 512, 1..16383, enforced per
//     chunk against the 16384 processor context bound), audio_chunk_mode
//     (auto/fixed/vad/none), audio_chunk_duration_sec (positive finite <= 120,
//     default 120, ignored in none).
//   session: granite_speech_2b.{encoder,projector,prefill,decode}_graph_arena_mb
//     (positive MiB, default 1024), granite_speech_2b.vad_model_path (optional
//     existing VAD weights file; auto falls back to fixed without one,
//     explicit vad requires it).
//   load: granite_speech_2b.weight_type (native default, f32 supported).
//
// Chunking: validate -> mono 16 kHz PCM on one sample clock (valid empty
// yields an empty transcript); reuse the parse/plan helpers; cap each span;
// join tiny (<= 256 post-resample samples) tails into a neighboring in-bound
// span and rebalance the last two; zero overlap; VAD with no speech yields an
// empty transcript. Per span: frontend -> encoder -> projector -> C1 prompt ->
// C2 greedy; temp per-clip buffers are released. Join trims outer whitespace
// and inserts a single separator only between ASCII word boundaries (JP
// script-aware: no injected spaces between Japanese neighbors). Result:
// TaskResult::text_output plain Transcript, asr/offline; output language is
// the normalized hint when known, empty when auto.

#include "engine/community_models/granite_speech_2b/assets.h"
#include "engine/community_models/granite_speech_2b/decoder.h"
#include "engine/community_models/granite_speech_2b/encoder.h"
#include "engine/community_models/granite_speech_2b/frontend.h"
#include "engine/community_models/granite_speech_2b/projector.h"
#include "engine/community_models/granite_speech_2b/prompt.h"
#include "engine/framework/audio/chunking.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session.h"
#include "engine/framework/runtime/session_base.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace engine::community_models::granite_speech_2b {

// Supported source-language hint codes (C1 contract). Empty = automatic
// recognition: no hint text, no language result.
inline constexpr const char * kGraniteSpeech2BSupportedLanguages[] = {
    "en", "fr", "de", "es", "pt", "ja",
};

// Request-option defaults and bounds (mirror model_specs/granite_speech_2b.json).
inline constexpr int64_t kGraniteSpeech2BDefaultMaxTokens = 512;
inline constexpr int64_t kGraniteSpeech2BMinMaxTokens = 1;
inline constexpr int64_t kGraniteSpeech2BMaxMaxTokens = 16383;
inline constexpr float kGraniteSpeech2BDefaultChunkSeconds = 120.0F;
inline constexpr float kGraniteSpeech2BMaxChunkSeconds = 120.0F;

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
struct GraniteSpeech2BRequestOptions {
    std::string language;
    std::string target_language;
    std::vector<std::string> keywords;
    int64_t max_tokens = kGraniteSpeech2BDefaultMaxTokens;
    audio::AudioChunkMode chunk_mode = audio::AudioChunkMode::Auto;
    float chunk_duration_sec = kGraniteSpeech2BDefaultChunkSeconds;
};

// Validates the request options: language (explicit request option wins over
// text_input.language; trim + lowercase ASCII; blank = auto; unsupported
// rejects), target_language (options-only request key; blank/absent = ASR,
// else a normalized target code, unsupported rejects; never read from
// text_input), keywords (parsed from text_input.text: comma-separated,
// trimmed, empties dropped; whitespace-only = none), max_tokens (1..16383),
// chunk mode (auto/fixed/vad/none only), chunk duration (positive finite <=
// 120, ignored in none).
GraniteSpeech2BRequestOptions granite_speech_2b_resolve_request_options(
    const runtime::TaskRequest & request);

// Fixed-mode span planner over post-resample 16 kHz mono samples: contiguous
// non-overlapping [start, end) spans of chunk_samples (last span shorter).
// Throws on non-positive inputs.
std::vector<runtime::TimeSpan> granite_speech_2b_plan_fixed_spans(
    int64_t total_samples, int64_t chunk_samples);

// Tail normalization shared by the fixed and VAD planners: spans with
// (0, kGraniteSpeech2BMinSamples] samples merge into a neighbor (next when
// one exists, else previous); any span left over the cap is split into
// near-equal in-bound parts (the common case rebalances the last two).
// A lone tiny span throws for fixed/whole inputs (too short to transcribe);
// VAD callers drop it instead (pass vad_single_tiny_ok=true) so a lone VAD
// blip yields no spans (empty transcript) rather than an error.
std::vector<runtime::TimeSpan> granite_speech_2b_merge_tiny_tails(
    std::vector<runtime::TimeSpan> spans,
    int64_t chunk_samples,
    bool vad_single_tiny_ok = false);

// Join rule: trims the chunk, then appends with a single space separator
// only when both boundary code points are ASCII (latin word separation);
// Japanese/non-ASCII neighbors concatenate directly.
void granite_speech_2b_append_chunk_text(std::string & merged, std::string chunk_text);

class GraniteSpeech2BSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    GraniteSpeech2BSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const GraniteSpeech2BAssets> assets,
        assets::TensorStorageType weight_storage);
    ~GraniteSpeech2BSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    std::string transcribe_span(
        const runtime::AudioBuffer & clip,
        const GraniteSpeech2BRequestOptions & options);
    runtime::IOfflineVoiceTaskSession & vad_session();

    runtime::TaskSpec task_;
    std::shared_ptr<const GraniteSpeech2BAssets> assets_;
    GraniteSpeech2BFrontend frontend_;
    GraniteSpeech2BEncoderRuntime encoder_;
    GraniteSpeech2BProjectorRuntime projector_;
    GraniteSpeech2BPromptBuilder prompt_builder_;
    std::vector<std::string> reserved_markers_;
    GraniteSpeech2BDecoderRuntime decoder_;
    std::string vad_model_path_;
    std::unique_ptr<runtime::ILoadedVoiceModel> vad_model_;
    std::unique_ptr<runtime::IOfflineVoiceTaskSession> vad_session_;
    // Sessions run under server/CLI leases; serialize run() so concurrent
    // calls cannot interleave the shared decoder graphs.
    std::mutex run_mutex_;
};

class GraniteSpeech2BLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    GraniteSpeech2BLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const GraniteSpeech2BAssets> assets,
        assets::TensorStorageType weight_storage);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const GraniteSpeech2BAssets> assets_;
    assets::TensorStorageType weight_storage_;
};

std::unique_ptr<GraniteSpeech2BLoadedModel> load_granite_speech_2b_model(
    const runtime::ModelLoadRequest & request);

std::shared_ptr<runtime::IVoiceModelLoader> make_granite_speech_2b_loader();

}  // namespace engine::community_models::granite_speech_2b
