/*
 * audiocpp.h — C ABI for embedding audio.cpp in-process.
 *
 * This is a thin facade over engine::runtime. It adds no behaviour of its own:
 * every call maps onto ModelRegistry -> ILoadedVoiceModel -> IVoiceTaskSession,
 * the same surfaces audiocpp_cli uses.
 *
 * Contract:
 *   - Opaque handles only. No C++ type crosses this boundary.
 *   - No exception escapes. Every entry point returns audiocpp_status; the
 *     detail for the last failing call on THIS thread is audiocpp_last_error().
 *   - `const char *` and `const float *` outputs are BORROWED. They stay valid
 *     until the handle that produced them is freed or mutated. Copy anything
 *     you intend to keep.
 *   - Handles are not thread-safe individually. Separate handles may be used
 *     concurrently from separate threads.
 *   - Freeing NULL is a no-op, so cleanup paths need no null checks.
 *
 * Lifetime note: handles keep their parents alive internally (a session holds
 * its model, a model holds its registry). Freeing out of order is therefore
 * safe, which matters for garbage-collected callers where finalizer order is
 * not deterministic.
 */

#ifndef AUDIOCPP_H
#define AUDIOCPP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  if defined(AUDIOCPP_BUILD_DLL)
#    define AUDIOCPP_API __declspec(dllexport)
#  elif defined(AUDIOCPP_USE_DLL)
#    define AUDIOCPP_API __declspec(dllimport)
#  else
#    define AUDIOCPP_API
#  endif
#elif defined(__GNUC__)
#  define AUDIOCPP_API __attribute__((visibility("default")))
#else
#  define AUDIOCPP_API
#endif

/* ------------------------------------------------------------------ */
/* Versioning                                                          */
/* ------------------------------------------------------------------ */

#define AUDIOCPP_ABI_VERSION_MAJOR 0
#define AUDIOCPP_ABI_VERSION_MINOR 1
#define AUDIOCPP_ABI_VERSION_PATCH 0

/* Packed as (major << 16) | (minor << 8) | patch. A caller built against a
 * different MAJOR must not use the library. */
AUDIOCPP_API uint32_t audiocpp_abi_version(void);

/* audio.cpp's own build version, e.g. "0.2.1". Borrowed, static lifetime. */
AUDIOCPP_API const char * audiocpp_build_version(void);

/* ------------------------------------------------------------------ */
/* Status and errors                                                   */
/* ------------------------------------------------------------------ */

typedef enum audiocpp_status {
    AUDIOCPP_OK = 0,
    AUDIOCPP_ERR_INVALID_ARGUMENT = 1,
    AUDIOCPP_ERR_UNSUPPORTED_FAMILY = 2,
    AUDIOCPP_ERR_LOAD_FAILED = 3,
    AUDIOCPP_ERR_RUNTIME = 4,
    AUDIOCPP_ERR_OUT_OF_MEMORY = 5,
    AUDIOCPP_ERR_OUT_OF_RANGE = 6,
    AUDIOCPP_ERR_NOT_AVAILABLE = 7
} audiocpp_status;

/* Detail for the most recent failure on this thread. Read it immediately after
 * a call returns non-OK; a later successful call may clear it. Never NULL, and
 * borrowed until this thread's next call. */
AUDIOCPP_API const char * audiocpp_last_error(void);

/* Human-readable name for a status code. Borrowed, static lifetime. */
AUDIOCPP_API const char * audiocpp_status_string(audiocpp_status status);

/* ------------------------------------------------------------------ */
/* Handles                                                             */
/* ------------------------------------------------------------------ */

typedef struct audiocpp_registry audiocpp_registry;
typedef struct audiocpp_model    audiocpp_model;
typedef struct audiocpp_session  audiocpp_session;
typedef struct audiocpp_options  audiocpp_options;
typedef struct audiocpp_request  audiocpp_request;
typedef struct audiocpp_result   audiocpp_result;
typedef struct audiocpp_event    audiocpp_event;

/* ------------------------------------------------------------------ */
/* Options — mirrors the framework's string->string option maps 1:1.   */
/* ------------------------------------------------------------------ */

AUDIOCPP_API audiocpp_options * audiocpp_options_create(void);
AUDIOCPP_API audiocpp_status    audiocpp_options_set(audiocpp_options * options,
                                                     const char * key,
                                                     const char * value);
AUDIOCPP_API void               audiocpp_options_free(audiocpp_options * options);

/* ------------------------------------------------------------------ */
/* Registry                                                            */
/* ------------------------------------------------------------------ */

/* config_path may be NULL for the default loader set. */
AUDIOCPP_API audiocpp_status audiocpp_registry_create(const char * config_path,
                                                      audiocpp_registry ** out_registry);
AUDIOCPP_API void            audiocpp_registry_free(audiocpp_registry * registry);

AUDIOCPP_API size_t          audiocpp_registry_family_count(const audiocpp_registry * registry);
AUDIOCPP_API audiocpp_status audiocpp_registry_family(const audiocpp_registry * registry,
                                                      size_t index,
                                                      const char ** out_family);

/* ------------------------------------------------------------------ */
/* Model                                                               */
/* ------------------------------------------------------------------ */

/* Model selection. Every field may be NULL:
 *   family_hint          --family, or NULL to identify the model from its path
 *   config_id            --config, for packages that ship several configs
 *   weight_id            --weight, for packages that ship several weight sets
 *   model_spec_override  --model-spec-override */
typedef struct audiocpp_model_config {
    const char * family_hint;
    const char * config_id;
    const char * weight_id;
    const char * model_spec_override;
} audiocpp_model_config;

/* config and options may both be NULL. options corresponds to --load-option. */
AUDIOCPP_API audiocpp_status audiocpp_model_load(audiocpp_registry * registry,
                                                  const char * model_path,
                                                  const audiocpp_model_config * config,
                                                  const audiocpp_options * options,
                                                  audiocpp_model ** out_model);
AUDIOCPP_API void            audiocpp_model_free(audiocpp_model * model);

AUDIOCPP_API const char * audiocpp_model_family(const audiocpp_model * model);
AUDIOCPP_API const char * audiocpp_model_description(const audiocpp_model * model);

/* Capability queries. task/mode are the same spellings the CLI accepts,
 * e.g. "tts", "asr", "vad", "diarization", "alignment" / "offline",
 * "streaming". Returns 1 when supported, 0 when not or when unrecognised. */
AUDIOCPP_API int audiocpp_model_supports(const audiocpp_model * model,
                                         const char * task,
                                         const char * mode);
AUDIOCPP_API int audiocpp_model_supports_timestamps(const audiocpp_model * model);
AUDIOCPP_API int audiocpp_model_supports_speaker_reference(const audiocpp_model * model);
AUDIOCPP_API int audiocpp_model_supports_style_condition(const audiocpp_model * model);

AUDIOCPP_API size_t          audiocpp_model_language_count(const audiocpp_model * model);
AUDIOCPP_API audiocpp_status audiocpp_model_language(const audiocpp_model * model,
                                                     size_t index,
                                                     const char ** out_language);

/* Which option map a CliOptionInfo belongs to. */
typedef enum audiocpp_option_scope {
    AUDIOCPP_OPTION_SCOPE_REQUEST = 0,
    AUDIOCPP_OPTION_SCOPE_SESSION = 1,
    AUDIOCPP_OPTION_SCOPE_LOAD    = 2
} audiocpp_option_scope;

/* Runtime introspection of a model's declared options, straight off
 * ModelInspection::cli. This is what lets a binding enumerate what a family
 * accepts instead of hardcoding per-family knowledge.
 *
 * Every out parameter is optional — pass NULL for anything you don't want.
 * Strings that the model did not declare come back as "" rather than NULL, so
 * callers never need a null check on them. out_required is 1 or 0. */
AUDIOCPP_API size_t          audiocpp_model_option_count(const audiocpp_model * model,
                                                         audiocpp_option_scope scope);
AUDIOCPP_API audiocpp_status audiocpp_model_option(const audiocpp_model * model,
                                                   audiocpp_option_scope scope,
                                                   size_t index,
                                                   const char ** out_name,
                                                   const char ** out_value_name,
                                                   const char ** out_description,
                                                   const char ** out_default_value,
                                                   const char ** out_min_value,
                                                   const char ** out_max_value,
                                                   int * out_required);

/* ------------------------------------------------------------------ */
/* Session                                                             */
/* ------------------------------------------------------------------ */

/* backend is "cpu", "cuda", "hip"/"rocm", "vulkan", "metal" or "best";
 * NULL means "cpu". device and threads mirror --device / --threads. */
typedef struct audiocpp_backend_config {
    const char * backend;
    int          device;
    int          threads;
} audiocpp_backend_config;

/* backend_config may be NULL for CPU / device 0 / 1 thread.
 * options may be NULL. Corresponds to --session-option. */
AUDIOCPP_API audiocpp_status audiocpp_session_create(const audiocpp_model * model,
                                                      const char * task,
                                                      const char * mode,
                                                      const audiocpp_backend_config * backend_config,
                                                      const audiocpp_options * options,
                                                      audiocpp_session ** out_session);
AUDIOCPP_API void            audiocpp_session_free(audiocpp_session * session);

AUDIOCPP_API const char * audiocpp_session_family(const audiocpp_session * session);

/* Optional. audiocpp_session_run() and audiocpp_stream_start() always prepare
 * with the request they are about to run -- preparation carries the input
 * length, so a session prepared for one request must not run another. Call this
 * only to force allocation early, with a request representative of what will
 * follow; the run still prepares. */
AUDIOCPP_API audiocpp_status audiocpp_session_prepare(audiocpp_session * session,
                                                      const audiocpp_request * request);

AUDIOCPP_API audiocpp_status audiocpp_session_run(audiocpp_session * session,
                                                  const audiocpp_request * request,
                                                  audiocpp_result ** out_result);

/* ------------------------------------------------------------------ */
/* Request                                                             */
/* ------------------------------------------------------------------ */

AUDIOCPP_API audiocpp_request * audiocpp_request_create(void);
AUDIOCPP_API void               audiocpp_request_free(audiocpp_request * request);

/* language may be NULL. A non-empty language is ALSO written to the request's
 * "language" option, because that is what audiocpp_cli's --language does and
 * some families read only the option. A later audiocpp_request_set_option with
 * the same key overrides it. */
AUDIOCPP_API audiocpp_status audiocpp_request_set_text(audiocpp_request * request,
                                                       const char * text,
                                                       const char * language);

/* Interleaved float PCM. Copied into the request, so `samples` need not
 * outlive the call. `frames` is per-channel. */
AUDIOCPP_API audiocpp_status audiocpp_request_set_audio(audiocpp_request * request,
                                                        const float * samples,
                                                        size_t frames,
                                                        int sample_rate,
                                                        int channels);

/* Speaker reference for cloning/conversion families. */
AUDIOCPP_API audiocpp_status audiocpp_request_set_voice_audio(audiocpp_request * request,
                                                              const float * samples,
                                                              size_t frames,
                                                              int sample_rate,
                                                              int channels);
AUDIOCPP_API audiocpp_status audiocpp_request_set_voice_id(audiocpp_request * request,
                                                           const char * cached_voice_id);

/* Style conditioning, for families that advertise it
 * (audiocpp_model_supports_style_condition). These mirror --style-language,
 * --emotion, --speaking-rate, --pitch-shift, --energy-scale and --style-tag.
 * Each is independent; set only the ones you need. */
AUDIOCPP_API audiocpp_status audiocpp_request_set_style_language(audiocpp_request * request,
                                                                 const char * language);
AUDIOCPP_API audiocpp_status audiocpp_request_set_emotion(audiocpp_request * request,
                                                          const char * emotion);
AUDIOCPP_API audiocpp_status audiocpp_request_set_speaking_rate(audiocpp_request * request,
                                                                float speaking_rate);
AUDIOCPP_API audiocpp_status audiocpp_request_set_pitch_shift(audiocpp_request * request,
                                                              float pitch_shift);
AUDIOCPP_API audiocpp_status audiocpp_request_set_energy_scale(audiocpp_request * request,
                                                               float energy_scale);
AUDIOCPP_API audiocpp_status audiocpp_request_set_style_tag(audiocpp_request * request,
                                                            const char * key,
                                                            const char * value);

/* Artifacts: opaque payloads a family produces or consumes -- speaker
 * embeddings, acoustic tokens, MIDI, diarization state. Persisting one from a
 * result and feeding it back into a later request is how voice state is reused
 * without re-deriving it. */
typedef enum audiocpp_artifact_kind {
    AUDIOCPP_ARTIFACT_SPEAKER_EMBEDDING = 0,
    AUDIOCPP_ARTIFACT_STYLE_EMBEDDING = 1,
    AUDIOCPP_ARTIFACT_PROMPT_EMBEDDING = 2,
    AUDIOCPP_ARTIFACT_ACOUSTIC_TOKENS = 3,
    AUDIOCPP_ARTIFACT_MIDI = 4,
    AUDIOCPP_ARTIFACT_TRANSCRIPT_ALIGNMENT = 5,
    AUDIOCPP_ARTIFACT_DIARIZATION_STATE = 6,
    AUDIOCPP_ARTIFACT_VAD_STATE = 7,
    AUDIOCPP_ARTIFACT_CUSTOM = 8
} audiocpp_artifact_kind;

/* The payload is copied, so it need not outlive the call. out_index (optional)
 * receives the artifact's position, for use with set_artifact_meta. */
AUDIOCPP_API audiocpp_status audiocpp_request_add_artifact(audiocpp_request * request,
                                                           audiocpp_artifact_kind kind,
                                                           const char * id,
                                                           const void * payload,
                                                           size_t payload_bytes,
                                                           size_t * out_index);
AUDIOCPP_API audiocpp_status audiocpp_request_set_artifact_meta(audiocpp_request * request,
                                                                size_t index,
                                                                const char * key,
                                                                const char * value);

/* Corresponds to --request-option. */
AUDIOCPP_API audiocpp_status audiocpp_request_set_option(audiocpp_request * request,
                                                         const char * key,
                                                         const char * value);

/* ------------------------------------------------------------------ */
/* Result                                                              */
/* ------------------------------------------------------------------ */
/* Accessors rather than a struct copy, so TaskResult can gain fields without
 * breaking the ABI. Every out parameter is optional. */

AUDIOCPP_API void audiocpp_result_free(audiocpp_result * result);

/* AUDIOCPP_ERR_NOT_AVAILABLE when the task produced no audio. */
AUDIOCPP_API audiocpp_status audiocpp_result_audio(const audiocpp_result * result,
                                                   const float ** out_samples,
                                                   size_t * out_frames,
                                                   int * out_sample_rate,
                                                   int * out_channels);

/* AUDIOCPP_ERR_NOT_AVAILABLE when the task produced no text. */
AUDIOCPP_API audiocpp_status audiocpp_result_text(const audiocpp_result * result,
                                                  const char ** out_text,
                                                  const char ** out_language);

AUDIOCPP_API size_t          audiocpp_result_segment_count(const audiocpp_result * result);
AUDIOCPP_API audiocpp_status audiocpp_result_segment(const audiocpp_result * result,
                                                     size_t index,
                                                     int64_t * out_start_sample,
                                                     int64_t * out_end_sample,
                                                     float * out_confidence,
                                                     const char ** out_text);

AUDIOCPP_API size_t          audiocpp_result_speaker_turn_count(const audiocpp_result * result);
AUDIOCPP_API audiocpp_status audiocpp_result_speaker_turn(const audiocpp_result * result,
                                                          size_t index,
                                                          int64_t * out_start_sample,
                                                          int64_t * out_end_sample,
                                                          const char ** out_speaker_id,
                                                          float * out_confidence,
                                                          const char ** out_text);

AUDIOCPP_API size_t          audiocpp_result_word_count(const audiocpp_result * result);
AUDIOCPP_API audiocpp_status audiocpp_result_word(const audiocpp_result * result,
                                                  size_t index,
                                                  const char ** out_word,
                                                  int64_t * out_start_sample,
                                                  int64_t * out_end_sample,
                                                  float * out_confidence);

/* Named audio outputs, for families that emit more than one stream
 * (source separation, multi-speaker TTS). */
AUDIOCPP_API size_t          audiocpp_result_named_audio_count(const audiocpp_result * result);
AUDIOCPP_API audiocpp_status audiocpp_result_named_audio(const audiocpp_result * result,
                                                         size_t index,
                                                         const char ** out_id,
                                                         const float ** out_samples,
                                                         size_t * out_frames,
                                                         int * out_sample_rate,
                                                         int * out_channels);

/* Artifacts the task produced. TaskResult carries a single artifact_output and
 * a list of output_artifacts; both are presented here as one flat list, the
 * single one first when present. The payload is borrowed. */
AUDIOCPP_API size_t          audiocpp_result_artifact_count(const audiocpp_result * result);
AUDIOCPP_API audiocpp_status audiocpp_result_artifact(const audiocpp_result * result,
                                                      size_t index,
                                                      audiocpp_artifact_kind * out_kind,
                                                      const char ** out_id,
                                                      const void ** out_payload,
                                                      size_t * out_payload_bytes);
AUDIOCPP_API size_t          audiocpp_result_artifact_meta_count(const audiocpp_result * result,
                                                                 size_t index);
AUDIOCPP_API audiocpp_status audiocpp_result_artifact_meta(const audiocpp_result * result,
                                                           size_t index,
                                                           size_t meta_index,
                                                           const char ** out_key,
                                                           const char ** out_value);

/* ------------------------------------------------------------------ */
/* Streaming                                                           */
/* ------------------------------------------------------------------ */
/* Pull-based, mirroring IStreamingVoiceTaskSession. No callback crosses the
 * FFI boundary. Requires a session created with mode "streaming". */

typedef enum audiocpp_stream_input_kind {
    AUDIOCPP_STREAM_INPUT_NONE = 0,
    AUDIOCPP_STREAM_INPUT_AUDIO_CHUNKS = 1
} audiocpp_stream_input_kind;

typedef enum audiocpp_stream_output_kind {
    AUDIOCPP_STREAM_OUTPUT_FINAL_RESULT = 0,
    AUDIOCPP_STREAM_OUTPUT_PULL_EVENTS = 1
} audiocpp_stream_output_kind;

/* The chunk size the family wants; push that many frames per call where you
 * can. Every out parameter is optional. */
AUDIOCPP_API audiocpp_status audiocpp_stream_policy(const audiocpp_session * session,
                                                    audiocpp_stream_input_kind * out_input,
                                                    audiocpp_stream_output_kind * out_output,
                                                    int64_t * out_preferred_chunk_samples,
                                                    double * out_preferred_chunk_seconds);

/* request may be NULL. Resets any stream already in flight. */
AUDIOCPP_API audiocpp_status audiocpp_stream_start(audiocpp_session * session,
                                                   const audiocpp_request * request);

/* Feeds one chunk and returns the event it produced. out_event may be NULL if
 * the caller only wants the final result. */
AUDIOCPP_API audiocpp_status audiocpp_stream_push(audiocpp_session * session,
                                                  const float * samples,
                                                  size_t frames,
                                                  int sample_rate,
                                                  int channels,
                                                  int64_t start_sample,
                                                  audiocpp_event ** out_event);

/* Drains events the family queued itself. Sets *out_event to NULL when the
 * queue is empty; that is AUDIOCPP_OK, not an error. */
AUDIOCPP_API audiocpp_status audiocpp_stream_next_event(audiocpp_session * session,
                                                        audiocpp_event ** out_event);

AUDIOCPP_API audiocpp_status audiocpp_stream_finish(audiocpp_session * session,
                                                    audiocpp_result ** out_result);
AUDIOCPP_API audiocpp_status audiocpp_stream_reset(audiocpp_session * session);

AUDIOCPP_API void audiocpp_event_free(audiocpp_event * event);
AUDIOCPP_API int  audiocpp_event_is_final(const audiocpp_event * event);

/* An event carries the same shapes a result does, so it is read through the
 * result accessors. Borrowed — valid until the event is freed. */
AUDIOCPP_API const audiocpp_result * audiocpp_event_as_result(const audiocpp_event * event);

typedef enum audiocpp_voice_activity_kind {
    AUDIOCPP_VOICE_ACTIVITY_SPEECH_START = 0,
    AUDIOCPP_VOICE_ACTIVITY_SPEECH_END = 1,
    AUDIOCPP_VOICE_ACTIVITY_SPEECH_SEGMENT = 2
} audiocpp_voice_activity_kind;

AUDIOCPP_API size_t          audiocpp_event_voice_activity_count(const audiocpp_event * event);
AUDIOCPP_API audiocpp_status audiocpp_event_voice_activity(const audiocpp_event * event,
                                                           size_t index,
                                                           audiocpp_voice_activity_kind * out_kind,
                                                           int64_t * out_sample,
                                                           float * out_probability);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* AUDIOCPP_H */
