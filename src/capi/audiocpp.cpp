/*
 * C ABI facade over engine::runtime.
 *
 * This file adds no behaviour. It owns three things and nothing else:
 *   1. translating C types to and from the framework's own types,
 *   2. stopping every exception at the boundary,
 *   3. keeping parent handles alive so callers can free in any order.
 *
 * It deliberately depends only on the framework (the same headers
 * audiocpp_cli uses) and never on app/, so the CLI and the C API stay
 * independent of each other.
 */

#include "audiocpp.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iterator>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace rt = engine::runtime;

namespace {

thread_local std::string g_last_error;

const char * const kEmptyString = "";

/* Every entry point funnels through here: the framework throws, the ABI
 * returns codes. `runtime_status` lets a caller classify its own failures
 * (a load failure is not a generic runtime failure) without parsing the
 * exception message. */
template <typename Fn>
audiocpp_status guard(Fn && fn, audiocpp_status runtime_status = AUDIOCPP_ERR_RUNTIME) noexcept {
    try {
        g_last_error.clear();
        return fn();
    } catch (const std::bad_alloc & e) {
        g_last_error = e.what();
        return AUDIOCPP_ERR_OUT_OF_MEMORY;
    } catch (const std::invalid_argument & e) {
        g_last_error = e.what();
        return AUDIOCPP_ERR_INVALID_ARGUMENT;
    } catch (const std::exception & e) {
        g_last_error = e.what();
        return runtime_status;
    } catch (...) {
        g_last_error = "unknown error";
        return runtime_status;
    }
}

audiocpp_status fail(audiocpp_status status, const char * message) noexcept {
    g_last_error = message;
    return status;
}

engine::core::BackendType parse_backend_type(const std::string & value) {
    /* Intentionally duplicated from app/cli/args.cpp rather than shared: the
     * C API must not depend on the CLI app. The spellings are the CLI's. */
    if (value == "cpu") return engine::core::BackendType::Cpu;
    if (value == "cuda") return engine::core::BackendType::Cuda;
    if (value == "hip" || value == "rocm") return engine::core::BackendType::Hip;
    if (value == "vulkan") return engine::core::BackendType::Vulkan;
    if (value == "metal") return engine::core::BackendType::Metal;
    if (value == "best") return engine::core::BackendType::BestAvailable;
    throw std::invalid_argument("unsupported backend: " + value);
}

std::unordered_map<std::string, std::string> option_map(const audiocpp_options * options);

void assign_out(const char ** out, const std::string & value) {
    if (out != nullptr) *out = value.c_str();
}

void assign_out(const char ** out, const std::optional<std::string> & value) {
    if (out != nullptr) *out = value.has_value() ? value->c_str() : kEmptyString;
}

rt::AudioBuffer make_audio_buffer(const float * samples, size_t frames, int sample_rate, int channels) {
    if (sample_rate <= 0) throw std::invalid_argument("sample_rate must be positive");
    if (channels <= 0) throw std::invalid_argument("channels must be positive");
    if (frames > 0 && samples == nullptr) throw std::invalid_argument("samples is null but frames > 0");
    if (frames > SIZE_MAX / static_cast<size_t>(channels)) {
        throw std::invalid_argument("frames * channels overflows");
    }

    rt::AudioBuffer buffer;
    buffer.sample_rate = sample_rate;
    buffer.channels = channels;
    if (frames > 0) {
        /* Interleaved, matching what the framework hands to the WAV sink.
         * Guarded because [null, null) is not a valid range to assign from,
         * even though an empty buffer is a legitimate request. */
        buffer.samples.assign(samples, samples + (frames * static_cast<size_t>(channels)));
    }
    return buffer;
}

size_t frame_count(const rt::AudioBuffer & audio) noexcept {
    const int channels = audio.channels > 0 ? audio.channels : 1;
    return audio.samples.size() / static_cast<size_t>(channels);
}

}  // namespace

/* ------------------------------------------------------------------ */
/* Handles                                                             */
/* ------------------------------------------------------------------ */

struct audiocpp_options {
    std::unordered_map<std::string, std::string> map;
};

struct audiocpp_registry {
    std::shared_ptr<rt::ModelRegistry> registry;
    std::vector<std::string> families;
};

struct audiocpp_model {
    /* Held so the registry cannot outlive its loaded models regardless of the
     * order the caller frees handles in. */
    std::shared_ptr<rt::ModelRegistry> registry;
    std::shared_ptr<rt::ILoadedVoiceModel> model;
    rt::ModelLoadRequest load_request;

    /* Inspection rescans the model directory, so it is deferred until a
     * caller actually asks about options. */
    mutable std::optional<rt::ModelInspection> inspection;

    const rt::ModelInspection & inspect() const {
        if (!inspection.has_value()) {
            inspection = registry->inspect(load_request);
        }
        return *inspection;
    }

    const std::vector<rt::CliOptionInfo> & options_for(audiocpp_option_scope scope) const {
        const auto & cli = inspect().cli;
        switch (scope) {
            case AUDIOCPP_OPTION_SCOPE_REQUEST: return cli.request_options;
            case AUDIOCPP_OPTION_SCOPE_SESSION: return cli.session_options;
            case AUDIOCPP_OPTION_SCOPE_LOAD:    return cli.load_options;
        }
        throw std::invalid_argument("unknown option scope");
    }
};

struct audiocpp_session {
    std::shared_ptr<rt::ModelRegistry> registry;
    std::shared_ptr<rt::ILoadedVoiceModel> model;
    std::unique_ptr<rt::IVoiceTaskSession> session;
    rt::RunMode mode = rt::RunMode::Offline;
    std::string family;

    rt::IOfflineVoiceTaskSession & offline() const {
        auto * offline = dynamic_cast<rt::IOfflineVoiceTaskSession *>(session.get());
        if (offline == nullptr) {
            throw std::runtime_error("session does not support offline execution");
        }
        return *offline;
    }

    rt::IStreamingVoiceTaskSession & streaming() const {
        auto * streaming = dynamic_cast<rt::IStreamingVoiceTaskSession *>(session.get());
        if (streaming == nullptr) {
            throw std::runtime_error("session does not support streaming execution");
        }
        return *streaming;
    }
};

struct audiocpp_request {
    rt::TaskRequest request;
};

struct audiocpp_result {
    rt::TaskResult result;
    /* An event exposes its contents through a result view that it owns. The
     * view has the same type as an owned result and there is a public free for
     * that type, so freeing one has to be a no-op rather than a double free. */
    bool owned = true;
};

struct audiocpp_event {
    audiocpp_result view;
    std::vector<rt::VoiceActivityEvent> voice_activity;
    bool is_final = false;
};

namespace {

std::unordered_map<std::string, std::string> option_map(const audiocpp_options * options) {
    return options != nullptr ? options->map : std::unordered_map<std::string, std::string>{};
}

/* A StreamEvent carries the same shapes a TaskResult does under different
 * names, so it is presented through the result accessors. partial_text maps
 * to text_output, and any voice-activity event that carries a segment is
 * surfaced as a speech segment. */
std::unique_ptr<audiocpp_event> wrap_event(rt::StreamEvent && event) {
    auto wrapped = std::make_unique<audiocpp_event>();
    wrapped->view.owned = false;
    wrapped->is_final = event.is_final;
    wrapped->voice_activity = event.voice_activity;

    auto & result = wrapped->view.result;
    result.audio_output = std::move(event.audio_output);
    result.named_audio_outputs = std::move(event.named_audio_outputs);
    result.text_output = std::move(event.partial_text);
    result.speaker_turns = std::move(event.speaker_turns);
    result.word_timestamps = std::move(event.word_timestamps);
    result.output_artifacts = std::move(event.output_artifacts);
    for (const auto & activity : wrapped->voice_activity) {
        if (activity.segment.has_value()) {
            result.speech_segments.push_back(*activity.segment);
        }
    }
    return wrapped;
}

}  // namespace

/* ------------------------------------------------------------------ */
/* Versioning                                                          */
/* ------------------------------------------------------------------ */

uint32_t audiocpp_abi_version(void) {
    return (static_cast<uint32_t>(AUDIOCPP_ABI_VERSION_MAJOR) << 16) |
           (static_cast<uint32_t>(AUDIOCPP_ABI_VERSION_MINOR) << 8) |
           static_cast<uint32_t>(AUDIOCPP_ABI_VERSION_PATCH);
}

const char * audiocpp_build_version(void) {
#ifdef AUDIOCPP_VERSION_STRING
    return AUDIOCPP_VERSION_STRING;
#else
    return "unknown";
#endif
}

const char * audiocpp_last_error(void) {
    return g_last_error.c_str();
}

const char * audiocpp_status_string(audiocpp_status status) {
    switch (status) {
        case AUDIOCPP_OK: return "ok";
        case AUDIOCPP_ERR_INVALID_ARGUMENT: return "invalid argument";
        case AUDIOCPP_ERR_UNSUPPORTED_FAMILY: return "unsupported family";
        case AUDIOCPP_ERR_LOAD_FAILED: return "load failed";
        case AUDIOCPP_ERR_RUNTIME: return "runtime error";
        case AUDIOCPP_ERR_OUT_OF_MEMORY: return "out of memory";
        case AUDIOCPP_ERR_OUT_OF_RANGE: return "index out of range";
        case AUDIOCPP_ERR_NOT_AVAILABLE: return "not available";
    }
    return "unknown status";
}

/* ------------------------------------------------------------------ */
/* Options                                                             */
/* ------------------------------------------------------------------ */

audiocpp_options * audiocpp_options_create(void) {
    try {
        return new audiocpp_options();
    } catch (...) {
        return nullptr;
    }
}

audiocpp_status audiocpp_options_set(audiocpp_options * options, const char * key, const char * value) {
    if (options == nullptr || key == nullptr || value == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "options, key and value must be non-null");
    }
    return guard([&] {
        options->map[key] = value;
        return AUDIOCPP_OK;
    });
}

void audiocpp_options_free(audiocpp_options * options) {
    delete options;
}

/* ------------------------------------------------------------------ */
/* Registry                                                            */
/* ------------------------------------------------------------------ */

audiocpp_status audiocpp_registry_create(const char * config_path, audiocpp_registry ** out_registry) {
    if (out_registry == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "out_registry must be non-null");
    }
    *out_registry = nullptr;
    return guard([&] {
        auto handle = std::make_unique<audiocpp_registry>();
        auto config = config_path != nullptr
            ? std::optional<std::filesystem::path>(std::filesystem::path(config_path))
            : std::nullopt;
        handle->registry = std::make_shared<rt::ModelRegistry>(rt::make_default_registry(config));
        handle->families = handle->registry->families();
        *out_registry = handle.release();
        return AUDIOCPP_OK;
    });
}

void audiocpp_registry_free(audiocpp_registry * registry) {
    delete registry;
}

size_t audiocpp_registry_family_count(const audiocpp_registry * registry) {
    return registry != nullptr ? registry->families.size() : 0;
}

audiocpp_status audiocpp_registry_family(const audiocpp_registry * registry, size_t index, const char ** out_family) {
    if (registry == nullptr || out_family == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "registry and out_family must be non-null");
    }
    if (index >= registry->families.size()) {
        return fail(AUDIOCPP_ERR_OUT_OF_RANGE, "family index out of range");
    }
    *out_family = registry->families[index].c_str();
    return AUDIOCPP_OK;
}

/* ------------------------------------------------------------------ */
/* Model                                                               */
/* ------------------------------------------------------------------ */

audiocpp_status audiocpp_model_load(audiocpp_registry * registry,
                                    const char * model_path,
                                    const audiocpp_model_config * config,
                                    const audiocpp_options * options,
                                    audiocpp_model ** out_model) {
    if (registry == nullptr || model_path == nullptr || out_model == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "registry, model_path and out_model must be non-null");
    }
    *out_model = nullptr;
    const char * family_hint = config != nullptr ? config->family_hint : nullptr;
    /* Checked up front so an unknown family is reported as such instead of
     * arriving as a generic load failure. */
    if (family_hint != nullptr && !registry->registry->supports_family(family_hint)) {
        return fail(AUDIOCPP_ERR_UNSUPPORTED_FAMILY, std::string("unsupported family: ").append(family_hint).c_str());
    }
    return guard([&] {
        auto handle = std::make_unique<audiocpp_model>();
        handle->registry = registry->registry;
        handle->load_request.model_path = std::filesystem::path(model_path);
        if (family_hint != nullptr) {
            handle->load_request.family_hint = std::string(family_hint);
        }
        if (config != nullptr) {
            if (config->config_id != nullptr) handle->load_request.config_id = std::string(config->config_id);
            if (config->weight_id != nullptr) handle->load_request.weight_id = std::string(config->weight_id);
            if (config->model_spec_override != nullptr) {
                handle->load_request.model_spec_override =
                    std::filesystem::path(config->model_spec_override);
            }
        }
        handle->load_request.options = option_map(options);
        handle->model = std::shared_ptr<rt::ILoadedVoiceModel>(
            handle->registry->load(handle->load_request).release());
        *out_model = handle.release();
        return AUDIOCPP_OK;
    }, AUDIOCPP_ERR_LOAD_FAILED);
}

void audiocpp_model_free(audiocpp_model * model) {
    delete model;
}

const char * audiocpp_model_family(const audiocpp_model * model) {
    if (model == nullptr) return kEmptyString;
    return model->model->metadata().family.c_str();
}

const char * audiocpp_model_description(const audiocpp_model * model) {
    if (model == nullptr) return kEmptyString;
    return model->model->metadata().description.c_str();
}

int audiocpp_model_supports(const audiocpp_model * model, const char * task, const char * mode) {
    if (model == nullptr || task == nullptr || mode == nullptr) return 0;
    try {
        const auto wanted_task = rt::parse_voice_task_kind(task);
        const auto wanted_mode = rt::parse_run_mode(mode);
        for (const auto & capability : model->model->capabilities().supported_tasks) {
            if (capability.task != wanted_task) continue;
            for (const auto & supported : capability.modes) {
                if (supported == wanted_mode) return 1;
            }
        }
    } catch (...) {
        /* An unrecognised task or mode is "not supported", not an error. */
        return 0;
    }
    return 0;
}

int audiocpp_model_supports_timestamps(const audiocpp_model * model) {
    if (model == nullptr) return 0;
    return model->model->capabilities().supports_timestamps ? 1 : 0;
}

int audiocpp_model_supports_speaker_reference(const audiocpp_model * model) {
    if (model == nullptr) return 0;
    return model->model->capabilities().supports_speaker_reference ? 1 : 0;
}

int audiocpp_model_supports_style_condition(const audiocpp_model * model) {
    if (model == nullptr) return 0;
    return model->model->capabilities().supports_style_condition ? 1 : 0;
}

size_t audiocpp_model_language_count(const audiocpp_model * model) {
    if (model == nullptr) return 0;
    return model->model->capabilities().languages.size();
}

audiocpp_status audiocpp_model_language(const audiocpp_model * model, size_t index, const char ** out_language) {
    if (model == nullptr || out_language == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "model and out_language must be non-null");
    }
    const auto & languages = model->model->capabilities().languages;
    if (index >= languages.size()) {
        return fail(AUDIOCPP_ERR_OUT_OF_RANGE, "language index out of range");
    }
    *out_language = languages[index].c_str();
    return AUDIOCPP_OK;
}

size_t audiocpp_model_option_count(const audiocpp_model * model, audiocpp_option_scope scope) {
    if (model == nullptr) return 0;
    try {
        return model->options_for(scope).size();
    } catch (...) {
        return 0;
    }
}

audiocpp_status audiocpp_model_option(const audiocpp_model * model,
                                      audiocpp_option_scope scope,
                                      size_t index,
                                      const char ** out_name,
                                      const char ** out_value_name,
                                      const char ** out_description,
                                      const char ** out_default_value,
                                      const char ** out_min_value,
                                      const char ** out_max_value,
                                      int * out_required) {
    if (model == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "model must be non-null");
    }
    return guard([&] {
        const auto & options = model->options_for(scope);
        if (index >= options.size()) {
            return fail(AUDIOCPP_ERR_OUT_OF_RANGE, "option index out of range");
        }
        const auto & option = options[index];
        assign_out(out_name, option.name);
        assign_out(out_value_name, option.value_name);
        assign_out(out_description, option.description);
        assign_out(out_default_value, option.default_value);
        assign_out(out_min_value, option.min_value);
        assign_out(out_max_value, option.max_value);
        if (out_required != nullptr) *out_required = option.required ? 1 : 0;
        return AUDIOCPP_OK;
    });
}

/* ------------------------------------------------------------------ */
/* Session                                                             */
/* ------------------------------------------------------------------ */

audiocpp_status audiocpp_session_create(const audiocpp_model * model,
                                        const char * task,
                                        const char * mode,
                                        const audiocpp_backend_config * backend_config,
                                        const audiocpp_options * options,
                                        audiocpp_session ** out_session) {
    if (model == nullptr || task == nullptr || mode == nullptr || out_session == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "model, task, mode and out_session must be non-null");
    }
    *out_session = nullptr;
    return guard([&] {
        const rt::TaskSpec spec{rt::parse_voice_task_kind(task), rt::parse_run_mode(mode)};

        rt::SessionOptions session_options;
        if (backend_config != nullptr) {
            if (backend_config->backend != nullptr) {
                session_options.backend.type = parse_backend_type(backend_config->backend);
            }
            if (backend_config->threads <= 0) {
                throw std::invalid_argument("threads must be positive");
            }
            session_options.backend.device = backend_config->device;
            session_options.backend.threads = backend_config->threads;
        }
        session_options.options = option_map(options);

        auto handle = std::make_unique<audiocpp_session>();
        handle->registry = model->registry;
        handle->model = model->model;
        handle->mode = spec.mode;
        handle->session = model->model->create_task_session(spec, session_options);
        handle->family = handle->session->family();
        *out_session = handle.release();
        return AUDIOCPP_OK;
    });
}

void audiocpp_session_free(audiocpp_session * session) {
    delete session;
}

const char * audiocpp_session_family(const audiocpp_session * session) {
    if (session == nullptr) return kEmptyString;
    return session->family.c_str();
}

audiocpp_status audiocpp_session_prepare(audiocpp_session * session, const audiocpp_request * request) {
    if (session == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "session must be non-null");
    }
    return guard([&] {
        const rt::TaskRequest empty;
        const auto & task_request = request != nullptr ? request->request : empty;
        session->session->prepare(rt::build_preparation_request(task_request));
        return AUDIOCPP_OK;
    });
}

audiocpp_status audiocpp_session_run(audiocpp_session * session,
                                     const audiocpp_request * request,
                                     audiocpp_result ** out_result) {
    if (session == nullptr || out_result == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "session and out_result must be non-null");
    }
    *out_result = nullptr;
    if (session->mode != rt::RunMode::Offline) {
        return fail(AUDIOCPP_ERR_NOT_AVAILABLE, "session was created in streaming mode");
    }
    return guard([&] {
        const rt::TaskRequest empty;
        const auto & task_request = request != nullptr ? request->request : empty;
        /* Always prepared with the request that is about to run, exactly as
         * audiocpp_cli does. build_preparation_request carries the input
         * length, so preparing once and then running a different request would
         * leave the session sized for the wrong input. */
        session->session->prepare(rt::build_preparation_request(task_request));

        auto handle = std::make_unique<audiocpp_result>();
        handle->result = session->offline().run(task_request);
        *out_result = handle.release();
        return AUDIOCPP_OK;
    });
}

/* ------------------------------------------------------------------ */
/* Request                                                             */
/* ------------------------------------------------------------------ */

audiocpp_request * audiocpp_request_create(void) {
    try {
        return new audiocpp_request();
    } catch (...) {
        return nullptr;
    }
}

void audiocpp_request_free(audiocpp_request * request) {
    delete request;
}

audiocpp_status audiocpp_request_set_text(audiocpp_request * request, const char * text, const char * language) {
    if (request == nullptr || text == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "request and text must be non-null");
    }
    return guard([&] {
        rt::Transcript transcript;
        transcript.text = text;
        if (language != nullptr) transcript.language = language;
        request->request.text_input = std::move(transcript);
        /* audiocpp_cli's --language sets BOTH the transcript language and
         * options["language"] (app/cli/request.cpp:336), and some families read
         * only the latter. Mirroring that is what keeps this API's answers
         * identical to the CLI's. A later audiocpp_request_set_option wins. */
        if (language != nullptr && *language != '\0') {
            request->request.options["language"] = language;
        }
        return AUDIOCPP_OK;
    });
}

audiocpp_status audiocpp_request_set_audio(audiocpp_request * request,
                                           const float * samples,
                                           size_t frames,
                                           int sample_rate,
                                           int channels) {
    if (request == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "request must be non-null");
    }
    return guard([&] {
        request->request.audio_input = make_audio_buffer(samples, frames, sample_rate, channels);
        return AUDIOCPP_OK;
    });
}

audiocpp_status audiocpp_request_set_voice_audio(audiocpp_request * request,
                                                 const float * samples,
                                                 size_t frames,
                                                 int sample_rate,
                                                 int channels) {
    if (request == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "request must be non-null");
    }
    return guard([&] {
        if (!request->request.voice.has_value()) request->request.voice = rt::VoiceCondition{};
        if (!request->request.voice->speaker.has_value()) request->request.voice->speaker = rt::VoiceReference{};
        request->request.voice->speaker->audio = make_audio_buffer(samples, frames, sample_rate, channels);
        return AUDIOCPP_OK;
    });
}

audiocpp_status audiocpp_request_set_voice_id(audiocpp_request * request, const char * cached_voice_id) {
    if (request == nullptr || cached_voice_id == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "request and cached_voice_id must be non-null");
    }
    return guard([&] {
        if (!request->request.voice.has_value()) request->request.voice = rt::VoiceCondition{};
        if (!request->request.voice->speaker.has_value()) request->request.voice->speaker = rt::VoiceReference{};
        request->request.voice->speaker->cached_voice_id = std::string(cached_voice_id);
        return AUDIOCPP_OK;
    });
}

namespace {

rt::StyleCondition & mutable_style(audiocpp_request * request) {
    if (!request->request.voice.has_value()) request->request.voice = rt::VoiceCondition{};
    if (!request->request.voice->style.has_value()) request->request.voice->style = rt::StyleCondition{};
    return *request->request.voice->style;
}

rt::ArtifactKind to_artifact_kind(audiocpp_artifact_kind kind) {
    switch (kind) {
        case AUDIOCPP_ARTIFACT_SPEAKER_EMBEDDING:    return rt::ArtifactKind::SpeakerEmbedding;
        case AUDIOCPP_ARTIFACT_STYLE_EMBEDDING:      return rt::ArtifactKind::StyleEmbedding;
        case AUDIOCPP_ARTIFACT_PROMPT_EMBEDDING:     return rt::ArtifactKind::PromptEmbedding;
        case AUDIOCPP_ARTIFACT_ACOUSTIC_TOKENS:      return rt::ArtifactKind::AcousticTokens;
        case AUDIOCPP_ARTIFACT_MIDI:                 return rt::ArtifactKind::Midi;
        case AUDIOCPP_ARTIFACT_TRANSCRIPT_ALIGNMENT: return rt::ArtifactKind::TranscriptAlignment;
        case AUDIOCPP_ARTIFACT_DIARIZATION_STATE:    return rt::ArtifactKind::DiarizationState;
        case AUDIOCPP_ARTIFACT_VAD_STATE:            return rt::ArtifactKind::VadState;
        case AUDIOCPP_ARTIFACT_CUSTOM:               return rt::ArtifactKind::Custom;
    }
    throw std::invalid_argument("unknown artifact kind");
}

audiocpp_artifact_kind from_artifact_kind(rt::ArtifactKind kind) noexcept {
    switch (kind) {
        case rt::ArtifactKind::SpeakerEmbedding:    return AUDIOCPP_ARTIFACT_SPEAKER_EMBEDDING;
        case rt::ArtifactKind::StyleEmbedding:      return AUDIOCPP_ARTIFACT_STYLE_EMBEDDING;
        case rt::ArtifactKind::PromptEmbedding:     return AUDIOCPP_ARTIFACT_PROMPT_EMBEDDING;
        case rt::ArtifactKind::AcousticTokens:      return AUDIOCPP_ARTIFACT_ACOUSTIC_TOKENS;
        case rt::ArtifactKind::Midi:                return AUDIOCPP_ARTIFACT_MIDI;
        case rt::ArtifactKind::TranscriptAlignment: return AUDIOCPP_ARTIFACT_TRANSCRIPT_ALIGNMENT;
        case rt::ArtifactKind::DiarizationState:    return AUDIOCPP_ARTIFACT_DIARIZATION_STATE;
        case rt::ArtifactKind::VadState:            return AUDIOCPP_ARTIFACT_VAD_STATE;
        case rt::ArtifactKind::Custom:              break;
    }
    return AUDIOCPP_ARTIFACT_CUSTOM;
}

/* TaskResult keeps a single artifact_output alongside a list. Callers see one
 * flat list, the single one first. */
const rt::VoiceArtifact * result_artifact_at(const audiocpp_result * result, size_t index) noexcept {
    const auto & task = result->result;
    if (task.artifact_output.has_value()) {
        if (index == 0) return &*task.artifact_output;
        index -= 1;
    }
    if (index >= task.output_artifacts.size()) return nullptr;
    return &task.output_artifacts[index];
}

}  // namespace

audiocpp_status audiocpp_request_set_style_language(audiocpp_request * request, const char * language) {
    if (request == nullptr || language == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "request and language must be non-null");
    }
    return guard([&] { mutable_style(request).language = std::string(language); return AUDIOCPP_OK; });
}

audiocpp_status audiocpp_request_set_emotion(audiocpp_request * request, const char * emotion) {
    if (request == nullptr || emotion == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "request and emotion must be non-null");
    }
    return guard([&] { mutable_style(request).emotion = std::string(emotion); return AUDIOCPP_OK; });
}

audiocpp_status audiocpp_request_set_speaking_rate(audiocpp_request * request, float speaking_rate) {
    if (request == nullptr) return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "request must be non-null");
    return guard([&] { mutable_style(request).speaking_rate = speaking_rate; return AUDIOCPP_OK; });
}

audiocpp_status audiocpp_request_set_pitch_shift(audiocpp_request * request, float pitch_shift) {
    if (request == nullptr) return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "request must be non-null");
    return guard([&] { mutable_style(request).pitch_shift = pitch_shift; return AUDIOCPP_OK; });
}

audiocpp_status audiocpp_request_set_energy_scale(audiocpp_request * request, float energy_scale) {
    if (request == nullptr) return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "request must be non-null");
    return guard([&] { mutable_style(request).energy_scale = energy_scale; return AUDIOCPP_OK; });
}

audiocpp_status audiocpp_request_set_style_tag(audiocpp_request * request, const char * key, const char * value) {
    if (request == nullptr || key == nullptr || value == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "request, key and value must be non-null");
    }
    return guard([&] { mutable_style(request).tags[key] = value; return AUDIOCPP_OK; });
}

audiocpp_status audiocpp_request_add_artifact(audiocpp_request * request,
                                              audiocpp_artifact_kind kind,
                                              const char * id,
                                              const void * payload,
                                              size_t payload_bytes,
                                              size_t * out_index) {
    if (request == nullptr || id == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "request and id must be non-null");
    }
    if (payload_bytes > 0 && payload == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "payload is null but payload_bytes > 0");
    }
    return guard([&] {
        rt::VoiceArtifact artifact;
        artifact.kind = to_artifact_kind(kind);
        artifact.id = id;
        if (payload_bytes > 0) {
            const auto * bytes = static_cast<const std::byte *>(payload);
            artifact.payload.assign(bytes, bytes + payload_bytes);
        }
        request->request.input_artifacts.push_back(std::move(artifact));
        if (out_index != nullptr) *out_index = request->request.input_artifacts.size() - 1;
        return AUDIOCPP_OK;
    });
}

audiocpp_status audiocpp_request_set_artifact_meta(audiocpp_request * request,
                                                   size_t index,
                                                   const char * key,
                                                   const char * value) {
    if (request == nullptr || key == nullptr || value == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "request, key and value must be non-null");
    }
    if (index >= request->request.input_artifacts.size()) {
        return fail(AUDIOCPP_ERR_OUT_OF_RANGE, "artifact index out of range");
    }
    return guard([&] {
        request->request.input_artifacts[index].meta[key] = value;
        return AUDIOCPP_OK;
    });
}

audiocpp_status audiocpp_request_set_option(audiocpp_request * request, const char * key, const char * value) {
    if (request == nullptr || key == nullptr || value == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "request, key and value must be non-null");
    }
    return guard([&] {
        request->request.options[key] = value;
        return AUDIOCPP_OK;
    });
}

/* ------------------------------------------------------------------ */
/* Result                                                              */
/* ------------------------------------------------------------------ */

void audiocpp_result_free(audiocpp_result * result) {
    if (result != nullptr && !result->owned) return;
    delete result;
}

audiocpp_status audiocpp_result_audio(const audiocpp_result * result,
                                      const float ** out_samples,
                                      size_t * out_frames,
                                      int * out_sample_rate,
                                      int * out_channels) {
    if (result == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "result must be non-null");
    }
    if (!result->result.audio_output.has_value()) {
        return fail(AUDIOCPP_ERR_NOT_AVAILABLE, "result carries no audio output");
    }
    const auto & audio = *result->result.audio_output;
    if (out_samples != nullptr) *out_samples = audio.samples.data();
    if (out_frames != nullptr) *out_frames = frame_count(audio);
    if (out_sample_rate != nullptr) *out_sample_rate = audio.sample_rate;
    if (out_channels != nullptr) *out_channels = audio.channels;
    return AUDIOCPP_OK;
}

audiocpp_status audiocpp_result_text(const audiocpp_result * result,
                                     const char ** out_text,
                                     const char ** out_language) {
    if (result == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "result must be non-null");
    }
    if (!result->result.text_output.has_value()) {
        return fail(AUDIOCPP_ERR_NOT_AVAILABLE, "result carries no text output");
    }
    assign_out(out_text, result->result.text_output->text);
    assign_out(out_language, result->result.text_output->language);
    return AUDIOCPP_OK;
}

size_t audiocpp_result_segment_count(const audiocpp_result * result) {
    return result != nullptr ? result->result.speech_segments.size() : 0;
}

audiocpp_status audiocpp_result_segment(const audiocpp_result * result,
                                        size_t index,
                                        int64_t * out_start_sample,
                                        int64_t * out_end_sample,
                                        float * out_confidence,
                                        const char ** out_text) {
    if (result == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "result must be non-null");
    }
    if (index >= result->result.speech_segments.size()) {
        return fail(AUDIOCPP_ERR_OUT_OF_RANGE, "segment index out of range");
    }
    const auto & segment = result->result.speech_segments[index];
    if (out_start_sample != nullptr) *out_start_sample = segment.span.start_sample;
    if (out_end_sample != nullptr) *out_end_sample = segment.span.end_sample;
    if (out_confidence != nullptr) *out_confidence = segment.confidence;
    assign_out(out_text, segment.text);
    return AUDIOCPP_OK;
}

size_t audiocpp_result_speaker_turn_count(const audiocpp_result * result) {
    return result != nullptr ? result->result.speaker_turns.size() : 0;
}

audiocpp_status audiocpp_result_speaker_turn(const audiocpp_result * result,
                                             size_t index,
                                             int64_t * out_start_sample,
                                             int64_t * out_end_sample,
                                             const char ** out_speaker_id,
                                             float * out_confidence,
                                             const char ** out_text) {
    if (result == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "result must be non-null");
    }
    if (index >= result->result.speaker_turns.size()) {
        return fail(AUDIOCPP_ERR_OUT_OF_RANGE, "speaker turn index out of range");
    }
    const auto & turn = result->result.speaker_turns[index];
    if (out_start_sample != nullptr) *out_start_sample = turn.span.start_sample;
    if (out_end_sample != nullptr) *out_end_sample = turn.span.end_sample;
    assign_out(out_speaker_id, turn.speaker_id);
    if (out_confidence != nullptr) *out_confidence = turn.confidence;
    assign_out(out_text, turn.text);
    return AUDIOCPP_OK;
}

size_t audiocpp_result_word_count(const audiocpp_result * result) {
    return result != nullptr ? result->result.word_timestamps.size() : 0;
}

audiocpp_status audiocpp_result_word(const audiocpp_result * result,
                                     size_t index,
                                     const char ** out_word,
                                     int64_t * out_start_sample,
                                     int64_t * out_end_sample,
                                     float * out_confidence) {
    if (result == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "result must be non-null");
    }
    if (index >= result->result.word_timestamps.size()) {
        return fail(AUDIOCPP_ERR_OUT_OF_RANGE, "word index out of range");
    }
    const auto & word = result->result.word_timestamps[index];
    assign_out(out_word, word.word);
    if (out_start_sample != nullptr) *out_start_sample = word.span.start_sample;
    if (out_end_sample != nullptr) *out_end_sample = word.span.end_sample;
    if (out_confidence != nullptr) *out_confidence = word.confidence;
    return AUDIOCPP_OK;
}

size_t audiocpp_result_named_audio_count(const audiocpp_result * result) {
    return result != nullptr ? result->result.named_audio_outputs.size() : 0;
}

audiocpp_status audiocpp_result_named_audio(const audiocpp_result * result,
                                            size_t index,
                                            const char ** out_id,
                                            const float ** out_samples,
                                            size_t * out_frames,
                                            int * out_sample_rate,
                                            int * out_channels) {
    if (result == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "result must be non-null");
    }
    if (index >= result->result.named_audio_outputs.size()) {
        return fail(AUDIOCPP_ERR_OUT_OF_RANGE, "named audio index out of range");
    }
    const auto & named = result->result.named_audio_outputs[index];
    assign_out(out_id, named.id);
    if (out_samples != nullptr) *out_samples = named.audio.samples.data();
    if (out_frames != nullptr) *out_frames = frame_count(named.audio);
    if (out_sample_rate != nullptr) *out_sample_rate = named.audio.sample_rate;
    if (out_channels != nullptr) *out_channels = named.audio.channels;
    return AUDIOCPP_OK;
}

size_t audiocpp_result_artifact_count(const audiocpp_result * result) {
    if (result == nullptr) return 0;
    return result->result.output_artifacts.size() +
           (result->result.artifact_output.has_value() ? 1u : 0u);
}

audiocpp_status audiocpp_result_artifact(const audiocpp_result * result,
                                         size_t index,
                                         audiocpp_artifact_kind * out_kind,
                                         const char ** out_id,
                                         const void ** out_payload,
                                         size_t * out_payload_bytes) {
    if (result == nullptr) return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "result must be non-null");
    const auto * artifact = result_artifact_at(result, index);
    if (artifact == nullptr) return fail(AUDIOCPP_ERR_OUT_OF_RANGE, "artifact index out of range");
    if (out_kind != nullptr) *out_kind = from_artifact_kind(artifact->kind);
    assign_out(out_id, artifact->id);
    if (out_payload != nullptr) *out_payload = artifact->payload.data();
    if (out_payload_bytes != nullptr) *out_payload_bytes = artifact->payload.size();
    return AUDIOCPP_OK;
}

size_t audiocpp_result_artifact_meta_count(const audiocpp_result * result, size_t index) {
    if (result == nullptr) return 0;
    const auto * artifact = result_artifact_at(result, index);
    return artifact != nullptr ? artifact->meta.size() : 0;
}

audiocpp_status audiocpp_result_artifact_meta(const audiocpp_result * result,
                                              size_t index,
                                              size_t meta_index,
                                              const char ** out_key,
                                              const char ** out_value) {
    if (result == nullptr) return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "result must be non-null");
    const auto * artifact = result_artifact_at(result, index);
    if (artifact == nullptr) return fail(AUDIOCPP_ERR_OUT_OF_RANGE, "artifact index out of range");
    if (meta_index >= artifact->meta.size()) {
        return fail(AUDIOCPP_ERR_OUT_OF_RANGE, "artifact meta index out of range");
    }
    /* unordered_map has no index; step to the requested position. Meta maps are
     * a handful of entries, so this stays cheap. */
    auto it = artifact->meta.begin();
    std::advance(it, static_cast<std::ptrdiff_t>(meta_index));
    assign_out(out_key, it->first);
    assign_out(out_value, it->second);
    return AUDIOCPP_OK;
}

/* ------------------------------------------------------------------ */
/* Streaming                                                           */
/* ------------------------------------------------------------------ */


namespace {

/* Mode is checked before the call rather than inferring it from a thrown
 * exception, so a real failure inside a family is reported as a runtime error
 * instead of being flattened into "not available". */
audiocpp_status require_streaming(const audiocpp_session * session) noexcept {
    if (session->mode != rt::RunMode::Streaming) {
        return fail(AUDIOCPP_ERR_NOT_AVAILABLE, "session was not created in streaming mode");
    }
    return AUDIOCPP_OK;
}

}  // namespace

audiocpp_status audiocpp_stream_policy(const audiocpp_session * session,
                                       audiocpp_stream_input_kind * out_input,
                                       audiocpp_stream_output_kind * out_output,
                                       int64_t * out_preferred_chunk_samples,
                                       double * out_preferred_chunk_seconds) {
    if (session == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "session must be non-null");
    }
    { const audiocpp_status s = require_streaming(session); if (s != AUDIOCPP_OK) return s; }
    return guard([&] {
        const auto policy = session->streaming().streaming_policy();
        if (out_input != nullptr) {
            *out_input = policy.input == rt::StreamingInputKind::AudioChunks
                ? AUDIOCPP_STREAM_INPUT_AUDIO_CHUNKS
                : AUDIOCPP_STREAM_INPUT_NONE;
        }
        if (out_output != nullptr) {
            *out_output = policy.output == rt::StreamingOutputKind::PullEvents
                ? AUDIOCPP_STREAM_OUTPUT_PULL_EVENTS
                : AUDIOCPP_STREAM_OUTPUT_FINAL_RESULT;
        }
        if (out_preferred_chunk_samples != nullptr) {
            *out_preferred_chunk_samples = policy.preferred_audio_chunk_samples;
        }
        if (out_preferred_chunk_seconds != nullptr) {
            *out_preferred_chunk_seconds = policy.preferred_audio_chunk_seconds;
        }
        return AUDIOCPP_OK;
    });
}

audiocpp_status audiocpp_stream_start(audiocpp_session * session, const audiocpp_request * request) {
    if (session == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "session must be non-null");
    }
    { const audiocpp_status s = require_streaming(session); if (s != AUDIOCPP_OK) return s; }
    return guard([&] {
        const rt::TaskRequest empty;
        const auto & task_request = request != nullptr ? request->request : empty;
        session->session->prepare(rt::build_preparation_request(task_request));
        session->streaming().start_stream(task_request);
        return AUDIOCPP_OK;
    });
}

audiocpp_status audiocpp_stream_push(audiocpp_session * session,
                                     const float * samples,
                                     size_t frames,
                                     int sample_rate,
                                     int channels,
                                     int64_t start_sample,
                                     audiocpp_event ** out_event) {
    if (session == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "session must be non-null");
    }
    if (out_event != nullptr) *out_event = nullptr;
    { const audiocpp_status s = require_streaming(session); if (s != AUDIOCPP_OK) return s; }
    return guard([&] {
        auto buffer = make_audio_buffer(samples, frames, sample_rate, channels);
        rt::AudioChunk chunk;
        chunk.sample_rate = buffer.sample_rate;
        chunk.channels = buffer.channels;
        chunk.start_sample = start_sample;
        chunk.samples = std::move(buffer.samples);

        auto event = session->streaming().process_audio_chunk(chunk);
        if (out_event != nullptr) {
            *out_event = wrap_event(std::move(event)).release();
        }
        return AUDIOCPP_OK;
    });
}

audiocpp_status audiocpp_stream_next_event(audiocpp_session * session, audiocpp_event ** out_event) {
    if (session == nullptr || out_event == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "session and out_event must be non-null");
    }
    *out_event = nullptr;
    { const audiocpp_status s = require_streaming(session); if (s != AUDIOCPP_OK) return s; }
    return guard([&] {
        auto event = session->streaming().next_stream_event();
        /* An empty queue is a normal outcome, not a failure. */
        if (event.has_value()) {
            *out_event = wrap_event(std::move(*event)).release();
        }
        return AUDIOCPP_OK;
    });
}

audiocpp_status audiocpp_stream_finish(audiocpp_session * session, audiocpp_result ** out_result) {
    if (session == nullptr || out_result == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "session and out_result must be non-null");
    }
    *out_result = nullptr;
    { const audiocpp_status s = require_streaming(session); if (s != AUDIOCPP_OK) return s; }
    return guard([&] {
        auto handle = std::make_unique<audiocpp_result>();
        handle->result = session->streaming().finish_stream();
        *out_result = handle.release();
        return AUDIOCPP_OK;
    });
}

audiocpp_status audiocpp_stream_reset(audiocpp_session * session) {
    if (session == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "session must be non-null");
    }
    { const audiocpp_status s = require_streaming(session); if (s != AUDIOCPP_OK) return s; }
    return guard([&] {
        session->streaming().reset();
        return AUDIOCPP_OK;
    });
}

void audiocpp_event_free(audiocpp_event * event) {
    delete event;
}

int audiocpp_event_is_final(const audiocpp_event * event) {
    return event != nullptr && event->is_final ? 1 : 0;
}

const audiocpp_result * audiocpp_event_as_result(const audiocpp_event * event) {
    return event != nullptr ? &event->view : nullptr;
}

size_t audiocpp_event_voice_activity_count(const audiocpp_event * event) {
    return event != nullptr ? event->voice_activity.size() : 0;
}

audiocpp_status audiocpp_event_voice_activity(const audiocpp_event * event,
                                              size_t index,
                                              audiocpp_voice_activity_kind * out_kind,
                                              int64_t * out_sample,
                                              float * out_probability) {
    if (event == nullptr) {
        return fail(AUDIOCPP_ERR_INVALID_ARGUMENT, "event must be non-null");
    }
    if (index >= event->voice_activity.size()) {
        return fail(AUDIOCPP_ERR_OUT_OF_RANGE, "voice activity index out of range");
    }
    const auto & activity = event->voice_activity[index];
    if (out_kind != nullptr) {
        switch (activity.kind) {
            case rt::VoiceActivityEvent::Kind::SpeechStart:
                *out_kind = AUDIOCPP_VOICE_ACTIVITY_SPEECH_START;
                break;
            case rt::VoiceActivityEvent::Kind::SpeechEnd:
                *out_kind = AUDIOCPP_VOICE_ACTIVITY_SPEECH_END;
                break;
            case rt::VoiceActivityEvent::Kind::SpeechSegment:
                *out_kind = AUDIOCPP_VOICE_ACTIVITY_SPEECH_SEGMENT;
                break;
        }
    }
    if (out_sample != nullptr) *out_sample = activity.sample;
    if (out_probability != nullptr) *out_probability = activity.probability;
    return AUDIOCPP_OK;
}
