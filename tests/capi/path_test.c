/*
 * C ABI path test.
 *
 * Deliberately compiled as C: the point is that audiocpp.h is consumable by a
 * C compiler, with no C++ runtime and no framework headers on the caller side.
 *
 * Runs Silero VAD over assets/resources/sample_16k.wav. Both ship in-repo, so
 * the test needs no model download and no network, and runs on CPU in well
 * under a second.
 *
 * It asserts the ABI contract: status codes, borrowed-string rules,
 * out-of-range handling, out-of-order frees, and that a session reused across
 * requests returns identical results. It asserts that speech is found, because
 * the input really is speech, but nothing about how much -- that would pin the
 * test to Silero's thresholds rather than to the ABI.
 */

#include "audiocpp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;

#define CHECK(cond, ...)                                         \
    do {                                                         \
        if (!(cond)) {                                           \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);  \
            fprintf(stderr, __VA_ARGS__);                        \
            fprintf(stderr, "\n");                               \
            g_failures++;                                        \
        }                                                        \
    } while (0)

#define CHECK_OK(expr)                                                    \
    do {                                                                  \
        audiocpp_status _s = (expr);                                      \
        CHECK(_s == AUDIOCPP_OK, "%s -> %s (%s)", #expr,                  \
              audiocpp_status_string(_s), audiocpp_last_error());         \
    } while (0)

#define MAX_SEGMENTS 256

typedef struct {
    float * samples;
    size_t  frames;
    int     sample_rate;
    int     channels;
} audio_clip;

static uint32_t read_u32(const unsigned char * p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint16_t read_u16(const unsigned char * p) {
    return (uint16_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8));
}

/* Minimal 16-bit PCM WAV reader. Only enough to walk the chunk list of the
 * in-repo fixture; the C API itself takes float PCM and has no opinion on
 * containers. */
static int load_wav(const char * path, audio_clip * clip) {
    unsigned char header[12];
    unsigned char chunk[8];
    unsigned char * data = NULL;
    uint32_t data_size = 0;
    int channels = 0, sample_rate = 0, bits = 0;
    size_t i;
    FILE * file = fopen(path, "rb");

    if (file == NULL) {
        fprintf(stderr, "cannot open %s\n", path);
        return 0;
    }
    if (fread(header, 1, sizeof(header), file) != sizeof(header) ||
        memcmp(header, "RIFF", 4) != 0 || memcmp(header + 8, "WAVE", 4) != 0) {
        fprintf(stderr, "%s is not a RIFF/WAVE file\n", path);
        fclose(file);
        return 0;
    }

    while (fread(chunk, 1, sizeof(chunk), file) == sizeof(chunk)) {
        const uint32_t size = read_u32(chunk + 4);
        if (memcmp(chunk, "fmt ", 4) == 0) {
            unsigned char fmt[16];
            if (size < sizeof(fmt) || fread(fmt, 1, sizeof(fmt), file) != sizeof(fmt)) break;
            channels = read_u16(fmt + 2);
            sample_rate = (int)read_u32(fmt + 4);
            bits = read_u16(fmt + 14);
            if (size > sizeof(fmt)) fseek(file, (long)(size - sizeof(fmt)), SEEK_CUR);
        } else if (memcmp(chunk, "data", 4) == 0) {
            data = (unsigned char *)malloc(size);
            if (data == NULL || fread(data, 1, size, file) != size) {
                free(data);
                data = NULL;
                break;
            }
            data_size = size;
            break;
        } else {
            fseek(file, (long)(size + (size & 1u)), SEEK_CUR);
        }
    }
    fclose(file);

    if (data == NULL || bits != 16 || channels < 1 || sample_rate <= 0) {
        fprintf(stderr, "%s: unsupported or truncated (bits=%d ch=%d rate=%d)\n",
                path, bits, channels, sample_rate);
        free(data);
        return 0;
    }

    clip->frames = data_size / (size_t)(2 * channels);
    clip->channels = channels;
    clip->sample_rate = sample_rate;
    clip->samples = (float *)malloc(sizeof(float) * clip->frames * (size_t)channels);
    if (clip->samples == NULL) {
        free(data);
        return 0;
    }
    for (i = 0; i < clip->frames * (size_t)channels; ++i) {
        const int16_t pcm = (int16_t)read_u16(data + (i * 2));
        clip->samples[i] = (float)pcm / 32768.0f;
    }
    free(data);
    return 1;
}

typedef struct {
    size_t  count;
    int64_t start[MAX_SEGMENTS];
    int64_t end[MAX_SEGMENTS];
} segment_set;

static void run_once(audiocpp_session * session, const audio_clip * clip, segment_set * out) {
    audiocpp_request * request = audiocpp_request_create();
    audiocpp_result * result = NULL;
    size_t i;

    out->count = 0;
    CHECK(request != NULL, "audiocpp_request_create returned NULL");
    if (request == NULL) return;

    CHECK_OK(audiocpp_request_set_audio(request, clip->samples, clip->frames,
                                        clip->sample_rate, clip->channels));
    CHECK_OK(audiocpp_session_run(session, request, &result));
    CHECK(result != NULL, "run produced no result handle");

    if (result != NULL) {
        const size_t count = audiocpp_result_segment_count(result);
        int64_t previous_end = -1;
        for (i = 0; i < count && i < MAX_SEGMENTS; ++i) {
            int64_t start = -1, end = -1;
            float confidence = -1.0f;
            const char * text = NULL;
            CHECK_OK(audiocpp_result_segment(result, i, &start, &end, &confidence, &text));
            CHECK(start >= 0 && end >= start, "segment %zu span is malformed [%lld,%lld]",
                  i, (long long)start, (long long)end);
            CHECK(end <= (int64_t)clip->frames,
                  "segment %zu ends at %lld, past the %zu-frame input",
                  i, (long long)end, clip->frames);
            CHECK(start >= previous_end, "segment %zu starts at %lld, before the previous end %lld",
                  i, (long long)start, (long long)previous_end);
            CHECK(text != NULL, "segment %zu text is NULL; absent should read as \"\"", i);
            previous_end = end;
            out->start[i] = start;
            out->end[i] = end;
        }
        out->count = count;
        /* Out-of-range access must be reported, not crash. */
        CHECK(audiocpp_result_segment(result, count, NULL, NULL, NULL, NULL)
              == AUDIOCPP_ERR_OUT_OF_RANGE, "out-of-range segment was not rejected");
        /* A VAD result carries no audio or text; NOT_AVAILABLE, not a crash. */
        CHECK(audiocpp_result_audio(result, NULL, NULL, NULL, NULL)
              == AUDIOCPP_ERR_NOT_AVAILABLE, "vad result claimed to have audio");
    }

    audiocpp_result_free(result);
    audiocpp_request_free(request);
}

static int same_segments(const segment_set * a, const segment_set * b) {
    size_t i;
    if (a->count != b->count) return 0;
    for (i = 0; i < a->count && i < MAX_SEGMENTS; ++i) {
        if (a->start[i] != b->start[i] || a->end[i] != b->end[i]) return 0;
    }
    return 1;
}

int main(int argc, char ** argv) {
    const char * model_path = (argc > 1) ? argv[1] : "assets/framework/models/silero_vad";
    const char * audio_path = (argc > 2) ? argv[2] : "assets/resources/sample_16k.wav";
    /* Defaults to cpu so the registered test is deterministic and needs no GPU;
     * pass "cuda"/"vulkan"/"metal" to exercise another backend through the ABI. */
    const char * backend_name = (argc > 3) ? argv[3] : "cpu";
    audiocpp_registry * registry = NULL;
    audiocpp_model * model = NULL;
    audiocpp_session * session = NULL;
    audio_clip clip;
    segment_set first, second, after_free;
    size_t families, options, i;
    int found_family = 0;

    memset(&clip, 0, sizeof(clip));

    printf("abi=%u build=%s backend=%s\n",
           audiocpp_abi_version(), audiocpp_build_version(), backend_name);
    CHECK(audiocpp_abi_version() == (((uint32_t)AUDIOCPP_ABI_VERSION_MAJOR << 16) |
                                     ((uint32_t)AUDIOCPP_ABI_VERSION_MINOR << 8) |
                                     (uint32_t)AUDIOCPP_ABI_VERSION_PATCH),
          "abi version mismatch between header and library");

    /* Freeing NULL is documented as a no-op; cleanup paths rely on it. */
    audiocpp_registry_free(NULL);
    audiocpp_model_free(NULL);
    audiocpp_session_free(NULL);
    audiocpp_result_free(NULL);
    audiocpp_request_free(NULL);
    audiocpp_options_free(NULL);
    audiocpp_event_free(NULL);

    if (!load_wav(audio_path, &clip)) return 1;
    printf("audio=%s frames=%zu rate=%d ch=%d\n",
           audio_path, clip.frames, clip.sample_rate, clip.channels);

    CHECK_OK(audiocpp_registry_create(NULL, &registry));
    if (registry == NULL) { fprintf(stderr, "no registry; aborting\n"); return 1; }

    families = audiocpp_registry_family_count(registry);
    CHECK(families > 0, "registry advertises no families");
    for (i = 0; i < families; ++i) {
        const char * family = NULL;
        CHECK_OK(audiocpp_registry_family(registry, i, &family));
        if (family != NULL && strcmp(family, "silero_vad") == 0) found_family = 1;
    }
    CHECK(found_family, "silero_vad is not in the registry");
    {
        const char * overflow = NULL;
        CHECK(audiocpp_registry_family(registry, families, &overflow) == AUDIOCPP_ERR_OUT_OF_RANGE,
              "out-of-range family was not rejected");
        CHECK(audiocpp_registry_family(registry, 0, NULL) == AUDIOCPP_ERR_INVALID_ARGUMENT,
              "NULL out parameter was not rejected");
    }

    /* An unknown family must be reported as such, not as a generic failure,
     * and must leave a message behind. */
    {
        audiocpp_model * bogus = NULL;
        audiocpp_model_config bogus_config;
        audiocpp_status status;
        memset(&bogus_config, 0, sizeof(bogus_config));
        bogus_config.family_hint = "no_such_family";
        status =
            audiocpp_model_load(registry, model_path, &bogus_config, NULL, &bogus);
        CHECK(status == AUDIOCPP_ERR_UNSUPPORTED_FAMILY,
              "unknown family gave %s, expected unsupported family",
              audiocpp_status_string(status));
        CHECK(strlen(audiocpp_last_error()) > 0, "failing call left no error detail");
        CHECK(bogus == NULL, "failed load still wrote an out handle");
    }

    {
        audiocpp_model_config config;
        memset(&config, 0, sizeof(config));
        config.family_hint = "silero_vad";
        CHECK_OK(audiocpp_model_load(registry, model_path, &config, NULL, &model));
    }
    if (model == NULL) { fprintf(stderr, "no model; aborting\n"); return 1; }

    printf("family=%s\n", audiocpp_model_family(model));
    CHECK(strcmp(audiocpp_model_family(model), "silero_vad") == 0,
          "unexpected family '%s'", audiocpp_model_family(model));
    CHECK(audiocpp_model_description(model) != NULL, "description is NULL");
    CHECK(audiocpp_model_supports(model, "vad", "offline"), "model does not advertise vad/offline");
    CHECK(!audiocpp_model_supports(model, "tts", "offline"), "vad model claims to do tts");
    CHECK(!audiocpp_model_supports(model, "nonsense", "offline"),
          "an unrecognised task should read as unsupported, not fail");

    /* Runtime introspection: a binding must be able to discover what a family
     * accepts without knowing anything about the family. Silero declares no
     * options, so what is asserted here is that the accessor is well behaved
     * at a count of zero. */
    for (i = 0; i < 3; ++i) {
        const audiocpp_option_scope scope = (audiocpp_option_scope)i;
        size_t j;
        options = audiocpp_model_option_count(model, scope);
        printf("options[scope=%zu]=%zu\n", i, options);
        for (j = 0; j < options; ++j) {
            const char * name = NULL;
            const char * value_name = NULL;
            const char * description = NULL;
            const char * default_value = NULL;
            int required = -1;
            const char * min_value = NULL;
            const char * max_value = NULL;
            CHECK_OK(audiocpp_model_option(model, scope, j, &name, &value_name,
                                           &description, &default_value,
                                           &min_value, &max_value, &required));
            CHECK(min_value != NULL && max_value != NULL,
                  "option %zu bounds are NULL; absent should read as \"\"", j);
            CHECK(name != NULL && strlen(name) > 0, "option %zu has no name", j);
            CHECK(value_name != NULL && description != NULL && default_value != NULL,
                  "option %zu returned a NULL string; absent should read as \"\"", j);
            CHECK(required == 0 || required == 1, "option %zu required flag is %d", j, required);
            printf("  %s (%s) default='%s' required=%d\n", name, value_name, default_value, required);
        }
        CHECK(audiocpp_model_option(model, scope, options, NULL, NULL, NULL, NULL, NULL, NULL, NULL)
              == AUDIOCPP_ERR_OUT_OF_RANGE, "out-of-range option was not rejected");
    }

    {
        audiocpp_backend_config backend;
        backend.backend = backend_name;
        backend.device = 0;
        backend.threads = 2;
        CHECK_OK(audiocpp_session_create(model, "vad", "offline", &backend, NULL, &session));
    }
    if (session == NULL) { fprintf(stderr, "no session; aborting\n"); return 1; }
    CHECK(strlen(audiocpp_session_family(session)) > 0, "session reports no family");

    run_once(session, &clip, &first);
    run_once(session, &clip, &second);
    printf("segments=%zu\n", first.count);
    CHECK(first.count > 0, "no speech found in %s, which is a speech recording", audio_path);
    CHECK(first.count <= MAX_SEGMENTS, "more segments than this test can record");
    /* Session reuse is the whole reason to embed rather than shell out, and it
     * is the one thing the CLI path can never exercise. */
    CHECK(same_segments(&first, &second), "reusing the session changed the result");

    /* An explicit prepare must not disturb a later, differently shaped run.
     * Preparation carries the input length, which is why run() re-prepares
     * rather than trusting an earlier prepare. Note this asserts the invariant
     * rather than detecting the hazard: Silero windows its input at a fixed 512
     * frames and is indifferent to the declared length, as is Parakeet, so
     * neither discriminates between the two implementations. */
    {
        audiocpp_request * warm = audiocpp_request_create();
        segment_set after_warm;
        CHECK(warm != NULL, "request alloc");
        if (warm != NULL) {
            CHECK_OK(audiocpp_request_set_audio(warm, clip.samples, clip.frames / 8,
                                                clip.sample_rate, clip.channels));
            CHECK_OK(audiocpp_session_prepare(session, warm));
            audiocpp_request_free(warm);
        }
        run_once(session, &clip, &after_warm);
        CHECK(same_segments(&first, &after_warm),
              "a prepare sized for a shorter request changed the full run's result");
    }

    /* An offline session must refuse streaming calls rather than misbehave. */
    CHECK(audiocpp_stream_push(session, clip.samples, 512, clip.sample_rate, clip.channels, 0, NULL)
          == AUDIOCPP_ERR_NOT_AVAILABLE, "offline session accepted a stream push");

    /* Streaming, over the same audio. Silero advertises vad/streaming, so this
     * needs no extra model. The streaming and offline algorithms differ, so the
     * segments are not required to match the offline run -- what is asserted is
     * that the pull-based surface behaves. */
    if (audiocpp_model_supports(model, "vad", "streaming")) {
        audiocpp_session * stream = NULL;
        audiocpp_backend_config backend;
        backend.backend = backend_name;
        backend.device = 0;
        backend.threads = 2;
        CHECK_OK(audiocpp_session_create(model, "vad", "streaming", &backend, NULL, &stream));
        if (stream != NULL) {
            audiocpp_stream_input_kind input = AUDIOCPP_STREAM_INPUT_NONE;
            int64_t chunk_frames = 0;
            size_t offset = 0;
            size_t activity_events = 0;
            size_t stream_segments = 0;
            audiocpp_result * stream_result = NULL;

            CHECK_OK(audiocpp_stream_policy(stream, &input, NULL, &chunk_frames, NULL));
            CHECK(input == AUDIOCPP_STREAM_INPUT_AUDIO_CHUNKS, "vad stream wants no audio chunks");
            CHECK(chunk_frames > 0, "stream policy gave a chunk size of %lld",
                  (long long)chunk_frames);
            if (chunk_frames <= 0) chunk_frames = 512;

            CHECK_OK(audiocpp_stream_start(stream, NULL));
            while (offset + (size_t)chunk_frames <= clip.frames) {
                audiocpp_event * event = NULL;
                CHECK_OK(audiocpp_stream_push(stream, clip.samples + offset,
                                              (size_t)chunk_frames, clip.sample_rate,
                                              clip.channels, (int64_t)offset, &event));
                if (event != NULL) {
                    const size_t activity = audiocpp_event_voice_activity_count(event);
                    size_t k;
                    for (k = 0; k < activity; ++k) {
                        audiocpp_voice_activity_kind kind = (audiocpp_voice_activity_kind)-1;
                        int64_t sample = -1;
                        float probability = -1.0f;
                        CHECK_OK(audiocpp_event_voice_activity(event, k, &kind, &sample, &probability));
                        CHECK(kind == AUDIOCPP_VOICE_ACTIVITY_SPEECH_START ||
                              kind == AUDIOCPP_VOICE_ACTIVITY_SPEECH_END ||
                              kind == AUDIOCPP_VOICE_ACTIVITY_SPEECH_SEGMENT,
                              "unrecognised voice activity kind %d", (int)kind);
                        CHECK(probability >= 0.0f && probability <= 1.0f,
                              "probability %f is outside [0,1]", (double)probability);
                    }
                    activity_events += activity;
                    CHECK(audiocpp_event_voice_activity(event, activity, NULL, NULL, NULL)
                          == AUDIOCPP_ERR_OUT_OF_RANGE, "out-of-range activity was not rejected");
                    /* An event reads through the result accessors. The view is
                     * owned by the event, and has the same type as an owned
                     * result, so freeing it must be a no-op rather than a
                     * double free -- the event stays usable afterwards. */
                    {
                        const audiocpp_result * view = audiocpp_event_as_result(event);
                        CHECK(view != NULL, "event exposed no result view");
                        if (view != NULL) {
                            const size_t before = audiocpp_result_segment_count(view);
                            audiocpp_result_free((audiocpp_result *)view);
                            CHECK(audiocpp_result_segment_count(view) == before,
                                  "freeing a borrowed event view damaged it");
                        }
                    }
                    audiocpp_event_free(event);
                }
                offset += (size_t)chunk_frames;
            }
            CHECK_OK(audiocpp_stream_finish(stream, &stream_result));
            if (stream_result != NULL) {
                stream_segments = audiocpp_result_segment_count(stream_result);
                audiocpp_result_free(stream_result);
            }
            printf("stream: chunk=%lld events=%zu segments=%zu\n",
                   (long long)chunk_frames, activity_events, stream_segments);
            CHECK(activity_events > 0 || stream_segments > 0,
                  "streaming produced neither events nor segments over speech audio");

            /* A streaming session must refuse the offline entry point. */
            CHECK(audiocpp_session_run(stream, NULL, &stream_result) == AUDIOCPP_ERR_NOT_AVAILABLE,
                  "streaming session accepted an offline run");
            CHECK_OK(audiocpp_stream_reset(stream));
            audiocpp_session_free(stream);
        }
    }

    /* Handles hold their parents alive, so freeing out of order is safe.
     * Without that, this would dangle. */
    audiocpp_model_free(model);
    audiocpp_registry_free(registry);
    run_once(session, &clip, &after_free);
    CHECK(same_segments(&first, &after_free),
          "session stopped agreeing with itself after its model and registry were freed");
    audiocpp_session_free(session);

    free(clip.samples);

    if (g_failures > 0) {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("c api path test OK\n");
    return 0;
}
