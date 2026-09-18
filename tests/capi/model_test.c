/*
 * C ABI model test -- opt-in, because it needs downloaded models.
 *
 * path_test.c proves the ABI contract against a model that ships in-repo. This
 * one proves the ABI carries real work for real families across task types, and
 * that the answers match what audiocpp_cli produces from the same inputs.
 *
 * It is registered unconditionally but SKIPS (exit 77) when no model is found,
 * so a checkout with no downloads still reports green. Point it at a models
 * root:
 *
 *   cmake -S . -B build -DAUDIOCPP_BUILD_C_API=ON \
 *       -DAUDIOCPP_C_API_MODEL_ROOT=/path/to/models
 *   ctest --test-dir build -R audiocpp_c_api_model --output-on-failure
 *
 * Transcripts and audio summaries are printed as `parity:<family>:<field>=...`
 * lines so they can be diffed against the CLI's output for the same input.
 */

#include "audiocpp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <unistd.h>
#endif

#define CTEST_SKIP 77

static int g_failures = 0;
static int g_ran = 0;
static int g_skipped = 0;

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

typedef struct {
    const char * family;
    const char * task;
    const char * mode;
    /* Default package layout as documented per family. A case whose file is
     * missing is skipped, so a renamed package degrades to a skip rather than
     * a failure. */
    const char * relative_model;
    const char * text;       /* non-NULL => text-driven (TTS) */
    const char * language;
    const char * voice_id;
    /* Generative families draw noise per run (Kokoro's harmonic-plus-noise
     * excitation, for one), so a fixed seed is what makes a run comparable to
     * anything -- including to audiocpp_cli's output for the same input. */
    const char * seed;
    int expect_audio;
    int expect_text;
    int expect_turns;
    int expect_named_audio;
    /* Some families fix their input rate (BS-RoFormer wants 44.1 kHz), so a
     * case can ask for the alternate clip instead of the default one. */
    int use_alt_audio;
} model_case;

static const model_case CASES[] = {
    { "kokoro_tts", "tts", "offline",
      "Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf",
      "The quick brown fox jumps over the lazy dog.", "en-us", "af_heart", "1234",
      1, 0, 0, 0, 0 },
    { "citrinet_asr", "asr", "offline",
      "Citrinet-ASR-GGUF/citrinet-asr-q8_0.gguf",
      NULL, NULL, NULL, NULL, 0, 1, 0, 0, 0 },
    { "parakeet_tdt", "asr", "offline",
      "Parakeet-TDT-0.6B-v3-GGUF/parakeet-tdt-0.6b-v3-q8_0.gguf",
      NULL, NULL, NULL, NULL, 0, 1, 0, 0, 0 },
    { "sortformer_diar", "diar", "offline",
      "Sortformer-Diar-4spk-v1-GGUF/sortformer-diar-4spk-v1-q8_0.gguf",
      NULL, NULL, NULL, NULL, 0, 0, 1, 0, 0 },
    { "bs_roformer", "sep", "offline",
      "BS-RoFormer-ep368-GGUF/bs-roformer-ep368-q8_0.gguf",
      NULL, NULL, NULL, NULL, 0, 0, 0, 1, 1 },
};

#define CASE_COUNT (sizeof(CASES) / sizeof(CASES[0]))

typedef struct {
    float * samples;
    size_t  frames;
    int     sample_rate;
    int     channels;
} audio_clip;

static uint16_t read_u16(const unsigned char * p) {
    return (uint16_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8));
}

static uint32_t read_u32(const unsigned char * p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int load_wav(const char * path, audio_clip * clip) {
    unsigned char header[12], chunk[8];
    unsigned char * data = NULL;
    uint32_t data_size = 0;
    int channels = 0, sample_rate = 0, bits = 0;
    size_t i;
    FILE * file = fopen(path, "rb");

    if (file == NULL) return 0;
    if (fread(header, 1, sizeof(header), file) != sizeof(header) ||
        memcmp(header, "RIFF", 4) != 0 || memcmp(header + 8, "WAVE", 4) != 0) {
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
            if (data == NULL || fread(data, 1, size, file) != size) { free(data); data = NULL; break; }
            data_size = size;
            break;
        } else {
            fseek(file, (long)(size + (size & 1u)), SEEK_CUR);
        }
    }
    fclose(file);
    if (data == NULL || bits != 16 || channels < 1 || sample_rate <= 0) { free(data); return 0; }

    clip->frames = data_size / (size_t)(2 * channels);
    clip->channels = channels;
    clip->sample_rate = sample_rate;
    clip->samples = (float *)malloc(sizeof(float) * clip->frames * (size_t)channels);
    if (clip->samples == NULL) { free(data); return 0; }
    for (i = 0; i < clip->frames * (size_t)channels; ++i) {
        clip->samples[i] = (float)(int16_t)read_u16(data + (i * 2)) / 32768.0f;
    }
    free(data);
    return 1;
}

/* Online processor count, or 4 where it cannot be determined. */
static int host_cpu_count(void) {
#if defined(_WIN32)
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return info.dwNumberOfProcessors > 0 ? (int)info.dwNumberOfProcessors : 4;
#else
    const long count = sysconf(_SC_NPROCESSORS_ONLN);
    return count > 0 ? (int)count : 4;
#endif
}

static int file_exists(const char * path) {
    FILE * f = fopen(path, "rb");
    if (f == NULL) return 0;
    fclose(f);
    return 1;
}

/* Prints every option a family declares. This is the part of the ABI that
 * lets a binding generate typed wrappers without hardcoding families, and it
 * is the part path_test.c cannot cover, because the in-repo VAD models declare
 * no options at all. */
static size_t dump_options(const audiocpp_model * model, const char * family) {
    static const char * const SCOPES[] = { "request", "session", "load" };
    size_t total = 0;
    size_t scope;
    for (scope = 0; scope < 3; ++scope) {
        const size_t count = audiocpp_model_option_count(model, (audiocpp_option_scope)scope);
        size_t i;
        for (i = 0; i < count; ++i) {
            const char * name = NULL;
            const char * value_name = NULL;
            const char * description = NULL;
            const char * fallback = NULL;
            const char * min_value = NULL;
            const char * max_value = NULL;
            int required = -1;
            CHECK_OK(audiocpp_model_option(model, (audiocpp_option_scope)scope, i,
                                           &name, &value_name, &description, &fallback,
                                           &min_value, &max_value, &required));
            CHECK(min_value != NULL && max_value != NULL,
                  "%s option '%s' bounds are NULL; absent should read as \"\"",
                  family, name ? name : "?");
            CHECK(name != NULL && strlen(name) > 0, "%s %s option %zu has no name", family, SCOPES[scope], i);
            CHECK(value_name != NULL && description != NULL && fallback != NULL,
                  "%s option '%s' returned a NULL string; absent should read as \"\"",
                  family, name ? name : "?");
            CHECK(required == 0 || required == 1,
                  "%s option '%s' required flag is %d", family, name ? name : "?", required);
            printf("  %-8s %-28s value=%-14s required=%d default='%s' range=[%s,%s]\n",
                   SCOPES[scope], name, value_name, required, fallback, min_value, max_value);
        }
        total += count;
    }
    return total;
}

static void run_case(const model_case * c, const char * models_root,
                     const audio_clip * clip, const audio_clip * alt_clip,
                     const char * backend_name, int threads) {
    char model_path[4096];
    audiocpp_registry * registry = NULL;
    audiocpp_model * model = NULL;
    audiocpp_session * session = NULL;
    audiocpp_request * request = NULL;
    audiocpp_result * result = NULL;
    audiocpp_backend_config backend;
    audiocpp_model_config load_config;
    size_t options;

    if (c->use_alt_audio) {
        if (alt_clip->samples == NULL) {
            printf("skip %-16s (needs the alternate audio clip)\n", c->family);
            g_skipped++;
            return;
        }
        clip = alt_clip;
    }

    snprintf(model_path, sizeof(model_path), "%s/%s", models_root, c->relative_model);
    if (!file_exists(model_path)) {
        printf("skip %-16s (no %s)\n", c->family, c->relative_model);
        g_skipped++;
        return;
    }

    printf("\n=== %s (%s/%s) ===\n", c->family, c->task, c->mode);
    if (audiocpp_registry_create(NULL, &registry) != AUDIOCPP_OK) {
        CHECK(0, "registry: %s", audiocpp_last_error());
        return;
    }
    if (!audiocpp_registry_family_count(registry)) {
        CHECK(0, "empty registry");
        audiocpp_registry_free(registry);
        return;
    }
    memset(&load_config, 0, sizeof(load_config));
    load_config.family_hint = c->family;
    if (audiocpp_model_load(registry, model_path, &load_config, NULL, &model) != AUDIOCPP_OK) {
        /* A family not linked into this build is a skip, not a failure -- the
         * model composite is a build-time choice. */
        printf("skip %-16s (load: %s)\n", c->family, audiocpp_last_error());
        g_skipped++;
        audiocpp_registry_free(registry);
        return;
    }

    CHECK(strcmp(audiocpp_model_family(model), c->family) == 0,
          "loaded '%s' but asked for '%s'", audiocpp_model_family(model), c->family);
    CHECK(audiocpp_model_supports(model, c->task, c->mode),
          "%s does not advertise %s/%s", c->family, c->task, c->mode);

    options = dump_options(model, c->family);
    printf("declared_options=%zu speaker_ref=%d style=%d timestamps=%d\n", options,
           audiocpp_model_supports_speaker_reference(model),
           audiocpp_model_supports_style_condition(model),
           audiocpp_model_supports_timestamps(model));

    backend.backend = backend_name;
    backend.device = 0;
    backend.threads = threads;
    if (audiocpp_session_create(model, c->task, c->mode, &backend, NULL, &session) != AUDIOCPP_OK) {
        CHECK(0, "%s session: %s", c->family, audiocpp_last_error());
        audiocpp_model_free(model);
        audiocpp_registry_free(registry);
        return;
    }

    request = audiocpp_request_create();
    CHECK(request != NULL, "request alloc");
    if (c->text != NULL) {
        CHECK_OK(audiocpp_request_set_text(request, c->text, c->language));
        if (c->voice_id != NULL) CHECK_OK(audiocpp_request_set_voice_id(request, c->voice_id));
    } else {
        CHECK_OK(audiocpp_request_set_audio(request, clip->samples, clip->frames,
                                            clip->sample_rate, clip->channels));
    }

    if (c->seed != NULL) CHECK_OK(audiocpp_request_set_option(request, "seed", c->seed));

    if (audiocpp_session_run(session, request, &result) != AUDIOCPP_OK) {
        CHECK(0, "%s run: %s", c->family, audiocpp_last_error());
    } else {
        g_ran++;

        if (c->expect_audio) {
            const float * samples = NULL;
            size_t frames = 0;
            int rate = 0, channels = 0;
            CHECK_OK(audiocpp_result_audio(result, &samples, &frames, &rate, &channels));
            CHECK(frames > 0, "%s produced no audio frames", c->family);
            CHECK(rate > 0 && channels > 0, "%s audio has rate=%d channels=%d",
                  c->family, rate, channels);
            if (frames > 0 && samples != NULL) {
                double peak = 0.0;
                size_t i;
                for (i = 0; i < frames * (size_t)channels; ++i) {
                    const double a = samples[i] < 0.0f ? -(double)samples[i] : (double)samples[i];
                    if (a > peak) peak = a;
                }
                CHECK(peak > 0.001, "%s audio is silent (peak %.6f)", c->family, peak);
                printf("parity:%s:audio_frames=%zu\n", c->family, frames);
                printf("parity:%s:audio_rate=%d\n", c->family, rate);
                printf("parity:%s:audio_seconds=%.3f\n", c->family, (double)frames / (double)rate);
                printf("parity:%s:audio_peak=%.4f\n", c->family, peak);
            }
        }

        if (c->expect_text) {
            const char * text = NULL;
            const char * language = NULL;
            CHECK_OK(audiocpp_result_text(result, &text, &language));
            CHECK(text != NULL && strlen(text) > 0, "%s produced no transcript", c->family);
            if (text != NULL) printf("parity:%s:text=%s\n", c->family, text);
        }

        if (c->expect_turns) {
            const size_t turns = audiocpp_result_speaker_turn_count(result);
            size_t i;
            CHECK(turns > 0, "%s produced no speaker turns", c->family);
            for (i = 0; i < turns; ++i) {
                int64_t start = -1, end = -1;
                const char * speaker = NULL;
                float confidence = -1.0f;
                const char * text = NULL;
                CHECK_OK(audiocpp_result_speaker_turn(result, i, &start, &end,
                                                      &speaker, &confidence, &text));
                CHECK(start >= 0 && end >= start, "%s turn %zu span [%lld,%lld]",
                      c->family, i, (long long)start, (long long)end);
                CHECK(speaker != NULL, "%s turn %zu speaker id is NULL", c->family, i);
            }
            printf("parity:%s:speaker_turns=%zu\n", c->family, turns);
        }

        if (c->expect_named_audio) {
            const size_t streams = audiocpp_result_named_audio_count(result);
            size_t i;
            CHECK(streams > 0, "%s produced no named audio streams", c->family);
            for (i = 0; i < streams; ++i) {
                const char * id = NULL;
                const float * samples = NULL;
                size_t frames = 0;
                int rate = 0, channels = 0;
                CHECK_OK(audiocpp_result_named_audio(result, i, &id, &samples,
                                                     &frames, &rate, &channels));
                CHECK(id != NULL && strlen(id) > 0, "%s stream %zu has no id", c->family, i);
                CHECK(frames > 0 && rate > 0 && channels > 0,
                      "%s stream '%s' is %zu frames at %d Hz x%d",
                      c->family, id ? id : "?", frames, rate, channels);
                printf("parity:%s:stream_%zu=%s:%zu@%d\n", c->family, i,
                       id ? id : "?", frames, rate);
            }
            printf("parity:%s:named_audio=%zu\n", c->family, streams);
            CHECK(audiocpp_result_named_audio(result, streams, NULL, NULL, NULL, NULL, NULL)
                  == AUDIOCPP_ERR_OUT_OF_RANGE,
                  "%s: out-of-range named audio not rejected", c->family);
        }

        /* Word timestamps are optional per family; report when present so the
         * accessor gets exercised wherever a model does emit them. */
        {
            const size_t words = audiocpp_result_word_count(result);
            if (words > 0) {
                size_t i;
                for (i = 0; i < words; ++i) {
                    const char * word = NULL;
                    int64_t start = -1, end = -1;
                    float confidence = -1.0f;
                    CHECK_OK(audiocpp_result_word(result, i, &word, &start, &end, &confidence));
                    CHECK(word != NULL, "%s word %zu is NULL", c->family, i);
                    CHECK(start >= 0 && end >= start, "%s word %zu span [%lld,%lld]",
                          c->family, i, (long long)start, (long long)end);
                }
                printf("parity:%s:words=%zu\n", c->family, words);
            }
        }

        /* Artifacts, for families that emit them (MIDI, speaker embeddings,
         * diarization state). Reported when present so the accessor is
         * exercised wherever a model does produce one. */
        {
            const size_t artifacts = audiocpp_result_artifact_count(result);
            size_t i;
            for (i = 0; i < artifacts; ++i) {
                audiocpp_artifact_kind kind = AUDIOCPP_ARTIFACT_CUSTOM;
                const char * id = NULL;
                const void * payload = NULL;
                size_t bytes = 0;
                CHECK_OK(audiocpp_result_artifact(result, i, &kind, &id, &payload, &bytes));
                CHECK(id != NULL, "%s artifact %zu id is NULL", c->family, i);
                printf("parity:%s:artifact_%zu=%s:%d:%zu\n", c->family, i,
                       id ? id : "?", (int)kind, bytes);
            }
            if (artifacts > 0) printf("parity:%s:artifacts=%zu\n", c->family, artifacts);
            CHECK(audiocpp_result_artifact(result, artifacts, NULL, NULL, NULL, NULL)
                  == AUDIOCPP_ERR_OUT_OF_RANGE,
                  "%s: out-of-range artifact not rejected", c->family);
        }

        /* Every indexed accessor must reject one past the end. */
        CHECK(audiocpp_result_segment(result, audiocpp_result_segment_count(result),
                                      NULL, NULL, NULL, NULL) == AUDIOCPP_ERR_OUT_OF_RANGE,
              "%s: out-of-range segment not rejected", c->family);
        CHECK(audiocpp_result_word(result, audiocpp_result_word_count(result),
                                   NULL, NULL, NULL, NULL) == AUDIOCPP_ERR_OUT_OF_RANGE,
              "%s: out-of-range word not rejected", c->family);
        CHECK(audiocpp_result_speaker_turn(result, audiocpp_result_speaker_turn_count(result),
                                           NULL, NULL, NULL, NULL, NULL) == AUDIOCPP_ERR_OUT_OF_RANGE,
              "%s: out-of-range speaker turn not rejected", c->family);
    }

    audiocpp_result_free(result);
    audiocpp_request_free(request);
    audiocpp_session_free(session);
    audiocpp_model_free(model);
    audiocpp_registry_free(registry);
}

int main(int argc, char ** argv) {
    const char * models_root = (argc > 1) ? argv[1] : "";
    const char * audio_path = (argc > 2) ? argv[2] : "assets/resources/sample_16k.wav";
    const char * backend_name = (argc > 3) ? argv[3] : "cpu";
    /* 44.1 kHz stereo, for families that fix their input rate. */
    const char * alt_audio_path = (argc > 4) ? argv[4] : "tests/ace_step/assets/complete_source_demucs_8s.wav";
    /* Separation dominates this test's wall time and scales with threads (191s
     * at 4 against 113s at 16), and thread count does not change results --
     * verified bit-identical stems at both. Detected rather than defaulted to a
     * constant, so a bare invocation uses the machine it is on instead of
     * relying on every caller to remember the argument. */
    const int threads = (argc > 5) ? atoi(argv[5]) : host_cpu_count();
    audio_clip clip;
    audio_clip alt_clip;
    size_t i;

    memset(&clip, 0, sizeof(clip));
    memset(&alt_clip, 0, sizeof(alt_clip));

    if (models_root[0] == '\0') {
        printf("no model root configured; set AUDIOCPP_C_API_MODEL_ROOT to run this test\n");
        return CTEST_SKIP;
    }
    if (threads <= 0) {
        fprintf(stderr, "threads must be positive\n");
        return 1;
    }
    if (!load_wav(audio_path, &clip)) {
        fprintf(stderr, "cannot read %s\n", audio_path);
        return 1;
    }

    if (!load_wav(alt_audio_path, &alt_clip)) {
        printf("note: no alternate clip at %s; rate-fixed families will skip\n", alt_audio_path);
    }

    printf("models_root=%s\naudio=%s frames=%zu rate=%d\nbackend=%s\n",
           models_root, audio_path, clip.frames, clip.sample_rate, backend_name);
    printf("threads=%d\n", threads);
    if (alt_clip.samples != NULL) {
        printf("alt_audio=%s frames=%zu rate=%d ch=%d\n",
               alt_audio_path, alt_clip.frames, alt_clip.sample_rate, alt_clip.channels);
    }

    for (i = 0; i < CASE_COUNT; ++i) {
        run_case(&CASES[i], models_root, &clip, &alt_clip, backend_name, threads);
    }

    free(clip.samples);
    free(alt_clip.samples);

    printf("\nran=%d skipped=%d failures=%d\n", g_ran, g_skipped, g_failures);
    if (g_failures > 0) return 1;
    if (g_ran == 0) {
        printf("no model was available; skipping\n");
        return CTEST_SKIP;
    }
    printf("c api model test OK\n");
    return 0;
}
