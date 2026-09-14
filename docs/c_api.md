# C API

`include/audiocpp.h` is a C ABI for embedding audio.cpp **in-process**. It is a
facade over `engine::runtime` and adds no behaviour of its own: every call maps
onto `ModelRegistry` → `ILoadedVoiceModel` → `IVoiceTaskSession`, the same
surfaces `audiocpp_cli` uses.

It is off by default.

```bash
cmake -S . -B build -DAUDIOCPP_BUILD_C_API=ON
cmake --build build --target audiocpp
```

This produces `libaudiocpp.so` / `libaudiocpp.dylib` / `audiocpp.dll` with
`SOVERSION 0`. Nothing changes in a build that leaves the option off.

## When to use it

| | best for | cost |
|---|---|---|
| `audiocpp_cli` | scripting, batch jobs | a process per request; weights reload every time |
| `audiocpp_server` | multi-client, remote, OpenAI-compatible clients | a port, a supervised process, audio serialization per call |
| C API | embedding in an application | you manage the handles yourself |

The C API is the only one of the three that keeps a session warm inside your own
process, so it is the one that benefits from graph and cache reuse across
requests.

## Contract

- **Opaque handles only.** No C++ type crosses the boundary.
- **No exception escapes.** Every entry point returns `audiocpp_status`.
  `audiocpp_last_error()` gives the detail for the last failing call *on the
  calling thread*.
- **Returned `const char *` and `const float *` are borrowed.** They are valid
  until the handle that produced them is freed or mutated. Copy anything you
  intend to keep.
- **Strings are never NULL.** A field the model did not populate reads as `""`,
  so callers need no null checks on returned strings.
- **Freeing NULL is a no-op**, so cleanup paths need no null checks either.
- **Out parameters are optional.** Pass NULL for anything you do not want.
- **Handles are not individually thread-safe.** Separate handles may be used
  concurrently from separate threads.

### Lifetime

Handles hold their parents alive internally: a session holds its model, and a
model holds its registry. **Freeing out of order is therefore safe.**

```c
audiocpp_model_free(model);        /* the session keeps working */
audiocpp_registry_free(registry);
audiocpp_session_run(session, request, &result);   /* still valid */
audiocpp_session_free(session);
```

This is deliberate. Callers from garbage-collected runtimes cannot control the
order finalizers run in, and requiring them to would make the ABI a trap. The
path test asserts it.

### Versioning

`audiocpp_abi_version()` returns `(major << 16) | (minor << 8) | patch`. A
caller built against a different **major** must not use the library. Check it
once at load time:

```c
if ((audiocpp_abi_version() >> 16) != AUDIOCPP_ABI_VERSION_MAJOR) {
    /* refuse to proceed */
}
```

## Usage

```c
#include "audiocpp.h"

audiocpp_registry * registry = NULL;
audiocpp_model    * model    = NULL;
audiocpp_session  * session  = NULL;
audiocpp_request  * request  = NULL;
audiocpp_result   * result   = NULL;

audiocpp_model_config config = { "kokoro_tts", NULL, NULL, NULL };

audiocpp_registry_create(NULL, &registry);
audiocpp_model_load(registry, "models/Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf",
                    &config, NULL, &model);

audiocpp_backend_config backend = { "cuda", 0, 4 };
audiocpp_session_create(model, "tts", "offline", &backend, NULL, &session);

request = audiocpp_request_create();
audiocpp_request_set_text(request, "Hello from audio.cpp.", "en-us");
audiocpp_request_set_option(request, "voice-id", "af_heart");

if (audiocpp_session_run(session, request, &result) == AUDIOCPP_OK) {
    const float * samples; size_t frames; int rate, channels;
    audiocpp_result_audio(result, &samples, &frames, &rate, &channels);
    /* samples is interleaved and borrowed: frames * channels floats. */
}

audiocpp_result_free(result);
audiocpp_request_free(request);
audiocpp_session_free(session);
audiocpp_model_free(model);
audiocpp_registry_free(registry);
```

`task` and `mode` take the same spellings as `--task` and `--mode`; `backend`
takes the same spellings as `--backend`. Options map 1:1 onto `--load-option`,
`--session-option`, and `--request-option`, and the fields of
`audiocpp_model_config` map onto `--family`, `--config`, `--weight` and
`--model-spec-override`.

Beyond text and audio, a request carries everything `TaskRequest` does: a
speaker reference (`audiocpp_request_set_voice_id` /
`..._set_voice_audio`), style conditioning (`..._set_speaking_rate`,
`..._set_pitch_shift`, `..._set_emotion`, `..._set_style_language`,
`..._set_energy_scale`, `..._set_style_tag`), and input artifacts
(`audiocpp_request_add_artifact`). Results expose the matching outputs,
including artifacts, so a speaker embedding can be read out of one result and
fed back into a later request instead of being re-derived.

## Discovering what a model accepts

`ModelInspection::cli` is exposed, so a binding can enumerate a family's options
at runtime instead of hardcoding per-family knowledge:

```c
size_t count = audiocpp_model_option_count(model, AUDIOCPP_OPTION_SCOPE_REQUEST);
for (size_t i = 0; i < count; ++i) {
    const char *name, *value_name, *description, *fallback, *min_value, *max_value;
    int required;
    audiocpp_model_option(model, AUDIOCPP_OPTION_SCOPE_REQUEST, i,
                          &name, &value_name, &description, &fallback,
                          &min_value, &max_value, &required);
}
```

Bounds come through too, so generated wrappers can validate rather than just
pass values along — Sortformer's `speaker_threshold` reports `[0, 1]`, Kokoro's
`text_chunk_size` reports a minimum of 32.

This is what keeps the ABI independent of the model surface: the option maps are
`string -> string` in the framework already, so **adding a model family never
changes this header**.

## Streaming

Pull-based, mirroring `IStreamingVoiceTaskSession`. No callback crosses the FFI
boundary.

```c
audiocpp_session_create(model, "vad", "streaming", &backend, NULL, &session);

int64_t chunk = 0;
audiocpp_stream_policy(session, NULL, NULL, &chunk, NULL);   /* frames per push */
audiocpp_stream_start(session, NULL);

for (...) {
    audiocpp_event * event = NULL;
    audiocpp_stream_push(session, block, (size_t)chunk, 16000, 1, offset, &event);
    if (event) {
        const audiocpp_result * view = audiocpp_event_as_result(event);
        /* read via the result accessors */
        audiocpp_event_free(event);
    }
}

audiocpp_stream_finish(session, &result);
```

A `StreamEvent` carries the same shapes a `TaskResult` does under different
names, so it is read through the result accessors: `partial_text` appears as
the result's text, and any voice-activity event carrying a segment appears as a
speech segment. `audiocpp_stream_next_event()` sets `*out_event` to NULL when
the queue is empty — that is `AUDIOCPP_OK`, not an error.

## Symbol surface

`libaudiocpp.so` exports exactly the 55 entry points in `audiocpp.h` and nothing
else. This needs a linker export list, not just `CXX_VISIBILITY_PRESET hidden`:
the visibility preset reaches only this target's own sources, while the static
libraries linked in (ggml, cJSON, sentencepiece, ...) are compiled without
`-fvisibility=hidden` and would otherwise be re-exported — 2767 dynamic symbols
instead of 55, any of which could collide with something the host application
has already loaded.

`src/capi/audiocpp.map` (ELF) and `src/capi/audiocpp.symbols` (Mach-O) hold that
list.

Windows works differently and is worth understanding, because the obvious
assumption is wrong. `__declspec(dllexport)` is **additive** — it says "also
export this", never "export only this" — so a DLL's surface is its own
annotations plus every annotation in the static libraries it absorbs. There is
no PE equivalent of an allowlist short of a `.def` file. Vendored cJSON defaults
to annotating all 78 of its public functions, which put the first Windows build
at 147 exports instead of 69; `cjson_vendor` is therefore compiled with
`CJSON_HIDE_SYMBOLS`. If a future dependency starts annotating its symbols, the
Windows surface will grow again and neither of the two allowlists will stop it —
which is why `audiocpp_c_api_exports` asserts the surface on every platform
rather than leaving it to this paragraph.

To check:

```bash
# ELF
nm -D --defined-only build/bin/libaudiocpp.so | awk '$2=="T"{print $3}' | grep -cv '^audiocpp_'
# Mach-O
nm -gU build/bin/libaudiocpp.dylib | awk '{print $NF}' | grep -cv '^_audiocpp_'
# PE
dumpbin /EXPORTS build\bin\Release\audiocpp.dll
# all three: 0 non-matching, 69 total
```

## Notes

- The library does **not** call `omp_set_num_threads()`. `audiocpp_cli` does,
  but that is process-global state and a library embedded in a host application
  has no business changing it. Set `threads` in `audiocpp_backend_config`.
- `audiocpp_session_run()` prepares the session implicitly. Call
  `audiocpp_session_prepare()` directly only to pay that cost ahead of a
  latency-sensitive run; the next run will then skip it.
- The facade depends only on framework headers, never on `app/`, so the CLI and
  the C API stay independent of each other. The one consequence is that the
  backend-name mapping is duplicated from `app/cli/args.cpp`.

## Test

`tests/capi/path_test.c` is compiled **as C**, which is the point: the header
must be consumable by a C compiler with no C++ runtime on the caller's side. It
runs against Silero VAD, which ships in-repo, so it needs no model download.

```bash
cmake -S . -B build -DAUDIOCPP_BUILD_C_API=ON
cmake --build build --target audiocpp_c_api_path_test
ctest --test-dir build -R audiocpp_c_api_path --output-on-failure
```

Note the `-R audiocpp_c_api_path` there rather than the wider `-R audiocpp_c_api`.
CTest treats a registered test whose executable has not been built as a hard
failure rather than a skip, so running the full selector after building only one
test binary reports red for a reason that is not a real failure. Either name the
binaries you want, as below, or just `cmake --build build` and let everything
build.

The registered test runs on CPU so it needs no GPU. A fourth argument selects
another backend through the ABI:

```bash
./build/bin/audiocpp_c_api_path_test \
    assets/framework/models/silero_vad assets/resources/sample_16k.wav cuda
```

### Tests

| test | needs | covers |
|---|---|---|
| `audiocpp_c_api_exports` | nothing | that the library exports exactly this header |
| `audiocpp_c_api_path` | nothing (in-repo model) | the ABI contract, offline + streaming |
| `audiocpp_c_api_model` | downloaded models | real families across TTS / ASR / diarization |
| `audiocpp_c_api_parity` | models + `audiocpp_cli` | that the C API and the CLI agree |

The last two **skip** (CTest exit 77) rather than fail when models are absent, so
a checkout with no downloads still reports green. Point them at a models root:

```bash
cmake -S . -B build -DAUDIOCPP_BUILD_C_API=ON \
    -DAUDIOCPP_C_API_MODEL_ROOT=/path/to/models
cmake --build build --target audiocpp_c_api_path_test audiocpp_c_api_model_test audiocpp_cli
ctest --test-dir build -R audiocpp_c_api --output-on-failure
```

`audiocpp_c_api_parity` is the one that answers "is this actually working": the
C API drives the same runtime the CLI does, so for identical inputs the two must
produce identical output.

Note that generative families draw noise per run — two unseeded `audiocpp_cli`
Kokoro runs over the same text produce different audio — so the parity check
seeds both sides. Without that the comparison would be meaningless.

### The path test

It covers both the offline and the streaming surface -- Silero advertises
`vad/streaming`, so no extra model is needed -- and asserts the ABI contract
rather than the model's judgement: status codes, borrowed-string rules,
out-of-range handling, out-of-order frees, each mode refusing the other's entry
point, and that a reused session gives the same answer twice. It deliberately does not assert that
synthetic audio is detected as speech, which would pin the test to Silero's
thresholds instead of to the ABI.
