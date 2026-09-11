# Shared eSpeak-ng phonemizer

`engine::audio::EspeakPhonemizer` is a reusable, optional runtime adapter for
eSpeak-ng. SanoTTS (E2M and Piper frontends) and Inflect v2 use it. Other models,
including the separate Kokoro preview, can use the same adapter without copying
dynamic-library loading or process-global state management.

By default users provide an installed shared library and its matching data.
Alternatively, `AUDIOCPP_STATIC_ESPEAK=ON` builds the pinned eSpeak-ng 1.52.0 source
and statically links its code into both CLI and server. No eSpeak DLL or `.so` is
required in that mode. Existing explicit library/data session options still work.

## Static build with separate data

```sh
cmake -S . -B build/static-espeak -DAUDIOCPP_STATIC_ESPEAK=ON
cmake --build build/static-espeak --config Release --target audiocpp_cli audiocpp_server
```

The first build downloads a SHA-256-verified upstream source archive. eSpeak is
built in an isolated CMake project; it does not change audio.cpp's shared-library
settings. Sonic, audio playback, MBROLA, asynchronous synthesis and speechPlayer
are disabled because this adapter only needs phonemization. The upstream tool
compiles the phoneme tables and all 114 language dictionaries. Cross-compiling
this option is currently rejected because generating data requires a host tool.

The build packs the data automatically. The executable output directory contains:

```text
audiocpp_cli[.exe]
audiocpp_server[.exe]
espeak-ng-data.bin
licenses/espeak-ng/
```

The code is linked separately into each executable. With no explicit session
paths, the static adapter finds `espeak-ng-data.bin` relative to the executable,
not the working directory. It also accepts the legacy `espeak-ng-data.gguf` name,
then an unpacked `espeak-ng-data` folder as fallbacks. Explicit model `espeak_data_path`
options accept a folder or a data package (`.bin` or `.gguf`), including in dynamic-library mode.
Keep the data package when moving the executables. Missing/invalid data produces an error;
it is not silently substituted with another installed version.

The `.bin` package internally uses data-only GGUF with zero tensors: dictionaries and phoneme tables, not
neural-model weights or audio recordings. It uses audio.cpp's binary embedded-file
metadata layout, format version 1, pinned to eSpeak-ng 1.52.0. The current package
is 18,384,736 bytes (about 17.5 MiB). It is not compressed. This does not add a
model-manager download package.

### Extraction cache

eSpeak still needs ordinary files. The adapter extracts into a per-user cache:

- Windows: `%LOCALAPPDATA%/audio.cpp/espeak-data/`
- Linux/macOS: `$XDG_CACHE_HOME/audio.cpp/espeak-data/`, or
  `$HOME/.cache/audio.cpp/espeak-data/` when XDG_CACHE_HOME is unavailable.

CLI and server share content-keyed cache entries. On reuse, every file is checked
against the package without rewriting unchanged files. Extraction publishes a
complete directory atomically, so concurrent processes never use half-written
files. Changed packages and damaged caches get new entries; existing entries are
not modified while another engine may be using them. Unsafe paths, duplicate or
case-colliding names, invalid byte ranges and unsupported versions are rejected.
The package and extracted files each occupy disk space. Old cache entries can be
removed manually when no audio.cpp process is using them; automatic eviction is
not implemented.

To repack a compatible 1.52.0 data directory manually:

```sh
audiocpp_espeak_pack /path/to/espeak-ng-data /path/to/espeak-ng-data.bin
```

The license directory includes upstream COPYING, the original source archive,
and our CMake integration/patch scripts. eSpeak-ng retains GPL-3.0-or-later terms.
Static builds are opt-in and distributors must meet the applicable requirements
for the combined work, including corresponding source; copying only the license
notice is not sufficient. This option does not relicense the upstream dependency.

Validated on Windows x64/MSVC: static CLI/server builds, no eSpeak DLL import,
21/21 frontend token sequences identical to dynamic eSpeak-ng 1.52.0 using the
same generated data, including an executable-plus-data-package-only portable directory,
and 200 concurrent frontend requests. Five focused tests pass, including binary
packing/extraction, cache reuse, recovery, concurrent extraction and traversal
rejection. Linux/macOS build
paths are provided but have not been validated locally.

## Model integration

```cpp
#include "engine/framework/audio/espeak_phonemizer.h"

engine::audio::EspeakPhonemizer phonemizer(
    library_path,          // empty: static engine if enabled, else library search
    espeak_data_directory, // empty: executable-local in static mode
    {"en-us"});            // ordered voice candidates, chosen by the model

const auto ipa = phonemizer.phonemize(text, 2);
```

The adapter accepts eSpeak's phoneme-mode integer, including IPA ties and
separators. An optional third argument to `phonemize` controls how clauses are
joined (default: one space). Model-specific normalization, punctuation handling,
voice fallback policy, IPA cleanup and token mapping stay in the model frontend.
For example, Kokoro can request its caret-tied IPA mode; SanoTTS keeps its
regional voice preference and different E2M/Piper modes.

The constructor validates paths, required exports and voice availability. Calls
throw exceptions for unavailable dependencies instead of silently substituting
another phonemizer. Initialization requests eSpeak's `DONT_EXIT` behavior.

## Lifetime and concurrency

eSpeak owns a process-global translator and output buffer. A single shared
service serializes initialization, voice selection, clause processing and copying
the returned text. Each request reselects its client's voice. Creating or
destroying another frontend cannot terminate a currently running request.

The service caches one runtime for the active library/data path pair. Changing
either closes the old runtime and initializes the requested one under the same
lock; failed switches can be retried. Relative explicit paths are resolved when
the client is constructed. Keep the library and data files available while clients
use them. No neural-model weights or audio buffers are cached here.

All in-process eSpeak consumers must use this service: independent direct eSpeak
calls cannot participate in its lock. Kokoro migration is intentionally left to
its separate preview PR rather than including a model port in this refactor.

## Validation

Configure with `ENGINE_BUILD_TESTS=ON` for `espeak_phonemizer_test`. Its small mock
shared libraries exercise missing dependencies/symbols/voices, failure recovery,
clause joining, modes, cursor progress, client destruction and 1,000 concurrent
calls with different voices. The mock is test-only and requires no eSpeak install.

For frontend tests and the optional real-library probe:

```sh
cmake -S . -B build/espeak-tests -DAUDIOCPP_MODEL_SET=custom \
  -DAUDIOCPP_MODELS="sanotts;inflect_v2" -DENGINE_BUILD_TESTS=ON \
  -DENGINE_BUILD_MODEL_TESTS=ON
cmake --build build/espeak-tests --target espeak_phonemizer_test \
  sanotts_frontend_test inflect_v2_frontend_test espeak_frontend_probe
ctest --test-dir build/espeak-tests --output-on-failure \
  -R '^(espeak_phonemizer|sanotts_frontend|inflect_v2_frontend)_test$'
```

Run `espeak_frontend_probe <library> <espeak-ng-data> <mode>` using modes
`sanotts`, `inflect`, `piper`, or `concurrent`. The first three print deterministic
token sequences for comparison with pre-refactor frontends. `piper` covers eleven
languages using a synthetic IPA-range vocabulary, not downloaded model weights.
`concurrent` checks 200 interleaved SanoTTS/Inflect requests against serial results.
These are frontend tests, not end-to-end audio quality or GPU performance tests.
