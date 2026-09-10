# Shared eSpeak-ng phonemizer

`engine::audio::EspeakPhonemizer` is a reusable, optional runtime adapter for
eSpeak-ng. SanoTTS (E2M and Piper frontends) and Inflect v2 use it. Other models,
including the separate Kokoro preview, can use the same adapter without copying
dynamic-library loading or process-global state management.

This change does not bundle eSpeak-ng source, binaries, headers or language data,
and does not introduce a build-time eSpeak dependency. Users still provide an
installed shared library and its matching data. Existing model-specific session
options remain supported. eSpeak-ng retains its upstream license.

## Model integration

```cpp
#include "engine/framework/audio/espeak_phonemizer.h"

engine::audio::EspeakPhonemizer phonemizer(
    library_path,          // empty: normal platform library search
    espeak_data_directory, // espeak-ng-data itself; empty: library default
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
