# Apps

Each app owns its UI, project schema and workflow in its own directory. Apps use
project storage, model execution and media utilities without importing
Studio or reading its form state. The app manifest is discovered by registry.ts.

The Text to speech app (`tts`) has this project pipeline:

1. Script: import TXT/Markdown, choose splitting settings, preview chunk boundaries.
2. Speakers: create or reuse a speaker preset combining a model, voice or
   reference recording, and generation settings. Assign speakers to passages,
   generate speech, compare takes and explicitly choose replacements. Optional
   reference transcription uses its own installed recognition model.
3. Assemble: chosen takes in document order, default/per-boundary pauses and output
   sample rate. Missing/changed takes must be resolved before building assembly.
4. Finish: optional enhancement model, gain or peak normalization, original
   and processed previews, WAV PCM16 or WebM Opus export. Opus uses the existing
   media library/browser encoder and reports unsupported encoding explicitly.

These are configurable features of a project, not a separate capability module
framework. Import/export formats can be extended in the app's file/media handling.
Model execution uses a ModelSettings snapshot built from an installed model.
Lib → Speakers manages speaker presets; the TTS app consumes them. Speech model presets
from older versions are imported into its speaker store once per ID, preserving
all saved settings. This compatibility read leaves IndexedDB versions unchanged.

Project configurations are snapshots. Editing a speaker preset does not mutate
another project. The shared library stores opaque app records; this app owns the data
schema. IndexedDB persists projects, reference audio, take histories and derived
outputs in this browser. Old chunks/takes are archived after split/merge/removal.

The app keeps chosen takes when regeneration fails or is cancelled. New takes
never replace an existing choice automatically. Editing text/settings marks older
takes and downstream audio out of date. Apps remain mounted when navigating to
Lib/Studio, and save before returning to the app list. Playback pauses when hidden.

One operation runs at a time, with sequential inference and cancellation. A
ModelRuntime applies a configuration once per consecutive group of requests;
changing configuration creates another runtime. Managed models are explicitly
reloaded because the server does not report load options. Server-configured
models must match and are not rewritten. Capacity/eviction remains server-owned.

Current formats are TXT/Markdown for text, browser-decodable reference audio,
WAV and WebM for export. EPUB/PDF, MP3 export and video dubbing are not included.
Assembly/finishing use browser memory and are not unlimited-duration renderers.

## Cross-language dubbing

The `cross-language-dubbing` app uses the existing server recipe through
`POST /v1/recipes/run`. Upload a recording, select an installed offline speech
translation model and an offline TTS model, then choose the source and target
language codes. Only ASR models advertising `target_language` appear as
translation choices. Canary, OWSM and OWSM-CTC expose this option.

Choose a TTS voice for the target language. If the TTS model requires a language
option, set its model-specific code in Speech model's generation options;
translator and TTS language identifiers are independent. For example, Canary
can translate German (`de`) to English (`en`) for an English TTS voice.
Models with speaker-reference support also accept an optional reference
recording and transcript. A source recording is not automatically used as a
cloning reference.

The app prepares model configurations using the same `ModelRuntime` as Text to
speech. The server recipe owns translation and synthesis; the browser uploads
audio and downloads the translated text and dubbed WAV. Generation time,
output duration and RTF come from the recipe response. Cancellation aborts the
browser request and retains the previous result; it does not promise to stop
an already-running offline server stage. There is no subtitle alignment,
video editing or persistent project state in this app. Use the default server
runtime, as required by the recipe endpoint.

## Browser regression checks

With Playwright available to Node and the built UI served by a local server:

```sh
UI_BASE_URL=http://127.0.0.1:8080 node webui/native/tests/ui-workflows.mjs
```

`UI_BROWSER_TOOLS` can point to a directory from which `playwright` resolves;
`UI_BROWSER_CHANNEL` defaults to `chrome`. The checks use isolated browser
storage and deterministic HTTP fixtures, exercising Studio status/native
chunking, Arena task changes, and dubbing request routing, recovery and downloads.
They do not load models. Actual model/audio validation is a separate check.
