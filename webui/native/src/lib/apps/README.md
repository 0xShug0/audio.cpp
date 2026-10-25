# Apps

Each app owns its UI, project schema and workflow in its own directory. Apps use
project storage, model execution and media utilities without importing
Studio or reading its form state. The app manifest is discovered by registry.ts.

The included app is Text to speech (`tts`). Its project pipeline is:

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
