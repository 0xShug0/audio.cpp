# Narration workspace

## Goal
A new user can always identify what to work on and the next action. Keep the script visually dominant, settings beside the work, and the main action in one anchored footer. Preserve project data, generation, take selection, audio editing, assembly, and export contracts.

## Reference study
- [yovoice](https://github.com/leemysw/yovoice), `web/src/features/create/studio.tsx` and `inspector.tsx`: a quiet writing surface with a separate voice inspector; progressive disclosure for generation parameters.
- [Speech Studio](https://github.com/timoncool/speech-studio), `frontend/src/components/speech/ProjectEditor.tsx` and `lib/simpleMode.ts`: guided text-to-file steps, passage readiness, local playback and explicit take selection.

## Layout and flow (designed before implementation)

    Project title                       Projects menu · Saved
    1 Script —— 2 Voice —— 3 Arrange —— 4 Export
    Step heading and one sentence of guidance
    ┌──────────────────────────────┬──────────────────────┐
    │ Script / selected passage    │ Contextual settings  │
    │                              │ (Voice / Arrange /   │
    │ Local playback and takes     │  Export only)        │
    └──────────────────────────────┴──────────────────────┘
    Back       Next-action guidance          Main action →

1. **Script**: paste or import text into a generous writing surface. Sensible split defaults; splitting options in a disclosure. Footer action becomes Review passages, then Continue to voice. Preview is part of the writing surface.
2. **Voice**: choose project model and voice in the right inspector. Read/edit one passage in the main surface, with previous/next controls and an expandable searchable passage browser. Generate missing audio from the footer. Listen using a simple player; compare takes, customize one passage, and edit waveform audio on demand. When every passage has a current chosen take, the footer advances to Arrange.
3. **Arrange**: review passage order and individual joins. Default pause and sample rate in the inspector; exact navigation and ordering tools disclosed. Footer combines audio, then advances to Export. Missing audio leads back to Voice.
4. **Export**: listen to combined audio. Audio enhancement and level adjustment are optional disclosures; format in the inspector. Footer downloads, applying processing automatically. Processed audio can be previewed explicitly.

## Interaction rules
- One filled primary action per screen, in the same sticky footer within document flow; no duplicate next/generate actions in headings or inspectors.
- Backward navigation preserves work; forward steps require their actual prerequisites. Explain the requirement next to the primary action.
- Status and cancellation live together in the footer. Errors have an alert and dismiss control. Existing chosen audio survives cancellation or failure.
- Save automatically; project management and deletion are grouped in a menu. Advanced, bulk and destructive actions use labeled disclosures.
- New generations retain explicit replacement selection. Changed text/settings visibly mark audio out of date and invalidate downstream output.
- Native labels, keyboard focus, step state, live status, and text readiness labels. On narrow screens, settings stack above the passage editor; controls wrap; the footer remains reachable.

## Verification
Use the real Svelte UI through browser clicks, text entry, upload, playback, and download. Exercise a fresh project, missing model guidance, generation, take replacement, text invalidation, cancellation/failure, ordering/pauses, processing, export, reload persistence, keyboard access, and narrow layouts. If inference is unavailable, disclose API fixtures separately from the real browser audio assembly/export path.

## Running the browser regression

From `webui/native`, start `npm run dev -- --host 127.0.0.1`. The regression uses Chrome and keeps browser tooling separate from application dependencies:

```sh
npm install --prefix /tmp/tts-browser-tools playwright @axe-core/playwright
TTS_BROWSER_TOOLS=/tmp/tts-browser-tools node src/lib/apps/tts/tests/workflow.mjs
```

Optional environment variables: `TTS_BASE_URL` (defaults to `http://127.0.0.1:5173`), `TTS_BROWSER_CHANNEL` (defaults to `chrome`), and `TTS_TEST_OUTPUT` (defaults to a temporary results directory). The test creates its own isolated browser context, mocks model API responses, and writes screenshots, exported audio, and a JSON report. It does not require model weights.

The deterministic model fixture returns four-second PCM audio. Playback, waveform edits, pause insertion, normalization/gain, WAV/WebM encoding, IndexedDB storage, and browser downloads all use the actual application implementations. Actual model inference and speech quality require a running audio.cpp backend.

Validated on 2026-10-25: 21 browser checks passed; all four steps fit 320–1440px and had zero WCAG A/AA scan violations. Svelte/TypeScript checks reported zero errors and warnings; the production build succeeded. The tracked `webui/native/dist/index.html` is regenerated because the native server embeds that bundle at configure time. Component source changes are restricted to this TTS directory.

## Iteration 2: speakers and a desktop workspace

The first iteration still required page scrolling and repeated model/voice controls. This iteration gives the full app a fixed viewport height on desktop, with independently scrollable panes and a persistent action bar:

    Project name                         Script · Speakers · Arrange · Export
    ┌─────────────────┬──────────────────────────────┬───────────────────────┐
    │ Passage search  │ Passage 1                     │ Speakers              │
    │ and outline     │                              │ Alex                  │
    │                 │ Editable passage text        │ Maya                  │
    │ 01 Alex · ready │                              │ + Add speaker         │
    │ 02 Maya · ready │ Playback and local actions   │ Assignment controls   │
    └─────────────────┴──────────────────────────────┴───────────────────────┘
    Progress and status                                        Main action →

A speaker combines a name, model configuration, and built-in or reference voice. Speaker creation/editing happens in one dialog. Speakers are stored in a TTS-owned library and copied into projects so editing one project does not unexpectedly change another. Existing model/voice configurations migrate into project speakers while preserving take signatures and audio. Passage assignment, splitting, joining, and restoration preserve speaker identity.

The center only contains the selected text and playback. Take comparison and waveform editing use dialogs. The left outline is always visible on desktop, including in Arrange; the right pane is the only place to assign speakers. On narrow screens, Passages / Editor / Speakers switch between panes so the editor and action bar remain reachable without a long stacked page.

Iteration 2 validation (2026-10-25): 19 browser scenarios pass against the production bundle, including speaker creation/assignment, cross-project reuse and isolation, explicit take replacement, cancellation, waveform editing, processing, downloads, and viewport checks at 320–1440px. All four stages pass WCAG A/AA scans. Four migration/routing checks (`node src/lib/apps/tts/tests/speakers.mjs`) verify that version 1 projects upgrade to version 2 without invalidating chosen takes or assembly. Svelte/TypeScript reports zero errors and warnings. Both UI and full-model Metal server builds pass. A separate live test creates a speaker, generates actual Kokoro speech, plays it, and confirms zero page scrolling at 1440×900.


## Iteration 3: speaker presets replace model presets

The reusable TTS unit is a speaker preset: name + model + built-in or reference
voice + generation settings. Its selector lives in Add speaker, and Model
settings is directly visible below the model selector. Existing projects retain
their own configuration snapshots.

Global model preset saving, Library browsing, and cross-page model preset routing
are removed. Recognition and enhancement selectors use installed models directly.
A compatibility read imports legacy speech settings once per stable ID. Database
versions, existing stores, project records, and audio signatures remain unchanged.

Iteration 3 validation: 23 production-browser checks, four project migration
checks, Svelte checks (zero errors and warnings), and UI/native builds pass. A
separate real Kokoro inference and playback check passes. The browser workflow
suite uses deterministic speech API fixtures; browser audio work is real.

## Iteration 4: manage speakers in Lib

Lib offers separate Voices and Speakers collections. Speakers can be searched,
created, edited (including model settings and reference voices), duplicated, and
deleted. Their use remains in TTS. The app's mounted preset picker refreshes after
library writes, and project speaker snapshots and recordings stay intact.

Legacy-import deletion uses a tombstone in the existing speakers store so the
compatibility read cannot resurrect a deleted preset. No database version or
store schema changes. The speaker editor is shared by Lib and the app.

Iteration 4 validation: 29 browser scenarios pass against the rebuilt native UI,
including Lib speaker CRUD, deletion of legacy imports, persistence, live picker
refresh, project isolation, mobile widths, and WCAG A/AA scans. All database
versions remain 1. The four migration checks, Svelte checks (zero errors and
warnings), production UI build, and full-model native server build pass.

## Upload cancellation regression

A file input emits a bubbling `cancel` event when its picker is dismissed. The
speaker dialog handles only its own cancellation event, so cancelling an upload
preserves the draft; Escape on the dialog still closes it.

The focused regression uses native, trusted Chrome file-input cancellation twice,
checks the name/model/reference/transcript are retained, and verifies dialog
Escape separately. From `webui/native`, run it with the same browser tools setup
as the workflow regression:

```sh
TTS_BROWSER_TOOLS=/tmp/tts-browser-tools node src/lib/apps/tts/tests/upload-cancel.mjs
```

Step-by-step live validation on 2026-10-25 used no speech API fixtures: import and
save a reference recording and matching transcript in Lib → Voices; create a
BreezeTTS 2 Clone Q8 speaker preset from that voice; cancel upload without losing
the draft; successfully upload a replacement; save with the library voice; import
the speaker preset in TTS; cancel upload there too; create a second app speaker
using Kokoro Q8 and randomly chosen `bm_lewis`; generate both passages; play each;
combine with the default 0.25-second pause; play across the join; download WAV;
reload and play the saved result. Both real model requests returned 200. The
24 kHz mono PCM16 WAV is 9.225 seconds (4.4 + 0.25 + 4.575), and both sections have
nonzero audio. No browser runtime errors occurred. Svelte and production/native
builds pass. Database versions remain 1; no commit was made for this fix.
