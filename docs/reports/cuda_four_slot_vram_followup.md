# CUDA four-slot VRAM follow-up: six models

Local Windows/MSVC tests on an RTX 3090 (24 GiB), 2026-09-27, eight CPU threads,
CUDA device 0. Source base: PR #706 commit `16e0fa0a`. Each model's
implementation, CUDA slot admission change and capacity assertion are committed
separately.

## Problem and change

All six families previously passed three slots. Four slots were skipped for
Confucius4 TTS, Echo TTS, MOSS VoiceGenerator and OuteTTS based on estimated
VRAM, and stopped near capacity for Fish Audio and SheetSage2. The skipped
estimates were not measured four-slot peaks or proof of allocation failure.

These families uploaded separate device copies of immutable weights for each
session. Reuse the existing package-owned `SharedWeightCache` for CUDA tensors:

| Family | Components shared between CUDA sessions |
|---|---|
| Confucius4 TTS | T2S and S2A weights |
| Echo TTS | DiT weights and their backing weight store |
| Fish Audio | Slow/fast AR weights |
| MOSS VoiceGenerator | Delay-backbone weights, using the existing optional decoder cache API |
| OuteTTS | Llama weights, keyed separately by storage type |
| SheetSage2 | Encoder and decoder weights |

Each slot retains its own execution backend, graphs, KV caches, reference state,
constants and sampling state. Only initial weight loading is coordinated by
the existing cache mutex; generation is not serialized. Keys distinguish the
component, CUDA device and storage type, including weight-context size where
configurable. The cache belongs to one fixed-config loaded package and is
released with that package. OuteTTS uses weak entries with per-runtime owners:
when the last slot stops using a storage format, its weights are released,
preserving the existing single-slot memory behavior when changing voice routes.
There are no CUDA kernel or generic cache
implementation changes, and no precision, prompt or generation-limit changes.

After validation, the server CUDA admission table allows four slots for
these six tested family/task combinations. Vulkan capacities and CPU fallback
remain unchanged. Other checkpoints, tasks, options and larger pools are not
certified by these tests.

## Sampled peak GPU memory

All values below are **GiB**, including model weights, caches, workspaces and
CUDA context/pool overhead. The before/after three-slot runs use the same
requests and quantify the saving without comparing different slot counts.

| Model | Before, 3 slots | After, 3 slots | Peak reduction at 3 slots | After, 4 slots |
|---|---:|---:|---:|---:|
| Confucius4 TTS | 19.17 | 13.46 | 29.8% | 16.89 |
| Echo TTS | 18.03 | 13.32 | 26.1% | 16.70 |
| Fish Audio S2 Pro | 18.59 | 9.10 | 51.1% | 10.47 |
| MOSS VoiceGenerator | 22.32 | 15.88 | 28.9% | 20.05 |
| OuteTTS | 19.92 | 10.61 | 46.7% | 12.52 |
| SheetSage2 | 20.17 | 15.12 | 25.0% | 19.24 |

These are sampled peaks over complete cold/warm/mixed test sequences, not just
retained memory after a response. GPU tests run sequentially, with no other
model server workload on the RTX 3090. A per-test VRAM guard stays enabled and
did not terminate any baseline or candidate case in this pass. Sampling can
miss shorter transient peaks; fitting these fixtures does not guarantee that
longer inputs or different options fit.

## Validation

The fresh baseline server is saved before modifying source. For every family,
baseline one-slot requests are A/A/B/B/A; three-slot baseline waves are cold A,
warm A, mixed A/B, reversed mixed A/B, and A after the mixed waves. Candidate
one-slot outputs are compared with that baseline. Four-slot validation is
repeated on independent fresh servers, including one run of the final build;
the final build also repeats the three-slot waves.

Prompt A is `This is a test of two independent audio requests.` and prompt B is
`A second request should keep its own voice and seed.`, with seeds 1234 and 5678.
OuteTTS uses `The train is here.` and `The bus is here.`. SheetSage2 uses the
saved 16 kHz speech fixture for A and `assets/resources/a.wav` for B. Reference
audio is `assets/resources/a.wav`; models requiring a reference transcript use
`This little work was finished in the year eighteen o three, and intended for immediate publication.`
MOSS VoiceGenerator receives an explicit seed for this pass. Full configs and
request limits are recorded in the local JSON files.

- 467 normalized candidate/baseline outputs match their fresh serial
  references exactly. Comparison includes WAV SHA-256 and response data, or
  SheetSage2 text and score/event artifact hashes. Of those outputs, 240 are
  four-slot outputs across the two independent candidate passes, excluding
  reloads. All waves reach their configured active-slot count.
- Every four-slot candidate unloads successfully, reports the model unloaded,
  then reloads and reproduces its serial reference output. Active and queued
  leases drain after each sequence. Some CUDA context/pool memory can remain
  after model unload; test-server exit returns GPU use to zero.
- An additional OuteTTS cloned/default-voice route-switch sequence produces
  26 exact candidate outputs, including simultaneous mixed routes at
  four slots and a reload. This checks separate storage-type cache entries.
  The one-slot route-switch peak is 6.81 GiB, matching the original
  one-slot route-switch run; the four-slot mixed-route peak is 13.48 GiB.
- Vulkan compatibility controls for all six models compare a saved older
  single-slot Vulkan executable with current two-slot cold/warm/mixed waves.
  All 36 parallel responses match their corresponding serial references and
  every wave reaches two active slots. These controls do not raise Vulkan limits.
  The older Vulkan executable already contains earlier fixes; it is not claimed
  to be pristine upstream.
- CUDA and Vulkan CLI/server builds succeed. Eight focused CTests pass per
  backend: shared weight cache, session pool, model slots, single-slot
  compatibility, busy guard, server configuration, SheetSage audio frontend
  and SheetSage processing. Capacity assertions cover the six new CUDA limits.

The retained validation records contain 582 successful requests across the
baseline, candidate, route-switch and Vulkan control runs. Superseded diagnostic
passes are excluded from this count. Counts above overlap. Exact parity applies to these
fixtures, not a universal quality or determinism guarantee. CPU inference was
not retested in this pass; non-CUDA load paths keep private weights.

## Local evidence

The audit directory is `outputs/model-two-slot-audit/cuda-four-slot-vram-followup/`
outside the repository. It contains per-request WAVs/artifacts, configs,
normalized responses, baseline/candidate results, binary SHA-256 provenance,
the source diff, sampled memory, build logs, focused CTest logs, Vulkan controls
and `summary.json`. Harnesses and strict result assertions are in the parent
audit directory. These binary/audio artifacts are retained locally and are not
included in the source or documentation commits.

## Separate model commits

| Model | Commit |
|---|---|
| Confucius4 TTS | [`f90e7004`](https://github.com/0xShug0/audio.cpp/commit/f90e7004abb632e7de4304bd0c5ca823b321b828) |
| Echo TTS | [`3a03429d`](https://github.com/0xShug0/audio.cpp/commit/3a03429d5b119930291cd4a4451faa522a6b1e86) |
| Fish Audio | [`faaf0583`](https://github.com/0xShug0/audio.cpp/commit/faaf0583519816ed915bc93899dd01ef43549431) |
| MOSS VoiceGenerator | [`cb67d853`](https://github.com/0xShug0/audio.cpp/commit/cb67d853f9b94df09f2b7d94b3923d99389a0e68) |
| OuteTTS | [`478df6e4`](https://github.com/0xShug0/audio.cpp/commit/478df6e49936074e0d9764536359b363e1fc6365) |
| SheetSage2 | [`bca38bc0`](https://github.com/0xShug0/audio.cpp/commit/bca38bc09c3cb2859bf95d818065575b24d00b4e) |
