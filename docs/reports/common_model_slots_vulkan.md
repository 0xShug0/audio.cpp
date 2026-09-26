# Vulkan compatibility validation for common model slots

Date: 2026-09-26. Slots source: `0da5b26b`, upstream reference: `ab862077`.
Windows/MSVC Release, Vulkan SDK 1.4.350.0, RTX 3090 24 GiB, eight CPU threads.
The reference includes both merged Higgs CUDA and Vulkan prefill memory fixes.

## Scope

The common framework compiles with Vulkan, but the Higgs adapter currently
advertises **one slot on Vulkan**. CUDA offline is the enabled parallel adapter.
This validates default single-slot Vulkan compatibility, queued requests and
clean rejection of unsupported multiple slots. It does not enable or establish
the safety of two through four concurrent Vulkan sessions.

## Build and unit tests

- Updated Vulkan build with the full model set, CLI, server, GGUF tool, default
  tests and extended tests: successful.
- Upstream Vulkan server with Higgs and BS-RoFormer: successful.
- Complete local Vulkan CTest suite: **68/70 passed**. All five slots/lifecycle
  tests passed (pool, scheduler, single-slot compatibility, busy guard, config).

Two extended tests fail on both the PR and unmodified upstream:

| Test | PR result | Upstream result |
|---|---|---|
| `gtcrn_utility_test` | GTCRN DNS3 frame case 0: max difference 0.00261319, tolerance 0.002 | Same case, difference and numerical values |
| `http_live_body_test` | Windows process exits with `0xc0000409` | Same exit code |

The GTCRN mean difference is 0.0000441502; the failing maximum is at index 110,
expected -0.0777373, actual -0.0803505. The HTTP framing test links `http.cpp`
rather than the server runtime/session pool. Neither its source nor GTCRN or
shared Vulkan operations change in the slots PR. The existing failures were
recorded without changing numerical tolerances or suppressing tests.

Vulkan device 0 on this machine is the AMD integrated GPU; device 1 is the
RTX 3090. The first unfiltered suite also passed 68/70, with the GTCRN failure on
AMD DNS3 case 1. The final suite explicitly selected the 3090 using
`GGML_VK_VISIBLE_DEVICES=1`, which presents it as Vulkan device 0 to unit tests.
The server comparisons used the original device list and explicit `device: 1`.

At the end of this validation, all nine GitHub checks for source revision
`0da5b26b` passed, including Linux Vulkan and Nix Vulkan. Those checks do not
replace local real-model inference or cover every GPU/driver. Adding this report
and harness triggers another CI run without changing C++ runtime behavior.

## Real inference and lifecycle checks

`server_backend_compat.py` ran upstream and updated executables on the same
Vulkan device with the same model, input, reference audio, request options and
cache-state sequence. It compared WAV bytes and error responses directly.

Higgs uses `higgs-audio-v3-tts.gguf`, the bundled `b.wav` reference/transcript,
and the Polish Apple sentence from the [CUDA benchmark](common_model_slots.md).
English requests use the diffusion-transformer/MLLM/VAE sentence supplied for
previous tests. Seed and token limits are fixed per case.

| Check | Result |
|---|---|
| Ten sequential cases: Polish cold/warm, English, cache shrinking/growing, unconditioned requests, restored reference, long text | Eight generated WAVs byte-identical; two token-limit errors identical |
| Four simultaneous requests to one Vulkan slot | One active request, three queued; all four WAVs match upstream sequential requests |
| Short busy timeout while the slot is occupied | HTTP 503; subsequent queued work completes |
| Unload during generation and subsequent reload | Waits for inference; warm and reloaded cold audio unchanged |
| Legacy BS-RoFormer default slot and idle model eviction | Vocals/instrumental byte-identical; status reports one-slot capability |
| Higgs configured with two Vulkan slots | Explicit capacity=1 failure; no published pool or leaked lease |
| Inference/load failure recovery | Later Higgs request succeeds and reproduces the upstream cold WAV |
| Targeted and all-model unload during first lazy load | Both wait for loading and inference, then clear loaded state/capability |

In total, **17 updated Higgs WAVs and two BS-RoFormer stems** match upstream
Vulkan reference bytes. The BS-RoFormer input is a three-second 44.1 kHz stereo
clip converted from the reference audio. Models use their local spec overrides.

The Polish single-slot reference SHA-256 values are:

| Cache state | SHA-256 |
|---|---|
| Cold | `27a8169d67ae19badaf8ea7d0d8406854877639f838127ba932095b65d710007` |
| Warm | `590a16c139590aed8b240ddf61207b87f81b077b032d7b7d6c1b6800d83b9f29` |

These Vulkan outputs differ from CUDA outputs in both upstream and the PR.
Backend numerical differences can change autoregressive sampling. Parity here
compares the same backend and cache state; it is not a cross-backend identity
claim. First-request timing also includes driver/pipeline compilation, so the
before/after sequence is not used to claim a Vulkan speedup.

## Reproduction

Build the upstream and updated executables with Vulkan enabled and CUDA disabled.
Use `--list-devices` to identify the Vulkan device; do not assume its index is
the CUDA device index. On this machine:

```powershell
$env:GGML_VK_VISIBLE_DEVICES = '1'
ctest --test-dir build/windows-vulkan-release --output-on-failure --parallel 4
```

Use a fresh shell (without that device filter) for the explicit device-1 server
comparison. Substitute paths to the executable and input files:

```sh
python tests/higgs_audio_tts/server_backend_compat.py --before-server ../audio.cpp/build/windows-vulkan-release/bin/audiocpp_server.exe --after-server build/windows-vulkan-release/bin/audiocpp_server.exe --backend vulkan --device 1 --model models/higgs-audio-v3-tts.gguf --spec model_specs/higgs_audio_tts.json --reference assets/resources/b.wav --separator-model models/bs-roformer-ep368-q8_0.gguf --separator-spec model_specs/bs_roformer.json --input-audio legacy-input.wav --output-dir vulkan-compat
```

The harness saves configs, logs, per-request JSON, WAVs and the result summary,
and terminates its servers even on failure. GGUF-backed checks are manual and
require the model files; they are not automatically run by the unit-test suite.
Other model families were compiled but were not exercised with real checkpoints.
