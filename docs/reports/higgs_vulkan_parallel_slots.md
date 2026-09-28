# Higgs Audio v3 parallel Vulkan slots

Local validation, 2026-09-27. Base source: `6dd717f3`. Windows/MSVC Release,
Vulkan SDK 1.4.350.0, NVIDIA driver 616.56, RTX 3090 24 GiB, eight CPU threads,
offline TTS.
Model: `higgs-audio-v3-tts.gguf` (5,095,340,672 bytes), with
`model_specs/higgs_audio_tts.json`. Reference audio: `assets/resources/b.wav`.

## Implementation

The existing Higgs factory rejected every non-CUDA backend. It now advertises
four slots for Vulkan offline sessions and retains its existing CUDA ceiling.
CPU and other unsupported combinations retain one slot. Each Vulkan slot owns
its backend context, command/graph execution state, KV cache, reference caches
and sampler. The factory shares the immutable AR and codec device weights.
The primary session outlives its clones through the existing session pool.

The runtime change is confined to Higgs's capability check and error message.
No Vulkan or CUDA kernels, numerical operations, generic model allowlist, or
server scheduler were changed. This enables concurrent sessions; it does not
batch tokens or promise simultaneous hardware execution of every GPU operation.

## Time and memory

Prompt: "Ptaszki ćwierkają, że Apple chce wrócić na rynek serwerów dostarczając
chłonnemu rynkowi AI swoje własne rozwiązania bazujące na zmodyfikowanych układach
z serii M." Seed 1234, `max_tokens=1024`, default reference caching.
One cold round followed by three warm rounds at each count.
Warm time is the median whole-batch HTTP time. Each round submits one request
per slot and verifies that the configured count is simultaneously active.

| Slots / concurrent requests | Cold batch (s) | Warm batch median (s) | Warm peak VRAM (GiB) | Peak across all rounds (GiB) |
|---|---:|---:|---:|---:|
| 1 | 2.496 | 2.244 | 5.285 | 5.285 |
| 2 | 4.730 | 4.327 | 6.063 | 6.063 |
| 3 | 6.934 | 6.462 | 6.132 | 6.176 |
| 4 | 9.387 | 8.695 | 6.553 | 6.656 |

VRAM includes model weights and runtime allocations above the sampled NVML idle
baseline (250 MiB). Loaded-model usage was 4.720 GiB at every slot count, which
is consistent with shared device weights. Memory is sampled, not an allocation
trace. Startup/model loading is excluded from the batch times.

Four times the single-slot warm median is 8.976 s, compared with 8.695 s for
four concurrent requests: approximately 3.1% lower batch time, or 3.2% higher
throughput. This sequential estimate is not a separately measured four-request
serial batch. The small timing difference should not be treated as a guaranteed
speedup; the clear benefits here are concurrent admission and sharing weights.
Independent server instances were not benchmarked in this Vulkan run.

All 40 WAVs in the benchmark match their corresponding cold/warm single-slot
reference bytes. The Polish hashes also match the earlier unmodified Vulkan
references recorded in [the original report](common_model_slots_vulkan.md).

## Mixed request histories

Four distinct requests use Polish and English, different seeds, a short prompt,
and a doubled English prompt with a 1536-token limit. Each slot runs five
requests, including rotations of those inputs. All 20 outputs match a separate
single-slot run with the same per-session request history.

An initial comparison against a generic warm reference failed because requests
can land on slots with different histories. Higgs may rebuild its KV cache after
a longer request, which changes whether the next prefill reuses reference KV
state. Cold and cached paths can already produce different seeded output in the
original implementation. The corrected harness controls slot admission order
and compares the complete histories; it does not relax byte equality or change
generation behavior. A fixed seed alone does not guarantee the same WAV across
arbitrary slot scheduling histories.

The mixed-request harness also passed six requests sharing four slots, a 503
busy timeout while all slots are occupied, unload waiting for all active slots,
parallel cold reload, invalid-request lease cleanup, and subsequent successful
generation. All compared WAVs remain byte-identical to the matching reference.

The separate single-slot regression harness also passed all ten inference/error
cases, queued execution, unload/reload, lazy-load management and failure recovery.
Its two unconditioned requests hit the same expected token-limit error before
and after the change. Five Vulkan slots reject cleanly with `capacity=4` and no
published pool or leaked lease. Legacy BS-RoFormer single-slot stems remain
byte-identical and its Vulkan capacity remains one.

## Vulkan validation layer

An additional one/four-slot cold/warm run used `VK_LAYER_KHRONOS_validation`,
with `thread_safety=true` and `validate_sync=true` in `vk_layer_settings.txt`.
The server logs confirm the layer was active, synchronization validation was
enabled and no validation checks were disabled. All ten WAVs matched; neither
server log reported a validation error, VUID violation, synchronization hazard
or threading error. This is runtime evidence for the tested paths, not a proof
of safety for every Vulkan driver or workload. Instrumented timings are excluded
from the benchmark table.

Validation-layer environment (settings file in the artifact directory):

```powershell
$env:VK_INSTANCE_LAYERS = 'VK_LAYER_KHRONOS_validation'
$env:VK_LAYER_PATH = 'C:\VulkanSDK\1.4.350.0\Bin'
$env:VK_LAYER_SETTINGS_PATH = (Resolve-Path ../outputs/higgs-vulkan-slots).Path
```

The file contains `khronos_validation.thread_safety = true`,
`khronos_validation.validate_sync = true`,
`khronos_validation.report_flags = error,warn,info`,
`khronos_validation.debug_action = VK_DBG_LAYER_ACTION_LOG_MSG`, and
`khronos_validation.log_filename = stdout`. The benchmark used `--slots 1 4
--iterations 1 --skip-instances` with output under `validation-layer/`.

## Reproduction and artifacts

Build with Vulkan enabled, then run the five focused session-pool, scheduler,
configuration, busy-guard and single-slot compatibility CTests. Both the Vulkan
and CUDA builds passed these five tests locally; both CLI and server rebuilt.
The existing full-suite Vulkan result (68/70, with two reproduced upstream
failures) is historical, not a new full-suite run for this change.

Identify the correct GPU with `--list-devices`. On this machine Vulkan device 1
is the RTX 3090; device 0 is the AMD integrated GPU. NVML benchmark monitoring
uses NVIDIA device 0, so this harness requires those selections to refer to the
same physical GPU.

```powershell
cmake --build build/windows-vulkan-release --target audiocpp_server audiocpp_cli session_pool_test server_model_slots_test server_busy_guard_test server_config_test server_single_slot_compat_test -j 12
ctest --test-dir build/windows-vulkan-release --output-on-failure -R '^(session_pool_test|server_model_slots_test|server_busy_guard_test|server_config_test|server_single_slot_compat_test)$'
python tests/higgs_audio_tts/server_slots_bench.py --server build/windows-vulkan-release/bin/audiocpp_server.exe --model ../models/higgs-audio-v3-tts.gguf --spec model_specs/higgs_audio_tts.json --request-json ../outputs/higgs-vulkan-slots/request.json --output ../outputs/higgs-vulkan-slots/benchmark --backend vulkan --device 1 --slots 1 2 3 4 --iterations 3 --skip-instances
python tests/higgs_audio_tts/server_parallel_compat.py --before-server build/windows-vulkan-release/bin/audiocpp_server_before_vulkan_slots.exe --after-server build/windows-vulkan-release/bin/audiocpp_server.exe --model ../models/higgs-audio-v3-tts.gguf --spec model_specs/higgs_audio_tts.json --request-json ../outputs/higgs-vulkan-slots/request.json --output-dir ../outputs/higgs-vulkan-slots/mixed-history-retry --backend vulkan --device 1 --slots 4
python tests/higgs_audio_tts/server_backend_compat.py --before-server build/windows-vulkan-release/bin/audiocpp_server_before_vulkan_slots.exe --after-server build/windows-vulkan-release/bin/audiocpp_server.exe --model ../models/higgs-audio-v3-tts.gguf --spec model_specs/higgs_audio_tts.json --reference assets/resources/b.wav --output-dir ../outputs/higgs-vulkan-slots/single-compat --backend vulkan --device 1 --expected-capacity 4 --separator-model ../models/bs-roformer-ep368-q8_0.gguf --separator-spec model_specs/bs_roformer.json --input-audio ../outputs/higgs-slots/common-framework/legacy-input.wav
```

Local artifacts are in the parent workspace's `outputs/higgs-vulkan-slots/`:
benchmark JSON, memory/status samples, all generated WAVs, model configurations,
server logs and build/test logs. The pre-change Vulkan executable was saved
before rebuilding; SHA-256:
`a52c13248c3876955b5f90ccc86f7ca0560f7f8e35dd2f4bef81a0826603b31f`.

Coverage is this model, backend, GPU/driver and these workloads. Other Vulkan
models, devices, streaming modes and more than four Higgs slots are not enabled
or certified by this experiment. Changes remain local and unpublished.
