# Vulkan connection-reset follow-up

Follow-up to PR #706 commit `44d0be78`, Windows/MSVC Release, RTX 3090
24 GiB, Vulkan device 1, eight CPU threads, 2026-09-27. Changes are organized
into separate model admission commits and a validation report for PR #706.

## Finding and retained changes

GLM TTS, OuteTTS, SheetSage2 and Stable Audio now pass two-slot execution on
the current regular server. This investigation adds their tested offline
family/task pairs to the Vulkan admission table at capacity two, taking that
table from 67 to 71 entries, and tests the policy. The CUDA table remains at
89 entries with unchanged capacities.

No additional model inference, precision, Vulkan kernel or synchronization
change was needed for these four models. The earlier shared tensor-mapping
protection (`0448deb1`) already applies to Vulkan as well as CUDA, guarding
mapped storage during copying, synchronous upload and release.

The preserved older audit executable reproduces the historical failures:
GLM TTS, SheetSage2 and Stable Audio each crashed during a fresh concurrent
cold-start pair. OuteTTS crashed in two of five fresh pairs; three succeeded.
All observed crashes exited with Windows `0xC0000005` (access violation), and
none triggered the VRAM guard. Their logs did not report a Vulkan allocation
failure. Thus OuteTTS's historical failure is intermittent, and these resets
must not be classified as confirmed out-of-memory failures.

These comparisons establish that the old failures reproduce and the current
code passes the tested sequences. They are consistent with the earlier
shared-loading race fix; the old executable contains other earlier differences,
so this follow-up does not claim to isolate the exact crashing instruction.

## Regular-server validation

| Model | Task | Exact paired outputs | One-slot peak (MiB) | Two-slot peak (MiB) | Warm single (s) | Warm pair wall time (s) |
|---|---|---:|---:|---:|---:|---:|
| GLM TTS | tts | 8/8 | 6762 | 12137 | 0.985 | 1.671 |
| OuteTTS | tts | 8/8 | 3863 | 7529 | 0.758 | 1.255 |
| SheetSage2 | midi | 8/8 | 10077 | 17654 | 2.095 | 3.615 |
| Stable Audio | gen | 8/8 | 3575 | 6996 | 4.961 | 7.520 |

Memory is sampled device-wide with `nvidia-smi`, including weights, runtime
buffers and driver/desktop usage; it is not isolated per-slot cache usage.
Sampling can miss brief peaks. No retained run reached the configured guard
(total GPU memory minus 512 MiB). Timings are individual observations, not
statistical performance benchmarks; the pair column covers both requests.

For each model, the saved pre-admission-change regular one-slot executable
and the updated one-slot server ran two different requests: **8/8 comparisons
were exact**, including WAV bytes and MIDI/artifacts. Each current catalog
request also matched its original Vulkan survey hash. Seeds were fixed at
1234 and 5678 where applicable; TTS/music used different prompts, and SheetSage2
used the original audio and a cropped second input. Reference voices were fixed.

Fresh updated two-slot servers then ran simultaneous cold requests, warm
identical requests, and mixed requests in both orders: **32/32 outputs exactly
matched their updated single-slot references**, with two active slots observed
in every wave. Serial repetitions before and after concurrency, drained leases,
explicit unload and lazy reload all passed. No generation limits were shortened.

Because the old OuteTTS crash was intermittent, four additional fresh updated
two-slot servers ran cold-start pairs. All eight responses returned HTTP 200,
matched the catalog reference exactly, and avoided the memory guard: **40/40
exact paired outputs across the retained primary and extra cold-start checks**.

The regular Vulkan server rejected three-slot loads for all four targets with
the expected capacity-two error (4/4); no model or lease remained published.
Three/four slots, other checkpoints/tasks, longer inputs and different reference
voices remain outside this validation. This admission change covers offline
tasks only; it does not enable streaming.

## Builds, tests and evidence

CUDA and Vulkan CLI/server Release builds passed. Nine focused CUDA CTests
and eight focused Vulkan CTests passed (17/17): slot policy/pool, scheduler,
single-slot compatibility, busy/configuration handling, mapped tensor-source
concurrency, immutable weight ownership, Vulkan weight lifetime, deterministic
CUDA ISTFT and FSQ codec reuse, as applicable to each backend. This task changes
no CUDA model inference or CUDA admission; it did not rerun a full CUDA model sweep.

Raw configurations, requests, original/updated outputs, server logs and
measurements are under the workspace's `outputs/model-two-slot-audit/`:

- `vulkan-resets-candidate-v1/`: four-model primary parity/reload validation.
- `vulkan-resets-legacy-recheck/`: preserved older executable diagnostics.
- `vulkan-resets-legacy-outetts-repeats/`: four further old OuteTTS cold starts.
- `vulkan-resets-current-outetts-repeats/`: four updated OuteTTS cold starts.
- `vulkan-resets-capacity/`: production rejection of unvalidated three-slot loads.

These large local artifacts and diagnostic scripts are not repository files.
