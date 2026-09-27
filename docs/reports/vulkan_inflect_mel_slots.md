# Inflect V2 and Mel-Band RoFormer Vulkan slots

Follow-up to PR #706 commit `c622b5e0`, tested 2026-09-27 on Windows/MSVC
Release, RTX 3090 24 GiB, Vulkan device 1 and CUDA device 0, eight CPU threads.
The normal server now admits the tested offline Inflect V2 TTS and Mel-Band
RoFormer separation packages at up to four Vulkan slots. The Vulkan admission
table increases from 71 to 73 entries; the CUDA table and capacities are unchanged.

## Findings and fixes

**Inflect V2:** the saved original Vulkan executable asserts even with one
slot, at the Vulkan matmul buffer-alignment check. Relative-value attention
slices probability rows at arbitrary token-stride offsets and passes those
views as matmul inputs. Materialize these rows using the existing backend
layout helper on Vulkan only. CPU and CUDA keep their original views.
Disabling Vulkan fusion did not avoid the original assertion.

Inflect's duration and decoder graphs also explicitly request FP32 matmul
accumulation on Vulkan. This improves agreement with the original CPU/CUDA
waveforms in the measured cases. It is scoped to Inflect graphs; it does not
change backend kernels or precision policy for other models. Vulkan may still
convert operands to FP16 in its cooperative-matrix path, so requesting FP32
accumulation does not imply fully FP32 operands or exact cross-backend output.

**Mel-Band RoFormer:** mask accumulation parallelized over overlapping bands,
which wrote to the same frequency/frame cells without synchronization. This
host OpenMP data race also occurs with a single inference slot. Parallelize
over frames instead, giving each cell one writer while retaining the original
serial band-addition order and FP32 arithmetic. There are no atomic additions
or model-wide inference locks. This host fix applies to the shared RoFormer
runtime on all backends; BS-RoFormer controls remain exact.

The saved original Mel-Band executable with both server threads and
`OMP_NUM_THREADS` set to one supplies a valid reference. Setting only server
threads did not prevent the host race on HTTP workers. Both corrected normal
eight-thread CUDA and Vulkan outputs exactly match their respective original
single-thread references for two distinct input clips. Racy original
multi-thread output is not an exactness reference.

## Vulkan validation

| Model/package | Slots | Exact concurrent outputs | Sampled peak VRAM (MiB) | Warm batch wall time (s) |
|---|---:|---:|---:|---:|
| Inflect Micro V2 original | 1 | serial reference | 261 | 0.050 |
| Inflect Micro V2 original | 2 | 8/8 | 495 | 0.065 |
| Inflect Micro V2 original | 3 | 12/12 | 649 | 0.091 |
| Inflect Micro V2 original | 4 | 16/16 | 802 | 0.127 |
| Mel-Band RoFormer Q8_0 | 1 | serial reference | 2658 | 0.575 |
| Mel-Band RoFormer Q8_0 | 2 | 8/8 | 6525 | 1.005 |
| Mel-Band RoFormer Q8_0 | 3 | 12/12 | 9160 | 1.494 |
| Mel-Band RoFormer Q8_0 | 4 | 16/16 | 10568 | 1.907 |

Each count ran cold identical requests, warm identical requests, and two
mixed-input/seed waves in opposite orders: **72/72 concurrent responses exactly
matched updated single-slot references**, including WAV bytes, with all
requested slots observed active. Serial repetitions, serial requests after
concurrency, drained leases, explicit unload and lazy reload also passed.
Both models reject five-slot loads with capacity four before publishing a
loaded model or active lease. These changes cover offline tasks only.

VRAM is sampled device-wide with `nvidia-smi`, including weights, buffers and
driver/desktop use; it is not isolated cache memory. Sampling may miss brief
peaks. Timings are individual observations, not repeated performance
benchmarks, and each multi-slot batch includes all requests. Tested prompts
use a fixed reference voice with seeds 1234/5678; separation uses an original
clip and a cropped second input. Larger inputs, other checkpoints/options and
more than four slots are outside this validation.

## Quality and CUDA controls

The original Inflect Vulkan executable crashes, so it supplies no valid
Vulkan waveform for before/after quality comparison. The repaired Vulkan
implementation produces repeatable output and exact single/parallel parity.
It is **not byte-identical to CPU or CUDA**. On the two tested prompts, explicit
FP32 accumulation improves waveform agreement relative to original CPU:

| Prompt | Alignment-only Vulkan SNR versus CPU (dB) | Retained Vulkan SNR versus CPU (dB) | Retained Vulkan SNR versus CUDA (dB) |
|---|---:|---:|---:|
| First | 19.37 | 32.00 | 32.29 |
| Second | 27.82 | 30.84 | 31.82 |

These are numerical waveform comparisons, not perceptual-quality scores or a
claim of identical sound across backends. Sample rates, channel counts and
frame counts match. A native Vulkan transposed-convolution experiment did not
improve agreement and was discarded.

Final CUDA controls for both target models passed **16/16 exact two-slot
responses**, serial/state/reload checks, and **4/4 exact original-versus-updated
single-slot comparisons**. Original Mel-Band references run with host OpenMP
threads fixed to one to avoid the original race. Inflect's CPU/CUDA graph
precision and layouts remain unchanged. BS-RoFormer CUDA and Vulkan controls
each passed eight exact paired responses, state/reload checks and their saved
catalog baseline. No shared CUDA or Vulkan kernel is modified.

## Builds, tests and evidence

CUDA and Vulkan CLI/server Release builds passed. Ten focused CTests per
backend passed (**20/20**): slot admission/pool/scheduling, single-slot
compatibility, busy/configuration handling, mapped tensor-source concurrency,
immutable weights, overlapping mask accumulation, and backend-specific weight
lifetime, attention alignment/precision, CUDA ISTFT and FSQ reuse as applicable.
The new mask regression covers cancellation-sensitive overlapping bands with
1/2/8 OpenMP threads and two simultaneous callers. The Inflect regression runs
CPU and Vulkan attention rows at lengths 3/5/7/17, verifies layouts and
backend-specific accumulation policy, and checks repeated dot products.
This is focused validation, not a rerun of the full model suite.

Local evidence is under `outputs/model-two-slot-audit/`:

- `vulkan-inflect-mel-investigation/`: original assertions, output variability
  and numerical experiments.
- `vulkan-inflect-mel-final/inflect_v2/` and
  `vulkan-inflect-mel-final-higher/inflect_v2/`: retained 1–4-slot validation
  and cross-backend waveform metrics.
- `vulkan-inflect-mel-candidate-v3/mel_band_roformer/` and
  `vulkan-inflect-mel-higher/mel_band_roformer/`: exact original and 1–4-slot
  Mel-Band comparisons.
- `cuda-inflect-mel-final-controls/`: final CUDA original/updated controls.
- `cuda-inflect-mel-bs-controls/`, `vulkan-inflect-mel-bs-controls/`:
  BS-RoFormer controls.
- `vulkan-inflect-mel-capacity/`: five-slot rejection checks.

Raw WAVs, test configurations and large logs are local artifacts, not repository
files.
