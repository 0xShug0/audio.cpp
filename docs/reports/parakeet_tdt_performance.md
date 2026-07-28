# Parakeet-TDT 0.6B v3 — performance and validation report

Detailed measurements, methodology, and the reasoning behind what was and
wasn't optimized. The user-facing model documentation lives in
[docs/community_models/parakeet_tdt.md](../community_models/parakeet_tdt.md);
this file is the evidence behind the summary numbers there.

Everything here was measured on one machine, so read the hardware section
first — the CPU results generalize, the CUDA results do not.

## Test hardware

The CPU is an Intel
i7-9750H (6C/12T, Coffee Lake, AVX2) — an ordinary 6-core, so the CPU results
below should carry over reasonably to comparable AVX2 machines.

The GPU does **not** generalize and every CUDA number here should be read with
that in mind: it is a **GTX 1650 with Max-Q Design** — Turing, compute
capability 7.5, 4 GB, and a **35 W** power limit. That is a thermally- and
power-constrained mobile part, roughly the weakest CUDA device this model
realistically runs on, and its compute capability sits below the 8.0 (Ampere)
threshold at which ggml turns on CUDA graph capture at all (see the CUDA
graph section below). Several optimizations here measured as *no change* on
CUDA; that is a statement about this card, not about the optimization. They
are worth re-measuring on anything Ampere or newer before concluding
anything.

## Performance

Measured on the reference test clip (`2086-149220-0033.wav`, 7.435s), 5 timed
iterations + 1 warmup, against the real NeMo model (`nemo_toolkit[asr]`,
torch 2.13+cu130) on the same machine. `parakeet_warm_bench` reports a full
frontend/encoder/decoder timing breakdown via `--timing-file` (real timing —
this previously silently produced an empty log; see the commit fixing
`configure_logging` wiring).

| | CPU | CUDA |
|---|---|---|
| NeMo/PyTorch (Python) | 857.7 ms (RTF 0.115, 8.7x real-time) | OOM — see note below |
| audio.cpp, default settings | 1247.5 ms (RTF 0.168, 6.0x real-time) | 170.6 ms (RTF 0.023, 43.6x real-time) |
| audio.cpp, `matmul_weight_type=q8_0` + tuned threads | 830.9 ms (RTF 0.112, 9.0x real-time) | 132.1 ms (RTF 0.018, 56.3x real-time) |
| audio.cpp, above + conv-pointwise reclassification fix | **~750 ms** (RTF 0.101, **9.9x** real-time) | ~138 ms (RTF 0.019, 53.9x real-time, no clear change) |

That table is a **historical progression**, each row measured in the session
that produced it — which on this machine means each row carries its own
thermal state. Treat it as the story of how the numbers moved, not as figures
comparable row to row.

Re-measured end to end on the current tree with the drift-cancelling harness
(ABBA, 4 passes, median of 5 iterations per run, machine idle):

| config | CPU | CUDA |
|---|---|---|
| default (8 threads, f32) | ~1245 ms (6.0x real-time) | ~169 ms (44.0x) |
| tuned (12 threads, q8_0) | **~715 ms** (**10.4x**) | ~131 ms (56.8x) |
| ratio | **1.72x** | **1.31x** |

The default rows reproduce the historical figures closely (1247.5 / 170.6),
which is a useful check that the table above is still meaningful. The CPU
tuned row was previously quoted as ~750 ms from a pre-optimization session;
~715 ms is the current measurement.

Separately from the table above (which is all same-length, matched-capacity
work), the largest single CPU win in this model's history was not a kernel or
a precision change at all: it was **not running the graph oversized**. A 7.4s
clip on a session prepared with a 60s audio contract went from 10928 ms to
1140 ms — 9.6x — and stopped dropping a sentence while it was at it. See the
graph-capacity section below.

This machine got noisier (shared background load) partway through this
history, enough that later single-shot end-to-end numbers aren't reliably
comparable table-row-to-table-row — the fused-QKV and graph-optimizer
findings below use an interleaved A/B (alternate config every run) instead of
separate before/after sessions specifically to cancel that out; see each
section for the actual methodology and numbers.

```bash
build/<preset>/bin/parakeet_warm_bench \
    --model models/parakeet-tdt-0.6b-v3 --audio tests/parakeet_tdt/assets/2086-149220-0033.wav \
    --backend cpu --threads 12 \
    --session-option parakeet_tdt.matmul_weight_type=q8_0
```

Two things drove essentially the entire gap between the "default settings" row
and PyTorch, both found by actually looking at the timing breakdown instead of
just the end-to-end number (which was previously impossible — see above):

- **Thread count.** This CPU is 6 physical / 12 logical cores, and 12 threads
  is fastest: 3.1% faster than 6 and 1.1% faster than 8, measured with the
  drift-cancelling A/B described below. (An earlier revision of this document
  claimed the opposite — "6 is consistently fastest, 12 is consistently worst
  (contention)". That conclusion came from sequential before/after runs on a
  thermally drifting machine and did not survive a controlled comparison. The
  standalone GEMM microbenchmark agrees with the corrected result: 95.7
  GFLOP/s at 6 threads versus 138.6 at 12.) The end-to-end gain is much
  smaller than the GEMM gain because the non-GEMM ops are memory-bound and do
  not benefit from hyperthreads. Sweep it on your own hardware — the optimum
  is core-count-dependent, not a fixed number — but sweep it *interleaved*,
  not sequentially.
- **Weight precision.** `parakeet_tdt.matmul_weight_type` defaults to
  `native` (F32, since the checkpoint ships F32 weights) and is a session
  option, not a fixed choice — `native|f32|f16|bf16|q8_0` are all available
  (`parakeet_tdt.conv_weight_type` separately, `native|f32|f16`). ggml's CPU
  backend has heavily hand-tuned Q8_0 dot-product kernels (the common case
  for llama.cpp-style inference), and Q8_0 is far and away the biggest win —
  1.79x. BF16 is a smaller but real 1.14x; F16 is not a speedup at all and is
  actually *slower* than F32 single-core. So the gain tracks which kernels
  happen to be well optimized, not "fewer bits" as a general principle — don't
  assume an option helps without measuring it. Encoder graph compute alone
  accounts for ~93-96% of total wall time in every configuration measured;
  that's where quantization pays off, and frontend and decoder are already
  only a few percent of the total each. See the measured accuracy cost of
  each below before picking one.

## Weight precision: speed and the measured accuracy cost

**What quantization actually costs, measured across 24 languages.** The
earlier version of this section justified keeping `native` as the default on
the grounds that quantization had "only been validated against this one clip".
That has now been done properly, with the
[multilingual harness](../../tests/parakeet_tdt/multilingual/README.md): 120
FLEURS clips, 5 per language, across the 24 European languages the model
supports that FLEURS covers, 21.8 minutes of audio. Every transcript from that
run — reference plus all four weight types, per clip — is checked in under
[`multilingual/results/`](../../tests/parakeet_tdt/multilingual/results/README.md),
so these numbers can be re-derived (or re-scored with a different WER
normalization) without re-running anything.

| weight type | encode speed | transcript identical to f32 | WER vs f32 | absolute WER |
|---|---|---|---|---|
| `native` (f32) | 1.00x | 100% | — | 0.1338 |
| `f16` | ~1.00x | 99.2% | 0.0003 | 0.1336 |
| `bf16` | **1.14x** | 99.2% | 0.0004 | 0.1334 |
| `q8_0` | **1.79x** | 91.7% | 0.0062 | 0.1331 |

The headline is that **q8_0's transcription churn is not the same thing as
quality loss.** It changes 8.3% of transcripts, but absolute WER against the
human reference is flat — 0.1331 vs f32's 0.1338, a difference far inside the
noise of a 120-clip sample. Inspecting the diffs shows why: many are cosmetic
(`T Rex` → `T-Rex`, `80 km/50 mil` → `80 km 50 mil`), and where real words
change, q8_0 is sometimes *right* where f32 was wrong (Hungarian
`ez a helys turista` → `ez a hely sok turista`, which is the correct reading).
Per-language it moves both ways: Estonian gets worse (0.164 → 0.184), Hungarian
(0.140 → 0.117) and Slovenian (0.299 → 0.284) get better. With 5 clips per
language those per-language numbers are individually noisy; treat the aggregate
as the real signal and the per-language column as a check that no single
language falls off a cliff. None does.

So the honest guidance is:

- **`bf16` is close to free**: 1.14x faster, 99.2% of transcripts byte-identical
  to f32. If you want a speedup without thinking about it, take this one.
- **`q8_0` is the real speedup**: 1.79x, with no measurable aggregate quality
  cost on this corpus, but ~8% of transcripts will differ textually from an f32
  run. That matters if you are diffing output against a stored f32 baseline; it
  mostly does not matter if you care about what the text says.
- **`f16` is pointless here**: same accuracy as bf16, no speedup (it is actually
  slower than f32 single-core — 25.5 vs 27.8 GFLOP/s). ggml's win comes from
  specific hand-tuned kernels, not from "fewer bits" as a general principle.

`native` remains the default because it is the only setting that reproduces
exactly, and because 120 clips at 5 per language is enough to rule out a cliff
but not enough to certify a default for every language and domain. Turn the
others on explicitly via the session option above.

### Safetensors, GGUF, and what was measured

The timing and multilingual tables above use the upstream F32 safetensors
package and select the in-memory storage type with
`parakeet_tdt.matmul_weight_type`. GGUF is a standalone package format, not a
separate runtime: both source formats execute through the same ggml graphs.

Path tests were also run against standalone original-F32, F16, BF16, and Q8_0
GGUF files with the schema, tokenizer, and configs embedded. Their measured
sizes were approximately 2.4 GB, 1.2 GB, 1.2 GB, and 874 MB respectively. All
four passed full-context, long-form, timestamp, and buffered-streaming
regressions while using the type stored in the GGUF. Those checks establish
format support and output stability on the fixture; they are not a separate
GGUF startup or throughput benchmark, and the 120-clip corpus was not rerun
from each GGUF.

## Where the CPU time actually goes

Profiling (`perf record`) the encoder on
this machine attributes ~74% of cycles to a single symbol: llamafile's
`tinyBLAS::gemm_bloc` f32 kernel, plus ~10% to `ggml_vec_dot_f32` and ~6% to
OpenMP barriers. It is overwhelmingly GEMM-bound, which bounds what any
graph-level change can win — the view/scale/copy eliminations documented above
are individually bit-exact and jointly worth only ~0.5% at 6 threads (~2% at 1
thread, where there is no parallelism to hide the memory work).

A standalone GEMM microbenchmark over the exact shapes this encoder issues
gives the ceiling:

| | 1 thread | 6 threads |
|---|---|---|
| f32 (`k=1024, m=4096, n=93`) | 27.8 GFLOP/s | 128.4 GFLOP/s |
| f16 | 25.5 | 128.6 |
| bf16 | 29.5 | 145.1 |
| q8_0 | 50.1 | 278.3 |

Two conclusions worth recording, both of which killed an optimization that
looked obvious beforehand:

- **The single-core GEMM is not memory-bound.** Sweeping the weight-matrix
  footprint from 0.25 MB (fits L2) to 32 MB (far past the 12 MB L3) leaves
  throughput flat at 28–31 GFLOP/s. So it is kernel-bound at roughly 22% of
  this core's AVX2 FMA peak, inside vendored llamafile code — not something
  reachable from this model's own files. Shrinking the weights helps only
  insofar as a *different, faster kernel* gets selected (which is exactly why
  q8_0 wins and f16 does not).
- **Padding the sequence to clear tinyBLAS's fast-path guards does not pay
  for itself.** All three attention matmuls miss the fast path at `T=93`,
  because tinyBLAS bails when `k % 8 != 0` or `m % 4 != 0`: AC has `m=93`,
  BD has `m=185`, AV has `k=93`. That is exactly the ~10% of cycles sitting
  in the `ggml_vec_dot_f32` fallback. Padding does fix it — with keys padded
  to 96 and the position axis to 192, per layer:

  | | 1 thread | 6 threads |
  |---|---|---|
  | AC `q·kᵀ` | 1.240 → 0.699 ms (1.8x) | 0.253 → 0.206 ms (1.2x) |
  | BD `q·pᵀ` | 2.354 → 1.340 ms (1.8x) | 0.499 → 0.399 ms (1.2x) |
  | AV `a·v` | 3.002 → 0.627 ms (4.8x) | 0.603 → 0.202 ms (3.0x) |
  | total ×24 layers | 158 → 64 ms | 32.5 → 19.4 ms |

  but the saving is only ~94 ms of ~4800 ms at 1 thread and ~13 ms of
  ~1020 ms at 6, and it has to be paid for. Padding the *token* axis makes
  every weight GEMM ~6% slower (~233 ms spent to save ~91 ms — a clear loss).
  Padding only the *key* axis avoids that, but still widens the QKV
  projection to `n=96`, which costs ~6% of that GEMM (~7.6 ms at 6 threads,
  ~40 ms at 1). Net: roughly +0.5% at 6 threads and +1% at 1. Not worth
  reworking the relative-position attention path for, particularly since
  changing AV's `k` changes its accumulation blocking and so gives up the
  bit-exactness every other change here maintains. Revisit if the fallback
  ever grows (longer sequences, or a machine where the fast/slow kernel gap
  is wider than the ~1.7x measured here).

## Graph capacity

`ensure_graph()` used to reuse any cached graph whose capacity merely *exceeded*
the request, and `encode()` zero-pads the input up to that capacity — so the
graph always runs at its built size regardless of how short the real audio is.
A session prepared with a long audio contract therefore paid the long price on
every short clip that followed:

| 7.4s clip, graph built for | encoder compute |
|---|---|
| 7.4s (matched) | 1018 ms |
| 60s (oversized) | 10928 ms — **10.7x** |
| 60s contract, after the fix | 1140 ms |

Worse, it was also an *accuracy* bug. Zero input does not stay zero: every
subsampling conv carries a bias, so the padded tail arrives at the encoder
stack as nonzero garbage. The attention mask was unconditionally all-zeros, so
real frames attended to that garbage, and because softmax normalizes across
keys it rescaled every real frame's attention. On the reference clip fed to a
60s-capacity graph this silently dropped an entire sentence:

```
matched    "...turning away her eyes. It is certainly very like the old portrait."
oversized  "...turning away her eyes."
```

Both are fixed: attention now masks padded key columns (by column only, never
by query row, so no softmax row degenerates to all `-inf` and yields NaN), and
`ensure_graph()` rebuilds rather than reusing a graph more than 10% oversized.
Rebuilding costs ~400 ms — dominated by recomputing the 24 per-layer positional
projections, the allocation itself is ~0.4 ms — so it wins outright on the
first call and by more on every call after. Note this means `prepare()`'s
capacity is a *memory* reservation, not a promise that no rebuild happens.

## The ggml graph optimizer

`src/framework/runtime/graph_optimizer.cpp` implements a real, unit-tested
graph rewrite pass (`engine::runtime::optimize_graph`) that folds broadcast
`ggml_repeat` nodes into the consuming op and elides pure metadata and no-op
nodes. Before the Parakeet work, grepping the codebase found **zero production
callers**: it was built and unit-tested (`encoder_module_test`) but unused by
model execution. Parakeet now calls it from `ensure_graph()` (`encoder.cpp`)
once per distinct input
length right after `ggml_build_forward_expand` and before `ggml_gallocr_alloc_graph`
— free to call since the graph is cached and reused across every subsequent
`encode()` at that length. Effect on the reference clip's encoder graph:
**2854 nodes → 1425** (1323 metadata-only ops elided, mostly redundant
reshape/view/cont chains from the hand-rolled attention permutes, plus 106
broadcast repeats folded). Validated via the parity harness (`enc_out` cosine
0.972258, bit-identical to before) and the golden-transcription test (exact
match, both backends). Speed, measured with an interleaved A/B via the
optimizer's existing `ENGINE_GRAPH_OPTIMIZER=0/1` env toggle (six runs each,
alternating, to cancel out this machine's background-load drift): CPU
q8_0 encoder time **~812ms → ~726ms, a real ~10% reduction**, ahead in 5 of 6
paired runs; CUDA: no measurable difference, expected, since GPU options
intentionally skip metadata-only elision (a scheduler-based backend still
needs those reshape/view nodes) and only broadcast-repeat folding applies
there, which touched a small fraction of this graph. The CPU win comes from
actually reducing scheduling/dispatch work, not FLOPs — consistent with the
whole encoder graph being just ~1400-2800 nodes for a 93-frame clip, small
enough that per-node overhead is a real fraction of wall time on CPU.

## Conv-pointwise weight misclassification

**Conv-pointwise weight misclassification (found by cross-referencing
[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR), a
whisper.cpp-derived multi-model ASR project with its own FastConformer/Parakeet
port).** Their `PERFORMANCE.md` documents a bug in their quantizer: the
Conformer conv module's `pointwise_conv1`/`pointwise_conv2` weights are
kernel_size=1 Conv1d — mathematically a Linear layer, and NeMo/most ports
(including this one) run them via a matmul, not a real conv op — but their
quantizer classified them as "conv" weights and left them at F16 even when
targeting Q8_0, costing ~35% of encoder time on their ARM CPU target
(~14% on x86, where OpenBLAS's F16 GEMM path was already reasonably fast).
Checking our own loader (`weights.cpp`'s `load_encoder_layer`) found the
identical classification bug: `conv_pw1`/`conv_pw2` were loaded under
`conv_weight_type` (capped at f16, no q8_0) instead of `matmul_weight_type`,
even though `build_fastconformer_conv_module` in `encoder.cpp` runs them
through `LinearModule` (`mul_mat`), exactly like every other weight bucketed
under `matmul_weight_type`. Reclassified to `matmul_weight_type` (they now
quantize to Q8_0 when that option is set, same as before when it isn't —
zero behavior change at the `native` default). Effect, on top of the
already-applied `q8_0` + thread tuning above: **~10% additional CPU wall-time
reduction** (830.9 ms → ~750 ms); no clear CUDA change (cuBLAS was already
handling these small matmuls efficiently regardless of storage bucket).
Verified via the numerical parity harness: `layer_0` cosine 0.999991,
`enc_out` cosine 0.967974 (was 0.968028 pre-fix, i.e. unchanged within noise
— quantizing more of the *same class* of already-quantized weights doesn't
meaningfully add new error here) and via the golden-transcription test,
which still matches exactly with `matmul_weight_type=q8_0` on both backends.

## Fused QKV projection

**Fused QKV projection.** CrispASR's FastConformer notes also credit a fused
Q/K/V projection (one matmul instead of three per layer) as part of their
combined encoder win, and their CUDA default deliberately uses *manual*
(non-flash) attention because `flash_attn_ext` rejects their per-head
relative-position mask on CUDA and silently falls back to CPU for all layers
— independent corroboration of this port's own flash-attention finding above.
Implemented here: `weights.cpp`'s `load_encoder_layer` now reads the raw F32
rows for `q_proj`/`k_proj`/`v_proj` and concatenates them into a single
`[3*hidden, hidden]` weight at load time (row-major `[out_features,
in_features]` layout — the three already-flat `[hidden, hidden]` row-major
buffers concatenate directly into the fused layout with no interleaving
needed), populating the `qkv_weight` field of the shared `AttentionWeights`
struct (previously dead — no model in this repo, including `nemotron_asr`
which uses the same shared relative-attention module, actually populated it).
`build_encoder_layer` in `encoder.cpp` now does one `Linear(hidden, 3*hidden)`
matmul and slices the result into Q/K/V instead of three separate
`Linear(hidden, hidden)` calls. This is a pure reorganization, not a
precision trade — validated via the parity harness to be **numerically
identical** to the pre-fusion path at both `native` (cosine 0.972258, exact
match) and `q8_0` (cosine 0.967974, exact match) precision, and the
golden-transcription test still matches exactly on both backends. Measured
effect on wall time: **no clear change** on this hardware (CPU and CUDA, both
within run-to-run noise) — dispatch/kernel-launch overhead isn't the
bottleneck at this graph size on this machine, so the three-matmul and
one-matmul forms cost about the same here. Kept anyway: it's free (zero
accuracy cost, fewer graph nodes, simpler code), and it removes per-layer
dispatch work that this benchmark happens not to be sensitive to — a single
clip, one at a time, on one machine. Backends or workloads where launch
overhead does matter (batched or concurrent serving, GPUs where graph replay
is available) would see it; this one doesn't. Cheap to carry once validated
correct, even without a local win to show for it today.

## Flash attention: tried, measured slower, kept opt-in

The FastConformer encoder's relative-position self-attention (Transformer-XL
style "AC"/"BD" score terms) can be fused into a single `ggml_flash_attn_ext_with_bias_mask`
op instead of a separate QK^T matmul + additive bias + `ggml_soft_max_ext` +
AV matmul — this is exactly the "dense additive attention bias" case that op
was built for (see `common_relative_attention.cpp`'s `use_specialized_flash_attention`
path, which already uses it, unused by any production model in this repo before
this). It was wired in here as `parakeet_tdt.perf_mode=flash_attention` and
validated correct via the numerical parity harness (`enc_out` cosine 0.972325
vs. 0.972258 for the non-flash path — no measurable accuracy difference).
But measured end to end, on this hardware, it was consistently a few percent
*slower*, not faster, on both CPU and CUDA, on both the 7.4s reference clip and
a synthetic 59.5s clip (ruling out "too short a sequence to matter"). Plausible
reason: this encoder's attention sequences are short (well under 1000 frames)
and `head_dim=128` isn't necessarily in the sweet spot of ggml's flash-attention
kernels on a 35 W Turing-generation mobile card (GTX 1650 Max-Q); the existing softmax+matmul path
is already efficient at this scale. Left in as an opt-in for anyone testing on
different hardware (newer GPU generations in particular) where the tradeoff
might flip, but it is not recommended and not the default based on what was
actually measured here.

## CUDA: why nothing here showed a dispatch-overhead win

**CUDA graph capture is enabled by the build but architecturally unavailable
on this test GPU.** Traced with `nsys profile --trace=cuda` against
`parakeet_warm_bench`:
16,068 individual `cudaLaunchKernel` calls and 4,265 `cudaStreamSynchronize`
calls (47% of total CUDA API time) across 6 encoder passes plus the full
per-token TDT decode loop — real per-launch/per-sync overhead that CUDA graph
capture-and-replay exists specifically to eliminate. `cuda_api_sum` shows
**zero** `cudaGraphLaunch`/`cudaGraphInstantiate` calls anywhere, even though
`GGML_CUDA_GRAPHS=ON` in this build (`ENGINE_DEFAULT_ENABLE_CUDA_GRAPHS` in
the top-level `CMakeLists.txt`) and the encoder/decoder graphs already satisfy the
capture prerequisite (same cached `ggml_cgraph*` reused across every call at
a given shape — see `ensure_graph()`/`ensure_step_graph()`/`ensure_joint_graph()`,
all check-and-return-early on cache hit, no rebuild per inference or per
token). Root cause, in `ggml_cuda_graph_set_enabled`
(`external/ggml/src/ggml-cuda/ggml-cuda.cu`): ggml unconditionally disables
CUDA graph capture below `GGML_CUDA_CC_AMPERE` (compute capability 8.0).
This test machine's GTX 1650 Max-Q is Turing, compute capability 7.5 — one
generation short. This is not a bug or a missing setting in this codebase;
it's ggml's own hardware gate, and there is nothing to change here — the
build already does everything right. It does, however, retroactively explain
why flash attention and fused-QKV (both above) measured no CUDA benefit: they
reduce exactly the per-op dispatch/launch overhead that graph replay would
otherwise hide, so on hardware where replay is active (Ampere or newer —
RTX 30/40/50-series, A100/H100, etc.) those two changes might show a real
CUDA win where they showed none here. Worth re-measuring on such hardware
rather than assuming this test machine's null result generalizes.

**Why CUDA has no PyTorch comparison point.** `nemo_asr` on this GPU hits
`torch.OutOfMemoryError` just loading the model — this 4GB card's VRAM
budget is entirely consumed by PyTorch's own overhead on top of the weights.
The C++/ggml path runs comfortably in that same budget. This isn't really a
"we're faster" comparison so much as "the reference implementation doesn't
run here at all" — worth knowing if you're targeting small/consumer GPUs
rather than datacenter cards.

## Backends and memory

Both backends were re-validated after the CPU optimization pass, on the merged
tree.

| | CPU (12 threads) | CUDA (GTX 1650 Max-Q) |
|---|---|---|
| encoder compute, 7.4s clip | ~1140 ms | ~124 ms |
| encoder compute, 60s clip | — | ~940 ms |
| peak host RSS, f32 | 3843 MiB | 1818 MiB |
| peak host RSS, q8_0 | 1078 MiB | — |
| peak VRAM | — | 2642 MiB of 4096 |

Notes:

- **q8_0 cuts host RSS by 3.6x** (3843 → 1078 MiB), which is a bigger practical
  difference than its 1.79x speed win on memory-constrained machines.
- **VRAM fits comfortably in 4 GB**, with headroom. Worth contrasting with the
  reference implementation: `nemo_asr` cannot load this model on this same card
  at all — PyTorch's own overhead on top of the weights exhausts the 4 GB
  budget before inference starts.
- The graph right-sizing fix applies on CUDA too, and by a similar factor: a
  7.4s clip on a 60s-capacity graph would run the full 60s cost (~940 ms) and
  now runs at ~126 ms, about **7.5x**, with the transcription corrected in the
  same way as on CPU.

**Parity against NeMo, re-verified after the optimization pass.** The
reference dump was regenerated from scratch (torch 2.13.0+cu130, nemo 2.7.3)
and all three stages pass on both backends:

| stage | CPU | CUDA | threshold |
|---|---|---|---|
| `mel_features` | 1.000000 | 1.000000 | >= 0.999 |
| `layer_0` | 1.000000 | 1.000000 | >= 0.999 |
| `enc_out` | 0.972258 | 0.972258 | >= 0.97 |

`enc_out` cosine is **identical to the value measured before any of the
optimizations** in this report, which is the strongest available statement
that they changed no math: correctness here is confirmed directly against
NeMo rather than inherited from an internal baseline.

**CPU and CUDA agree numerically.** Dumping `enc_out` on each backend through
the parity harness (`--backend cpu` / `--backend cuda`) gives cosine
**0.99999988**, relative RMS 3.5e-06, max absolute difference 2.6e-06, with
bit-identical mel features. That is float32 accumulation-order noise between
different kernel implementations, and it is the check that confirms the
view-based attention rewrite — verified bit-exact on CPU, where operands get
materialized — is also correct under CUDA's kernels.

## Benchmark methodology

Wall-clock numbers here move
by 20–30% between a cold and a thermally saturated run — the same binary
measured 979 ms cold and 1264 ms at steady state within a single session, and
background load moves it further. Any A/B run as "config A for a while, then
config B" is therefore meaningless at the few-percent scale. The comparisons
above use [`tests/parakeet_tdt/bench/ab.sh`](../../tests/parakeet_tdt/bench/README.md):
a discarded burn-in run to reach steady state,
then ABBA ordering per pass scored by the *mean* of each side's two slots, so
linear drift cancels exactly (slots 1&4 vs 2&3 share a midpoint). Scoring by
`min()` instead — which an earlier iteration of this harness did — silently
hands the win to whichever binary occupied the coldest slot, and produced a
confident 5–8% "regression" that reversed under the corrected estimator.
Retired instruction counts (`perf stat`) are reproducible to ~0.01% here and
are a good cross-check, but they are insensitive to memory-movement changes
and so cannot be the only metric.
