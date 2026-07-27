# Parakeet-TDT 0.6B v3

FastConformer-TDT ASR port of NVIDIA's [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
(0.6B params, 25 European languages with auto language detection), loaded directly
from the repo's Transformers-compatible `model.safetensors` checkpoint (not the
`.nemo` archive).

## Status

Offline (full-context) transcription only — see [Known limitations](#known-limitations)
for streaming. Verified correct end to end on CPU and CUDA against the real NeMo
reference model, both by final transcription and by numerical (per-layer
activation) comparison; see [Validation](#validation).

## Architecture

```
Frontend: 16kHz -> 128 mel bins, preemphasis=0.97, NeMo per-feature normalization
Encoder:  3-stage Conv2D subsampling (8x) -> 24-layer FastConformer -> 1024-dim
Decoder:  2-layer LSTM predictor + joint network (TDT: 5 duration classes [0..4])
```

## Build and run

```bash
cmake --build build/<preset> --target parakeet_warm_bench

build/<preset>/bin/parakeet_warm_bench \
    --model models/parakeet-tdt-0.6b-v3 \
    --audio tests/parakeet_tdt/assets/2086-149220-0033.wav \
    --backend cpu   # or: --backend cuda
```

Install the model with:

```bash
python3 tools/model_manager.py install parakeet_tdt --models-root models
```

## Performance

Measured on the reference test clip (`2086-149220-0033.wav`, 7.435s), 5 timed
iterations + 1 warmup, against the real NeMo model (`nemo_toolkit[asr]`,
torch 2.13+cu130) on the same machine (Intel i7-9750H, 6C/12T; GTX 1650, 3.6GB
VRAM). `parakeet_warm_bench` reports a full frontend/encoder/decoder timing
breakdown via `--timing-file` (real timing — this previously silently
produced an empty log; see the commit fixing `configure_logging` wiring).

| | CPU | CUDA |
|---|---|---|
| NeMo/PyTorch (Python) | 857.7 ms (RTF 0.115, 8.7x real-time) | OOM — see note below |
| audio.cpp, default settings | 1247.5 ms (RTF 0.168, 6.0x real-time) | 170.6 ms (RTF 0.023, 43.6x real-time) |
| audio.cpp, `matmul_weight_type=q8_0` + tuned threads | 830.9 ms (RTF 0.112, 9.0x real-time) | 132.1 ms (RTF 0.018, 56.3x real-time) |
| audio.cpp, above + conv-pointwise reclassification fix | **~750 ms** (RTF 0.101, **9.9x** real-time) | ~138 ms (RTF 0.019, 53.9x real-time, no clear change) |

This machine got noisier (shared background load) partway through this
history, enough that later single-shot end-to-end numbers aren't reliably
comparable table-row-to-table-row — the fused-QKV and graph-optimizer
findings below use an interleaved A/B (alternate config every run) instead of
separate before/after sessions specifically to cancel that out; see each
section for the actual methodology and numbers.

```bash
build/<preset>/bin/parakeet_warm_bench \
    --model models/parakeet-tdt-0.6b-v3 --audio tests/parakeet_tdt/assets/2086-149220-0033.wav \
    --backend cpu --threads 6 \
    --session-option parakeet_tdt.matmul_weight_type=q8_0
```

Two things drove essentially the entire gap between the "default settings" row
and PyTorch, both found by actually looking at the timing breakdown instead of
just the end-to-end number (which was previously impossible — see above):

- **Thread count.** This CPU is 6 physical / 12 logical cores. 8 threads (an
  arbitrary default) oversubscribes; 6 is consistently fastest, 12 is
  consistently worst (contention). Sweep it on your own hardware — the
  optimum is core-count-dependent, not a fixed number.
- **Weight precision.** `parakeet_tdt.matmul_weight_type` defaults to
  `native` (F32, since the checkpoint ships F32 weights) and is a session
  option, not a fixed choice — `native|f32|f16|bf16|q8_0` are all available
  (`parakeet_tdt.conv_weight_type` separately, `native|f32|f16`). ggml's CPU
  backend has heavily hand-tuned Q8_0 dot-product kernels (the common case
  for llama.cpp-style inference); F16/BF16 were measured *slower* than F32
  here, not faster — the win is specifically from Q8_0's kernels, not from
  "less precision" in general, so don't assume the other options help without
  measuring. Encoder graph compute alone accounts for ~93-96% of total wall
  time in every configuration measured; that's where quantization actually
  pays off; frontend and decoder are already a few percent of the total each.

**Why this isn't the default (yet).** Quantizing matmul weights is a real,
measured accuracy trade — small, but not zero: comparing against the NeMo
reference via the [numerical parity harness](../../tests/parakeet_tdt/parity/README.md),
`enc_out` cosine similarity drops from 0.972258 (native) to 0.968028 (q8_0)
— comparable in size to the float32 accumulation noise already present
between the isolated single-layer graph and the full 24-layer production
graph (see that harness's README for why that gap exists at all), not a
cliff, and transcription was still exactly correct in every configuration
tested. But it's only been validated against this one clip; defaulting
everyone into reduced precision without broader validation would be the
wrong call. If you want the speed and can tolerate (or want to verify for
your own audio) a small accuracy trade, turn it on explicitly with the
session option above.

**Flash attention was tried and did not help — kept as an opt-in, not a default.**
The FastConformer encoder's relative-position self-attention (Transformer-XL
style "AC"/"BD" score terms) can be fused into a single `ggml_flash_attn_ext_with_bias_mask`
op instead of a separate QK^T matmul + additive bias + `ggml_soft_max_ext` +
AV matmul — this is exactly the "dense additive attention bias" case that op
was built for (see `common_relative_attention.cpp`'s `use_specialized_flash_attention`
path, which already uses it, unused by any production model in this repo before
this). It was wired in here as `parakeet_tdt.encoder_flash_attention=true` and
validated correct via the numerical parity harness (`enc_out` cosine 0.972325
vs. 0.972258 for the non-flash path — no measurable accuracy difference).
But measured end to end, on this hardware, it was consistently a few percent
*slower*, not faster, on both CPU and CUDA, on both the 7.4s reference clip and
a synthetic 59.5s clip (ruling out "too short a sequence to matter"). Plausible
reason: this encoder's attention sequences are short (well under 1000 frames)
and `head_dim=128` isn't necessarily in the sweet spot of ggml's flash-attention
kernels on a Turing-generation card (GTX 1650); the existing softmax+matmul path
is already efficient at this scale. Left in as an opt-in for anyone testing on
different hardware (newer GPU generations in particular) where the tradeoff
might flip, but it is not recommended and not the default based on what was
actually measured here.

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
accuracy cost, fewer graph nodes, simpler code), and CrispASR's own numbers
suggest it matters more on other backends (their win was measured on Metal/
ARM CPU) or under conditions this single-clip single-machine benchmark
doesn't exercise (batched/concurrent serving, different GPU generations) —
exactly the kind of thing that's cheap to carry once validated correct, even
without a local win to show for it today.

**Dead framework infrastructure, switched on: the ggml graph optimizer.**
`src/framework/runtime/graph_optimizer.cpp` implements a real, unit-tested
graph rewrite pass (`engine::runtime::optimize_graph`) — folds broadcast
`ggml_repeat` nodes into the consuming op, elides pure metadata ops
(reshapes/views that don't move data), elides no-op nodes — but grepping the
whole codebase turned up **zero callers** anywhere: not `graph_executor.cpp`,
not any of the ~40 model families, nothing. It's built, tested
(`encoder_module_test`), and entirely unused in production. Wired into
Parakeet's `ensure_graph()` (`encoder.cpp`), called once per distinct input
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

**Why the CUDA side hasn't shown a dispatch-overhead win from any of the
above: CUDA graph capture is already on, but architecturally disabled on this
test GPU.** Traced with `nsys profile --trace=cuda` against `parakeet_warm_bench`:
16,068 individual `cudaLaunchKernel` calls and 4,265 `cudaStreamSynchronize`
calls (47% of total CUDA API time) across 6 encoder passes plus the full
per-token TDT decode loop — real per-launch/per-sync overhead that CUDA graph
capture-and-replay exists specifically to eliminate. `cuda_api_sum` shows
**zero** `cudaGraphLaunch`/`cudaGraphInstantiate` calls anywhere, even though
`GGML_CUDA_GRAPHS=ON` in this build (`ENGINE_DEFAULT_ENABLE_CUDA_GRAPHS` in
the top-level `CMakeLists.txt`, matching `tmp/repos/parakeet.cpp`'s own
`GGML_CUDA_GRAPHS ON`) and the encoder/decoder graphs already satisfy the
capture prerequisite (same cached `ggml_cgraph*` reused across every call at
a given shape — see `ensure_graph()`/`ensure_step_graph()`/`ensure_joint_graph()`,
all check-and-return-early on cache hit, no rebuild per inference or per
token). Root cause, in `ggml_cuda_graph_set_enabled`
(`external/ggml/src/ggml-cuda/ggml-cuda.cu`): ggml unconditionally disables
CUDA graph capture below `GGML_CUDA_CC_AMPERE` (compute capability 8.0).
This test machine's GTX 1650 is Turing, compute capability 7.5 — one
generation short. This is not a bug or a missing setting in this codebase;
it's ggml's own hardware gate, and there is nothing to change here — the
build already does everything right. It does, however, retroactively explain
why flash attention and fused-QKV (both above) measured no CUDA benefit: they
reduce exactly the per-op dispatch/launch overhead that graph replay would
otherwise hide, so on hardware where replay is active (Ampere or newer —
RTX 30/40/50-series, A100/H100, etc.) those two changes might show a real
CUDA win where they showed none here. Worth re-measuring on such hardware
rather than assuming this test machine's null result generalizes.

**Graph capacity is load-bearing: an oversized graph is both slow and wrong.**
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

**Where the CPU time actually goes.** Profiling (`perf record`) the encoder on
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

**Benchmark methodology (this machine drifts).** Wall-clock numbers here move
by 20–30% between a cold and a thermally saturated run — the same binary
measured 979 ms cold and 1264 ms at steady state within a single session, and
background load moves it further. Any A/B run as "config A for a while, then
config B" is therefore meaningless at the few-percent scale. The comparisons
above use `tmp/bench/ab.sh`: a discarded burn-in run to reach steady state,
then ABBA ordering per pass scored by the *mean* of each side's two slots, so
linear drift cancels exactly (slots 1&4 vs 2&3 share a midpoint). Scoring by
`min()` instead — which an earlier iteration of this harness did — silently
hands the win to whichever binary occupied the coldest slot, and produced a
confident 5–8% "regression" that reversed under the corrected estimator.
Retired instruction counts (`perf stat`) are reproducible to ~0.01% here and
are a good cross-check, but they are insensitive to memory-movement changes
and so cannot be the only metric.

**Why CUDA has no PyTorch comparison point.** `nemo_asr` on this GPU hits
`torch.OutOfMemoryError` just loading the model — this 3.6GB card's VRAM
budget is entirely consumed by PyTorch's own overhead on top of the weights.
The C++/ggml path runs comfortably in that same budget. This isn't really a
"we're faster" comparison so much as "the reference implementation doesn't
run here at all" — worth knowing if you're targeting small/consumer GPUs
rather than datacenter cards.

## Validation

- **Golden-transcription regression test** (`ctest -R parakeet_golden_transcription_test`,
  wired into the normal `ENGINE_BUILD_TESTS` suite, skips cleanly when the model
  isn't downloaded): runs the offline pipeline against a checked-in LibriSpeech
  test-clean clip (`tests/parakeet_tdt/assets/2086-149220-0033.wav`) and asserts the
  decoded text matches the real NeMo model's transcription for that clip exactly:
  *"Well, I don't wish to see it any more, observed Phoebe, turning away her eyes.
  It is certainly very like the old portrait."*
- **Numerical parity harness** (`tests/parakeet_tdt/parity/`, manual/pre-release
  check, not a CI gate — needs a NeMo install): compares mel features, a single
  isolated encoder layer, and the full encoder output directly against real NeMo
  activations via forward hooks. See `tests/parakeet_tdt/parity/README.md` for
  exact setup and run commands. This is what actually caught the encoder
  conv-module bug below — the golden-transcription test alone did not, since
  greedy decoding happened to be robust enough to it on that one clip.
- Backends tested: CPU and CUDA (GTX 1650), both producing the exact same
  transcription.

Two real bugs were found and fixed getting to this state, worth knowing about if
you're touching this code:

1. The frontend was missing NeMo's per-feature mel normalization (subtract the
   per-mel-bin mean, divide by the per-mel-bin unbiased std over valid frames).
   Without it the encoder saw wildly out-of-distribution input and produced
   garbage (~500x too large in magnitude by the end of the conv subsampling
   stack).
2. The FastConformer encoder layer's conv module used causal (left-only)
   padding, and separately built the depthwise conv with `use_bias=false`,
   silently dropping the folded batch-norm bias term. The padding mode is
   wrong for this model's full-context (non-streaming) configuration — NeMo's
   `CausalConv1D` is only actually causal when constructed with `padding=None`;
   this model constructs it with an explicit symmetric `padding=(kernel-1)//2`.
   The dropped bias corrupted every layer's output by a per-channel constant
   offset; because greedy/argmax decoding is somewhat robust to numerical
   drift, this second bug did not visibly break transcription on the one test
   clip and was only caught by the numerical parity harness comparing actual
   activations against NeMo, not just final decoded text.

## Known limitations

- **No streaming support, and this checkpoint is not a good fit for it.**
  `nvidia/parakeet-tdt-0.6b-v3` was trained and exported with
  `att_context_style="regular"` and `att_context_size=[-1, -1]` — unlimited,
  fully bidirectional attention context, not NeMo's `"chunked_limited"` style
  that cache-aware streaming depends on. Calling NeMo's own
  `encoder.setup_streaming_params()` on this checkpoint does not error, but
  produces a degenerate configuration (~5.8 second chunks, a ~10000-frame
  attention cache) that provides essentially none of the latency or bounded-
  memory benefit real streaming is for. NVIDIA's own maintainers, asked
  directly about streaming with this model on its Hugging Face discussion
  page, pointed users at a different, dedicated cache-aware streaming
  architecture rather than confirming this checkpoint for the purpose.
  Implementing a genuine chunked/causal streaming path against a model whose
  attention was trained with an unrestricted receptive field risks silently
  degrading accuracy in ways that would be easy to miss without extensive
  validation, for a "streaming" mode that wouldn't provide much practical
  benefit even if it worked. Real streaming support would need a different,
  purpose-trained checkpoint (`att_context_style="chunked_limited"`) as
  either a variant option on this loader or a separate family — this is not
  a matter of finishing an implementation, it needs a different model file
  to target correctly. Only `RunMode::Offline` is supported.
  **If you need streaming ASR, this framework already has it**: the
  `nemotron_asr` family (`nvidia/nemotron-3.5-asr-streaming-0.6b`) is a
  same-size-class NVIDIA checkpoint actually trained cache-aware, with
  configurable chunk sizes down to 80ms, and already implements
  `IStreamingVoiceTaskSession` in this codebase.
- Validated against a single test clip end to end; the numerical parity
  harness has not been run across a broader validation set.
- The frontend does not implement NeMo's preprocessor `dither` (small random
  waveform noise, standard to disable at inference time in most ASR
  pipelines) — no observed effect on transcription correctness.
