# Drama 3 local validation (2026-10-14)

Experimental implementation based on upstream `f2881677`, on Windows 11,
Ryzen 7950X3D, 64 GiB RAM, RTX 3090 24 GiB, MSVC 14.43, CUDA 12.4, and Vulkan
SDK 1.4.350. No GGML/backend kernels or the dense Qwen3.5 module were modified.
The existing Fish/S2 Pro path remains selected for its original configurations.

The draft PR is rebased onto `67960689`; the three intervening upstream commits
do not touch the model, codec, or MoE runtime. This remains a work in progress:
the backend results below describe the tested scope, not full-model perceptual
parity or readiness for every platform.

After rebasing, both CLI/server builds, all four reference CTests, the CPU-only
reference/prompt checks, both model-spec/GGUF CTests, the codec checks on all
three backends, and loader/catalog sync (including its self-tests) pass again.
The benchmark timings below were measured before this unrelated upstream rebase.

Latest Vulkan pass (2026-10-15): warm short generation is about 1.03–1.08 s
against the previous 1.43–1.46 s; long generation is 8.26 s against 10.47 s.
Outputs remain byte-identical. The detailed follow-up below separates repeat
requests from different texts and shows the remaining CUDA gap.

Model revision: `e0a9218aa48888270f06ef1ed9ea67622c0e17d2`.
Fish Speech reference revision: `541566932abbaebcb0a40e562a1f846738383f38`.
All official downloaded file checksums were verified. The native mixed Q4_K
package contains 1380 tensors and occupies 20.315 GiB. All 616 quantized tensors
were additionally checked for finite Q4_K/Q8_0 scale fields.

## Tests

| Check | Result |
|---|---|
| CPU CLI/server build | Pass |
| Combined CUDA/Vulkan CLI/server build | Pass |
| Original prompt/tokenizer cases | 28/28 exact, including invalid-input cases |
| Original small architecture: CPU / CUDA / Vulkan | All three pass |
| New reference CTests | 4/4 pass |
| Existing GGUF source and model-spec CTests | 2/2 pass |
| Real Q4_K TTS | CPU, CUDA, Vulkan all produce the requested short sentence |
| Long text and reference cloning | CUDA/Vulkan HTTP 200; ASR recognizes all requested words |
| Repeat after long text, cloning, and a rejected request | Same short-request WAV bytes on both GPU backends |
| Invalid structural prompt | HTTP 400; following valid request succeeds |
| HTTP SSE audio streaming | CUDA 39 / Vulkan 40 deltas, followed by completion |
| S2 Pro regression against unmodified upstream | CPU WAV bytes exactly match |

The 129-token small fixture executes original Fish Speech classes with seeded
F32 weights, multiple key/value heads, grouped MoE routing, and partial RoPE.
It compares whole prefill, cached continuation, reset, speech-head scores,
tower causal state, and frame embeddings. CPU maximum hidden error was
`8.52e-6`; final Vulkan maximum hidden error was `2.60e-4`, with exact frame
embeddings and tower maximum error `1.43e-6`. The assertions use `1e-4` CPU and
`1e-3` GPU maximum absolute tolerances.

Batched Vulkan math initially changed close expert-routing choices. The new
runtime uses F32 projection accumulation and vector kernels for Vulkan prefill.
Tokenizer tests additionally caught inherited Qwen2 splitting and missing NFC
normalization; Drama 3 uses Qwen3.5 splitting and model-private utf8proc 2.11.3
(`e5e799221b45bbb90f5fdc5c69b6b8dfbf017e78`), with its source/license retained.
Vulkan whole-utterance codec graphs also exceeded the remaining VRAM on long
text; Drama 3 now uses the existing causal frame decoder for Vulkan offline
output, keeping its history across frames. These policies affect only Drama 3.

Streaming and offline samples have equal lengths. Vulkan PCM matches exactly;
CUDA's batched/causal codec paths differ by 49.42 dB signal-to-difference ratio
on the short fixture. This is not a claim of exact waveform parity across modes.

## Generation time and memory

Short prompt: `Hello from Drama three.` Seed 1234; server max_tokens 512;
eight threads; one loaded model and one session. Times are HTTP generation
wall time with weights already loaded; first request includes graph setup.
Warm values average the next two requests. VRAM is NVML whole-device usage
sampled every 20 ms, including weights, caches, driver memory, and a roughly
0.36 GiB idle baseline. It is not a weights-only measurement.

| Backend | First request | Warm mean | Audio length | Short peak VRAM | Mixed-request peak VRAM |
|---|---:|---:|---:|---:|---:|
| CUDA | 1.274 s | 0.807 s | 1.811 s | 21.450 GiB | 22.696 GiB |
| Vulkan | 1.971 s | 1.531 s | 1.858 s | 21.531 GiB | 21.667 GiB |

The CPU CLI smoke test used twelve threads: 3.554 s generation for 1.579 s audio.
Its initial loading/generation peak RSS was 42,386 MiB and peak committed
process memory 24,696 MiB; mapped checkpoint pages contribute to RSS.

| Request | CUDA time / audio length | Vulkan time / audio length |
|---|---:|---:|
| Long text through the framework chunker | 6.009 s / 12.307 s | 11.349 s / 11.935 s |
| Reference clone | 1.582 s / 2.926 s | 3.352 s / 3.297 s |
| Short request after cloning | 0.850 s / 1.811 s | 1.488 s / 1.858 s |

The long prompt was:

> The model checkpoint contains the diffusion transformer and layer-fusion weights. The encoder and decoder are loaded from separate files at runtime. This native runtime keeps the loaded model ready for the next request.

The reference was the CUDA short sample with its transcript. Cloning's requested
text was `The next sentence uses the reference voice.` Phonon-2 ASR recognized
both clone transcripts and every word of both final long samples. This checks
spoken content, not perceptual speaker similarity or a human-labelled WER score.

## Full checkpoint reference scope

An independent CPU reference loaded one original FP8 layer at a time and
evaluated the 26-row short prompt with F32 arithmetic. The native Q4_K runtime
used those exact input embeddings. This separates conversion/graph checks from
tokenizer and random-sampling differences.

| Backend | Speech-score cosine | Relative score RMS error | Same top-ranked token | Top-10 overlap |
|---|---:|---:|---|---:|
| CUDA | 0.998932 | 5.83% | Yes | 8/10 |
| Vulkan | 0.998823 | 4.86% | No | 7/10 |

Quantized full-model scores are approximate. Neither this comparison nor ASR
establishes perceptual equivalence to original FP8 synthesis. Native seeded
sampling does not reproduce PyTorch's random draws. Full Q8/BF16 packages,
Metal/AMD hardware, broad multilingual coverage, and multi-slot execution were
not validated here; the published-package catalog is unchanged.

## Optimization pass on the same Q4_K package

Compared with the initial native implementation above, using a saved executable
from before this pass. Both versions use the same model file, prompt, seed,
eight threads, max_tokens 512, and serial HTTP requests. GPU runs were performed
one server at a time; VRAM was sampled with NVML every 10 ms. These are local
measurements on the same RTX 3090, not a comparison with full original FP8
PyTorch synthesis. Early profiling runs with overlapping model loads and the
experimental smaller-context/scratch-allocation runs are excluded.

| Backend / request | Before | After | Less generation time |
|---|---:|---:|---:|
| Vulkan, warm short mean (5 requests) | 1.509 s | 1.421 s | 5.8% |
| Vulkan, long mean (3 requests) | 11.281 s | 10.492 s | 7.0% |
| CUDA, warm short mean (5 requests) | 0.812 s | 0.773 s | 4.8% |
| CUDA, long mean (3 requests) | 5.853 s | 5.484 s | 6.3% |

The first request did not improve: Vulkan was 1.819 / 1.872 s and CUDA
1.239 / 1.309 s before/after. The decoder graph is now created lazily, moving
some setup from model loading to the first request. Vulkan remains about 1.8
times CUDA's warm short-request time on this GPU.

| Backend | Before mixed peak | After mixed peak | Before held after short | After held after short | After with mem_saver held after short |
|---|---:|---:|---:|---:|---:|
| Vulkan | 21.665 GiB | 20.722 GiB | 21.414 GiB | 20.460 GiB | 20.031 GiB |
| CUDA | 22.726 GiB | 21.761 GiB | 21.134 GiB | 20.173 GiB | 19.655 GiB |

All VRAM values include weights and the approximately 0.359 GiB device baseline.
The package size/quantization is unchanged: about 0.96 GiB of text embeddings
moved from GPU memory into native host memory. This is a GPU-to-host memory
tradeoff, not a smaller checkpoint. CPU RSS also includes mapped checkpoint
pages; Windows committed process memory can include backend allocations.

With mem_saver enabled, mixed peaks were 20.544 GiB Vulkan / 21.214 GiB CUDA.
Warm short means (2 requests) were 1.508 s Vulkan / 0.884 s CUDA; long means
(3 requests) were 10.972 s Vulkan / 5.607 s CUDA. Normal builds keep mem_saver
off by default to retain faster cache reuse.

Kept changes: shared tower graphs, backend-to-backend hidden-state handoffs,
omission of unused teacher-forced heads, exact sampler RNG advancement without
noise calculations for discarded candidates, pinned/bounded Vulkan vector
prefill, host embeddings with matching conversion, lazy decoder ownership, and
complete AR graph/cache release before offline codec decoding with mem_saver.
Attention width and precision remain the validated settings. The original
dense Qwen3.5 implementation and shared CUDA/Vulkan kernels remain unchanged.

All 50 server output comparisons were byte-identical to their previous native
backend outputs, including cold/warm repeats, long chunked text, cloning, three
seeds, recovery after rejection, and both memory policies. The reference tests
also cover generation tower handoffs and release/recreation. CPU CLI output
with mem_saver remains byte-identical to the previous CPU sample.
Final streaming PCM also exactly matches the previous streaming samples on
CUDA (39 deltas) and Vulkan (40 deltas), with normal completion events. A fresh
CPU S2 Pro regression uses the same 23-character prompt on upstream `f2881677`
and this version: WAV SHA256 matches
`de537ddfed79f92ac49afe923601e943ddd55933aeab7f81ec8fa0d6d57f0704`.
All four model CTests, both CPU-only reference checks, and both GGUF/spec CTests
pass after the optimization pass.

Artifacts are under `outputs/drama3-development-20261014/optimization`:
`baseline-clean-{backend}`, `final-{backend}`, `final-mem-saver-{backend}`,
`summary.json`, and the associated configs, timing logs, WAVs, and host-memory
measurements. Run `optimization_suite.py` and `summarize_optimization.py` from the
workspace with the saved baseline binary in `optimization/baseline-bin`.
`extra-checks.json`, `drama3-cpu.wav`, `s2-upstream-cpu.wav`, and
`{backend}-stream.wav` record the CLI/regression/streaming checks.

## Vulkan follow-up (2026-10-15)

The previous pass still left Vulkan much slower than CUDA. Timing breakdowns
identified vector prefill and causal codec decode as the main differences.
An isolated GPU timestamp profile of the 26-row backbone prompt found 49,660
concatenation dispatches. Linear-attention head duplication chained per-head
concatenations; one interleaved F32 repeat reduces concatenations to 1,300
without changing values. Timing instrumentation adds overhead, so dispatch
counts diagnose the workload; unprofiled HTTP wall times measure speed.

Additional changes retain an exact 14-row prefix state when embeddings match
and queue offline codec frames in bounded groups of eight. The prefix cache
stores recurrent/conv state, the prefix KV rows, and hidden values; it does not
cache generated audio or RNG state. This speeds different texts sharing the
fixed instruction/header. Cache hits, changed suffixes, invalidation, and
release/recreation have exact native hidden-value checks. Prefix state is owned
by the private decoder and is freed by mem_saver together with other AR graphs.

| Case | Vulkan before this pass | Vulkan after | CUDA control | Less Vulkan time |
|---|---:|---:|---:|---:|
| Warm short, paired recheck (7 requests) | 1.463 s | 1.029 s | 0.774 s | 29.6% |
| Five different texts, paired recheck | 1.606 s | 1.148 s | 0.859 s | 28.5% |
| Long, full matrix mean (3 requests) | 10.475 s | 8.256 s | 5.455 s | 21.2% |

The full matrix's five warm short requests measured 1.433 / 1.082 s; the
separate paired recheck measured 1.463 / 1.029 s. CUDA controls used two warm
short requests, five different texts, and three long repeats. These are local
means, with run-to-run variation. Relative to CUDA wall time, Vulkan still needs
about 33–40% longer for the short prompt and 51% longer for the long prompt.
CUDA and Vulkan already generated slightly different audio lengths before this
pass (1.811 / 1.858 s short, 12.307 / 11.935 s long); every output is preserved
within its backend. Comparisons are prompt-generation times, not equal-duration
audio throughput.

First Vulkan request was 1.848 / 1.521 s before/after. Mixed peak VRAM was
20.722 / 20.761 GiB, including weights and the same approximately 0.359 GiB
baseline. The new prefix snapshot takes about 64 MiB; removing intermediate
head concatenations offsets part of that cost. The Q4_K file is unchanged.

Validation after this pass:

- 19/19 full-matrix server WAVs exactly match the saved pre-pass executable.
- 16/16 mem_saver server WAVs exactly match that same backend baseline.
- 14/14 paired timing-recheck WAVs exactly match; five cases use different text.
- Vulkan streaming PCM remains exact (40 deltas and a completion event).
- CPU, CUDA, Vulkan reference CTests and CPU-only reference checks pass.
- GGUF tensor-source and model-spec CTests pass; both CLI/server builds pass.
- Direct codec checks pass on CPU, CUDA, Vulkan: exact serial/queued float
  samples over 137 frames, crossing the 128-frame attention window, mixed
  1/7/8/9/112-frame calls, release/recreation, rejection, and recovery.

The additive `decode_stream_frames` codec API is used only by Drama 3's Vulkan
offline path. Other codec entry points and shared CUDA/Vulkan kernels remain
unchanged. This does not validate Metal, other Vulkan hardware, multi-slot
execution, or equivalence to full original FP8 synthesis.

Reproduce with `benchmark_vulkan_gap.py`, `final_vulkan_gap_checks.py`,
`recheck_gap_timing.py`, and `summarize_vulkan_gap.py` under the local workspace.
Artifacts are in `optimization/vulkan-gap-*`, including
`vulkan-gap-summary.json`, `vulkan-gap-checks.json`, codec-test logs, saved
baseline binaries, and before/after GPU operation profiles.

## Reproduce and artifacts

Build/conversion/reference commands are in [the model guide](../models/fish_drama3.md).
The local test workspace is `outputs/drama3-development-20261014`:

- `source`: implementation on `agent/fish-drama3-20261014`.
- `build-cpu/bin` and `build-gpu/bin`: CLI, server, and parity executables.
- `server_validation.py`: exact server configs, serial request cases, and NVML sampling.
- `server-cuda-results.json` / `server-vulkan-results.json`: final timings, memory, and WAV hashes.
- `server-{backend}-cold.wav`, `server-{backend}-long-chunked.wav`, `server-{backend}-clone.wav`: final GPU samples.
- `drama3-{backend}-stream.wav`: reconstructed SSE samples.
- `real_reference.py`, `real-reference`, `real-cuda`, `real-vulkan-final`: full checkpoint reference and comparisons.
- `s2-baseline-cpu.wav` and `s2-final-cpu.wav`: identical SHA256 `90c022b920470d67d42038f9357723d1bcf3f1f003e9db3da7a7821edc4e91ed`.

For the existing checks:

```bash
cmake --build build --parallel --target model_spec_system_test gguf_tensor_source_test
ctest --test-dir build -R '^(model_spec_system_test|gguf_tensor_source_test)$' --output-on-failure
```

These are local model and relevant framework checks, not a full-project CI run.
The benchmark scripts and generated WAVs listed above are local validation
artifacts, not files bundled with the PR. The committed model guide includes
conversion, CLI/server, reference-fixture, and codec-check commands that another
contributor can run with the pinned official checkpoint and reference checkout.
