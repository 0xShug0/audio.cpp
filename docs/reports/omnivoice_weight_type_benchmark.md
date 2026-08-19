# OmniVoice Runtime Weight-Type Benchmark

This report sweeps OmniVoice voice cloning over every combination of
`omnivoice.generator_weight_type`, `omnivoice.audio_tokenizer_weight_type`, and
`--num-inference-steps` on CUDA. In the measured routes, an `f16` generator is
up to **1.31x faster** than the native path, while the audio tokenizer weight
type changes wall time by at most about **9%**.

The practical finding is that `q8_0` is a reasonable choice for the generator
but **not** for the audio tokenizer: runtime `q8_0` tokenizer weights produced
unusable audio at every step count while buying no measurable speed.

Notes:

- Each single-request row is a single sample taken after 10 warmup runs, so
  individual rows carry run-to-run noise. The aggregate means below average 75
  runs per weight type and are the more reliable comparison.
- Wall time is the CLI-reported `session.wall_ms`.
- This sweep uses **runtime** quantization of SafeTensors weights through the
  session options. It is not the same as loading a prebuilt Q8 GGUF package,
  which quantizes far more conservatively. See the
  [Q8 performance report](gguf_q8_performance.md) for GGUF package results.
- Reference audio and text are Bengali, so absolute times reflect that input
  rather than a general-purpose baseline.

## Setup

| Item | Value |
|---|---|
| Backend | CUDA, NVIDIA GeForce RTX 4090, 24 GiB |
| Model | `models/OmniVoice` SafeTensors |
| Task | `tts` voice cloning, `--voice-ref` plus `--reference-text` |
| Language | Bengali |
| Seed | `42` |
| Steps swept | 8 to 64 in steps of 4 |
| Weight types swept | `native`, `f32`, `f16`, `bf16`, `q8_0` |
| Combinations | 375 single requests, 375 batch runs of 10 prompts each |

Scripts: [`tests/omnivoice/run_clone_benchmark.sh`](../../tests/omnivoice/run_clone_benchmark.sh)
and [`tests/omnivoice/run_clone_batch_benchmark.sh`](../../tests/omnivoice/run_clone_batch_benchmark.sh).

## Summary

Averaged over all audio tokenizer weight types and step counts:

| `omnivoice.generator_weight_type` | Mean wall time | Speedup vs `native` |
|---|---:|---:|
| `native` | 1008 ms | 1.00x |
| `f32` | 1005 ms | 1.00x |
| `bf16` | 951 ms | 1.06x |
| `q8_0` | 834 ms | 1.21x |
| `f16` | 798 ms | **1.26x** |

| `omnivoice.audio_tokenizer_weight_type` | Mean wall time | Speedup vs `native` |
|---|---:|---:|
| `native` | 950 ms | 1.00x |
| `f32` | 943 ms | 1.01x |
| `q8_0` | 935 ms | 1.02x |
| `f16` | 899 ms | 1.06x |
| `bf16` | 869 ms | 1.09x |

The generator dominates wall time, so it is the option worth tuning. The audio
tokenizer spans only 869-950 ms across all five settings.

## Generator Weight Type

Audio tokenizer held at `f16`:

| Steps | `native` | `f32` | `f16` | `bf16` | `q8_0` |
|---:|---:|---:|---:|---:|---:|
| 8 | 274 ms | 265 ms | 228 ms | 390 ms | 227 ms |
| 16 | 480 ms | 471 ms | 371 ms | 590 ms | 394 ms |
| 20 | 574 ms | 589 ms | 453 ms | 636 ms | 478 ms |
| 32 | 875 ms | 874 ms | 689 ms | 872 ms | 715 ms |
| 48 | 1281 ms | 1299 ms | 984 ms | 1202 ms | 1039 ms |
| 64 | 1704 ms | 1713 ms | 1298 ms | 1519 ms | 1346 ms |

`f16` is fastest at every step count, from 1.20x at 8 steps to 1.31x at 64
steps versus `native`. The `bf16` rows at 8 and 16 steps are visibly noisy
single samples; the aggregate above places `bf16` between `native` and `q8_0`.

## Audio Tokenizer Weight Type

Generator held at `f16`:

| Steps | `native` | `f32` | `f16` | `bf16` | `q8_0` |
|---:|---:|---:|---:|---:|---:|
| 8 | 298 ms | 302 ms | 228 ms | 222 ms | 265 ms |
| 16 | 458 ms | 459 ms | 371 ms | 383 ms | 405 ms |
| 20 | 527 ms | 530 ms | 453 ms | 462 ms | 494 ms |
| 32 | 768 ms | 762 ms | 689 ms | 689 ms | 725 ms |
| 48 | 1050 ms | 1066 ms | 984 ms | 1007 ms | 1035 ms |
| 64 | 1362 ms | 1369 ms | 1298 ms | 1289 ms | 1318 ms |

`q8_0` is never the fastest tokenizer setting, and it is slower than both `f16`
and `bf16` at every step count measured here.

## Runtime Q8_0 Audio Tokenizer Produces Unusable Audio

With `omnivoice.audio_tokenizer_weight_type=q8_0`, every run completed without
an error, wrote a WAV, and reported normal timings. The generated audio was not
listenable: it did not carry the intended speech at any step count from 8 to
64, independent of the generator weight type.

This is a listening judgement made while reviewing the sweep outputs. The
generated WAVs were not retained, so this report does not attach a numeric
similarity metric for the failure.

Because runtime `q8_0` tokenizer weights are also not faster than `f16` or
`bf16`, there is no measured reason to select them. Recommended pairing for
this route:

```bash
--session-option omnivoice.generator_weight_type=f16 \
--session-option omnivoice.audio_tokenizer_weight_type=f16
```

Runtime `q8_0` on the **generator** was fine in this sweep: it produced
listenable audio and ran 1.21x faster than `native`.

## Step Count Scaling

Generator and audio tokenizer both `f16`:

| Steps | Wall time | Per step | `omnivoice.session.generate_ms` | `omnivoice.session.decode_ms` |
|---:|---:|---:|---:|---:|
| 8 | 228 ms | 28.5 ms | 206 ms (90%) | 10 ms (4%) |
| 32 | 689 ms | 21.5 ms | 668 ms (97%) | 9 ms (1%) |
| 64 | 1298 ms | 20.3 ms | 1277 ms (98%) | 10 ms (1%) |

Wall time is essentially linear in step count and is almost entirely generator
diffusion. Audio tokenizer decode stays flat near 9-10 ms, which is why the
tokenizer weight type barely moves the total. Lowering
`--num-inference-steps` is therefore the largest single latency lever: 16 steps
runs 1.9x faster than the default 32.

## Batch Inference Is Sequential

Each `--batch-text-file` run of 10 prompts emitted **10 separate
`session.wall_ms` values**, one per prompt, across all 375 batch combinations
(3,750 requests total). This confirms that batched CLI input is executed as
sequential requests in one session rather than as a fused batch.

Generator and audio tokenizer both `f16`:

| Steps | Single request | Batch first request | Batch mean per request | Batch total (10) |
|---:|---:|---:|---:|---:|
| 8 | 228 ms | 197 ms | 167 ms | 1673 ms |
| 16 | 371 ms | 346 ms | 305 ms | 3046 ms |
| 32 | 689 ms | 596 ms | 555 ms | 5552 ms |
| 64 | 1298 ms | 1173 ms | 1078 ms | 10777 ms |

Batch totals scale linearly with prompt count. The per-request improvement of
1.20x-1.36x over an isolated single request comes from session and graph reuse
after the first prompt, not from batching: the first request in each batch is
consistently the slowest. A long-lived session that reuses one loaded model
gives the same benefit without a batch file.

## Known Limitations

- Single-request rows are one sample each; only the aggregate means average out
  run-to-run noise.
- Measured on one GPU with one Bengali reference voice and one short prompt.
- The `q8_0` audio tokenizer finding is qualitative. Reproducing it requires
  listening to the generated WAVs.
- No CPU, Vulkan, or Metal coverage.
