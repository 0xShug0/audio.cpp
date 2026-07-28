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
python3 tools/model_manager_v2.py install parakeet_tdt --models-root models
```

Word timestamps are built from the decoder's actual nonblank token-emission
frames. SentencePiece fragments and following punctuation are merged into
human-readable words. Each completed word ends at the next word's emission
boundary; the final word ends at the final token's predicted duration boundary,
clamped to the encoded audio.

## Performance

Reference clip (`2086-149220-0033.wav`, 7.435s) on an Intel i7-9750H (6C/12T)
and a GTX 1650 Max-Q, against the real NeMo model on the same machine:

| implementation | settings | CPU | CUDA |
|---|---|---|---|
| **NeMo / PyTorch** (the Python reference) | as shipped | 857.7 ms (8.7x real-time) | **OOM** — won't load on this 4GB card |
| **audio.cpp** (this port) | untuned: 8 threads, f32 | ~1245 ms (6.0x real-time) | ~169 ms (44.0x real-time) |
| **audio.cpp** (this port) | tuned: 12 threads, `q8_0` | **~715 ms** (**10.4x** real-time) | **~131 ms** (56.8x real-time) |

Read down the CPU column: **untuned, this port loses to PyTorch** (~1245 ms vs
857.7 ms). Tuned, it wins by ~1.2x — and on this GPU the comparison doesn't
exist at all, because the PyTorch reference cannot load the model in 4 GB
while the native path runs it in 2642 MiB. "Untuned" here means the two
defaults, not a deliberately handicapped configuration: `--threads` defaults
to 8, and `parakeet_tdt.matmul_weight_type` defaults to `native`, which for
this checkpoint means f32 (`config.json` declares `dtype: float32`).

Re-measured on the current tree with the drift-cancelling harness (ABBA, 4
passes, median of 5 iterations per run): tuned is **1.72x** default on CPU and
**1.31x** on CUDA. Both defaults reproduce earlier figures closely; the CPU
tuned row previously read ~750 ms, which was measured before the optimization
pass and is now slightly conservative.

```bash
build/<preset>/bin/parakeet_warm_bench \
    --model models/parakeet-tdt-0.6b-v3 --audio tests/parakeet_tdt/assets/2086-149220-0033.wav \
    --backend cpu --threads 12 \
    --session-option parakeet_tdt.matmul_weight_type=q8_0
```

Two settings account for essentially all of the tunable gap:

- **Thread count.** 12 threads is fastest on this 6C/12T CPU — 3.1% over 6,
  1.1% over 8. Sweep it on your own hardware, and sweep it *interleaved*
  rather than as sequential before/after runs (this machine drifts 20-30%
  between cold and thermally saturated, which is enough to invert the answer).
- **Weight precision.** `parakeet_tdt.matmul_weight_type` accepts
  `native|f32|f16|bf16|q8_0` (default `native`; `parakeet_tdt.conv_weight_type`
  separately accepts `native|f32|f16`). Encoder graph compute is ~93-96% of
  wall time, so this is the setting that matters.

Measured over 120 FLEURS clips across 24 languages
([harness](../../tests/parakeet_tdt/multilingual/README.md),
[raw transcripts](../../tests/parakeet_tdt/multilingual/results/README.md)):

| weight type | encode speed | transcript identical to f32 | absolute WER |
|---|---|---|---|
| `native` (f32) | 1.00x | 100% | 0.1338 |
| `f16` | ~1.00x | 99.2% | 0.1336 |
| `bf16` | **1.14x** | 99.2% | 0.1334 |
| `q8_0` | **1.79x** | 91.7% | 0.1331 |

- **`bf16`** is close to free: 1.14x, 99.2% of transcripts byte-identical.
- **`q8_0`** is the real speedup at 1.79x. It changes ~8% of transcripts, but
  absolute WER does not move — the churn is mostly formatting (`T Rex` →
  `T-Rex`), and where words change it goes both directions. It costs
  reproducibility against an f32 baseline, not quality.
- **`f16`** is not a speedup here at all; don't assume fewer bits means faster.

`native` stays the default because it is the only setting that reproduces
exactly. Full methodology, profiling, and the optimizations that were tried
and rejected are in
[docs/reports/parakeet_tdt_performance.md](../reports/parakeet_tdt_performance.md).

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
  Currently passing on both backends: `mel_features` and `layer_0` at cosine
  1.000000, `enc_out` at 0.972258 on CPU and CUDA alike.
- Backends tested: CPU and CUDA (GTX 1650 Max-Q), both producing the exact same
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
