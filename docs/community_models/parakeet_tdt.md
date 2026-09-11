# Parakeet-TDT 0.6B v3

Default model-manager downloads use the published GGUF package when available;
the original source/conversion instructions below remain valid for manual use.

FastConformer-TDT ASR port of NVIDIA's [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
(0.6B params, 25 European languages with auto language detection). The default
package uses the repository's Transformers-compatible `model.safetensors`
checkpoint rather than the `.nemo` archive; standalone audio.cpp GGUF packages
are also supported.

## Status

Offline full-context, bounded-window long-form, and buffered-streaming sessions
are implemented. The full-context path is verified end to end on CPU and CUDA
against the real NeMo reference model, both by final transcription and by
numerical comparison at selected intermediate and full-encoder boundaries; see
[Validation](#validation).
Buffered modes deliberately re-encode fixed bidirectional windows and are not
native cache-aware streaming.

| Mode | Context and memory | Result timing |
|---|---|---|
| Offline `full_context` | Whole utterance; quadratic attention grows with duration | One final result |
| Offline `long_form` | Bounded overlapping windows; center regions decoded once | One final result |
| Streaming | Same bounded windows with retained predictor state | Partial snapshots after right-context lookahead, then a final result |

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

Equivalent CLI commands:

```bash
# Offline full-context
build/<preset>/bin/audiocpp_cli \
    --task asr --family parakeet_tdt \
    --model models/parakeet-tdt-0.6b-v3 --backend cpu \
    --audio tests/parakeet_tdt/assets/2086-149220-0033.wav

# Offline bounded-window long-form
build/<preset>/bin/audiocpp_cli \
    --task asr --family parakeet_tdt \
    --model models/parakeet-tdt-0.6b-v3 --backend cpu \
    --session-option parakeet_tdt.offline_mode=long_form \
    --audio tests/parakeet_tdt/assets/2086-149220-0033.wav

# Buffered streaming from raw mono 16 kHz PCM
ffmpeg -loglevel error -i input.wav -f f32le -ac 1 -ar 16000 - \
  | build/<preset>/bin/audiocpp_cli \
      --task asr --family parakeet_tdt \
      --model models/parakeet-tdt-0.6b-v3 --backend cpu \
      --mode streaming --audio - --input-format f32le \
      --input-rate 16000 --input-channels 1
```

Install the model with:

```bash
python3 tools/model_manager_v2.py install parakeet_tdt --models-root models
```

This installs the upstream F32 safetensors package. The repository does not
ship a Parakeet weight file; build `audiocpp_gguf` and convert the installed
package to create a standalone GGUF with the model spec, tokenizer, and
configs embedded:

```bash
cmake --build build/<preset> --target audiocpp_gguf

# Original F32 weights
build/<preset>/bin/audiocpp_gguf \
    --input models/parakeet-tdt-0.6b-v3/model.safetensors \
    --root models/parakeet-tdt-0.6b-v3 \
    --family parakeet_tdt \
    --model-spec model_specs/parakeet_tdt.json \
    --type orig --overwrite \
    --output models/parakeet-tdt-0.6b-v3-f32.gguf

# F16 weights (use `--type bf16` for BF16)
build/<preset>/bin/audiocpp_gguf \
    --input models/parakeet-tdt-0.6b-v3/model.safetensors \
    --root models/parakeet-tdt-0.6b-v3 \
    --family parakeet_tdt \
    --model-spec model_specs/parakeet_tdt.json \
    --type f16 --overwrite \
    --output models/parakeet-tdt-0.6b-v3-f16.gguf

# Q8_0 weights
build/<preset>/bin/audiocpp_gguf \
    --input models/parakeet-tdt-0.6b-v3/model.safetensors \
    --root models/parakeet-tdt-0.6b-v3 \
    --family parakeet_tdt \
    --model-spec model_specs/parakeet_tdt.json \
    --type q8_0 --overwrite \
    --output models/parakeet-tdt-0.6b-v3-q8_0.gguf
```

Pass any `.gguf` file directly to `--model`. The tested files are about
2.4 GB for original F32, 1.2 GB for F16 or BF16, and 874 MB for Q8_0. All four
storage variants reproduce the checked-in golden transcription in offline
full-context, bounded-window long-form, and buffered-streaming tests. Q8_0 has
the broader multilingual accuracy tradeoff described under
[Performance](#performance).

GGUF is the package container, not a different execution engine: both the
safetensors and GGUF paths execute through ggml. With a safetensors source,
`parakeet_tdt.matmul_weight_type` converts eligible weights when the session is
created. With a pre-converted GGUF, the default `native` setting uses the
types stored in that file. The performance study below used the safetensors
package plus session-time storage conversion; the GGUF checks validate
standalone loading and output, not a separate GGUF-vs-safetensors speed claim.

## Options

Request options are bare names. Session options use the
`parakeet_tdt.` prefix.

| Option | Scope | Values/default | Meaning |
|---|---|---|---|
| `max_tokens` | request | integer >= 0; `0` | TDT token limit; `0` uses the model-derived limit |
| `keep_language_tags` | request | boolean; `false` | Preserve language-tag tokens in decoded text |
| `audio_chunk_mode` | request | `auto`, `fixed`, `vad`, `none`; `auto` | Offline chunking mode; `vad` skips silent spans with Silero VAD |
| `audio_chunk_duration_sec` | request | float >= 0.001; session value | Request-level chunk duration for fixed or VAD offline chunking |
| `weight_type` | session | `native`, `f32`, `f16`, `bf16`, `q8_0`; `native` | Fallback storage type for matmul weights |
| `matmul_weight_type` | session | same values; inherits `weight_type` | Encoder and decoder matmul storage type |
| `conv_weight_type` | session | `native`, `f32`, `f16`; `native` | True convolution-weight storage type |
| `perf_mode` | session | `off`, `flash_attention`; `off` | Attention implementation; flash attention was slower on the tested hardware |
| `weight_context_mb` | session | integer >= 1; `3072` | Weight arena size in MiB |
| `encoder_graph_arena_mb` | session | integer >= 1; `1024` | Encoder graph arena size in MiB |
| `decoder_graph_arena_mb` | session | integer >= 1; `256` | Decoder graph arena size in MiB |
| `audio_chunk_duration_sec` | session | float >= 0.001; `2` | Center-region duration for long-form and buffered streaming |
| `left_context_sec` | session | float >= 0; `10` | Past context included in each bounded window |
| `right_context_sec` | session | float >= 0; `2` | Future context/lookahead included in each bounded window |
| `streaming_attention_mode` | session | `full_context`; `full_context` | Bidirectional attention within each bounded streaming window |
| `offline_mode` | session | `full_context`, `long_form`, `auto`; `full_context` | Offline scheduling policy |
| `audio_chunk_threshold_sec` | session | float >= 0.001; `30` | `auto` threshold for switching to long-form execution |
| `vad_model_path` | session | path; `assets/framework/models/silero_vad` | Silero VAD model directory for `audio_chunk_mode=vad` |

Word timestamps are built from the decoder's actual nonblank token-emission
frames. SentencePiece fragments and following punctuation are merged into
human-readable words. Each completed word ends at the next word's emission
boundary; the final word ends at the final token's predicted duration boundary,
clamped to the encoded audio.

## Buffered streaming

`RunMode::Streaming` provides bounded-window buffered streaming for callers
that need incremental results. The default window uses a 2-second center
region, 10 seconds of left context, and 2 seconds of right context. Each fixed
window is re-encoded with full bidirectional attention, then only its center
frames are decoded while the TDT predictor state is retained. This is not
cache-aware streaming: right context adds lookahead latency, partial text can
differ from offline full-context output, and recent text remains provisional
until the next center boundary. Use `nemotron_asr` when native cache-aware,
lower-latency ASR is required.

The window is controlled with `parakeet_tdt.audio_chunk_duration_sec`,
`parakeet_tdt.left_context_sec`, and `parakeet_tdt.right_context_sec`.
Streaming currently requires contiguous mono 16 kHz chunks. `finalize()`
flushes a short tail; `reset()` retains the loaded weights and reusable graphs.

## Long-form audio

Offline mode remains full-context by default. Set
`parakeet_tdt.offline_mode=long_form` to process arbitrarily long mono 16 kHz
audio with the same bounded overlapping-window scheduler, or use `auto` to
switch after `parakeet_tdt.audio_chunk_threshold_sec` (30 seconds by
default). Long-form mode preserves global timestamps and predictor state while
decoding every center region exactly once. Because each region sees bounded
rather than utterance-wide context, its transcript can differ from the default
full-context result.

Set `--request-option audio_chunk_mode=vad` to use Silero VAD-planned speech
spans instead of fixed windows. VAD mode uses
`audio_chunk_duration_sec` as the maximum speech chunk duration, falling back to
`parakeet_tdt.audio_chunk_duration_sec` when omitted. `parakeet_tdt.vad_model_path`
locates the bundled Silero VAD model. `auto` keeps the existing Parakeet
`offline_mode` behavior.

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
  isn't downloaded): runs full-context and bounded-window long-form decoding
  against a checked-in LibriSpeech test-clean clip
  (`tests/parakeet_tdt/assets/2086-149220-0033.wav`) and asserts the decoded
  text matches the real NeMo model's transcription for that clip exactly:
  *"Well, I don't wish to see it any more, observed Phoebe, turning away her eyes.
  It is certainly very like the old portrait."*
- **Buffered-streaming regression test**
  (`ctest -R parakeet_streaming_transcription_test`): checks partial output,
  finalization, reset/reuse, option validation, and the final transcript on the
  same fixture.
- **Standalone GGUF path tests:** original F32, F16, BF16, and Q8_0 GGUFs with
  embedded specs and sidecars pass the offline, long-form, timestamp, and
  buffered-streaming regression paths while using their stored tensor types.
- **Multilingual precision study:** 120 FLEURS clips across the 24 supported
  languages covered by FLEURS measure transcript and WER movement for
  `native`, F16, BF16, and Q8_0. This is a precision comparison, not a general
  benchmark of Parakeet's absolute WER.
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

- **Buffered streaming is not native cache-aware streaming.**
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
  The implemented `RunMode::Streaming` therefore uses bounded, overlapping
  full-attention windows and retains only the TDT predictor state between
  center regions. It bounds graph and live-buffer size and produces partial
  results, but re-encodes context, incurs configured right-lookahead latency,
  and can differ from utterance-wide offline output. Genuine causal streaming
  would need a purpose-trained `att_context_style="chunked_limited"`
  checkpoint.
  **If you need streaming ASR, this framework already has it**: the
  `nemotron_asr` family (`nvidia/nemotron-3.5-asr-streaming-0.6b`) is a
  same-size-class NVIDIA checkpoint actually trained cache-aware, with
  configurable chunk sizes down to 80ms, and already implements
  `IStreamingVoiceTaskSession` in this codebase.
- Direct layer-by-layer numerical parity against NeMo is limited to one
  English fixture. The end-to-end multilingual study is broader (120 clips,
  24 languages) but shallow at five clips per language and is intended to
  compare storage precisions, not certify production WER for every domain.
- The frontend does not implement NeMo's preprocessor `dither` (small random
  waveform noise, standard to disable at inference time in most ASR
  pipelines) — no observed effect on transcription correctness.
