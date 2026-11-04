# Parakeet-TDT 0.6B v3

Default model-manager downloads use the published GGUF package when available;
the original source/conversion instructions below remain valid for manual use.

FastConformer-TDT ASR port of NVIDIA's [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
(0.6B params, 25 European languages with auto language detection). The default
model-manager package is the standalone Q8_0 GGUF. The repository also supports
a Transformers-compatible `model.safetensors` checkpoint rather than a `.nemo`
archive.

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

## Model

| Field | Value |
|---|---|
| Family | `parakeet_tdt` |
| Tasks | `asr` |
| Modes | `offline`, `streaming` |
| Languages | `bg`, `cs`, `da`, `de`, `el`, `en`, `es`, `et`, `fi`, `fr`, `hr`, `hu`, `it`, `lt`, `lv`, `mt`, `nl`, `pl`, `pt`, `ro`, `ru`, `sk`, `sl`, `sv`, `uk` |
| Default package | `parakeet_tdt_q8_0` |
| Input | Speech audio |
| Output | Transcript text and optional timestamps |

“Model/session default” means no explicit override. The runtime uses the
checkpoint or session setting, or leaves the optional value unset as described.

## Common Options (use directly)

| Option | Meaning |
|---|---|
| `--model` | Model file or directory shown in the examples. |
| `--backend` | Select an available backend supported by this model. |
| `--threads` | CPU thread count. |
| `--log` | Enable runtime logging. |
| `--audio` | Input audio, where required by the selected task. |

## Request Options (use with `--request-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `return_timestamps` | `bool` | Model/session default | Accepted for CLI --words-out compatibility. Word timestamps are always returned, including when false or omitted. |
| `max_tokens` | `int` | `0` | Maximum TDT generated tokens; 0 or omitted uses the model-derived limit. Declared range: minimum 0. |
| `keep_language_tags` | `bool` | `false` | Keep language tag tokens in decoded text; default false. |
| `audio_chunk_mode` | `auto`, `fixed`, `vad`, `none` | `auto` | Offline audio chunking mode. fixed selects overlapping long-form windows regardless of offline_mode; vad uses the configured VAD model (Silero by default) to skip silence; auto follows offline_mode. Offline input is downmixed and resampled as needed. |
| `audio_chunk_duration_sec` | `float` | Model/session default | VAD maximum chunk duration; omitted inherits the session duration. Fixed/long_form windows use the session setting, not this request override. Declared range: minimum 0.001. |

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `parakeet_tdt.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Shared matmul weight storage type; default native. |
| `parakeet_tdt.matmul_weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | Model/session default | Encoder and decoder matmul weight storage type; defaults to weight_type, which defaults to native. Q8_0 measured 1.79x faster on the tested CPU and changed roughly 8 percent of transcripts without moving aggregate word error rate. |
| `parakeet_tdt.conv_weight_type` | `native`, `f32`, `f16` | `native` | Convolution weight storage type; default native. |
| `parakeet_tdt.perf_mode` | `off`, `flash_attention` | `off` | Encoder attention implementation. Default off uses the validated relative-attention path; flash_attention enables the fused implementation, which was numerically validated but slower on the tested hardware. |
| `parakeet_tdt.weight_context_mb` | `int` | `3072` | Weight context arena size in MiB; default 3072. Declared range: minimum 1. |
| `parakeet_tdt.encoder_graph_arena_mb` | `int` | `1024` | Encoder graph arena size in MiB; default 1024. Declared range: minimum 1. |
| `parakeet_tdt.decoder_graph_arena_mb` | `int` | `256` | Decoder graph arena size in MiB; default 256. Declared range: minimum 1. |
| `parakeet_tdt.audio_chunk_duration_sec` | `float` | `2` | Center-region duration for buffered streaming in seconds; default 2. Fixed context windows are re-encoded rather than cache-aware. Also sizes offline fixed/long_form windows. Declared range: minimum 0.001. |
| `parakeet_tdt.left_context_sec` | `float` | `10` | Past context included when re-encoding each buffered-streaming window in seconds; default 10. Also applies to offline fixed/long_form windows. Declared range: minimum 0. |
| `parakeet_tdt.right_context_sec` | `float` | `2` | Future lookahead included when re-encoding each buffered-streaming window in seconds; default 2 and adds equivalent partial-result latency. Also applies to offline fixed/long_form windows. Declared range: minimum 0. |
| `parakeet_tdt.streaming_attention_mode` | `full_context` | `full_context` | Attention policy inside each buffered window. full_context preserves bidirectional attention over the bounded window. |
| `parakeet_tdt.offline_mode` | `full_context`, `long_form`, `auto` | `full_context` | Offline encoder scheduling. full_context encodes the whole utterance, long_form uses bounded overlapping windows, and auto selects long_form beyond audio_chunk_threshold_sec. |
| `parakeet_tdt.audio_chunk_threshold_sec` | `float` | `30.0` | Duration threshold used by offline_mode=auto before switching to bounded-window long-form execution; default 30 seconds. Declared range: minimum 0.001. |
| `parakeet_tdt.vad_model_path` | `path` | `assets/framework/models/silero_vad` | VAD model path used by audio_chunk_mode=vad (any built-in VAD family); default assets/framework/models/silero_vad. |

## Status

Offline full-context, bounded-window long-form, and buffered-streaming sessions
are implemented. The full-context path is verified end to end on CPU and CUDA
against the NeMo reference model.
Buffered modes deliberately re-encode fixed bidirectional windows and are not
native cache-aware streaming.

| Mode | Context and memory | Result timing |
|---|---|---|
| Offline `full_context` | Whole utterance; quadratic attention grows with duration | One final result |
| Offline `long_form` | Bounded overlapping windows; center regions decoded once | One final result |
| Streaming | Same bounded windows with retained predictor state | Partial snapshots after right-context lookahead, then a final result |

## Architecture

[Underdog Bark 3.0 Pro](underdog_bark3.md) is another English-only Parakeet-TDT
variant. Its converter and model-local adapter preserve the publisher's weight
scaling and 250 ms input prefix; use that guide rather than loading its original
BF16 checkpoint as stock Parakeet.

The same runtime can load the [Phonon-2 English ASR variant](phonon2_asr.md) after
exact reconstruction of its custom five-value container into a dense F32
package. The Phonon package supports English-only offline and streaming ASR; it does
not inherit this base checkpoint's 25-language claims.

```
Frontend: 16kHz -> 128 mel bins, preemphasis=0.97, NeMo per-feature normalization
Encoder:  3-stage Conv2D subsampling (8x) -> 24-layer FastConformer -> 1024-dim
Decoder:  2-layer LSTM predictor + joint network (TDT: 5 duration classes [0..4])
```

## Orukeet r3 checkpoint

[Orukeet r3](https://huggingface.co/oruk/orukeet) (Oruk AI) is a fine-tune of
`parakeet-tdt-0.6b-v3` that replaces half of the encoder's temporal depthwise
filters with fitted, frozen Gabor kernels and continues training on
multilingual and multi-accent data. It keeps the exact architecture,
tensor shapes, preprocessor, and tokenizer of the stock checkpoint — the
fitted kernels are materialized as ordinary convolution weights — so it runs
through this same `parakeet_tdt` loader with no engine changes. It is a
weight variant, not a new model family.

Orukeet publishes a `.nemo` checkpoint rather than a Transformers-compatible
safetensors package, so conversion goes through
`tools/community_models/convert_orukeet.py`, which maps the 651 NeMo tensors
to the audio.cpp layout, reuses the stock sidecars (verified against the
`.nemo` config; the r3 export audit certifies the tokenizer is unchanged),
and packs a standalone GGUF via `audiocpp_gguf`. The script never downloads
anything; point it at a local copy of the pinned r3 file:

```bash
uv run --python 3.12 --with torch --with pyyaml --with safetensors \
    --with sentencepiece --with numpy \
  tools/community_models/convert_orukeet.py \
    --checkpoint path/to/orukeet-v0.1.0.nemo \
    --converter build/<preset>/bin/audiocpp_gguf \
    --gguf-output models/Orukeet-GGUF/orukeet-q8_0.gguf \
    --type q8_0
```

The stock install (`models/parakeet-tdt-0.6b-v3`, from
`model_manager_v2.py install parakeet_tdt`) supplies the reference sidecars.
Source checkpoint SHA-256:
`031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56`.

Validated on CUDA (RTX 4070 Ti SUPER) against the golden clip
(`tests/parakeet_tdt/assets/2086-149220-0033.wav`): all four storage
variants reproduce the checked-in stock transcription byte-identically with
word timestamps — F32 (~210 ms, ~35x realtime), F16 (~141 ms, ~53x),
BF16 (~186 ms, ~40x), Q8_0 (~162 ms, ~46x); tested files are about 2.5 GB
for F32, 1.3 GB for F16 or BF16, and 916 MB for Q8_0. This checks port
correctness on one English clip; Orukeet's vendor-reported multilingual
gains (pooled FLEURS WER 9.85% vs 11.01%) are not independently validated
here, and its final adaptation trained on LibriSpeech test-other, so treat
the vendor's LibriSpeech numbers with caution.

License: Orukeet r3 weights and fitted kernels are CC BY-SA 4.0, retaining
NVIDIA's CC BY 4.0 foundation attribution; redistribution of converted GGUFs
carries the ShareAlike terms. See the upstream
[NOTICE](https://huggingface.co/oruk/orukeet/blob/main/NOTICE.md) and
[weight license](https://huggingface.co/oruk/orukeet/blob/main/LICENSE-WEIGHTS).
The converter records both in the staging `provenance.json`.

## Usage Notes

Request options are bare names. Session options use the
`parakeet_tdt.` prefix.

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
`parakeet_tdt.offline_mode=long_form` to process long audio with the same
bounded overlapping-window scheduler, or use `auto` to
switch after `parakeet_tdt.audio_chunk_threshold_sec` (30 seconds by
default). Long-form mode preserves global timestamps and predictor state while
decoding every center region exactly once. Because each region sees bounded
rather than utterance-wide context, its transcript can differ from the default
full-context result.

Offline long-form input is downmixed and resampled to mono 16 kHz before
windowing, using the same preprocessing as full-context mode. Streaming still
requires mono 16 kHz input. Request `audio_chunk_mode=fixed` selects this
bounded-window scheduler even when `offline_mode=full_context`. Set the session
option `parakeet_tdt.audio_chunk_duration_sec` for its center-window duration;
the request option `audio_chunk_duration_sec` only controls VAD chunk duration.

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

## Related Families

[Phonon-2 ASR](phonon2_asr.md), [Underdog Bark 3](underdog_bark3.md), and
[Haas Japanese ASR](haas.md) have dedicated families and usage guides while
sharing the Parakeet implementation. Select their own `--family` values.
