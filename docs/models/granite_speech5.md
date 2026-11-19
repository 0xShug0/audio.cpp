# IBM Granite Speech 5.0 2B in audio.cpp

IBM Granite Speech 5.0 2B is a ~2-billion-parameter autoregressive multilingual
Automatic Speech Recognition (ASR) model: a Conformer encoder, a Q-Former
projector, and a Granite text decoder. It transcribes English, French, German,
Spanish, Portuguese, and Japanese, and it writes punctuation and capitals
itself. A language hint selects the prompt language; without one the model
recognizes automatically with no language label.

Install the pinned IBM snapshot package (`granite_speech5_2b_safetensors`),
or convert it locally to GGUF using the recipe below. The default mixed-Q8 package requires upload before it can be installed
from the download catalog; the pinned native snapshot is available now.

## CLI Usage

### Offline Transcription

```bash
# Transcribe audio using the native safetensors checkpoint
audiocpp_cli --task asr --family granite_speech5 --model path/to/checkpoint --audio sample.wav --request-option language=en

```

### Quantized GGUF Loading

```bash
# Transcribe using a converted Q8_0 GGUF package
audiocpp_cli --task asr --family granite_speech5 --model models/Granite-Speech-5.0-2B-GGUF/granite-speech-5.0-2b-q8_0.gguf --audio sample.wav --request-option language=en

# The format-control orig GGUF preserves checkpoint tensor types
audiocpp_cli --task asr --family granite_speech5 --model models/Granite-Speech-5.0-2B-GGUF/granite-speech-5.0-2b-orig.gguf --audio sample.wav --request-option language=en
```

### Inspect And Install

```bash
# Inspect a standalone GGUF (embedded spec, sidecars, weights)
audiocpp_cli --inspect --family granite_speech5 --model models/Granite-Speech-5.0-2B-GGUF/granite-speech-5.0-2b-q8_0.gguf

# Force the embedded spec instead of the checkout's model_specs copy
audiocpp_cli --inspect --family granite_speech5 --model models/Granite-Speech-5.0-2B-GGUF/granite-speech-5.0-2b-q8_0.gguf --model-spec-override embedded

# Install the pinned IBM snapshot package (works today), or convert first
python3 tools/model_manager_v2.py info granite_speech5
python3 tools/model_manager_v2.py install granite_speech5_2b_safetensors --models-root models
# After the prebuilt Q8_0 remote upload, the default package installs directly:
# python3 tools/model_manager_v2.py install granite_speech5 --models-root models
```

### Conversion

Stage the checkpoint first: a directory with exactly the 8 loader-consumed
sidecars (`config.json`, `preprocessor_config.json`, `processor_config.json`,
`tokenizer.json`, `tokenizer_config.json`, `generation_config.json`,
`special_tokens_map.json`, `chat_template.jinja`) plus `model.safetensors`.
Do not stage `vocab.json`, `merges.txt`, `added_tokens.json` (content already
in `tokenizer.json`), `out_llm.safetensors` (unused CTC side output), or
sample audio.

```bash
# Format-control orig GGUF (native tensor keys preserved, checkpoint dtypes kept)
audiocpp_gguf \
  --input /path/to/stage/model.safetensors \
  --root /path/to/stage \
  --output /path/to/dist/granite-speech-5.0-2b-orig.gguf \
  --type orig \
  --family granite_speech5 \
  --overwrite

# Mixed Q8_0 GGUF: text-decoder matmuls quantize to Q8_0; the encoder,
# Q-Former, embedding table, and lm head stay BF16; 1-D tensors and the
# BatchNorm counters keep high precision
audiocpp_gguf \
  --input /path/to/stage/model.safetensors \
  --root /path/to/stage \
  --output /path/to/dist/granite-speech-5.0-2b-q8_0.gguf \
  --type q8_0 \
  --family granite_speech5 \
  --keep-type 'encoder.layers.0.conv.batch_norm.num_batches_tracked=native' \
  --keep-type 'encoder.layers.1.conv.batch_norm.num_batches_tracked=native' \
  --keep-type 'encoder.layers.2.conv.batch_norm.num_batches_tracked=native' \
  --keep-type 'encoder.layers.3.conv.batch_norm.num_batches_tracked=native' \
  --keep-type 'encoder.layers.4.conv.batch_norm.num_batches_tracked=native' \
  --keep-type 'encoder.layers.5.conv.batch_norm.num_batches_tracked=native' \
  --keep-type 'encoder.layers.6.conv.batch_norm.num_batches_tracked=native' \
  --keep-type 'encoder.layers.7.conv.batch_norm.num_batches_tracked=native' \
  --keep-type 'encoder.layers.8.conv.batch_norm.num_batches_tracked=native' \
  --keep-type 'encoder.layers.9.conv.batch_norm.num_batches_tracked=native' \
  --keep-type 'encoder.layers.10.conv.batch_norm.num_batches_tracked=native' \
  --keep-type 'encoder.layers.11.conv.batch_norm.num_batches_tracked=native' \
  --keep-type 'encoder.layers.12.conv.batch_norm.num_batches_tracked=native' \
  --keep-type 'encoder.layers.13.conv.batch_norm.num_batches_tracked=native' \
  --keep-type 'encoder.layers.14.conv.batch_norm.num_batches_tracked=native' \
  --keep-type 'encoder.layers.15.conv.batch_norm.num_batches_tracked=native' \
  --keep-type 'encoder.*=bf16' \
  --keep-type 'projector.*=bf16' \
  --keep-type 'language_model.model.embed_tokens.weight=bf16' \
  --keep-type 'language_model.lm_head.weight=bf16' \
  --overwrite
```

Stored-type layout of the Q8_0 artifact (954 tensors): 280 Q8_0
(the 40 text layers x q/k/o/v/gate/up/down projections), 658 BF16 (whole
encoder, Q-Former, embedding table, lm head, norms, biases), 16 I64
(BatchNorm `num_batches_tracked` counters). The orig artifact holds
938 BF16 + 16 I64.

## Model

| Field | Value |
|---|---|
| Family | `granite_speech5` |
| Tasks | `asr` |
| Modes | `offline` |
| Languages | `en`, `fr`, `de`, `es`, `pt`, `ja` |
| Default package | `granite_speech5_2b_q8_0` (prebuilt GGUF; installable once uploaded — until then install `granite_speech5_2b_safetensors` or convert locally with the recipe below) |
| Input | Speech audio |
| Output | Transcript text, no timestamps |

“Model/session default” means no explicit override. The runtime uses the
checkpoint or session setting, or leaves the optional value unset as described.

## Common Options (use directly)

| Option | Meaning |
|---|---|
| `--model` | Model checkpoint directory, standalone GGUF file, or installed package directory. |
| `--backend` | Select an available backend supported by this model. |
| `--threads` | CPU thread count. |
| `--log` | Enable runtime logging. |
| `--audio` | Input audio, where required by the selected task. |

## Request Options (use with `--request-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `language` | `string` | auto (blank) | Source language hint: en, fr, de, es, pt, or ja (case-insensitive ASCII). The explicit request option wins over text_input.language. Blank (or omitted) means automatic recognition: no hint text and no language label. |
| `target_language` | `string` | auto (blank = ASR) | Translation target: en, fr, de, es, pt, ja, it, or zh (case-insensitive ASCII; Italian renders `Italian`, Mandarin renders `Mandarin` in the prompt). Blank (or omitted) means ASR transcription; setting a target selects speech translation into that language and the output is labeled with the normalized target. The source side is described by `language` (blank = auto). Unsupported codes reject loudly. |
| `max_tokens` | `int` | `512` | Maximum generated tokens per audio chunk. The processor-derived context bound prompt_len + max_tokens <= 16384 (from actual token IDs and resampled audio) is enforced per chunk; hitting the budget returns the generated prefix. Declared range: 1..16383. |
| `audio_chunk_mode` | `auto`, `fixed`, `vad`, `none` | `auto` | Offline chunk mode: auto (short inputs whole, long inputs VAD when a VAD model is configured, else fixed), fixed, vad (requires granite_speech5.vad_model_path), or none. |
| `audio_chunk_duration_sec` | `float` | `120` | Positive maximum audio chunk duration in seconds, in (0, 120]. Ignored in none mode. |

## Keyword Biasing & Speech Translation

Both prompting features use the model card's preferred-prompt table verbatim;
the C++ prompts are verified token-for-token against the pinned processor
before any quality comparison (see `tests/granite_speech5/test_granite_speech5_f_units.cpp`
prompt-ID parity, 20 fixtures).

### Keyword biasing

Biasing terms travel in the transcript-text channel on every transport:

```bash
# CLI: terms are the --text value, comma-separated
audiocpp_cli --task asr --family granite_speech5 --model path/to/checkpoint \
  --audio sample.wav --request-option language=en \
  --text "Concord, tents, Tlalnepantla"

# JSON: the same terms in the "text" field
{"model": "...", "audio": "sample.wav", "language": "en",
 "text": "Concord, tents, Tlalnepantla"}
# Multipart: the "prompt" (or "text") part carries the terms
```

Rules: split on `,`, trim ASCII whitespace per term, drop empties,
preserve UTF-8 and multiword terms verbatim (no case folding, no dedup).
Whitespace-only text means no biasing (plain ASR path). The same term list
is applied to every audio chunk. Caller terms are checked against the 85
reserved control markers up front (a smuggled marker rejects loudly); the
per-chunk encode re-checks. IBM cites up to 2000 keywords: a 2000-term list
on a short clip (A=36, max_tokens=512) expands to prompt_len 7067 (text
7031), comfortably inside the runtime budget (7067 + 512 << 16384). The bound
is per-chunk: longer audio (larger A) or a larger max_tokens reduces the
usable term budget, so a 2000-term list that fits a 3.5 s clip can overflow
a 60 s one; a 20000-term list overflows even the short clip and is rejected
naming the 16384 limit. Padding trims ASCII whitespace only (see the
prompt.h contract): Unicode padding such as a fullwidth space is preserved
inside the term verbatim.

Performance and reference-comparison results are recorded in the
[validation report](../reports/granite_speech5_validation.md).

### Speech translation

```bash
audiocpp_cli --task asr --family granite_speech5 --model path/to/checkpoint \
  --audio sample.wav --request-option language=en --request-option target_language=fr
```

`target_language` travels in request options (CLI `--request-option`,
JSON `options`); the single-file multipart transcription shape has no
options channel. Use JSON for translation requests; do not pass
`target_language` as a multipart field. This family has no native batching. `target_language` absent/blank = plain ASR (unchanged behavior);
unsupported targets reject loudly. With keywords set, translation combines
with biasing (`translate ... Keywords: ...`); source hint adds a bare
`The speech is in .... ` prefix (the ASR do-not-translate clause is never
used for translation). The output language label is the normalized target.

Validated translation directions in the original port are English to French,
German, Spanish, Portuguese or Italian, and Japanese to English. See the
[validation report](../reports/granite_speech5_validation.md) for coverage
and numerical limitations.

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `granite_speech5.vad_model_path` | `path` | Model/session default | Path to a VAD weight file of any built-in VAD family (e.g. silero_vad_16k.safetensors), not a directory; VAD runs only when this path is a regular file. Defaults to assets/framework/models/silero_vad. |
| `granite_speech5.encoder_graph_arena_mb` | `int` | `1024` | Maximum encoder graph metadata budget in MiB; allocation is shape-sized. Must be positive. |
| `granite_speech5.projector_graph_arena_mb` | `int` | `1024` | Maximum projector graph metadata budget in MiB; allocation is shape-sized. Must be positive. |
| `granite_speech5.prefill_graph_arena_mb` | `int` | `1024` | Decoder prefill graph arena size in MB. Must be positive. |
| `granite_speech5.decode_graph_arena_mb` | `int` | `1024` | Decoder step graph arena size in MB. Must be positive. |

## Load Options (use with `--load-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `granite_speech5.weight_type` | `native`, `f32` | `native` | Shared model weight storage type. Native keeps the stored dtype (native checkpoint or GGUF mixed package types); f32 upcasts the BF16 snapshot for parity. |

## Architecture

- **Audio Frontend**: 16 kHz, 80-bin mel spectrogram (512 FFT, 400-sample window, 160-sample hop), stacked to 160-dim frames.
- **Audio Encoder**: 16-layer Conformer ($d_{model}=1024$, $d_{ff}=4096$, $d_{conv}=2048$, 8 heads, $head_{dim}=128$) with Shaw relative positional embeddings, depthwise convolution with eval batch-norm, and dual CTC feedback projections (`encoder.out`, `encoder.out_mid`).
- **Projector**: 2-layer Q-Former ($hidden=1024$, 16 heads) with 3 learnable queries, window 15, downsample 5, projecting to the 2048-dim text space.
- **Text Decoder**: 40-layer Granite decoder ($hidden=2048$, 16 query heads, 4 KV heads, $intermediate=4096$) with a 100353-token vocabulary, YARN RoPE (factor 4, 4096 base), and a 16384-token context bound. Greedy offline decoding, one chunk at a time.

## Audio Chunking & Join

Short inputs (at most 120 s) run whole for auto/fixed/vad; `none` always runs
whole. Long inputs split into at most 120 s spans (VAD half-second merge gaps
with quarter-second padding when a VAD file is configured, else fixed spans).
Tiny tails (at most 256 post-resample samples) merge into a neighbor; the
per-chunk transcripts join with a single space only between ASCII word
boundaries, so Japanese neighbors concatenate directly. Chunk boundaries and VAD planning are covered by model-specific tests.

## Validation

See the [validation report](../reports/granite_speech5_validation.md) for
original CPU/CUDA measurements and coverage limits. Those measurements
precede the graph-ownership cleanup and must not be treated as current
performance measurements of the refactored code.

## Word Timestamps

Word timestamps are not produced. `supports_timestamps` is false.

## License

Apache-2.0 source checkpoint
([ibm-granite/granite-speech-5.0-2b](https://huggingface.co/ibm-granite/granite-speech-5.0-2b));
see [Model licenses](../model_licenses.md). The Q8_0 GGUF package converts
this checkpoint and keeps its license.
