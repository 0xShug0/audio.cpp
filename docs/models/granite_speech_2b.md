# IBM Granite Speech 5.0 2B in audio.cpp

IBM Granite Speech 5.0 2B is a ~2-billion-parameter autoregressive multilingual
Automatic Speech Recognition (ASR) model: a Conformer encoder, a Q-Former
projector, and a Granite text decoder. It transcribes English, French, German,
Spanish, Portuguese, and Japanese, and it writes punctuation and capitals
itself. A language hint selects the prompt language; without one the model
recognizes automatically with no language label.

> Experimental: native safetensors plus staged GGUF packages. The
> `granite_speech_2b_q8_0` package entry is staged (remote upload and the
> community flip land only after the runtime validation gates pass).

## CLI Usage

### Offline Transcription

```bash
# Transcribe audio using the native safetensors checkpoint
audiocpp_cli --task asr --family granite_speech_2b --model path/to/checkpoint --audio sample.wav --request-option language=en

```

### Quantized GGUF Loading

```bash
# Transcribe using a converted Q8_0 GGUF package
audiocpp_cli --task asr --family granite_speech_2b --model models/Granite-Speech-5.0-2B-GGUF/granite-speech-5.0-2b-q8_0.gguf --audio sample.wav --request-option language=en

# The format-control orig GGUF (all tensors in checkpoint precision) behaves identically
audiocpp_cli --task asr --family granite_speech_2b --model models/Granite-Speech-5.0-2B-GGUF/granite-speech-5.0-2b-orig.gguf --audio sample.wav --request-option language=en
```

### Inspect And Install

```bash
# Inspect a standalone GGUF (embedded spec, sidecars, weights)
audiocpp_cli --inspect --family granite_speech_2b --model models/Granite-Speech-5.0-2B-GGUF/granite-speech-5.0-2b-q8_0.gguf

# Force the embedded spec instead of the checkout's model_specs copy
audiocpp_cli --inspect --family granite_speech_2b --model models/Granite-Speech-5.0-2B-GGUF/granite-speech-5.0-2b-q8_0.gguf --model-spec-override embedded

# Staged install target (remote upload pending validation)
python3 tools/model_manager_v2.py info granite_speech_2b
python3 tools/model_manager_v2.py install granite_speech_2b --models-root models
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
  --family granite_speech_2b \
  --overwrite

# Mixed Q8_0 GGUF: text-decoder matmuls quantize to Q8_0; the encoder,
# Q-Former, embedding table, and lm head stay BF16; 1-D tensors and the
# BatchNorm counters keep high precision
audiocpp_gguf \
  --input /path/to/stage/model.safetensors \
  --root /path/to/stage \
  --output /path/to/dist/granite-speech-5.0-2b-q8_0.gguf \
  --type q8_0 \
  --family granite_speech_2b \
  --keep-type 'encoder.*=bf16' \
  --keep-type 'projector.*=bf16' \
  --keep-type 'language_model.model.embed_tokens.weight=bf16' \
  --keep-type 'language_model.lm_head.weight=bf16' \
  --overwrite
```

Stored-type layout of the staged Q8_0 artifact (954 tensors): 280 Q8_0
(the 40 text layers x q/k/o/v/gate/up/down projections), 658 BF16 (whole
encoder, Q-Former, embedding table, lm head, norms, biases), 16 I64
(BatchNorm `num_batches_tracked` counters). The orig artifact holds
938 BF16 + 16 I64.

## Model

| Field | Value |
|---|---|
| Family | `granite_speech_2b` |
| Tasks | `asr` |
| Modes | `offline` |
| Languages | `en`, `fr`, `de`, `es`, `pt`, `ja` |
| Default package | `granite_speech_2b_q8_0` (staged, experimental) |
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
| `max_tokens` | `int` | `512` | Maximum generated tokens per audio chunk. The processor-derived context bound prompt_len + max_tokens <= 16384 (from actual token IDs and resampled audio) is enforced per chunk; hitting the budget returns the generated prefix. Declared range: 1..16383. |
| `audio_chunk_mode` | `auto`, `fixed`, `vad`, `none` | `auto` | Offline chunk mode: auto (short inputs whole, long inputs VAD when a VAD model is configured, else fixed), fixed, vad (requires granite_speech_2b.vad_model_path), or none. |
| `audio_chunk_duration_sec` | `float` | `120` | Positive maximum audio chunk duration in seconds, in (0, 120]. Ignored in none mode. |

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `granite_speech_2b.vad_model_path` | `path` | Model/session default | Path to a VAD weight file of any built-in VAD family (e.g. silero_vad_16k.safetensors), not a directory; VAD runs only when this path is a regular file. Defaults to assets/framework/models/silero_vad. |
| `granite_speech_2b.encoder_graph_arena_mb` | `int` | `1024` | Conformer encoder graph arena size in MB. Must be positive. |
| `granite_speech_2b.projector_graph_arena_mb` | `int` | `1024` | Q-Former projector graph arena size in MB. Must be positive. |
| `granite_speech_2b.prefill_graph_arena_mb` | `int` | `1024` | Decoder prefill graph arena size in MB. Must be positive. |
| `granite_speech_2b.decode_graph_arena_mb` | `int` | `1024` | Decoder step graph arena size in MB. Must be positive. |

## Load Options (use with `--load-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `granite_speech_2b.weight_type` | `native`, `f32` | `native` | Shared model weight storage type. Native keeps the stored dtype (native checkpoint or GGUF mixed package types); f32 upcasts the BF16 snapshot for parity. |

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
boundaries, so Japanese neighbors concatenate directly. The longest fixture
transcribed whole here is the 14.2 s Librispeech clip; VAD planning was gated
on the same clip in the session tests.

## Measured Hardware Usage

CPU-only, 8 threads, Q8_0 GGUF, 14.2 s English clip
(`librispeech_test_clean_6930-75918-0001.wav`, `language=en`):

- Peak RSS: 4.2 GiB (4,429,628 kB)
- Wall: 9.7 s end to end (RTF ~0.68 on this box)

The decoder weight context reserves 8 GiB of virtual address space up front
(C2 default); resident use stays near 4 GiB with the default four 1 GiB graph
arenas. The arbiter is the 2 B-parameter text decoder: the Q8_0 package keeps
its resident weights near 2.6 GB on top of the arenas. Native-checkpoint runs
additionally hold the 4.6 GB BF16 snapshot mapping. No CUDA/Metal run has
been measured yet; backend tags stay CPU-only until the validation gates
measure them.

## Word Timestamps

Word timestamps are not produced. `supports_timestamps` is false.

## License

Apache-2.0 source checkpoint
([ibm-granite/granite-speech-5.0-2b](https://huggingface.co/ibm-granite/granite-speech-5.0-2b));
see [Model licenses](../model_licenses.md). The staged GGUF packages convert
this checkpoint and keep its license.
