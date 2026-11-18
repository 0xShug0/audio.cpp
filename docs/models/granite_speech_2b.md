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
| `target_language` | `string` | auto (blank = ASR) | Translation target: en, fr, de, es, pt, ja, it, or zh (case-insensitive ASCII; Italian renders `Italian`, Mandarin renders `Mandarin` in the prompt). Blank (or omitted) means ASR transcription; setting a target selects speech translation into that language and the output is labeled with the normalized target. The source side is described by `language` (blank = auto). Unsupported codes reject loudly. |
| `max_tokens` | `int` | `512` | Maximum generated tokens per audio chunk. The processor-derived context bound prompt_len + max_tokens <= 16384 (from actual token IDs and resampled audio) is enforced per chunk; hitting the budget returns the generated prefix. Declared range: 1..16383. |
| `audio_chunk_mode` | `auto`, `fixed`, `vad`, `none` | `auto` | Offline chunk mode: auto (short inputs whole, long inputs VAD when a VAD model is configured, else fixed), fixed, vad (requires granite_speech_2b.vad_model_path), or none. |
| `audio_chunk_duration_sec` | `float` | `120` | Positive maximum audio chunk duration in seconds, in (0, 120]. Ignored in none mode. |

## Keyword Biasing & Speech Translation (Task F)

Both prompting features use the model card's preferred-prompt table verbatim;
the C++ prompts are verified token-for-token against the pinned processor
before any quality comparison (see `tests/granite_speech_2b/test_granite_speech_2b_f_units.cpp`
prompt-ID parity, 20 fixtures).

### Keyword biasing

Biasing terms travel in the transcript-text channel on every transport:

```bash
# CLI: terms are the --text value, comma-separated
audiocpp_cli --task asr --family granite_speech_2b --model path/to/checkpoint \
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

Measured (paired C++-native-CLI vs pinned-Python runs on identical audio;
short-form keyword prompts byte-identical everywhere):

- Rare-name recall (espeak-synthesized `f_kw_rare.wav`, ref exact by
  construction): baseline mangles all three names; with keywords the model
  recovers `quetzal` and `tlalnepantla` (case-insensitive) in every variant,
  and the full `Xochitl Quetzal` in the hinted run. `Zbigniew Brzezinski`
  never recovers (espeak renders it poorly — paired with/without delta on
  identical audio, not absolute quality).
- No-insertion probes (Librispeech 0000): absent terms
  (`xylophone, quantum`; accented `café, têntes, 日本語`) produce zero
  insertions — WER 0.00% vs the plain-ASR reference in all four rows, and
  the whitespace-only run is byte-identical to the golden transcript.
- Present terms (`Concord, tents`; multiwords; repeats) keep the transcript
  at WER 0.00% (recall 1.0 on in-vocabulary names).
- Aggregate over the 8 keyword manifest rows: precision 100.0%, recall
  75.0%, F1 85.7% (tp=9, fp=0, fn=3; fn are the unrecovered `Brzezinski`
  in two runs plus hintless `Xochitl`). Scored by
  `tests/granite_speech_2b/eval_metrics.py` keyword stats with transcript
  WER alongside.
- Parity note: on matched 16 kHz inputs the hinted rare run is byte-identical
  C++ vs Python; the hintless/baseline variants differ in filler words only
  while recovering the same keywords — chaotic greedy decode on robotic
  audio with flat posteriors, not a code defect (the committed 22050 Hz
  clip through the app resampler vs sox-16k wobbles the same way;
  clean-audio clips never do).

### Speech translation

```bash
audiocpp_cli --task asr --family granite_speech_2b --model path/to/checkpoint \
  --audio sample.wav --request-option language=en --request-option target_language=fr
```

`target_language` travels in request options (CLI `--request-option`,
JSON `options`); the single-file multipart transcription shape has no
options channel, so a `target_language` part there is rejected with a 400
naming the field (translation over HTTP is JSON-only by design; the batch
multipart `options` object is accepted by the parser but this family has no
native batching, so it cannot run there either). `target_language` absent/blank = plain ASR (unchanged behavior);
unsupported targets reject loudly. With keywords set, translation combines
with biasing (`translate ... Keywords: ...`); source hint adds a bare
`The speech is in .... ` prefix (the ASR do-not-translate clause is never
used for translation). The output language label is the normalized target.

Tested directions (short + long-form fixtures, implementer-authored refs;
C++ output vs pinned Python output per direction, chrF/BLEU vs refs reported
for both — corpus over the 12 manifest rows: chrF 66.04, BLEU 47.73):

| Direction | Short (chrF / BLEU) | Long (chrF / BLEU) | Notes |
|---|---|---|---|
| en→fr | 82.41 / 58.77 | 71.08 / 59.58 (63.9 s) | C++ == Python both |
| en→de | 55.49 / 31.24 | 71.63 / 59.60 | C++ == Python both |
| en→es | 79.87 / 80.71 | 64.75 / 45.13 | Long: one late word-order flip vs Python (chrF 64.75 vs 65.01, BLEU identical) — greedy cascade, see parity note |
| en→pt | 59.10 / 20.90 | 60.23 / 33.64 | Long: mid-text paraphrase flip vs Python (BLEU 33.64 vs 37.52) — greedy cascade, see parity note |
| en→it | 54.85 / 20.56 | 60.57 / 34.74 | Prompt-table target; eval-claims row covers en→it. C++ == Python both |
| ja→en | 83.80 / 61.05 | 65.48 / 49.95 (12.6 s synth) | Source hint ja exercised. C++ == Python both |
| en→ja, en→zh, fr/de/es/pt→en | smoke only | — | Implemented + prompt-parity covered; no qualified ref author (ja/zh) or no in-tree audio (fr/de/es/pt sources) — NOT advertised |

Parity note: 10 of 12 manifest directions are byte-identical C++ vs Python
(all shorts, fr/de/it longs, both ja→en). The en→es / en→pt longs diverge
late in the 235–242-token decode (same prefix, one paraphrase flip each) —
the fr/de/it longs match over the same length, so this is per-prompt greedy
cascade from sub-logit implementation noise (E2-gated math), not a defect;
scores are reported for both implementations. Combined keywords+translation
(`Concord, tents` + en→fr) is byte-identical with chrF 74.71 / BLEU 53.48.
The en→zh smoke run diverges at the first token (C++ `协议返回…` vs Python
`康科德回到…`) and is excluded from every claim — zh stays unadvertised.

Unmeasured directions stay unadvertised. The fr/de/es/pt→en directions have no
in-tree source audio (espeak French was tried: fluent-but-unfaithful output,
so synthetic speech is not a valid fixture source and is not used).
Plain ASR with no target set is unchanged (golden en/ja transcripts
identical).

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
