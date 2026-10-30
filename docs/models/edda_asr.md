# Edda ASR (experimental)

The `edda_asr` community family runs
[Edda v0.2 large](https://huggingface.co/danish-foundation-models/edda-v0.2-large), a Danish
Whisper-large-v3 fine-tune by the Alexandra Institute and Danish Foundation Models.
It also runs [edda-v0.2-large-hurtig](https://huggingface.co/syvai/edda-v0.2-large-hurtig),
which adds a 170M-parameter DFlash2 drafter trained by syv.ai. The edda weights in that
repository are byte-identical to the original release; the drafter only speeds up
greedy decoding.

| Field | Value |
|---|---|
| Family | `edda_asr` |
| Task | `asr` |
| Modes | `offline`, `streaming` (complete input; one event per audio chunk) |
| Language | Danish (`da`). Other Whisper language tokens are accepted, but the drafter was trained on Danish only. |
| Decoding | Greedy with optional lossless speculative decoding, or Transformers-compatible beam search |
| Long-form audio | Split at silence into windows of at most 30 seconds |
| Timestamps | Not supported |

## How speculative decoding works

Edda's 32-layer decoder normally produces one token per forward pass. With the drafter,
each step drafts a block: the last verified token followed by 15 mask slots, all predicted
in one drafter pass. Edda then verifies the whole block in one forward pass. It keeps the
longest drafted prefix it agrees with, plus its own next token. Edda's greedy choice still
decides every token, so the transcript is edda's greedy transcript.

The drafter reads hidden states from edda decoder layers 1, 8, 15, 22 and 29 as extra
attention context, cross-attends to edda's encoder output, and uses DFlash2's grouped
dynamic causal convolutions and candidate selector. The whole drafter runs as one GGML
graph. The selector's greedy walk over the top 16 candidates per slot runs on the host.

Verifying 16 tokens costs little more than decoding one. On gfx1151 a 16-token verify pass
takes about 17.5 ms against 13.4 ms for a single token, and a drafter pass takes about
5.4 ms. With 5 to 10 tokens accepted per pass, the decoder runs 3 to 7 times faster.

Speculative and plain greedy output can differ when two tokens are almost equally likely.
A 16-token verify pass runs different kernels than a single-token pass, and the small
numeric difference can flip such a tie. The model card reports the same effect in fp16
(299 of 300 clips identical).

## Models

Download either Hugging Face repository as-is; the runtime reads the safetensors
directory directly.

| Package id | Source | Contents |
|---|---|---|
| `edda_v0_2_large_hurtig_safetensors` | [syvai/edda-v0.2-large-hurtig](https://huggingface.co/syvai/edda-v0.2-large-hurtig) | Edda plus the drafter (about 3.4 GB) |
| `edda_v0_2_large_safetensors` | [danish-foundation-models/edda-v0.2-large](https://huggingface.co/danish-foundation-models/edda-v0.2-large) | Edda only, greedy decoding (about 3.1 GB) |

No audio.cpp GGUF is published yet. The `edda_v0_2_large_hurtig_q8_0` and
`edda_v0_2_large_hurtig_f16` package entries describe the layout of a local conversion.

```bash
huggingface-cli download syvai/edda-v0.2-large-hurtig --local-dir models/edda-v0.2-large-hurtig
```

## CLI

```bash
audiocpp_cli --task asr --family edda_asr \
  --model models/edda-v0.2-large-hurtig \
  --backend cuda --audio danish.wav --log
```

With `--log`, each audio chunk reports its token count, target passes, tokens per pass and
accepted drafted tokens, plus `edda.encoder_ms`, `edda.decoder_ms` and `edda.drafter_ms`.

Request options:

| Option | Default | Meaning |
|---|---|---|
| `language` | `da` | Whisper language token in the decoder prompt. |
| `speculative` | `true` | Use the drafter when the model ships one. |
| `draft_block` | `16` | Anchor token plus up to `draft_block - 1` drafted tokens per verify pass (2 to 16). |
| `draft_selector` | `true` | Pick drafted tokens with the DFlash2 candidate selector instead of per-slot argmax. |
| `num_beams` | `1` | Beam-search width (up to 16). Above 1 the drafter is not used. Edda's own generation config uses 5. |
| `length_penalty` | `1.0` | Exponent that normalizes finished beam scores by generated length. |
| `max_tokens` | `444` | Generated-token limit per chunk, bounded by the 448-position decoder context. |
| `audio_chunk_mode` | `auto` | `auto` splits audio longer than 30 seconds at silence; `silence`, `fixed` and `none` as in other ASR families. |
| `audio_chunk_duration_sec` | `30` | Maximum chunk length when splitting. |

Session options:

| Option | Default | Meaning |
|---|---|---|
| `weight_type` | `native` | Matmul weight storage, for example `q8_0`. Applies to edda and the drafter. |
| `drafter` | `true` | Load the drafter. `false` saves its memory (about 340 MB in F16) and decodes greedily. |

## GGUF conversion

The converter packs edda, the drafter (under the `drafter.` prefix), the tokenizer and
both configs into one GGUF:

```bash
cmake --build build/release --target audiocpp_gguf
audiocpp_gguf --family edda_asr \
  --input models/edda-v0.2-large-hurtig/model.safetensors \
  --input drafter=models/edda-v0.2-large-hurtig/drafter.safetensors \
  --root models/edda-v0.2-large-hurtig \
  --output models/Edda-v0.2-Large-Hurtig-GGUF/edda-v0.2-large-hurtig-q8_0.gguf \
  --type q8_0
```

Drop the `drafter=` input to convert the plain edda release. On HIP, the encoder is faster
with F16 than with Q8_0 weights. Add `--keep-type 'model.encoder.*=f16'` to quantize only
the decoder and the drafter.

## Beam search

Edda's published WER uses 5-beam search, which is slower but somewhat more accurate than
greedy decoding. `--request-option num_beams=5` follows the Transformers beam search used by
`WhisperForConditionalGeneration.generate`: log-softmax before the suppress-token processors,
the top `2 * num_beams` continuations per step, length-normalized finished scores and the
default `early_stopping=False` stopping heuristic.

The beams share one KV cache as a token tree. Each beam's newest token takes a fresh cache
slot, and its attention mask admits only that beam's ancestors, so no cache is copied or
reordered when beams are pruned. The tree cache holds `prompt + num_beams * max_tokens`
slots (about 380 MB for 5 beams) and is allocated on the first beam-search request.

## Performance

Mean per clip over 59 FLEURS `da_dk` clips (11.3 seconds of audio on average), after one
warm-up clip, on an AMD Radeon 8060S (gfx1151, HIP). Encoder times vary by about ±30 ms
between runs.

| Mode | Weights | Encoder | Decoder | Total | Tokens per edda pass |
|---|---|---|---|---|---|
| Greedy (`speculative=false`) | F16 | 265 ms | 597 ms | 862 ms | 1.00 |
| Speculative | F16 | 265-325 ms | 193 ms | 519 ms | 5.24 |
| Speculative | Q8_0 decoder, F16 encoder GGUF | 277 ms | 152 ms | 428 ms | 5.24 |
| Beam search, 5 beams | F16 | 304 ms | 929 ms | 1234 ms | 0.98 |

The encoder always processes a full 30-second window, so it dominates short clips once
decoding is speculative. It uses flash attention over F16 keys and values; on this GPU,
ggml's HIP flash-attention tile kernel is its largest single cost.

## Validation

The validation clips are 5 NST Danish test sentences and the first 60 FLEURS `da_dk`
test clips by file name. References come from PyTorch FP32 on the CPU: syv.ai's
`hurtig.spec_generate` for speculative and greedy decoding (it equals Transformers greedy
`generate()` on all 60 FLEURS clips), and `generate(num_beams=5)` for beam search.

| Check | Result |
|---|---|
| NST, 5 clips, HIP F16, speculative | 5/5 identical to the reference |
| NST, 5 clips, F16 GGUF, Q8_0 GGUF, Q8_0 decoder with F16 encoder, HIP | 5/5 identical |
| NST, 5 clips, HIP F16, 5-beam search | 5/5 identical to Transformers beam search |
| NST, 1 clip, CPU backend | Identical |
| FLEURS, 60 clips, HIP F16, speculative | 59/60 identical to the reference |
| FLEURS, 60 clips, HIP F16, greedy | 59/60 identical to the reference |
| FLEURS, 60 clips, Q8_0 decoder with F16 encoder, speculative | 55/60 identical to the reference |
| NST sentences concatenated into 180 seconds, default silence chunking | 0.0% WER against the source sentences |

The single FLEURS difference in each F16 mode is a near-tie between two spellings of a name.
Speculative and greedy runs each miss a different one of the two such clips, in the same way
the model card reports 299 of 300 identical clips in fp16.

Known limitations:

- No timestamps.
- One request at a time per session (batch size 1).
