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
| Decoding | Greedy, with optional lossless speculative decoding |
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

## Validation

The validation clips are 5 NST Danish test sentences and the first 60 FLEURS `da_dk`
test clips by file name. Speed is per clip, averaged after one warm-up clip.

| Check | Result |
|---|---|
| NST, 5 clips, HIP F16, against the reference `hurtig.spec_generate` (PyTorch FP32 CPU) | 5/5 identical transcripts |
| NST, 5 clips, F16 GGUF, Q8_0 GGUF, Q8_0 decoder with F16 encoder, HIP | 5/5 identical transcripts |
| NST, 1 clip, CPU backend | Identical transcript |
| FLEURS, 60 clips, HIP F16, speculative against greedy | 58/60 identical; both differences start at a near-tie |
| FLEURS tokens per edda pass | 5.24 (model card: 5.56 for its FLEURS sample) |

Known limitations:

- Greedy decoding only. Edda's published WER uses 5-beam search, which is somewhat more
  accurate than greedy.
- No timestamps.
- One request at a time per session (batch size 1).
