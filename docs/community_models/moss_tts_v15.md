# MOSS-TTS-v1.5

Zero-shot voice cloning from a short reference recording, on the 8B delay-pattern
member of the MOSS-TTS family.

- Family: `moss_tts_v15`
- Tasks: `tts`, `clon` (cloning), offline
- Languages: English, Chinese
- Weights: [OpenMOSS-Team/MOSS-TTS-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5), Apache-2.0
- Codec weights: [OpenMOSS-Team/MOSS-Audio-Tokenizer](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer), Apache-2.0
- Architecture: `moss_tts_delay` — a Qwen3-8B backbone with 32 audio codebook
  embeddings and `1 + 32` output heads, decoded on a delay pattern
- Codec: MOSS-Audio-Tokenizer v1, 24 kHz mono, hop 1920 (12.5 frames a second)

It shares its architecture with `moss_voicegen`, which is the same design at
Qwen3-1.7B and 16 codebooks. The backbone, heads, delay decoder and config parser
are shared framework code (`engine/framework/decoders/moss_tts_delay/`); what is
specific to this checkpoint is the prompt and the session.

## Usage

Cloning, which is what this checkpoint is for:

```bash
audiocpp_cli --family moss_tts_v15 --model <model-dir> --task clon \
  --voice-ref reference.wav \
  --text "This sentence should be spoken in the voice from the reference recording." \
  --out out.wav
```

Without a reference, it is plain TTS and the voice is whatever the model picks:

```bash
audiocpp_cli --family moss_tts_v15 --model <model-dir> --task tts \
  --text "The quick brown fox jumps over the lazy dog." --out out.wav
```

The reference is resampled and downmixed to the codec's 24 kHz mono and trimmed to
whole frames, so any sample rate and channel count will do. It is encoded once per
request, not once per text chunk.

### Options

| option | meaning |
|---|---|
| `--voice-ref <wav>` | Reference recording to clone. |
| `--tokens <int>` | Duration budget in codec frames at 12.5 a second — the model's own `- Tokens:` field. 40 tokens is about 3.2 s. |
| `--instruct <text>` | Voice description. **See the limitation below.** |
| `--language <name>` | Full language name; the model does not understand codes like `en`. |
| `--seed`, `--temperature`, `--top_p`, `--top_k`, `--repetition_penalty` | Sampling. |
| `moss_tts_v15.weight_type` | `native`, `f32`, `bf16` (default) or `q8_0`. |

`f16` is rejected for the backbone: its attention-sink activations run far past
f16's range and produce NaN from the first position. This is inherited from the
family and is the same on `moss_voicegen`.

## ⚠ Limitation: voice-attribute instructions are not reliably followed

`--instruct` is accepted, but on this checkpoint a description of the *speaker* is
followed only loosely. Measured on the reference implementation, four prompts
differing only in the requested speaker, median F0 of the result:

```
"A high-pitched young woman's voice, clearly female."   -> 216.3 Hz   plausible
"A woman speaking softly and warmly."                   -> 184.5 Hz   borderline
"A man speaking in a low register."                     -> 187.8 Hz   miss
"A very deep, low-pitched man's voice."                 -> 180.8 Hz   clear miss
```

Five samples with median F0 as a proxy is not a rigorous evaluation, and one of the
four was plausible — so this supports "unreliable", not "never works". But a "very
deep, low-pitched man's voice" at 181 Hz is wrong by any measure.

**Use a reference recording when the voice matters.** Cloning does work: a clone of
a 189.4 Hz reference came back at 184.0 Hz, where the same model without a reference
produced 117.3 Hz. If you want a voice from a written description rather than a
recording, `moss_voicegen` is the model built for that.

## Performance and VRAM

Measured on an RTX 3090 (24 GiB) and a 16-thread CPU. The times are
`session.wall_ms` — the work itself, excluding model load, which for a 9 GB
package is tens of seconds of disk I/O and swamps everything else if you time
the whole process.

| package | backend | RTF | peak VRAM (cloning) |
|---|---|---|---|
| q4_k | CUDA | 0.65 | 14.1 GiB |
| q8_0 | CUDA | 0.77 | 17.9 GiB |
| bf16 | CUDA | — | does not fit in 24 GiB when cloning |
| q4_k | CPU, 16 threads | ~16 | — |

CUDA is roughly 25-30x faster than the CPU path here, and comfortably faster
than real time; the CPU path is not.

**`weight_type` defaults to `native`, and you want to keep it there.** A GGUF
package carries its own type; forcing `bf16` dequantises it on load, so a
quantised package costs exactly as much memory as bf16 and the quantisation
buys nothing. Measured: q4_k peaks at 21.2 GiB forced to `bf16` against
10.6 GiB left `native`. For safetensors it makes no difference — those are bf16
upstream anyway.

**bf16 cannot clone on a 24 GiB card.** Cloning holds both halves of the codec
resident, where plain TTS needs only the decoder, and the encoder's weights are
the allocation that does not fit. Plain TTS works at every size. Use q8_0 on a
24 GiB card, or bf16 on something larger.

## What works

Measured on the reference implementation and reproduced through audio.cpp:

| | |
|---|---|
| voice cloning | tracks the reference closely |
| Chinese | works, from an English instruction |
| long form | 20.4 s from four sentences; RTF *improves* with length |
| `tokens` budget | honoured — 40 tokens produced 41 frames |

## Packaging

For safetensors, the model and the audio tokenizer are separate Hugging Face
repositories. Place the tokenizer snapshot under `audio_tokenizer/` inside the model
root — a real directory, not a symlink, which the GGUF converter does not follow:

```text
MOSS-TTS-v1.5/
  config.json
  tokenizer.json
  tokenizer_config.json
  merges.txt
  model.safetensors.index.json
  model-0000{1..4}-of-00004.safetensors
  audio_tokenizer/
    config.json
    model.safetensors.index.json
    model-0000{1,2}-of-00002.safetensors
```

Building the GGUF package:

```bash
audiocpp_gguf \
  --input model_weights=<root>/model.safetensors.index.json \
  --input audio_tokenizer_weights=<root>/audio_tokenizer/model.safetensors.index.json \
  --output moss_tts_v15_bf16_codec_f16.gguf \
  --type bf16 --keep-type "audio_tokenizer_weights*=f16" \
  --family moss_tts_v15 --root <root>
```

That produces about 20.5 GB with 2063 tensors, the model spec and 22 sidecars
embedded. Unlike `moss_voicegen`, the codec encoder is included as well as the
decoder, because cloning needs it.
