# Ming 3.0 TTS 0.5B

Ming uses a Qwen backbone, a continuous-latent flow-matching head, a waveform
VAE and CAMPPlus speaker conditioning. It supports Chinese, English, Japanese,
Korean, German, French and Spanish, with voice cloning and voice design.

GGUF packages: [audio-cpp/Ming-3.0-TTS-0.5B-GGUF](https://huggingface.co/audio-cpp/Ming-3.0-TTS-0.5B-GGUF).

Upstream: [inclusionAI/Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts).
Output is mono 44.1 kHz. Inference supports CUDA, Vulkan and CPU; only offline
generation is currently exposed.

## Usage

```bash
audiocpp_cli --family ming3_tts --task tts --backend cuda \
  --model /path/to/ming-3.0-tts-0.5b-bf16.gguf \
  --text "Good evening. The harbor is quiet tonight." \
  --request-option language=en --out speech.wav --log
```

The GGUF includes the backbone, flow head, audio VAE, CAMPPlus speaker encoder,
configuration and tokenizer. No separate codec or Python runtime is needed.
BF16 preserves the original checkpoint dtypes. Q8 retains the small stop head
in BF16 and the speaker encoder in F32.

### Voice Cloning

```bash
audiocpp_cli --family ming3_tts --task clon --backend cuda \
  --model /path/to/ming-3.0-tts-0.5b-bf16.gguf \
  --voice-ref reference.wav \
  --request-option reference_text="Words spoken in the reference." \
  --text "Please leave the lamp beside the window." \
  --request-option language=en --out clone.wav --log
```

With `reference_text`, the reference supplies both speaker identity and an
acoustic latent prefix. Omit the transcript to use speaker conditioning only.
Reference encoding samples the VAE posterior, matching the official pipeline's
RNG consumption.

### Voice Design

```bash
audiocpp_cli --family ming3_tts --task vdes --backend cuda \
  --model /path/to/ming-3.0-tts-0.5b-bf16.gguf \
  --text "Good evening. The harbor is quiet tonight." \
  --request-option language=en \
  --request-option instruct="A calm adult man with a clear, warm voice." \
  --out designed.wav --log
```

## Request Options

Use `--request-option name=value`, or the server request's `options` map.

| Option | Default | Description |
| --- | --- | --- |
| `template_name` | `tts`; `voice_design` on `vdes` | `tts`, `multilingual`, `basic`, `emotion`, `dialect`, `ip`, `voice_design`, or `freeform`. |
| `language` | Unset | Target language name or code: `zh`, `en`, `ja`, `ko`, `de`, `fr`, `es`. Required for `multilingual`; ignored by `dialect`. |
| `instruct` | Empty | Voice description for `voice_design`, required raw instruction for `freeform`, or optional raw instruction for `tts`. |
| `instruction_schema` | `en` for `dialect`, otherwise `zh` | Language of the structured instruction field names, not the spoken language. Values: `en`, `zh`. |
| `speed` | Unset | `basic` template: `慢速`, `中速`, `快速`. Categorical conditioning, not waveform resampling. |
| `pitch` | Unset | `basic` template: `低`, `中`, `高`. |
| `volume` | Unset | `basic` template: `低`, `中`, `高`. |
| `emotion` | Unset | Emotion label for the `emotion` template. |
| `dialect` | Unset | `dialect` template: `chuanyu dialect`, `cantonese`. |
| `voice_id` | Unset | `ip` template: `US_Emma`, `US_James`, `UK_George`, `UK_Charlotte`. |
| `reference_text` | Empty | Transcript accompanying `voice_ref`, enabling its acoustic prefix. |
| `speaker_embedding_file` | Unset | Precomputed CAMPPlus embedding file or list; see below. |
| `use_speaker_embedding` | `true` | Enable reference-speaker embedding conditioning. |
| `zero_speaker_embedding` | `false` | Use a zero projected speaker row, overriding file and reference conditioning. |
| `max_tokens` | `1000` | Maximum generated acoustic patches per text chunk, also bounded by the 512-position context. |
| `guidance_scale` | `2` | Conditioned flow prediction plus this scale times its difference from the null prediction. |
| `sigma` | `0.25` | Nonnegative flow SDE noise scale. |
| `temperature` | `0` | Nonnegative flow sampling temperature; zero still uses seeded initial noise. |
| `seed` | `1895` | Nonnegative reference-posterior and generation noise seed. |
| `raw_text` | `false` | Bypass text chunking and synthesize one fragment. |
| `text_chunk_size` | `300` | Framework text chunk budget in Unicode codepoints. Design/freeform remain unchunked unless explicitly set. |
| `text_chunk_mode` | `tag_aware` | `default`, `tag_aware`, `japanese`, or `endline`. |
| `trim_silence` | `true` | Trim trailing low-energy 100-ms frames per fragment, retaining a 300-ms tail. |

Structured templates accept only their matching controls: `basic` uses
`speed`/`pitch`/`volume`, `emotion` uses `emotion`, `dialect` uses `dialect`,
and `ip` uses `voice_id`. Supply at least one matching control. Do not combine
these templates with raw `instruct`; use `voice_design` for a voice description.

### Speaker Embedding Files

`speaker_embedding_file` accepts a 192-dimensional CAMPPlus embedding instead
of reference audio: a Torch checkpoint containing a root tensor or a dictionary
with `spk_emb`, `speaker_embedding`, `embedding` or `emb`, or the same named tensor
in Safetensors. Shapes `[192]` and `[1, 192]` are supported.

Supply one path with `--request-option speaker_embedding_file=embedding.pt`,
or a list in the server's `options` map. Files must be accessible to the server
process. Files supersede reference audio and its transcript;
`zero_speaker_embedding=true` supersedes both. With files provided,
`use_speaker_embedding=false` disables their conditioning without falling back
to reference audio.

## Session Options

Use `--session-option ming3_tts.name=value`, or the server model's
`session_options` map with the same family-prefixed key.

| Option | Default | Description |
| --- | --- | --- |
| `reference_cache` | `true` | Cache prepared conditioning for the most recently used reference waveform. |

There are no model-specific load options.

## Long Text

Normal synthesis uses the framework text chunker with a 300-codepoint budget.
This differs from the official Python semantic word/character chunker.
Voice design and freeform remain unchunked unless `text_chunk_size` is
explicitly supplied. `raw_text=true` bypasses chunking. The native frontend
does not run the optional official FST text normalizer; spell out numbers
and abbreviations when pronunciation matters.

Generation has a 512-position context shared between the prompt and emitted
acoustic patches. Reduce `text_chunk_size` or reference duration if the
conditioning leaves insufficient generation space. Each acoustic patch
represents approximately 80 ms. The default `max_tokens=1000` is additionally
bounded by that context. `trim_silence=true` trims each fragment's trailing
low-energy audio and retains a short tail.

The flow sampler remains stochastic at `temperature=0`; use `seed` for
controlled comparisons. `guidance_scale` controls flow guidance (default 2).
New and cached references consume different amounts of posterior noise,
as in the official implementation.

## Conversion

Use the complete official checkpoint directory, including `campplus.onnx`,
`model.safetensors` and the tokenizer files. Build `audiocpp_gguf` first.
The Python converter requires NumPy, ONNX and Safetensors; inference does not
require ONNX Runtime.

```bash
python tools/models/convert_ming3_tts_gguf.py \
  --source /path/to/Ming-3.0-tts-0.5B \
  --output /path/to/ming-3.0-tts-0.5b-bf16.gguf --type orig

python tools/models/convert_ming3_tts_gguf.py \
  --source /path/to/Ming-3.0-tts-0.5B \
  --output /path/to/ming-3.0-tts-0.5b-q8_0.gguf --type q8_0
```

Conversion extracts CAMPPlus parameters and embeds all required assets in each
GGUF. Model weights retain their upstream MIT license; see
[model licenses](../model_licenses.md).
