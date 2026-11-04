# Plapre 3

`plapre` is a Danish TTS model with a 335M-parameter Llama decoder,
Kanade 25-Hz speech tokens, a WavLM-based reference encoder, and a HiFT vocoder.
Output is mono 24 kHz. Upstream: [syvai/plapre-3](https://huggingface.co/syvai/plapre-3).

## Usage

```bash
audiocpp_cli --family plapre --task tts --backend cuda \
  --model /path/to/plapre-3-bf16.gguf \
  --text "Hej, hvordan har du det?" --request-option voice_id=tor --log

audiocpp_cli --family plapre --task clon --backend cuda \
  --model /path/to/plapre-3-bf16.gguf --voice-ref reference.wav \
  --reference-text "Ordene i referenceoptagelsen." \
  --text "I morgen kommer jeg igen." --log
```

`tts` and `clon` accept the same controls. Supplying reference audio uses its
speaker identity and audio-token prefix; the reference transcript is optional
but should match the recording when supplied. `x_vector_only_mode=true` uses
only the speaker identity, corresponding to upstream `speak(speaker_wav=...)`.
Built-in voices are `tor` (default), `ida`, `liv`, `ask`, and `kaj`.
The server's standard `"voice":"default"` selects the configured/default voice.

## Model

| Field | Value |
|---|---|
| Family | `plapre` |
| Tasks | `tts`, `clon` |
| Modes | `offline`, `streaming` |
| Languages | `da` |
| Default package | `plapre_3_bf16` |
| Input | Text; voice conditioning as shown in the examples |
| Output | Generated speech audio |

## Common Options (use directly)

| Option | Meaning |
|---|---|
| `--model` | Model file or directory shown in the examples. |
| `--backend` | Select an available backend supported by this model. |
| `--threads` | CPU thread count. |
| `--log` | Enable runtime logging. |
| `--text` | Input text; see the examples for the expected prompt format. |
| `--voice-ref` | Reference audio for voice cloning. |
| `--out` | Output WAV file. |

## Request Options (use with `--request-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `voice_id` | `tor`, `ida`, `liv`, `ask`, `kaj` | `tor` | Built-in speaker. |
| `reference_text` | `string` | `` | Transcript of the reference audio, used for voice cloning. |
| `x_vector_only_mode` | `bool` | `false` | Use only the reference speaker embedding, without the reference transcript/audio-token prefix. |
| `word_duration_frames` | `string` | Unset | JSON array of per-word integer durations, from 0 to 399 frames at 25 Hz. Duration-controlled text is generated in one chunk. |
| `context_text` | `string` | Unset | Previous utterance for prosodic context. |
| `context_audio_file` | `audio_path` | Unset | Optional previous-utterance WAV accompanying context_text. |
| `same_speaker` | `bool` | `true` | Whether context_text belongs to the target speaker. |
| `pronunciations` | `string` | Unset | Ordered JSON array of pronunciation references: [{"word":"morgen","audio_file":"morgen.wav"}]. Up to 10 references, 60 codes each and 200 total codes are used. |
| `edit_start_frame` | `int` | Unset | Start of the masked span in voice_ref, in 25-Hz codec frames. Supply both edit boundaries and the complete desired text. Declared range: minimum 0. |
| `edit_end_frame` | `int` | Unset | Exclusive end of the masked span in voice_ref. Editing runs in one text chunk. Declared range: minimum 0. |
| `max_tokens` | `int` | `500` | Maximum generated tokens per text chunk; each audio token represents 40 ms. Declared range: minimum 1. |
| `min_tokens` | `int` | `0` | Minimum generated tokens before a stop token can be selected. Declared range: minimum 0. |
| `temperature` | `float` | `0.8` | Sampling temperature; zero selects greedy decoding. Declared range: minimum 0. |
| `top_p` | `float` | `0.95` | Nucleus sampling probability in (0, 1]. Declared range: minimum 0, maximum 1. |
| `top_k` | `int` | `50` | Top-k sampling; -1 or zero disables filtering. Declared range: minimum -1. |
| `seed` | `int` | `0` | Sampling and vocoder noise seed. Declared range: minimum 0. |
| `text_chunk_size` | `int` | `300` | Maximum Unicode codepoints per long-form text chunk. Declared range: minimum 1. |
| `text_chunk_mode` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework text chunking policy. |
| `silence_duration_sec` | `float` | `0.1` | Silence inserted between text chunks. Declared range: minimum 0. |

## Usage Notes

Set model controls with `--request-option name=value`, or the server request's
`options` map. There are no model-specific load or session options.

Context/pronunciation files must be accessible to the CLI/server process.
Numbers are expanded into Danish words before tokenization. Chunking and
conditioning do not guarantee that the model follows every word or duration;
greedy generation can repeat or fail to stop on some inputs.
Short reference-conditioned utterances can also be sensitive to resampling
and encoder rounding. Use normal text chunking for longer passages; forcing
one large chunk can omit text in both C++ and official Python.

## Streaming

Both routes support `--mode streaming`. Audio events contain completed text
chunks, not incremental Kanade frames. The noncausal codec needs each chunk's
complete token sequence before decoding it. Offline and streaming use the same
synthesis path; concatenating audio deltas includes the inter-chunk gaps.

For the server, configure `"mode":"streaming"` and send a speech request with
`"response_format":"pcm","stream_format":"sse"`. See the
[server examples](../../examples/server/README.md#streaming-and-option-placement).

## Conversion

The BF16 (default) and F32 packages include the AR checkpoint, speaker projection and presets,
tokenizer, Kanade, the required WavLM layers, and CosyVoice2 HiFT tensors.
Python is needed for conversion, not C++ inference.

```bash
python tools/models/convert_plapre_gguf.py \
  --source /path/to/plapre-3 \
  --kanade /path/to/kanade-25hz-clean \
  --hift /path/to/CosyVoice2-0.5B/hift.pt \
  --speakers /path/to/plapre/plapre/speakers.json \
  --output /path/to/Plapre-3-GGUF/plapre-3-f32.gguf
```

The preset JSON comes from the official Plapre Python package. Without
`--speakers`, conversion reads that package's installed presets. Torchaudio
downloads WavLM Base+ weights on first conversion if they are not cached.

Convert the original package to the recommended BF16 package with:

```bash
audiocpp_gguf --input /path/to/Plapre-3-GGUF/plapre-3-f32.gguf \
  --output /path/to/Plapre-3-GGUF/plapre-3-bf16.gguf --type bf16 \
  --model-spec model_specs/plapre.json
```

BF16 and original F32 storage are validated. BF16 passed voice cloning and
built-in voice checks on CUDA, plus cloning checks on CPU and Vulkan. The tested
F16 package failed on a paragraph with nonfinite AR logits and is not offered.
CUDA uses F16 K/V for fused attention and reproduces codec BF16 activation
boundaries. The available flash-attention kernel uses F16 intermediates, so
native codec waveforms need not match Python exactly. Model weights retain
their upstream licenses; see
[model licenses](../model_licenses.md).
