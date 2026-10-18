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

## Controls

Set model controls with `--request-option name=value`, or the server request's
`options` map. There are no model-specific load or session options.

| Option | Default | Meaning |
|---|---|---|
| `voice_id` | `tor` | Built-in speaker; reference audio overrides its embedding. |
| `reference_text` | Empty | Transcript accompanying the reference audio. |
| `x_vector_only_mode` | `false` | Omit the reference text/audio-token prefix. |
| `word_duration_frames` | Unset | JSON integer array, one duration per word; 25 frames/second, 0-399 frames per word. Generates the text in one chunk. |
| `context_text` | Unset | Previous utterance for prosodic conditioning. |
| `context_audio_file` | Unset | Previous-utterance WAV accompanying `context_text`. |
| `same_speaker` | `true` | Whether the previous utterance belongs to the target speaker. |
| `pronunciations` | Unset | JSON array of `{"word":"morgen","audio_file":"morgen.wav"}`. Uses at most 10 references, 60 codes each, 200 codes total. |
| `edit_start_frame`, `edit_end_frame` | Unset | Replace this half-open span of `voice_ref`, in 25-Hz frames. Supply both and the complete desired text. Returns the spliced full utterance; no text splitting. |
| `temperature` | `0.8` | Zero selects greedy decoding. |
| `top_k`, `top_p` | `50`, `0.95` | Sampling filters; `top_k=0` or `-1` disables top-k filtering. |
| `seed` | `0` | AR sampling and vocoder noise seed. |
| `max_tokens`, `min_tokens` | `500`, `0` | Per-chunk generated token limits. One audio token represents 40 ms. |
| `text_chunk_size` | `300` | Framework text-chunk budget in Unicode codepoints. |
| `text_chunk_mode` | `default` | `default`, `tag_aware`, `japanese`, or `endline`. |
| `silence_duration_sec` | `0.1` | Gap between generated text chunks. |

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
