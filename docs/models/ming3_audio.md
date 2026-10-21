# Ming 3.0 Audio Tiny

Ming 3.0 Audio combines a Whisper audio encoder, a hybrid KDA/MLA
mixture-of-experts Thinker, and an independent continuous-latent Talker with
an audio VAE. The Thinker produces text; the Talker synthesizes that text.

Upstream checkpoint: [inclusionAI/Ming-3.0-audio-tiny](https://huggingface.co/inclusionAI/Ming-3.0-audio-tiny/tree/25b96e30be3739849bf76c0f48ac11db4a262fb3).
Code: [inclusionAI/Ming-3.0-audio-flash](https://github.com/inclusionAI/Ming-3.0-audio-flash).
GGUF: [audio-cpp/Ming-3.0-Audio-Tiny-GGUF](https://huggingface.co/audio-cpp/Ming-3.0-Audio-Tiny-GGUF).

The standalone GGUF includes the Thinker, Talker, audio VAE, CAMPPlus speaker
encoder, tokenizers, and configuration. No separate Python runtime or codec
download is required. Speech output is mono 44.1 kHz. Only offline sessions
are exposed.

## Usage

### Transcription And Audio Understanding

```bash
audiocpp_cli --family ming3_audio --task asr --backend cuda \
  --model /path/to/ming-3.0-audio-tiny-bf16.gguf \
  --audio recording.wav --log
```

The default prompt returns a language label followed by a tab and the transcript.
Use `--request-option prompt="Describe the sounds in this recording."` for
audio understanding instead of transcription.

### Named Voice

```bash
audiocpp_cli --family ming3_audio --task tts --backend cuda \
  --model /path/to/ming-3.0-audio-tiny-bf16.gguf \
  --text "The sun warms the earth, and a soft breeze moves through the garden." \
  --request-option voice_id=US_James --request-option language=en \
  --out speech.wav --log
```

### Reference Voice

```bash
audiocpp_cli --family ming3_audio --task clon --backend cuda \
  --model /path/to/ming-3.0-audio-tiny-bf16.gguf \
  --voice-ref reference.wav --text "Please leave the lamp beside the window." \
  --request-option language=en --out clone.wav --log
```

Reference audio supplies speaker identity. Adding
`--request-option reference_text="Words spoken in the reference."` also supplies
an acoustic prefix. Named voices are used only when reference audio is absent.
Both `tts` and `clon` accept reference audio.

Prefer a reference in the target language: cross-language cloned pronunciation
can drift even when the generated text is correct.

### Spoken Response

Use `--task s2s --audio question.wav --out response.wav` with the same model
and backend. The Thinker answers the question in the recording, and the Talker
speaks its answer. `prompt` overrides the Thinker instruction; `voice_id` or
`--voice-ref` controls the response voice.

## Controls

Use `--request-option name=value`, or the server request's `options` map.

| Option | Default | Meaning |
| --- | --- | --- |
| `prompt` | Task-dependent | Thinker instruction for input audio. |
| `system_prompt` | Empty | Thinker system instruction; selects the official chat-style text-before-audio message layout. |
| `enable_thinking` | `false` | Enable detailed thinking in the Thinker chat template. |
| `max_tokens` | `256` | Maximum generated Thinker text tokens. |
| `max_audio_tokens` | `1000` | Maximum generated Talker acoustic patches per fragment. |
| `voice_id` | `灵小柔` | Official named voice, used without reference audio; see the spec for accepted names. |
| `language` | Unset | Target Talker language; Chinese and English are the primary supported languages. |
| `reference_text` | Empty | Reference transcript enabling acoustic-prefix conditioning. |
| `seed` | `42` | Talker and reference-posterior sampling seed; Thinker decoding is greedy. |
| `guidance_scale` | `2` | Conditioned flow prediction plus this scale times its difference from the null prediction. |
| `sigma` | `0.25` | Flow SDE noise scale. |
| `temperature` | `0` | Flow sampling temperature; zero still uses seeded initial noise. |
| `raw_text` | `false` | Synthesize one fragment without text chunking. |
| `text_chunk_size` | `300` | Framework chunk budget in Unicode codepoints. |
| `text_chunk_mode` | `tag_aware` | `default`, `tag_aware`, `japanese`, or `endline`. |
| `trim_silence` | `true` | Trim trailing low-energy frames, retaining a short tail. |

Session option `ming3_audio.reference_cache=true` caches prepared conditioning
for the most recent reference waveform. There are no model-specific load options.

`--session-option ming3_audio.disk_experts=true` keeps Thinker expert weights in
host memory and stages selected experts into a bounded GPU cache. Inference
still runs on the selected GPU. `ming3_audio.pin_experts=true` retains expert
payloads in pinned RAM; set it to `false` for range reads from the GGUF instead.
`ming3_audio.expert_cache_size` selects 8 to 128 slots per layer (default 8).
Spoken-response sessions release Thinker workspaces before allocating the speech
stage in this mode. The offloaded Thinker allocates its KV cache in growing
buckets rather than reserving the entire `max_tokens` allowance upfront.
Growth preserves the existing KV and recurrent state without changing precision
or reducing the token limit. Memory still grows with actual context length.
The normal resident path remains the default.

The native frontend uses the framework text chunker rather than Python's
word/character buffering and does not run the optional Python FST normalizer.
Spell out numbers and abbreviations when pronunciation matters. Long text is
synthesized as independent fragments. Reference conditioning is retained across
fragments; generated conversation history is not.

## Conversion

Build the Debug `audiocpp_gguf` target, then convert the complete official snapshot:

```bash
python tools/models/convert_ming3_audio_gguf.py \
  --source /path/to/Ming-3.0-audio-tiny \
  --output /path/to/ming-3.0-audio-tiny-bf16.gguf \
  --type orig --temporary-dir /path/to/conversion-work --log
```

The converter stacks MoE expert banks without changing their BF16 values and
keeps expert routers in F32. Training-only next-token prediction weights are
not included. Conversion requires PyTorch, NumPy, ONNX, Safetensors, and GGUF
Python packages; inference does not require them. The checkpoint is MIT-licensed.

`--type q8_0` quantizes only Thinker expert banks. `--type q4_k` uses Q4_K
experts plus a Q8_0 text output head to reduce GPU-resident weights during
expert offload. Both retain source precision for token embeddings, attention and
dense/shared MLP projections: broader projection quantization caused repetition
or prolonged generation on the full meeting examples. Both
preserve source precision for the audio encoder, Talker, VAE and speaker
encoder, and keep routers F32. TTS-only memory usage therefore does not shrink
with these Thinker quantizations.
