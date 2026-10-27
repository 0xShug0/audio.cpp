# Audio Flamingo 3, Next, and Music Flamingo

The `audio_flamingo` family supports Audio Flamingo 3, Audio Flamingo Next, and Music Flamingo.
Audio Flamingo checkpoints answer questions about speech, music, and environmental sounds,
and support transcription and audio captioning through the `asr` task. The input
is an audio file plus an instruction; the output is text.

| Variant | Audio limit | Architecture difference |
| --- | --- | --- |
| Audio Flamingo 3 | 10 minutes | Whisper-style encoder, two-layer projector, Qwen2 decoder. |
| Audio Flamingo Next | 30 minutes | Adds Rotary Time Embeddings and explicit audio boundary tokens. |
| Music Flamingo 2601 | 20 minutes | Uses the Next architecture with music-specialized weights, its own system prompt, and checkpoint-specific context/RoPE settings. |

## Usage

```bash
build/debug/bin/audiocpp_cli \
  --task asr --family audio_flamingo \
  --model /path/to/audio-flamingo-next-bf16.gguf \
  --backend cuda --audio input.wav \
  --request-option "instruct=Describe the sounds in this recording." \
  --request-option max_tokens=512 \
  --text-out response.txt --log
```

For transcription, use `instruct=Transcribe the input speech.`. For music, ask a
specific question, such as `Describe the instruments and tempo.`. Instructions
requesting speaker labels or timestamps produce model-generated text, not structured
diarization or forced-alignment results.

## Music Flamingo

Music Flamingo analyzes music; it does not generate audio. It shares the existing
encoder, projector, and decoder implementations. The packaged chat template
supplies its system prompt.

```bash
build/debug/bin/audiocpp_cli \
  --task asr --family audio_flamingo \
  --model models/Audio-Flamingo-GGUF/music-flamingo-2601-q8_0.gguf \
  --backend cuda --audio music.wav \
  --request-option "instruct=Describe the instruments, rhythm, and mood of this music." \
  --text-out description.txt --log
```

## SpotSound Temporal Grounding

SpotSound is a PEFT LoRA for **Audio Flamingo 3**, not Music Flamingo or Next.
Download `adapter_model.safetensors` and `adapter_config.json` from
[Loie/SpotSound](https://huggingface.co/Loie/SpotSound) into the same directory.
Supply the sound event as `instruct` (or `--text`) and select its prompt template:

```bash
build/debug/bin/audiocpp_cli \
  --task asr --family audio_flamingo \
  --model models/Audio-Flamingo-GGUF/audio-flamingo-3-bf16.gguf \
  --backend cuda --audio recording.wav \
  --session-option audio_flamingo.lora=adapters/SpotSound \
  --request-option template_name=temporal_grounding \
  --request-option "instruct=ambulance siren" \
  --text-out intervals.txt --log
```

This template adds the upstream task instruction and timestamp labels for each
group of 25 audio tokens. It retains complete seconds of features, matching
upstream processing; a final fractional second is omitted, and recordings shorter
than one second are rejected. Results are model-generated timestamp text, not
guaranteed boundaries or structured alignment results. Ordinary requests use
`template_name=default` and keep the original prompt layout.

Adapters are merged into session weights using the shared LoRA loader, with no
extra inference graph operations. The same session/request options work in the
server. Omit `audio_flamingo.lora` to restore the base model. Other matching PEFT
adapters can use the default template. See [LoRA](../lora.md) for merged-GGUF export.

## Audio Input

Use a 16-kHz mono WAV when comparing with Python. Python's optional TorchCodec
loader and its librosa fallback use different downmixing and resampling paths,
which can change this model's response substantially. Pass the same decoded WAV
to both implementations for a meaningful comparison. audio.cpp averages channels
and uses SOXR for inputs that require resampling.

## Server

Configure the model with family `audio_flamingo`, task `asr`, and mode
`offline`. Send audio and an instruction through the transcription endpoint:

```bash
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model=audio-flamingo-next \
  -F file=@input.wav \
  -F "prompt=Describe the sounds in this recording."
```

The response contains `text`. For additional request options, use the endpoint's
JSON form with an `audio_path` and an `options` object.

## Common Options

Use these as ordinary CLI flags.

| Option | Values | Description |
| --- | --- | --- |
| `--audio` | Audio file | Input recording. Audio is mixed to mono and resampled to 16 kHz. |
| `--text` | Text | Instruction when `instruct` is not supplied. |
| `--text-out` | File path | Save the response as text. |

## Request Options

Use these with `--request-option key=value`.

| Option | Values | Default | Description |
| --- | --- | --- | --- |
| `instruct` | Text | `Transcribe the input speech.` | Question or instruction about the audio. |
| `template_name` | `default`, `temporal_grounding` | `default` | Use ordinary audio questions or the SpotSound event-query layout. |
| `max_tokens` | Positive integer | see description | Maximum generated response tokens. Defaults to checkpoint generation configuration (2048 fallback); prompt plus output budget must fit model positions. |
| `do_sample` | Boolean | `false` | Enable sampling instead of greedy decoding. |
| `temperature` | Positive number | `1` | Sampling temperature. Used only when do_sample=true; must be positive when sampling. |
| `top_k` | Signed integer | `50` | Sampling candidate count; nonpositive values disable top-k filtering. Used only when sampling. |
| `top_p` | Non-negative number | `1` | Nucleus sampling probability. Used only when sampling; zero is accepted and values >=1 disable nucleus filtering. |
| `repetition_penalty` | Positive number | `1` | Token repetition penalty. Must be positive. |
| `seed` | Signed integer | `42` | Sampling seed. Parsed as a signed integer and converted to uint32. |

Temperature, top-k, top-p, and seed affect sampled generation only.

## Session Options

Use these with `--session-option key=value`.

| Option | Values | Default | Description |
| --- | --- | --- | --- |
| `audio_flamingo.weight_type` | Framework weight storage types | `native` | Tensor storage type for the encoder, projector, and language model. |
| `audio_flamingo.lora` | File or directory path | Unset | Matching PEFT adapter; keep its configuration beside the weights. |
| `audio_flamingo.lora_scale` | Finite number | `1` | Multiplier for the adapter's alpha/r coefficient. |

## Long Audio

Audio Flamingo 3 accepts up to 10 minutes, Audio Flamingo Next up to
30 minutes, and Music Flamingo up to 20 minutes. The encoders process 30-second windows, and the language model attends
to the combined audio context. This is offline processing, not live audio streaming.
Longer input increases context memory and prefill time. Inputs beyond the selected
checkpoint's duration limit are rejected rather than silently truncated.

## Conversion

Download the official Transformers checkpoint, including its config and tokenizer
files. No model-specific conversion script is needed.

```bash
# Audio Flamingo 3
build/debug/bin/audiocpp_gguf \
  --input weights=/path/to/audio-flamingo-3-hf/model.safetensors \
  --root /path/to/audio-flamingo-3-hf \
  --family audio_flamingo \
  --model-spec model_specs/audio_flamingo.json \
  --type orig --output audio-flamingo-3-bf16.gguf

# Audio Flamingo Next
build/debug/bin/audiocpp_gguf \
  --input weights=/path/to/audio-flamingo-next-hf/model.safetensors \
  --root /path/to/audio-flamingo-next-hf \
  --family audio_flamingo \
  --model-spec model_specs/audio_flamingo.json \
  --type orig --output audio-flamingo-next-bf16.gguf

# Music Flamingo
build/debug/bin/audiocpp_gguf \
  --input weights=/path/to/music-flamingo-2601-hf/model.safetensors \
  --root /path/to/music-flamingo-2601-hf \
  --family audio_flamingo \
  --model-spec model_specs/audio_flamingo.json \
  --type q8_0 --output music-flamingo-2601-q8_0.gguf
```

The original tensors are BF16. The resulting GGUF includes the tokenizer, configs,
and model spec, so inference does not require the original checkpoint directory.

## Upstream

- [NVIDIA Audio Flamingo 3](https://huggingface.co/nvidia/audio-flamingo-3-hf)
- [NVIDIA Audio Flamingo Next Instruct](https://huggingface.co/nvidia/audio-flamingo-next-hf)
- [NVIDIA Music Flamingo 2601](https://huggingface.co/nvidia/music-flamingo-2601-hf)
- [SpotSound implementation](https://github.com/LoieSun/SpotSound)
- The model cards specify noncommercial terms. Review the upstream licenses before
  use or redistribution.
