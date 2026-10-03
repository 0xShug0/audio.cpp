# Audio Flamingo 3 and Next

The `audio_flamingo` family supports Audio Flamingo 3 and Audio Flamingo Next.
Both checkpoints answer questions about speech, music, and environmental sounds,
and support transcription and audio captioning through the `asr` task. The input
is an audio file plus an instruction; the output is text.

| Variant | Audio limit | Architecture difference |
| --- | --- | --- |
| Audio Flamingo 3 | 10 minutes | Whisper-style encoder, two-layer projector, Qwen2 decoder. |
| Audio Flamingo Next | 30 minutes | Adds Rotary Time Embeddings and explicit audio boundary tokens. |

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
| `max_tokens` | Positive integer | `2048` | Maximum number of generated response tokens. |
| `do_sample` | Boolean | `false` | Enable sampling instead of greedy decoding. |
| `temperature` | Positive number | `1` | Sampling temperature. |
| `top_k` | Non-negative integer | `50` | Sampling candidate count; `0` disables top-k filtering. |
| `top_p` | Number in `(0, 1]` | `1` | Nucleus sampling probability. |
| `repetition_penalty` | Positive number | `1` | Token repetition penalty. |
| `seed` | Non-negative integer | `42` | Sampling seed. |

Temperature, top-k, top-p, and seed affect sampled generation only.

## Session Options

Use these with `--session-option key=value`.

| Option | Values | Default | Description |
| --- | --- | --- | --- |
| `audio_flamingo.weight_type` | Framework weight storage types | `native` | Tensor storage type for the encoder, projector, and language model. |

## Long Audio

Audio Flamingo 3 accepts up to 10 minutes and Audio Flamingo Next accepts up to
30 minutes. Both encoders process 30-second windows, and the language model attends
to the combined audio context. This is offline processing, not live audio streaming.
Longer input increases context memory and prefill time. Inputs beyond the selected
checkpoint's duration limit are rejected rather than silently truncated.

## Conversion

Download either official Transformers checkpoint, including its config and tokenizer
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
```

The original tensors are BF16. The resulting GGUF includes the tokenizer, configs,
and model spec, so inference does not require the original checkpoint directory.

## Upstream

- [NVIDIA Audio Flamingo 3](https://huggingface.co/nvidia/audio-flamingo-3-hf)
- [NVIDIA Audio Flamingo Next Instruct](https://huggingface.co/nvidia/audio-flamingo-next-hf)
- The model cards specify noncommercial terms. Review the upstream licenses before
  use or redistribution.
