# VoxMel

`voxmel_tts` runs [VoxMel](https://github.com/VoxDance-Lab/VoxMel) 0.6B for English
and Mandarin speech synthesis and reference-audio voice cloning. Output is mono
24 kHz audio. Generation is offline; long text uses the framework text chunker.

## Downloads

GGUF package paths are under
[audio-cpp/audio.cpp-gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf), in
`VoxMel-TTS-GGUF/`. Each package includes the Qwen tokenizer, reference-audio frontend
constants, model configuration and Vocos weights. Python is not needed at inference.

| Package | Model weights | Vocos |
|---|---|---|
| `voxmel-tts-0.6b-f16.gguf` | F16, default | F32 |
| `voxmel-tts-0.6b-q8_0.gguf` | Q8 where supported | F32 |

## Usage

Voice cloning needs the exact transcript of the reference recording:

```bash
audiocpp_cli --family voxmel_tts --task clon --backend cuda \
  --model /path/to/voxmel-tts-0.6b-f16.gguf \
  --voice-ref reference.wav --reference-text "The words in the recording." \
  --text "The morning train arrived beside the river." \
  --request-option seed=2026 --threads 8 --out speech.wav --log
```

For unconditioned speech use `--task tts` and omit both reference options.
Use `--backend cpu` or `--backend vulkan` for another compiled backend.
Language follows the input text. Reference audio is mixed to mono and resampled
to 24 kHz. Each text chunk starts a separate generation with the same reference;
the seed increments between chunks.

## Model

| Field | Value |
|---|---|
| Family | `voxmel_tts` |
| Tasks | `tts`, `clon` |
| Mode | `offline` |
| Languages | English, Mandarin Chinese |
| Input | Text; optional reference audio with matching transcript |
| Output | Mono 24 kHz WAV |
| License | Apache-2.0 model; MIT Vocos |

The model uses a Qwen2 causal backbone, continuous flow heads for wavelet Mel
coefficients, two residual reconstruction tiers and the shared Vocos vocoder.
The 0.6B label describes the backbone; package size includes the other components.

## Common Options (use directly)

| Option | Meaning |
|---|---|
| `--model` | Self-contained GGUF file. |
| `--backend`, `--threads` | Execution backend and CPU thread count. |
| `--text` | English or Mandarin text. |
| `--voice-ref`, `--reference-text` | Reference recording and its accurate transcript. |
| `--log` | Component timings and `session.wall_ms`. |

## Session Options (`--session-option`)

| Option | Default | Meaning |
|---|---|---|
| `voxmel_tts.weight_type` | `native` | Use package storage, or override with `f32`, `f16`, `bf16`, `q8_0`. |
| `voxmel_tts.attention` | `auto` | `auto`, `eager` or `flash` attention. |

## Request Options (`--request-option`)

| Option | Default | Meaning |
|---|---|---|
| `reference_text` | empty | Required with reference audio. |
| `seed` | random | Nonnegative random seed. |
| `guidance_scale` | `3.5` | Text guidance; zero disables it. |
| `speaker_guidance_scale` | `3.0` | Reference-speaker guidance; zero disables it. |
| `num_inference_steps` | `32` | Euler steps per coarse Mel frame, 1–256. |
| `residual_num_inference_steps` | same as above | Euler steps for residual tiers. |
| `min_duration_sec` | `0.3` | Minimum seconds per text chunk. |
| `max_duration_sec` | `30` | Maximum seconds per text chunk; prompt and generation must fit the context. |
| `text_chunk_size` | `400` | Maximum Unicode codepoints per text chunk. |
| `text_chunk_mode` | `default` | `default`, `tag_aware`, `japanese`, `endline`. |

## Conversion

Download the official VoxMel-0.6B and `charactr/vocos-mel-24khz` checkpoints,
and clone the official VoxMel repository. With its Python dependencies installed:

```bash
python tools/models/convert_voxmel_tts.py \
  --source /path/to/VoxMel-0.6B --upstream /path/to/VoxMel \
  --vocos /path/to/vocos-mel-24khz --type f16 \
  --output /path/to/voxmel-tts-0.6b-f16.gguf --log-file conversion.log
```

Build `audiocpp_gguf` first. Select `--type q8_0` for the smaller quantized
package. Vocos and reference frontend constants stay F32.

## Numerical validation

With identical FP32 conditioning, noise and teacher-forced AR history, CPU
waveform cosine against Python is **0.9999944**. CUDA AR and flow checks pass;
propagated waveform cosine is **0.9998667**. Decoding the identical Python Mel
through CUDA Vocos alone gives **0.9999907**. This CUDA drift is accepted;
it does not meet the strict 0.99999 propagated-waveform gate. Normal English and
Mandarin outputs on CUDA, Vulkan and CPU have matching checked transcripts.
Seeds do not guarantee identical free-running waveforms across backends.
