# Magenta RealTime 2

Magenta RealTime 2 generates 48 kHz stereo instrumental music from a text style
prompt or an audio example. Both Small (230M) and Base (2.4B) use the same
`gen` route in offline and streaming modes. Text describes the music; it is not
a lyric transcript.

Download one self-contained F16 or Q8_0 package from
[audio-cpp/Magenta-RealTime-2-GGUF](https://huggingface.co/audio-cpp/Magenta-RealTime-2-GGUF).
Each GGUF includes the generator, SpectroStream codec, MusicCoCa style runtime
weights, configuration, vocabulary, and v1 model spec.

## Quick Start

```bash
audiocpp_cli --model Magenta-RealTime-2-GGUF/magenta-rt2-small-f16.gguf \
  --family magenta_realtime_2 --task gen --backend cuda \
  --text "disco funk" --request-option duration_sec=10 \
  --out music.wav --log
```

Use `magenta-rt2-base-f16.gguf` for Base, or `--backend cpu` / `vulkan`
for another backend. Use the `-q8_0.gguf` packages for quantized transformer
matrices. Add `--mode streaming` to emit audio incrementally;
the same request produces the same final waveform in either mode. Streaming
emits up to four codec frames (160 ms) per audio event. Each request starts a
fresh piece; it does not continue the previous request.

For audio style conditioning, replace `--text` with `--audio style.wav`.
MusicCoCa embeds the recording's style rather than continuing its waveform.

Set these through `--request-option name=value`. Guidance is discretized to
the model's trained control tokens. Piano states are `-1` (unspecified), `0`
(off), `1` (sustain), `2` (onset), and `3` (model chooses sustain/onset).

All generation graphs and bounded attention/codec state are cached per session.
MusicCoCa has its own framework runtime and reads its tensors from the same
GGUF. Inference does not require Python, JAX, or TensorFlow Lite.

## Model

| Field | Value |
|---|---|
| Family | `magenta_realtime_2` |
| Tasks | `music` |
| Modes | `offline`, `streaming` |
| Languages | `auto` |
| Default package | `magenta_realtime_2_small_f16` |
| Input | Text style prompt or reference audio |
| Output | 48 kHz stereo music audio |

“Model/session default” means no explicit override. The runtime uses the
checkpoint or session setting, or leaves the optional value unset as described.

## Common Options (use directly)

| Option | Meaning |
|---|---|
| `--model` | Model file or directory shown in the examples. |
| `--backend` | Select an available backend supported by this model. |
| `--threads` | CPU thread count. |
| `--log` | Enable runtime logging. |
| `--text` | Input text; see the examples for the expected prompt format. |
| `--audio` | Input audio, where required by the selected task. |

## Request Options (use with `--request-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `duration_sec` | `float` | `10` | Requested music duration in seconds. Declared range: minimum 0.04. |
| `temperature` | `float` | `1.3` | Sampling temperature; zero selects greedy decoding. Declared range: minimum 0. |
| `top_k` | `int` | `40` | Sampling candidate limit; zero keeps all 1024 valid codes. Declared range: minimum 0, maximum 1024. |
| `seed` | `int` | `0` | Generation seed. Declared range: minimum 0, maximum 4294967295. |
| `guidance_scale` | `float` | `3` | MusicCoCa style guidance, discretized to the model's trained conditioning tokens. Declared range: minimum -1, maximum 7. |
| `note_guidance_scale` | `float` | `1` | MIDI note guidance. Declared range: minimum -1, maximum 7. |
| `drum_guidance_scale` | `float` | `1` | Drum guidance. Declared range: minimum -1, maximum 7. |
| `use_mapper` | `bool` | `true` | Map text style embeddings into the audio embedding space. |
| `pianoroll_file` | `string` | Model/session default | Optional JSON array of 128 MIDI pitch states per frame at 25 Hz: -1 masked, 0 off, 1 sustain, 2 onset, 3 model chooses. |
| `drums` | `int` | `-1` | Drum conditioning: -1 leaves drums unspecified, 0 requests no drums, 1 requests drums. Declared range: minimum -1, maximum 1. |

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `magenta_realtime_2.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Transformer weight storage type. |

## Conversion

Download the original [Google checkpoint and resources](https://huggingface.co/google/magenta-realtime-2),
then run the converter with NumPy, safetensors, and `ai-edge-litert` installed:

```bash
python tools/models/convert_magenta_realtime_2_gguf.py \
  --source Magenta-RealTime-2 --output Magenta-RealTime-2-GGUF \
  --component small --type f16
python tools/models/convert_magenta_realtime_2_gguf.py \
  --source Magenta-RealTime-2 --output Magenta-RealTime-2-GGUF \
  --component base --type f16
```

Use `--type q8_0` instead for Q8_0. Each conversion embeds all MusicCoCa
resources in the selected generator GGUF. SpectroStream, MusicCoCa, and
sensitive normalization, bias, scale, and sink tensors remain F32.

Original model: [Google DeepMind Magenta RealTime 2](https://github.com/magenta/magenta-realtime).
Weights retain CC-BY-4.0 and the upstream usage terms.
