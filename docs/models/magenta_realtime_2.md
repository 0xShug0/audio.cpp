# Magenta RealTime 2

Magenta RealTime 2 generates 48 kHz stereo instrumental music from a text style
prompt or an audio example. Both Small (230M) and Base (2.4B) use the same
`gen` route in offline and streaming modes. Text describes the music; it is not
a lyric transcript.

Download one self-contained F16 or Q8_0 package from
[audio-cpp/Magenta-RealTime-2-GGUF](https://huggingface.co/audio-cpp/Magenta-RealTime-2-GGUF).
Each GGUF includes the generator, SpectroStream codec, MusicCoCa style runtime
weights, configuration, vocabulary, and v1 model spec.

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

| Request option | Default | Meaning |
|---|---:|---|
| `duration_sec` | `10` | Duration, rounded up to 40 ms frames; minimum 0.04 seconds. |
| `temperature` | `1.3` | Sampling temperature; `0` selects greedy decoding. |
| `top_k` | `40` | Candidate limit from 1–1024; `0` keeps all valid codes. |
| `seed` | `0` | Unsigned 32-bit generation seed. |
| `guidance_scale` | `3` | Style guidance from -1–7. |
| `note_guidance_scale` | `1` | Note guidance from -1–7. |
| `drum_guidance_scale` | `1` | Drum guidance from -1–7. |
| `use_mapper` | `true` | Map text embeddings into audio style space. Applies to text input. |
| `drums` | `-1` | Unspecified (`-1`), no drums (`0`), or drums (`1`). |
| `pianoroll_file` | unset | JSON array with one row per 25 Hz frame and 128 integer pitch states per row. |

Set these through `--request-option name=value`. Guidance is discretized to
the model's trained control tokens. Piano states are `-1` (unspecified), `0`
(off), `1` (sustain), `2` (onset), and `3` (model chooses sustain/onset).

| Session option | Default | Meaning |
|---|---|---|
| `magenta_realtime_2.weight_type` | `native` | Transformer weight storage type. Codec and MusicCoCa remain F32. |

All generation graphs and bounded attention/codec state are cached per session.
MusicCoCa has its own framework runtime and reads its tensors from the same
GGUF. Inference does not require Python, JAX, or TensorFlow Lite.

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
