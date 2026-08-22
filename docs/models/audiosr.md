# AudioSR

AudioSR performs audio super-resolution from an input waveform.

## Run

```bash
build/debug/bin/audiocpp_cli \
  --task gen \
  --family audiosr \
  --model /path/to/AudioSR-GGUF/audiosr-basic-f32.gguf \
  --backend cuda \
  --audio input.wav \
  --num-inference-steps 50 \
  --guidance-scale 3.5 \
  --seed 42 \
  --out output.wav
```

`ddim_eta` defaults to `1.0`, matching the official AudioSR inference default. Use `--request-option ddim_eta=0` only for deterministic sampler debugging.

Long audio is processed with bounded overlapping chunks once the input exceeds
`audio_chunk_duration_sec`. The defaults match the official Python long-audio
path:

```bash
--request-option audio_chunk_duration_sec=15 \
--request-option audio_chunk_overlap_sec=2
```
