# Dettato

Experimental multi-instrument audio-to-MIDI transcription, including instrument
grouping and drums. Upstream: [KobiMusic/Dettato](https://github.com/kobimusic/dettato).

The F16 package is `Dettato-GGUF/dettato-f16.gguf` in
[audio-cpp/audio.cpp-gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf).

```bash
./build/debug/bin/audiocpp_cli \
  --family dettato --model models/Dettato-GGUF/dettato-f16.gguf \
  --task midi --backend cuda --threads 8 --audio music.wav \
  --out-dir output --log --log-file dettato.log
```

Options:

- Session `dettato.window_hop_sec`: `2` (default) or `4`. The default matches
  upstream Python's transcription schedule; `4` processes fewer windows.
- Request `output_format`: `midi` (default) or `json`.

> [!WARNING]
> Instrument grouping is numerically sensitive. Instrument labels and track
> membership can differ from upstream Python's BF16 inference, even when note
> detection agrees closely.

> [!WARNING]
> If SOXR is unavailable, built-in linear resampling is used with a runtime
> warning. Output may differ from SOXR.

Convert the four upstream checkpoints with
[`tools/community_models/convert_dettato.py`](../../tools/community_models/convert_dettato.py).
The converter accepts `--source`, `--upstream`, `--output`, `--weight-type` and
`--log-file`; use `--help` for the full command syntax.

## WebUI

Select **Dettato (experimental audio to MIDI)** in Studio, upload music and run
transcription. The Downloads page uses the same F16 package from the model spec.
