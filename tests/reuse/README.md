# RE-USE Checks

Use the official `nvidia/RE-USE` checkpoint at revision
`022e920d727347a64d6c21fbf0628f2a5f37ad78`. Keep original weights outside the
GGUF output directory. The conversion script verifies shapes and retained F32
tensor bytes against the original safetensors.

```bash
python tests/reuse/convert_reuse_gguf.py \
  --source /path/to/RE-USE --output /path/to/RE-USE-GGUF/reuse-f32.gguf \
  --log build/logs/reuse/conversion.log
```

## Python Reference

The runner calls the official SEMamba model and STFT/ISTFT implementations.
Run in the `qwen3-tts` environment with PyTorch, `mamba_ssm`, `causal_conv1d`,
safetensors, einops, and soundfile installed.

```bash
python tests/reuse/run_python_reference.py \
  --reference-root /path/to/RE-USE \
  --audio /path/to/RE-USE/noisy_audio/mic_test2.wav \
  --repeats 4 --out build/logs/reuse/python.wav \
  --log build/logs/reuse/python.log
```

Ordinary performance runs leave TF32 and model precision at upstream defaults.
For separate controlled parity runs, set `NVIDIA_TF32_OVERRIDE=0` for both
implementations and pass `--no-tf32` to Python. Do not report controlled-run
timings as ordinary performance. Use `--out-format float32` in the CLI when
comparing final waveforms. Keep timing measurements sequential.

Compare full-recording restoration at multiple sample rates, stereo channels,
and repeated requests. Long-form checks must use the same chunk duration and
overlap in both implementations. Python exposes
`--audio-chunk-duration-sec` and `--audio-chunk-overlap-sec`; the corresponding
C++ request options are `audio_chunk_duration_sec` and
`audio_chunk_overlap_sec`.

## Vulkan Scan

`test_reuse_scan` exercises Mamba-1 state, batch, and strided B/C layouts without
model weights. Build it with model tests enabled and select `--backend vulkan`.
This does not replace final-waveform checks or validate Mamba-2 regressions.
