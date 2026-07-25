# BS-RoFormer validation

The native `bs_roformer` family was validated with
`model_bs_roformer_ep_368_sdr_12.9628.ckpt` and its matching YAML config from
audio-separator.

## Convert the reference checkpoint

```powershell
python tests\bs_roformer\convert_reference_ckpt.py `
  --ckpt model_bs_roformer_ep_368_sdr_12.9628.ckpt `
  --config-path model_bs_roformer_ep_368_sdr_12.9628.yaml `
  --output-dir models\BS-RoFormer-ep368
```

## Run the Python reference

```powershell
python tests\bs_roformer\run_python_reference.py `
  --audio-separator-root path\to\python-audio-separator `
  --ckpt model_bs_roformer_ep_368_sdr_12.9628.ckpt `
  --config-path model_bs_roformer_ep_368_sdr_12.9628.yaml `
  --audio input_8s.wav `
  --out outputs\python\vocals.wav `
  --device cuda
```

## Run audio.cpp

SafeTensors:

```powershell
audiocpp_cli.exe --task sep --family bs_roformer `
  --model models\BS-RoFormer-ep368 --backend cuda `
  --audio input_44k.wav --out-dir outputs\cpp-f32
```

Standalone Q8 GGUF:

```powershell
audiocpp_cli.exe --task sep `
  --model models\BS-RoFormer-ep368_Q8\BS-RoFormer-ep368_Q8.gguf `
  --backend cuda --audio input_44k.wav --out-dir outputs\cpp-q8
```

## Local validation result

Backend: CUDA, NVIDIA GeForce RTX 3090. Input: stereo, 44.1 kHz, 20 seconds.

| Comparison | Waveform cosine |
|---|---:|
| audio.cpp Q8 vs audio.cpp F32 | 0.999985368 |
| audio.cpp F32 vs audio-separator Python | 0.996474981 |
| audio.cpp Q8 vs audio-separator Python | 0.996387362 |

An exact-frame 8-second test produced 352,800 frames from both runtimes. Its
F32 Python comparison measured waveform cosine `0.989149342` and log-mel
cosine `0.998817655`; the lower waveform metric reflects the different
edge/chunk overlap policies used by the complete runtime paths.

For the same exact-frame input, Q8 versus F32 measured:

| Metric | Result |
|---|---:|
| Output frames | 352,800 / 352,800 |
| Waveform cosine | 0.999994784 |
| Log-mel cosine | 0.999898629 |
| Mean absolute sample error | 0.0002294 |
| Maximum absolute sample error | 0.0044556 |

This makes the tested Q8 output perceptually and numerically nearly identical
to the native F32 path while reducing the weight file from about 639 MB to
about 173 MB.

## Server and WebUI validation

The CUDA server was built with:

```powershell
.\scripts\build_windows.ps1 -Target audiocpp_server -Jobs 16
```

An offline `sep` model entry pointing directly at
`models\BS-RoFormer-ep368_Q8\BS-RoFormer-ep368_Q8.gguf` was exercised through
`POST /v1/tasks/run` with:

```json
{
  "model": "bs-roformer-q8",
  "request": {
    "audio": "E:\\path\\to\\input_8s.wav"
  }
}
```

The request returned `vocals` and `instrumental` named audio outputs in
3,032.1 ms including HTTP, lazy loading, base64 response serialization, and
file decoding. Both returned WAV files were byte-identical to the CLI outputs.

The WebUI was launched with the CUDA backend, selected `bs-roformer` beside
HTDemucs and Mel-Band RoFormer in the Source separation tab, uploaded the same
8-second WAV, and ran the visible Separate action. It loaded the standalone
GGUF in 0.5 seconds and displayed two playable/downloadable tracks after a
2.7-second separation run.
