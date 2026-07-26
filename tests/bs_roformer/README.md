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

## CUDA optimization validation

The CUDA graph now lowers non-causal axial attention through the framework's
view-preserving Flash Attention path with F32 accumulation. Gate values are
broadcast directly instead of materializing a repeated gate tensor. CPU and
other backends retain the explicit attention implementation.

RTX 3090 results for the same 8-second, 44.1 kHz stereo input:

| Route | `session.wall_ms` | Speed vs audio duration |
|---|---:|---:|
| Pre-change Q8, overlap 4 | 2,643.1 | 3.03x real time |
| Optimized Q8, overlap 4 | 1,440.4 | 5.55x real time |
| Optimized F32, overlap 4 | 1,874.3 | 4.27x real time |
| Optimized Q8, overlap 2 | 734.3 | 10.89x real time |
| Optimized Q8, overlap 1 | 384.0 | 20.84x real time |

The default remains overlap 4. Lower overlap is available through
`--session-option bs_roformer.num_overlap=2` or `=1`, but is deliberately
opt-in because it changes boundary blending. Overlap 1 versus the default
measured waveform cosine `0.989162147` and log-mel cosine `0.999110937`.

Default optimized Q8 versus the saved pre-change Q8 output measured waveform
cosine `0.999996424` and log-mel cosine `0.999937057`. Optimized Q8 versus the
current F32 output measured `0.999993563` and `0.999881983`, respectively.
The explicit CPU fallback also completed successfully; with overlap 1 its
output versus CUDA measured `0.999995887` waveform cosine and `0.999922335`
log-mel cosine.

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

The optimized resident server returned `vocals` and `instrumental` named audio
outputs with internal `session.wall_ms` values of 1,731.0 ms on the first
request and 1,401.1 / 1,422.0 ms on the next two. Corresponding end-to-end HTTP
times were 2,235.7 / 1,496.0 / 1,520.0 ms, including JSON and base64 response
serialization. Both returned WAV files were byte-identical to the CLI outputs.

The WebUI was launched with the CUDA backend, selected `bs-roformer` beside
HTDemucs and Mel-Band RoFormer in the Source separation tab, uploaded the same
8-second WAV, and ran the visible Separate action. It loaded the standalone
GGUF in 0.5 seconds and displayed two playable/downloadable tracks after a
2.7-second separation run.
