# FireRed VAD

Native GGML inference for the [FireRedTeam/FireRedVAD](https://github.com/FireRedTeam/FireRedVAD)
offline VAD and causal Stream-VAD checkpoints. Both models take 16 kHz mono audio
and return speech segments. Stream-VAD also emits speech-start and speech-end events.
The original code and weights are Apache-2.0 licensed.

## Build and Convert

Run these commands from the repository root:

```bash
scripts/build_metal.sh --build-dir build/firered-vad --build-type Release \
  --model-set custom --models firered_vad --with-tests \
  --target audiocpp_cli --target audiocpp_gguf --target firered_vad_probe
```

Download `VAD/model.pth.tar`, `VAD/cmvn.ark`, `Stream-VAD/model.pth.tar` and
`Stream-VAD/cmvn.ark` from [FireRedTeam/FireRedVAD on Hugging Face](https://huggingface.co/FireRedTeam/FireRedVAD).
Preserve the `VAD/` and `Stream-VAD/` directories under `<source-dir>`.
The converter needs `torch`, `numpy`, `safetensors` and `kaldiio`.

```bash
python tests/firered_vad/convert_gguf.py \
  --source <source-dir>/VAD --type f32 \
  --converter build/firered-vad/bin/audiocpp_gguf \
  --output models/FireRedVAD-GGUF/firered-vad-f32.gguf

python tests/firered_vad/convert_gguf.py \
  --source <source-dir>/Stream-VAD --type f32 \
  --converter build/firered-vad/bin/audiocpp_gguf \
  --output models/FireRedVAD-GGUF/firered-stream-vad-f32.gguf
```

Only F32 packages are provided: F16 and Q8 offer no meaningful performance or
peak-VRAM benefit for these small models, and Q8 adds numerical drift.
The converter still accepts `f32`, `f16` and `q8_0` for experimentation.
Each GGUF contains its weights, FP32 CMVN
coefficients, architecture configuration and v1 model spec. The family is marked
experimental while converted packages are prepared for publication.

## Run

Offline VAD:

```bash
build/firered-vad/bin/audiocpp_cli --task vad --family firered_vad \
  --model models/FireRedVAD-GGUF/firered-vad-f32.gguf \
  --backend metal --audio assets/resources/sample_16k.wav --metrics --log
```

Streaming with the causal checkpoint:

```bash
build/firered-vad/bin/audiocpp_cli --task vad --mode streaming --family firered_vad \
  --model models/FireRedVAD-GGUF/firered-stream-vad-f32.gguf \
  --backend metal --audio assets/resources/sample_16k.wav --log
```

Use `--backend cpu --threads <count>` for CPU inference. Thread count controls
parallelism; the CPU graph and cache layout are the same across thread counts.
CPU depthwise convolution uses channel-contiguous time blocks, and eligible matrix
multiplications use GGML BLAS when available, following Moonshine's scheduler pattern.
The requested thread count is passed to both backends; vendor BLAS libraries may
manage their own worker threads. Input chunks use 160-sample hops with a retained 240-sample overlap for
the 400-sample Kaldi window. The default transport chunk is 2560 samples.

The offline checkpoint includes lookahead; the causal checkpoint supports both
full-input and streaming runs. Streaming resets caches and segmentation state at
the start of each request.

## Options

Pass options with `--request-option name=value`. Duration options are rounded to
the nearest 10 ms model frame; halfway values round up.

| Option | VAD default | Stream-VAD default |
|---|---:|---:|
| `threshold` | 0.4 | 0.5 |
| `smooth_window_ms` | 50 | 50 |
| `min_speech_duration_ms` | 200 | 80 |
| `max_speech_duration_sec` | 20 | 20 |
| `min_silence_duration_ms` | 200 | 200 |
| `merge_silence_duration_ms` | 0 | Offline VAD only |
| `speech_pad_ms` | 0 | Offline VAD only |
| `speech_start_pad_ms` | Stream-VAD only | 50 |
| `audio_chunk_duration_sec` | 300 | 300 for full-input runs |

Full-input runs split network input at `audio_chunk_duration_sec`, resetting network history
between chunks as upstream does. Live streaming retains network history.

## Validation

The reference tools use the official Python implementation, including Kaldi
features, CMVN, frame probabilities and segment rules. Install `soundfile` and
`kaldi-native-fbank` as well as the conversion dependencies. `--gguf` additionally
requires the Python `gguf` package and compares against dequantized package weights
to separate quantization drift from backend numerical error.

```bash
python tests/firered_vad/python_reference.py \
  --reference <FireRedVAD-source-repo> --source <source-dir>/VAD \
  --audio assets/resources/sample_16k.wav --threads 1 \
  --output-prefix build/firered-vad/reference --device cpu --log

build/firered-vad/bin/firered_vad_probe \
  models/FireRedVAD-GGUF/firered-vad-f32.gguf \
  assets/resources/sample_16k.wav metal 8 \
  build/firered-vad/native build/firered-vad/reference --log

python tests/firered_vad/check_parity.py \
  --reference build/firered-vad/reference --native build/firered-vad/native
```

The probe checks streaming/full-input agreement for causal checkpoints and exports
features, probabilities and timestamps. It measures frontend, network and complete
detection separately, with five warmups and the median of 21 measured runs.
It accepts `cpu`, `cuda`, `vulkan`, and `metal`; the Python reference accepts
`--device cpu` or `--device cuda` and synchronizes CUDA timing. For causal
regression coverage, also run the official `TEST_MEETING_T0000000001_S00000.wav`
example with the Stream-VAD checkpoint: its near-threshold speech boundary exposes
small-matrix precision loss that the shorter examples do not.
The default parity checks bound normalized frontend RMSE by `1e-5`, frontend maximum
error by `1e-3`, frame-probability error by `0.002`, and require exact timestamps.

Unit tests cover upstream postprocessing fixtures and sparse/dense Kaldi frontend
equivalence. Sparse projection is opt-in and disabled by default for existing callers.

```bash
cmake --build build/firered-vad --target \
  firered_vad_rules_test kaldi_fbank_sparse_test
ctest --test-dir build/firered-vad \
  -R 'firered_vad_rules_test|kaldi_fbank_sparse_test' \
  --output-on-failure
```

Measured results are recorded in [validation.md](../../tests/firered_vad/validation.md).
