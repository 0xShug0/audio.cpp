# Kitten TTS 2 validation

Powered by Stellon Labs.

Run commands from the audio.cpp repository root. Replace
`/absolute/path/kitten-tts-2` with the local upstream model directory and
`/absolute/path/KittenTTS` with the upstream source checkout. They need not be
siblings of audio.cpp. Point GGUF arguments at the downloaded package or your
own converted file.

Commands below use POSIX-shell `\` line continuations. In PowerShell, join
the lines or replace those continuations with backticks. Environment-variable
setup is shown separately for both shells.

Build per the [model guide](../../docs/community_models/kitten_tts2.md), adding
`-DENGINE_BUILD_TESTS=ON`. Build `kitten_tts2_session_probe`,
`kitten_tts2_components_probe`, and `model_spec_system_test`.
The [fresh CUDA validation record](validation.md#fresh-cuda-validation-on-bf50ab82)
includes the Windows configure/build commands and results for the current port.
On Visual Studio, use `bin/Release/` and append `.exe` below.

```sh
python tools/check_loader_catalog_sync.py
build-kitten2/bin/model_spec_system_test
```

## Validation assets

The published GGUF is sufficient for native preset inference and, with a user
reference clip/transcript, cloning. The developer tests below additionally use:

- **Session cloning checks:** the upstream `voices/voices.json` index and its
  Bruno/Bella reference WAVs. Without the optional reference-model argument,
  the session probe runs only the preset and invalid-voice checks.
- **LM parity:** the upstream source checkout, full `lm/model.safetensors`,
  tokenizer/configuration, and the prepared voice index. Packed ternary/emb4
  source weights are not the reference input for this script.
- **Component parity:** `speaker/model.safetensors`, the LM speaker-projection
  weights, `native/s3gen_meanflow.safetensors`, `cpp/default/voices.json`, and
  the upstream **`cpp/default/decoder.pt`** TorchScript export. That export is
  a Python comparison reference only; the native decoder does not load it.
- **Multilingual checks:** the original tokenizer/configuration, voice index,
  referenced WAV/NPZ files, and a separate complete prepared voice JSON saved
  during conversion as shown below.

Follow the model guide for the additional S3 checkpoint. The validated upstream
snapshot/revisions are recorded in [validation.md](validation.md); component
graph extraction relies on the matching default `decoder.pt` export.
Python reference comparisons use NumPy, SoundFile, PyTorch, Transformers, and
safetensors. These are validation-only dependencies.

## Session regression

Enable tracing **before** the probe so later parity commands have their inputs.
In a POSIX shell:

```sh
export AUDIOCPP_KITTEN_TTS2_TRACE_DIR="build-kitten2/native-trace"
```

Or, in PowerShell:

```powershell
$env:AUDIOCPP_KITTEN_TTS2_TRACE_DIR = "build-kitten2/native-trace"
```

Then run the probe, keeping this environment variable set in that shell:

```sh
build-kitten2/bin/kitten_tts2_session_probe \
  models/kitten-tts2/kitten-tts2-native-q8-multilingual.gguf build-kitten2/native-session \
  8 native /absolute/path/kitten-tts-2 cpu
```

One session checks exact seeded repeats, voice switching, three-chunk synthesis,
unknown voice rejection, cloning from Bruno/Bella WAVs, cloned-voice cached
repeats, reference changes, and missing-transcript/short-clip rejection.
The optional reference model directory supplies WAVs/transcripts; inference
weights come from the first argument. A final `cpu` or `cuda` argument selects
the backend (default: `cpu`). Output WAVs are float32, mono, 24 kHz.
Reported RTF excludes loading, except the first clone includes lazy reference
encoder loading and conditioning.

Tracing captures prompt IDs, logits, speech codes, mel output, and cloned speaker
intermediates. The probe copies preset-request LM traces into
`native-session/request_N_trace/`. The `native-trace/` directory retains the
latest request, including speaker intermediates from the final clone. Run the
component comparison before reusing this trace directory for another process.
Chunked traces contain only the last chunk.

## Language model parity

Use the assets listed above and the trace from the session probe's first request.

```sh
python tests/kitten_tts2/check_parity.py \
  --model /absolute/path/kitten-tts-2 --python-package /absolute/path/KittenTTS \
  --trace build-kitten2/native-session/request_0_trace \
  --wav build-kitten2/native-session/request_0.wav \
  --text "Hello there. This is a test of Kitten speech." \
  --output build-kitten2/native-lm-parity.json
```

Checks exact prompt IDs, equal highest-scoring speech token, maximum logit error
under 0.5, and finite non-silent audio with the expected frame count. Use
`--emotion` for expression prompts. Native/PyTorch RNGs differ, so independently
sampled waveforms are not compared.

## Native components

```sh
build-kitten2/bin/kitten_tts2_components_probe \
  /absolute/path/kitten-tts-2/native/s3gen_meanflow.safetensors \
  /absolute/path/kitten-tts-2/cpp/default/voices.json \
  build-kitten2/native-session/request_0_trace/codes.json \
  build-kitten2/native-components
python tests/kitten_tts2/check_native_components.py \
  --model /absolute/path/kitten-tts-2 --upstream /absolute/path/KittenTTS \
  --clone-trace build-kitten2/native-trace \
  --decoder-output build-kitten2/native-components \
  --codes build-kitten2/native-session/request_0_trace/codes.json
```

The clone trace must contain an actual clone request. The script checks
XVectorSincNet using the exact resampled waveform, then the BF16 speaker
projection independently. It compares encoder hidden states and two-step
meanflow mel with the published decoder's Gaussian draws fixed to zero.
Finally, it supplies identical mel and zero random draws to HiFT and checks
pitch and waveform. The graph extraction rejects incompatible exports.
Append `cuda` to the component probe command to check NVIDIA execution. Use the
same numerical comparison limits on CPU and CUDA; no TF32 override is needed.

This regression catches the shared S3 relative-position zero-column placement
and layer-norm epsilon bugs fixed by the port. Max-error limits are 0.00005
for speaker identity, 0.0001 for encoder hidden states, 0.005 for mel, 0.032
for BF16 projection, and 0.01 for pitch/waveform.

## Server and dependencies

Register the GGUF as `kitten_tts2`, task `tts`, mode `offline`.
POST to `/v1/audio/speech` with a preset `voice` or reference audio:

```json
{
  "model": "kitten-tts2",
  "input": "Hello from native voice cloning.",
  "voice_ref": {"type": "path", "path": "/absolute/path/reference.wav"},
  "reference_text": "The words spoken in that recording.",
  "seed": 1234
}
```

Inspect executable imports (`dumpbin /dependents` on Windows) to verify no
Torch/Python/ONNX DLL. Inspect the GGUF for its three tensor namespaces,
license/NOTICE sidecars, and absence of TorchScript resources.

CPU and NVIDIA CUDA are tested backends. Student decoders, streaming, exhaustive
testing of every English voice, and automatic transcript generation are outside
the implemented/validated scope.
Numerical checks do not measure intelligibility or speaker similarity;
listen to generated samples as well.

## Multilingual presets

Build `audiocpp_kitten_tts2_prepare_voices` alongside the converter. If you already
saved the complete prepared index during packaging, pass its path to `--voices`
below. Otherwise, create a local GGUF and retain its index with:

```sh
python tools/community_models/prepare_kitten_tts2_gguf.py \
  --model /absolute/path/kitten-tts-2 \
  --converter build-kitten2/bin/audiocpp_gguf --type q8_0 \
  --output build-kitten2/kitten-tts2-native-q8-multilingual.gguf \
  --voices-output build-kitten2/kitten-tts2-multilingual-voices.json
```

The helper completes the older English-only index using the original WAV/NPZ
assets and rejects a package missing any registered preset. The following test
uses that local GGUF; substitute the downloaded GGUF path to validate the
published package with the same complete index.

```sh
python tests/kitten_tts2/check_multilingual.py \
  --server build-kitten2/bin/audiocpp_server \
  --model build-kitten2/kitten-tts2-native-q8-multilingual.gguf \
  --reference-model /absolute/path/kitten-tts-2 \
  --voices build-kitten2/kitten-tts2-multilingual-voices.json \
  --output build-kitten2/multilingual-validation --backend cpu
```

Use `--backend cuda` and the CUDA server to repeat on NVIDIA. This starts and
stops a temporary localhost server. NumPy, SoundFile and Transformers are used
only by the test; the original tokenizer is loaded locally without downloads.
It checks all nine language presets plus Bruno, 48 registered voices, exact
upstream reference tokens, reference/target BPE across scripts, finite non-silent
24 kHz output, generation ending before the token limit, Chinese chunking, and
German/Chinese cloning. WAVs, prompt traces and a JSON report remain in the output
directory. These are execution and text-preservation checks, not an ASR or
native-speaker pronunciation assessment.

`check_parity.py --voices <complete-index> --voice German` can additionally
compare multilingual LM logits against the full upstream weights.
