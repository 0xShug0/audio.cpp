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
The [validation record](validation.md)
includes the Windows configure/build commands and results for the current port.
On Visual Studio, use `bin/Release/` and append `.exe` below.

[Chatterbox S3 encoder correctness PR #778](https://github.com/0xShug0/audio.cpp/pull/778)
and the Kitten model PR #776 are included in the upstream base for this follow-up.
No additional correctness patch is needed on that base. The validation record
also retains the earlier pre-merge results and their prerequisite.

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
- **Component parity:** `speaker/model.safetensors`, the LM speaker-projection
  weights, `native/s3gen_meanflow.safetensors`, `cpp/default/voices.json`, and
  the upstream **`cpp/default/decoder.pt`** TorchScript export. That export is
  a Python comparison reference only; the native decoder does not load it.
- **Multilingual checks:** the original voice index, referenced WAV/NPZ files,
  and a separate complete prepared voice JSON saved during conversion as shown
  below.

Follow the model guide for the additional S3 checkpoint. The validated upstream
snapshot/revisions are recorded in [validation.md](validation.md); component
graph extraction relies on the matching default `decoder.pt` export.
Python component comparisons use NumPy, SoundFile, PyTorch, and safetensors.
These are validation-only dependencies.

## Session regression

```sh
build-kitten2/bin/kitten_tts2_session_probe \
  models/kitten-tts2/kitten-tts2-native-q8-multilingual.gguf build-kitten2/native-session \
  8 native /absolute/path/kitten-tts-2 cpu > build-kitten2/session.log 2>&1
```

One session checks exact seeded repeats, voice switching, three-chunk synthesis,
unknown voice rejection, cloning from Bruno/Bella WAVs, cloned-voice cached
repeats, reference changes, and missing-transcript/short-clip rejection.
The optional reference model directory supplies WAVs/transcripts; inference
weights come from the first argument. A final `cpu` or `cuda` argument selects
the backend (default: `cpu`). Output WAVs are float32, mono, 24 kHz.
Reported RTF and the framework's `session.wall_ms` cover the entire `run()` call,
including reference conditioning and first-clone lazy encoder loading. Initial
model/session construction is outside this timer.

The production model has no custom tensor-dump environment variable or hooks.
Normal framework timing/profiling remains available. `check_parity.py` is retained
only for comparing archived prompt/logit traces from the original port; it cannot
capture fresh traces from this runtime. Those historical comparisons additionally
require Transformers and the full upstream LM weights/tokenizer/configuration.

## Native components

For controlled **CUDA parity**, disable TF32 before launching the component
probe. Setting it only for the later Python comparison cannot change captured
tensors. This is a test setting; ordinary synthesis uses the default backend
precision policy.

```sh
export NVIDIA_TF32_OVERRIDE=0
```

```powershell
$env:NVIDIA_TF32_OVERRIDE = "0"
```

```sh
build-kitten2/bin/kitten_tts2_components_probe \
  /absolute/path/kitten-tts-2/native/s3gen_meanflow.safetensors \
  /absolute/path/kitten-tts-2/cpp/default/voices.json \
  tests/kitten_tts2/component_codes.json \
  build-kitten2/native-components cpu /absolute/path/kitten-tts-2
python tests/kitten_tts2/check_native_components.py \
  --model /absolute/path/kitten-tts-2 --upstream /absolute/path/KittenTTS \
  --speaker-output build-kitten2/native-components \
  --decoder-output build-kitten2/native-components \
  --codes tests/kitten_tts2/component_codes.json
```

The optional reference-model directory makes the test-only probe exercise
the public speaker API with Bruno's reference WAV. It writes the resampled
input with the framework WAV writer and embedding/projection vectors as JSON.
The script checks XVectorSincNet using that exact input, then the BF16 speaker
projection independently. The fixed 72-token fixture was captured from a seeded
Bruno request during the original port; both decoder implementations receive
the same fixture plus the usual three lookahead tokens.
The script compares encoder hidden states and two-step
meanflow mel with the published decoder's Gaussian draws fixed to zero.
Finally, it supplies identical mel and zero random draws to HiFT and checks
pitch and waveform. The graph extraction rejects incompatible exports.
Replace `cpu` with `cuda` in the component probe command to check NVIDIA execution. Use the
same numerical comparison limits on CPU and CUDA, with TF32 disabled for the
controlled CUDA run as described above.

The encoder comparison fails with the original relative-shift padding order.
The shift fix alone passes its tolerance; using the upstream LayerNorm epsilon
further reduces the error (see the ablation in [PR #778](https://github.com/0xShug0/audio.cpp/pull/778)).
Max-error limits are 0.00005 for speaker identity, 0.0001 for encoder hidden
states, 0.005 for mel, 0.032 for BF16 projection, and 0.01 for pitch/waveform.

## Server and dependencies

Run ordinary synthesis/server checks without the parity override, in a fresh
shell or after `unset NVIDIA_TF32_OVERRIDE` (POSIX) /
`Remove-Item Env:NVIDIA_TF32_OVERRIDE -ErrorAction SilentlyContinue` (PowerShell).

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
stops a temporary localhost server. NumPy and SoundFile are used only by the test.
It checks all nine language presets plus Bruno, 48 registered voices, exact
prepared reference tokens/transcripts against upstream assets, finite non-silent
24 kHz output, Chinese chunking, and German/Chinese cloning. WAVs and a JSON
report remain in the output directory. Without runtime dump hooks this test
does not inspect internal BPE IDs, logits, or generated token counts. These are
execution checks, not an ASR or native-speaker pronunciation assessment.
