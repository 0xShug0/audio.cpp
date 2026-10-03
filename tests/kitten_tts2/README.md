# Kitten TTS 2 validation

Powered by Stellon Labs.

Build per the [model guide](../../docs/community_models/kitten_tts2.md), adding
`-DENGINE_BUILD_TESTS=ON`. Build `kitten_tts2_session_probe`,
`kitten_tts2_components_probe`, and `model_spec_system_test`.
On Visual Studio, use `bin/Release/` and append `.exe` below.

```sh
python tools/check_loader_catalog_sync.py
build-kitten2/bin/model_spec_system_test
```

## Session regression

```sh
build-kitten2/bin/kitten_tts2_session_probe \
  build-kitten2/kitten-tts2-native-q8.gguf build-kitten2/native-session \
  8 native ../kitten-tts-2
```

One session checks exact seeded repeats, voice switching, three-chunk synthesis,
unknown voice rejection, cloning from Bruno/Bella WAVs, cloned-voice cached
repeats, reference changes, and missing-transcript/short-clip rejection.
The optional reference model directory supplies WAVs/transcripts; inference
weights come from the first argument. A final `cpu` or `cuda` argument selects
the backend (default: `cpu`). Output WAVs are float32, mono, 24 kHz.
Reported RTF excludes loading, except the first clone includes lazy reference
encoder loading and conditioning.

Set `AUDIOCPP_KITTEN_TTS2_TRACE_DIR` to capture prompt IDs, logits, speech codes,
mel output, and cloned speaker intermediates. The probe copies preset-request
LM traces to its output directory. Chunked traces contain only the last chunk.

## Language model parity

The reference scripts need local upstream sources, NumPy, SoundFile, PyTorch,
Transformers, and safetensors. These are validation-only dependencies.

```sh
python tests/kitten_tts2/check_parity.py \
  --model ../kitten-tts-2 --python-package ../KittenTTS \
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
  ../kitten-tts-2/native/s3gen_meanflow.safetensors \
  ../kitten-tts-2/cpp/default/voices.json \
  build-kitten2/native-session/request_0_trace/codes.json \
  build-kitten2/native-components
python tests/kitten_tts2/check_native_components.py \
  --model ../kitten-tts-2 --upstream ../KittenTTS \
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

Build `audiocpp_kitten_tts2_prepare_voices` alongside the converter. Convert with
`--voices-output build-kitten2/kitten-tts2-multilingual-voices.json` to keep the
prepared index for validation. The converter requires every registered preset;
an older English-only prepared index must be completed before packaging.

```sh
python tests/kitten_tts2/check_multilingual.py \
  --server build-kitten2/bin/audiocpp_server \
  --model build-kitten2/kitten-tts2-native-q8-multilingual.gguf \
  --reference-model ../kitten-tts-2 \
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
