# Reproducing repaired component checks

These checks use explicit local checkpoints and external output directories.
Run from the repository root in the model environment. C++ builds use only
`build/debug`, with parallel compilation. Save each invoked check's output to
an ignored log directory. No model weights, generated references or WAVs need
to be added to Git, and these repairs do not change model/framework math.

## Standalone framework checks

```sh
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug \
  -DENGINE_BUILD_TESTS=ON -DENGINE_BUILD_EXTENDED_TESTS=ON
cmake --build build/debug --parallel 12 --target gated_delta_net_module_test stft_parity_dump
build/debug/bin/gated_delta_net_module_test cpu --log
python tests/audio/stft_torch_parity.py assets/resources/sample_16k.wav --log
```

Gated DeltaNet retains its independent scalar/reference calculations and state
reuse checks. It is now registered with CTest; no model/checkpoint is required.
Its explicit `cuda` and `vulkan` arguments remain optional backend controls.

STFT compares against `torch.stft`, checks shape and finiteness, and returns
nonzero on a failed numerical comparison. The FP32 FFT comparison defaults to
`atol=1e-4`, `rtol=1e-5` (both explicitly configurable). `--periodic-window`
now selects the same window on both sides, rather than changing only Python.
`--pad-mode reflect` and `--preemphasis` exercise different processing paths.
This is not a waveform-generation or performance benchmark.

## MOSS context configuration

```sh
python tools/models/moss_codec_contexts_check.py --v1-config /path/to/v1/config.json
python tools/models/moss_codec_contexts_check.py --v2-config /path/to/v2/config.json \
  --nano-config /path/to/nano/config.json
```

Only explicitly supplied variants are checked. Use the actual codec variant's
config, not its TTS model config. Context or frame-rate mismatches return nonzero;
missing/unreadable input is an error. Source lookup is relative to this repo,
with `--source` available when checking a separately preserved source snapshot.

## MOSS prompt and hidden references

Existing fixture schemas and tensor computations are unchanged:

```sh
python tests/moss_tts_v15/reference/dump_fixtures.py \
  --model /path/to/MOSS-TTS-v1.5 --codec /path/to/codec \
  --output /path/to/external/v15 --device cpu
python tests/moss_tts_v15/reference/dump_clone_fixture.py \
  --model /path/to/MOSS-TTS-v1.5 --codec /path/to/codec \
  --reference-audio /path/to/reference.wav --output /path/to/external/v15
python tests/moss_ttsd/reference/dump_prompt_fixtures.py \
  --model /path/to/MOSS-TTSD --codec /path/to/codec --output /path/to/external/ttsd
python tests/moss_ttsd/reference/dump_real_clone_fixture.py \
  --model /path/to/MOSS-TTSD --codec /path/to/codec \
  --speaker-1 /path/to/ref_s1.wav --speaker-2 /path/to/ref_s2.wav \
  --output /path/to/external/ttsd
```

Use the original upstream/checkpoint revisions and exact reference recordings
when reproducing an existing golden; changing paths does not make a different
checkpoint/audio interchangeable. The real-clone script retains its fixed
speaker transcripts: the supplied recordings must match those transcripts.
The hidden-state generator retains its BF16 reference precision. Do not label
it an FP32 control. Document hashes, dependency versions and precision beside
generated outputs; do not overwrite a golden with candidate C++ outputs.

## MOSS codec encode/decode

The existing v1 encoder reference generator also accepts `--variant v2` for
stereo/48 kHz/3840-sample-frame inputs. It does not resample input; provide
native-rate audio. Mono v2 input is duplicated into stereo; native stereo is
preserved. The same prepared planar samples and quantizer rows feed the C++
encoder comparison.

```sh
python tests/moss_voicegen/reference/moss_v1_encode_ref.py --variant v2 \
  --codec /path/to/v2-codec --wav /path/to/native-48k.wav \
  --out-prefix /path/to/external/enc --device cpu
cmake --build build/debug --parallel 12 \
  --target moss_tts_local_codec_encode_parity moss_tts_local_codec_decode_parity
build/debug/bin/moss_tts_local_codec_encode_parity --codec /path/to/v2-codec \
  --prepared /path/to/external/enc_prepared.f32 --ref-codes /path/to/external/enc_codes.csv \
  --samples NUMBER_OF_PREPARED_SAMPLES_PER_CHANNEL --channels 2 --quantizers 12
build/debug/bin/moss_tts_local_codec_decode_parity --codec /path/to/v2-codec \
  --out /path/to/external/cpp.wav
```

The encoder checks codes against the independent Python capture. The decoder
capture checks 48 kHz stereo format, exact frame count and finite samples, using
the framework WAV writer. Compare its complete WAV against the independent
Python decoding (no truncation to the shorter output):

```sh
: "${MOSS_MIN_COSINE:?set a reviewed cosine acceptance bound}"
: "${MOSS_MAX_ABS_ERROR:?set a reviewed absolute-error acceptance bound}"
conda run --no-capture-output -n qwen3-tts python \
  tests/moss_tts_local/moss_tts_local_codec_parity.py --codec /path/to/v2-codec \
  --cpp-wav /path/to/external/cpp.wav --out /path/to/external/python.wav \
  --min-cosine "$MOSS_MIN_COSINE" --max-abs-error "$MOSS_MAX_ABS_ERROR" \
  > /path/to/external/codec-comparison.log 2>&1
```

Use bounds established for the same checkpoint/backend/precision; there is no
unverified default. Missing, malformed, wrong-length or numerically mismatched
outputs fail. This is a teacher-forced codec comparison, not free-running TTS
waveform similarity.

## MOSS generation and prompt captures

Generation smoke requests use the ordinary CLI/session path. The maintained
catalog `tools/audiocpp_cli/audiocpp_cli_path_cases.json` contains
`moss_tts_local_long_lived_session` and `moss_voicegen_radio_voice_design`.
The duplicate standalone generation binaries and the broken model-specific
memory wrapper are removed. Shared resource monitoring reports per-case peaks;
it does not claim the deleted wrapper's inferred per-request attribution.

The retained capture paths do not orchestrate another production pipeline:

- `moss_tts_local_prompt_probe --model MODEL --clone-ref CODES_CSV --text TEXT
  --dump PREFIX_CSV` captures the production text processor's clone prefix for
  comparison with the upstream processor. It does not construct a backbone.
- `moss_voicegen_generation_parity` retains the independent prompt/generated-row
  reference assertions. Optional `--dump-codes CODES_JSON` saves its de-delayed
  codes for independent codec replay; it does not replace those assertions.

## Conversion scripts

- `tools/gguf/f5_tts_convert.py` and
  `tools/gguf/minimax_music3/minimax_music3_convert.py` now resolve specs and
  converter defaults from the actual repository root.
- `tools/gguf/voxcpm1_convert.py` requires `--source` and `--output`;
  no personal filesystem destinations are assumed.
- `tools/gguf/liveavatar/liveavatar_components_convert.py` requires `--root`. Omitted
  `--support-output` and `--vae-output` are derived from that parsed root;
  explicit output paths retain precedence.

See each script's `--help` for its remaining checkpoint and quantization options.
For F5-specific controlled-reference generation and probe invocations, see
[f5_tts/README.md](f5_tts/README.md).

## SheetSage2 component assertions

`tests/yue2/sheetsage2_decoder_parity_probe.cpp` validates finite inputs and
outputs and the runtime's single-batch tensor shapes in capture mode. Encoder
output is `[1, subsampled_frames, 1024]`; decoder output is the final token's
`[31678]` logits, not every teacher-forced row. Every encoder `--runs` iteration
is checked, including against the reference when supplied.

Numerical comparison requires both `--reference FILE` and
`--max-abs-error VALUE`. VALUE must be finite and nonnegative; there is no
passing default. Choose and record an independently justified bound for the
checkpoint, precision, backend and component before assessing candidate output.
The probe rejects empty/mismatched/non-finite tensors, zero-energy comparisons,
non-finite metrics and max absolute error greater than VALUE. RMSE and cosine
are diagnostics, not additional calibrated thresholds.

Keep `tests/yue2/sheetsage2_reference_dump.py` as the independent Python oracle.
It writes mixed encoder state, decoder IDs, logits and `meta.json`. Match
`--memory-steps` to that metadata and use the final-token logits as the decoder
reference: if `logits_shape` includes multiple token rows, select its final row
externally without regenerating values from C++. The probe requires exactly
one vocabulary row. Encoder comparison uses the mixed-state oracle and the
corresponding normalized mel input; the generator does not currently export
that mel input. A capture without a reference establishes shape/finiteness
only, never numerical parity. Save output and `--log` diagnostics to an external
log file when invoking the probe.
