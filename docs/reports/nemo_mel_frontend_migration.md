# Configured NeMo-style mel frontend migration

The implementation in `engine/framework/audio/nemo_mel_frontend.h` owns the
complete waveform-to-features sequence: channel handling and resampling when
requested, waveform scaling, deterministic dither, preemphasis, the exact
window and filterbank, STFT/mel/log math, valid-frame calculation,
normalization, masking, padding, and output layout. Models supply a typed
configuration and checkpoint tensors. Model streaming code still decides
which samples form a chunk and, when necessary, its expected feature length.

The contract preserves distinct math that exists in audio.cpp today. In
particular, precise symmetric Hann construction differs from the cached
float Hann; sparse magnitude-squared mel differs from the existing
`LogMelSpectrogram` path; Parakeet's double-accumulator normalization differs
from the framework float normalizer; and Hviske's mixed-precision
normalization differs from both. These are explicit settings, not
family-name conditionals. The constructor validates the canonical runtime
shapes: a window has `win_length` values and a real-FFT filterbank has
`[feature_size, n_fft / 2 + 1]` values. Model loaders explicitly map raw
checkpoint layouts to those shapes; this check does not constrain how a future
checkpoint stores its tensors. A model with different frontend math needs an
explicit contract extension. Reusable frontends are owned by model assets or model
frontend objects, so Canary, Cohere, Citrinet, Sortformer v2, Parakeet, and
Hviske do not rebuild their filterbanks for each extraction. Nemotron 3.5 and
Granite Speech own a configured frontend in their frontend objects as well.
The Granite settings add its periodic Hann, HTK filterbank, dense power,
base-10 log, dynamic-range floor, deltas, and frame stacking to this contract.

## Migration and parity record

The first nine paths were captured before migration from commit
`a7b58a6d3d6ae4143c485266b1c6c09898ad8c72`; Nemotron 3.5 and Granite
Speech were captured before their migration from commit `33841bab`.
Each before/after pair used the same model and audio input. The dedicated probe
writes every F32 value and frame/layout metadata; `compare.py` checks the
complete tensors. All probes ran sequentially in a Debug build with saved
`--log` output.

| Production path | Frames / valid / bins | Values compared | Max absolute difference |
| --- | ---: | ---: | ---: |
| Parakeet centered | 744 / 743 / 128 | 95,232 | 0 |
| Parakeet uncentered | 741 / 741 / 128 | 94,848 | 0 |
| Sortformer v1 offline | 752 / 743 / 80 | 60,160 | 0 |
| Sortformer v2 offline | 752 / 744 / 128 | 96,256 | 0 |
| Sortformer v2 uncentered stream window | 752 / 741 / 128 | 96,256 | 0 |
| Hviske | 744 / 743 / 128 | 95,232 | 0 |
| Citrinet | 752 / 744 / 80 | 60,160 | 0 |
| Canary | 744 / 743 / 128 | 95,232 | 0 |
| Cohere | 744 / 743 / 128 | 95,232 | 0 |
| Nemotron 3.5 centered | 744 / 743 / 128 | 95,232 | 0 |
| Nemotron 3.5 uncentered | 741 / 741 / 128 | 94,848 | 0 |
| Granite Speech 5.0 | 372 / 372 / 320 | 119,040 | 0 |

All twelve paths matched bit for bit: 1,097,728/1,097,728 values, zero
mismatches, and identical frame counts and layouts.

To repeat a capture, create a JSON object mapping each family key in
`tests/nemo_mel_migration/run.py` to a model file path. Keep that mapping and
capture outputs outside the repository. Paths in the mapping may be absolute
or relative to the mapping file. Set the path variables below to your own
model map, audio input, and capture directories. Use a Python environment with
NumPy installed.

```bash
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug \
  -DAUDIOCPP_MODEL_SET=custom \
  -DAUDIOCPP_MODELS=parakeet_tdt,sortformer_diar,sortformer_diar_v2,hviske_asr,citrinet_asr,canary_asr,cohere_asr,nemotron_asr,granite5asr
cmake --build build/debug --target nemo_mel_migration_probe -j 8
python tests/nemo_mel_migration/run.py capture \
  --probe build/debug/bin/nemo_mel_migration_probe \
  --models-json "$MODELS_JSON" --audio "$AUDIO_INPUT" --out "$AFTER_DIR"
python tests/nemo_mel_migration/run.py compare \
  --before "$BEFORE_DIR" --after "$AFTER_DIR" --max-abs 0
```

`run.py index` can build a manifest from previously captured `.f32` and
`.meta` files using the same `--models-json` and `--audio` arguments. The
manifest records model and audio hashes rather than local file paths. A second
24 kHz input exercised resampling and different frame counts for Nemotron and
Granite; it was a shape/run check, not a before/after parity comparison.

## Long-input end-to-end parity

The refactor at `51b8f1d3` was compared with main at `6c70f32d` in 13
supported modes across the nine migrated
model families. Both Debug builds ran on CUDA with `--threads 8`. Runs were
issued manually and sequentially with `--log` and saved stdout. Each pair used
the same model, input, and options. Logs were retained outside the repository.

| Model | Modes and compared output | Result |
| --- | --- | --- |
| Nemotron 3.5 ASR | Offline transcript and 451 token timestamps; streaming transcript, 444 token timestamps, and 444 partial updates | Byte-identical in both modes |
| Granite Speech 5.0 | Offline transcript; streaming final transcript | Byte-identical in both modes |
| Parakeet TDT v3 | Offline transcript and 164 word timestamps; streaming transcript, eight partial updates, and 29 cumulative timestamp updates | Byte-identical in both modes; streaming truncation noted below |
| Sortformer v1 | Offline, 12 timed speaker turns | Byte-identical |
| Sortformer v2 | Offline, 16 timed speaker turns; streaming, nine final turns and four turn updates | Byte-identical in both modes |
| Hviske v5.3 | Offline transcript | Byte-identical |
| Citrinet | Offline transcript | Byte-identical |
| Canary 180M Flash | Offline transcript | Byte-identical |
| Cohere Transcribe | Offline transcript | Byte-identical |

The common input was 56 seconds of repeated four-speaker speech; Hviske used
55.95 seconds of Danish speech.
Sortformer v1 used `sortformer_diar.session_len_sec=60` because its default
20-second graph capacity rejected the long input. Parakeet streaming processed
28 frontend windows in both builds, but both transcripts and timestamp sets
ended around 15 seconds into the 56-second input. Thus its parity result does
not establish complete long-input streaming transcription. Granite streaming
emitted only a final transcript and neither Granite mode exposed timestamps.
These are functional parity comparisons, not quality or performance
measurements.

The feature-tensor comparisons establish frontend parity for the listed
checkpoint and fixture combinations, including the first uncentered
Sortformer v2 window. The separate long-input comparisons establish
end-to-end output parity for their recorded modes and inputs, subject to the
Parakeet streaming limitation above. Neither is a performance comparison; no
warmbench result is used as parity evidence.
