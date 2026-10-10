# Phonon-2 hotwords and live transcription

Validated follow-up for [PR #863](https://github.com/0xShug0/audio.cpp/pull/863),
extending model-only branch `1e8ebc65`. Runtime validation used integration base
`cb7441e6`, including the separate CUDA precision dependency in
[PR #861](https://github.com/0xShug0/audio.cpp/pull/861). CPU/Vulkan do not need
that dependency. Windows, Ryzen 7950X3D with 8 threads, RTX 3090 24 GiB (CUDA
device 0, Vulkan device 1). This follow-up does not change CUDA/Vulkan kernels
or enable parallel jobs.

## Changes

- Port Phonon-2's optional piece-level hotword policy: spelling alternatives,
  case variants, spoken alternatives, at most 25 terms, adjustable logit bonus.
  Blank and duration scores remain unbiased; each request owns its policy.
- Add a Phonon-specific rolling live session, using the original 50 ms energy
  gate, 350/500 ms partial cadence, 700 ms silence boundary and 30 s phrase cap.
  It re-decodes the current phrase and offsets committed word timestamps.
- Add an optional `partial_text_snapshot` event and CLI/server rendering for
  revisable text. Existing append events carry committed text. Default event
  behavior for other models is unchanged; stock Parakeet uses its old session.
- Update converter capability metadata and usage documentation. Existing model
  tensors are unchanged. Existing GGUFs can use an updated package-spec override.

## Actual validation

| Check | Result |
|---|---|
| Original hotword automaton | 9 lists, 329 transition/bonus vectors exactly match |
| Previous native offline output | 180/180 identical text and word timestamps: 20 recordings x 3 backends x 3 storage variants |
| Original decoder final text | 54/54 primary cases match; 6 additional named-term cases match |
| Original streaming final segments | 18/18 match |
| Stock Parakeet regression | CPU/CUDA/Vulkan, offline and buffered streaming: 6/6 identical previous text/timestamps |
| Real server | CPU/CUDA/Vulkan: JSON, multipart, streaming upload, live PCM with unaligned byte chunks, isolated hotwords and reuse pass |
| Session lifecycle | CPU/CUDA/Vulkan: 13/2049/whole-file chunks, invalid ordering, repeated finalization, sink failure, nonfinite audio, restart and 33 s idle input pass |
| Multiple phrases | CPU/CUDA/Vulkan: committed deltas equal final text; absolute word offsets agree |
| 30 s voiced cap | CUDA: 32 s input commits two segments, with offsets on the correct side of sample 480000 |
| Invalid hotwords/strength | Rejected; following valid requests succeed. Repeated multipart fields and silence with bias pass |
| CPU CTest | 54/54 pass |
| Converter/timestamp Python tests | 12/12 pass |

The reference is the original `fermion-research==0.2.11` Python policy and its
dense Torch CPU decoder, using the pinned Phonon artifact from the initial
integration. Inputs and model hashes are recorded in the previous Phonon
validation report. Raw cases, configs, outputs and logs are retained under
`outputs/phonon2-features-20261010/`; [summary JSON](phonon2_features.json) records
the implementation hashes and check counts.

Original CPU word timings match in 27/36 hotword cases. In the remaining nine
(one recording in each backend/storage combination), text matches but some
boundaries differ. The native timestamps are identical to the preceding native
implementation for that recording. Provisional sequences match in 6/18 stream
cases; other short hypotheses differ. A 1.6 s prefix demonstrates the discrepancy
before this change too: native says `see a`, original dense CPU says `see it.`.
These are not new streaming/hotword regressions. This is not a general WER claim
or a promise that every chunk boundary preserves full-context output.

## Illustrative request times

Uploaded 3 s LibriSpeech fixture, Q8, five provisional updates plus finalization.
Median of two warm requests after one cold request. Audio is uploaded immediately,
not paced in real time; TTFT is the server's inference measurement. These are
small functional-run samples, not a throughput benchmark or a speedup claim.

| Backend | Warm whole request (ms) | First provisional text (ms) |
|---|---:|---:|
| cpu | 837.3 | 98.0 |
| cuda | 202.9 | 52.8 |
| vulkan | 268.4 | 48.7 |

## Reproduction

Build the existing CPU or CUDA/Vulkan preset with `parakeet_tdt` enabled and
`ENGINE_BUILD_TESTS=ON`; build `audiocpp_cli`, `audiocpp_server`,
`phonon_hotwords_test`, and `phonon_features_probe`. Run:

```bash
ctest --test-dir build-phonon2-cpu --output-on-failure
python tests/parakeet_tdt/test_phonon2_conversion.py
python tests/parakeet_tdt/test_phonon2_timestamps.py
python tests/parakeet_tdt/validate_phonon2_hotwords.py \
  --probe build-phonon2/bin/phonon_features_probe \
  --original-hotwords ORIGINAL_WHEEL/fermion/_speech/hotwords.py \
  --source-config models/Phonon-2-source/extracted/config.json
phonon_features_probe request.json
```

Example probe input (add `.exe` for Windows executables):

```json
{"model":"models/Phonon-2-GGUF/phonon-2-q8_0.gguf","audio":"recording.wav",
 "backend":"cuda","device":0,"streaming":true,"chunk_samples":800,
 "hotwords":["Ada Lovelace","Quillon"],"strength":2,"lifecycle":true}
```

The local complete runs use `validate.py --reference`, `validate_native.py`,
`check_disabled.py`, `check_server.py`, `check_extra.py`, and `check_routes.py`
in the artifact directory. See [model usage](../community_models/phonon2.md)
for CLI/server commands and the spec override needed by old GGUF packages.
Metal, Linux/macOS, and the separate original Phonon WebSocket protocol were
not validated/implemented in this pass.
