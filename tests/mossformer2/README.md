# MossFormer2 validation

`convert_gguf.py` packages SS, SE, and SR checkpoints and verifies every stored
tensor against the source (including folded SR vocoder weight normalization).
See [conversion commands](../../docs/models/mossformer2.md#conversion).

Build the model integration runner in a Debug build with MossFormer2 enabled:

```bash
cmake --build build/debug --target mossformer2_validation model_spec_system_test -j8
build/debug/bin/model_spec_system_test
```

The runner loads the GGUF through the registry, without a spec override, and
uses one eight-thread session for the entire case list. It saves float32 WAVs and request
timings, rejects non-finite/silent results, and checks exact output equality
when an input and its options recur. Use different input lengths followed by
repeats to exercise graph replacement and reuse.

```json
[
  {"id": "warmup", "audio": "short.wav"},
  {"id": "long", "audio": "long.wav"},
  {"id": "short_repeat", "audio": "short.wav"},
  {"id": "whole_file", "audio": "short.wav", "options": {"audio_chunk_duration_sec": 0}}
]
```

Paths in the case list are relative to that JSON file. For SE/SR, prepare the
same 48 kHz mono inputs for both runtimes. Include input longer than 20 seconds
to exercise default windowing. SS uses `sep`; SE and SR use `s2s`.

```bash
OMP_NUM_THREADS=8 conda run --no-capture-output -n qwen3-tts \
  build/debug/bin/mossformer2_validation \
  MossFormer2-GGUF/mossformer2-se-48k-f32.gguf cuda s2s \
  cases.json results/cpp --log results/cpp.log

OMP_NUM_THREADS=8 conda run --no-capture-output -n qwen3-tts \
  python tests/mossformer2/reference.py \
  --upstream ClearerVoice-Studio --source MossFormer2_SE_48K --variant se \
  --cases reference-cases.json --output results/python --log results/python.log
```

`reference.py` calls the unmodified upstream NumPy decode functions and applies
the same output-length trim as ClearVoice's public file wrapper. Its case list
must omit request options, since it tests upstream defaults. For SR, use
`--variant sr` and the directory containing both original checkpoints.

For controlled F32 CUDA parity, set `NVIDIA_TF32_OVERRIDE=0` for both commands
and compare complete float32 waveforms, including output length. SE uses Kaldi
dither; the random streams differ, so waveform cosine is the comparison rather
than byte identity. Remove precision overrides for performance measurements,
match thread counts, run sequentially, and exclude the first warmup request.
CPU and Vulkan use the same C++ runner with their backend name.

## Restoration boundary regressions

For SE, include a 22-second recording with default chunking and a 900 ms
recording with `audio_chunk_duration_sec=0.5` and
`audio_chunk_overlap_sec=0.1`. Both inputs must contain signal through their
final sample. Set `"require_signal_at_end": true` on each case to reject a
zero-filled output tail.

Apply the same check to both SS output tracks: use a 3.5-second input with
default chunking, plus the 900 ms custom-chunk case above, for F32 and F16.
Also compare single-window, whole-file and zero-overlap outputs against the
pre-fix outputs; these unaffected paths must remain unchanged. Disable
`normalize_output` when comparing the unchanged prefix of a corrected chunked
output, since restoring the tail also changes whole-recording RMS normalization.

For SR, include silence and a signal whose estimated bandwidth reaches Nyquist
(for example, alternating positive/negative samples at 48 kHz). Set
`"expect_input_equal": true` to require exact pass-through, including length;
only such explicitly marked cases permit silent output. Run ordinary narrowband
speech before and after these cases to verify session reuse and unchanged
restoration of non-boundary inputs.
