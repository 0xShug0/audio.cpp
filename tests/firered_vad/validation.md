# FireRed VAD Validation

Measured on Apple M4 Pro, macOS, Release build, PyTorch 2.7.0 and
kaldi-native-fbank 1.22.3. Official reference revision:
`c30ec49e8cc69642b0ee65362eba11b9d11c6e54`; checkpoint revision:
`7990aaccc6b7aec1e527743bd30201f2c4a03b8c`.

## Graph and Framework Audit

The CPU graph and cache layout are identical across thread counts. Channel-contiguous
depthwise convolution uses fixed 32-frame blocks with 19-frame input overlap;
GGML distributes these blocks across workers. The final partial block is padded
and its extra outputs discarded. This preserves the filter math and causal history.
Eligible matrix multiplications use a model-local BLAS scheduler following
Moonshine's pattern, with the ordinary CPU allocator as fallback.
The Metal graph is unchanged. The convolution framework files remain at their
pre-port versions; the CPU rewrite is entirely model-local.
The graph optimizer retains the framework's CPU/GPU presets, with metadata-op
elision and identity-materialization folding disabled to preserve allocator view
dependencies.

The remaining framework addition is opt-in sparse Kaldi mel projection: 20 added
and two removed lines across its header and implementation. It skips zero filter
coefficients while retaining the order of nonzero accumulation. Existing callers
keep the dense default. The accompanying unit test requires exact equality.

## Performance

FP32 weights, the same 14.07194-second repository recording and default detection
options. Five warmups followed by 21 measured runs; table entries are medians.
Network timing excludes the frontend; total includes frontend and postprocessing
but excludes model loading. Python timings are retained from the preceding audit
on the same input and machine. Python and native CPU use matching requested thread
counts. GGML forwards the count to BLAS; Apple Accelerate manages its own workers.

| Model | Runtime | Threads | Network (ms) | Total (ms) | RTF |
|---|---|---:|---:|---:|---:|
| VAD | Python CPU | 1 | 15.19 | 24.00 | 0.001706 |
| VAD | Native CPU | 1 | 14.03 | 15.21 | 0.001081 |
| VAD | Python CPU | 8 | 68.34 | 77.24 | 0.005489 |
| VAD | Native CPU | 8 | 5.76 | 7.17 | 0.000510 |
| VAD | Native Metal | 8 | 1.83 | 3.14 | 0.000223 |
| Stream-VAD | Python CPU | 1 | 9.49 | 19.19 | 0.001364 |
| Stream-VAD | Native CPU | 1 | 6.78 | 8.13 | 0.000578 |
| Stream-VAD | Python CPU | 8 | 35.58 | 45.95 | 0.003265 |
| Stream-VAD | Native CPU | 8 | 4.48 | 5.73 | 0.000408 |
| Stream-VAD | Native Metal | 8 | 1.45 | 2.64 | 0.000188 |

### CPU Change Breakdown

Fresh measurements against commit `aa82c1f8`, with separate binaries for each
variant. All columns are network medians in milliseconds; no test-only switches
are present in the final implementation.

| Model | Threads | Before | BLAS only | Tiled graph only | Both | Network reduction |
|---|---:|---:|---:|---:|---:|---:|
| VAD | 1 | 69.17 | 33.04 | 50.97 | 14.03 | 79.7% |
| VAD | 8 | 10.83 | 7.76 | 8.96 | 5.76 | 46.8% |
| Stream-VAD | 1 | 51.80 | 16.09 | 44.06 | 6.78 | 86.9% |
| Stream-VAD | 8 | 8.68 | 6.19 | 6.88 | 4.48 | 48.3% |

On the 120-second input at eight threads, VAD network time falls from 89.74 to
28.30 ms, and Stream-VAD from 73.41 to 17.00 ms. These are fresh before/after runs.

There is a short-input regression at eight threads. On the official 2.24-second
English example, VAD network time increases from 2.160 to 2.863 ms (+32.5%), and
Stream-VAD from 1.728 to 2.463 ms (+42.5%). BLAS alone also regresses short VAD
(3.511 ms). The combined path is not a universal speedup. An earlier 256-frame
tile left short inputs with one convolution work unit; the final 32-frame tile
exposes more parallelism but does not eliminate this short-input cost. No
thread-dependent graph selection or input-specific benchmark gate is used.

Metal network medians were 1.776 -> 1.834 ms for VAD and 1.420 -> 1.447 ms for
Stream-VAD. The graph is unchanged and both probabilities and timestamps are
bit-identical before/after; these small timing differences do not establish a
performance change.

### Remaining Framework Optimization

A separate component probe used the production frontend settings and GGUF CMVN,
alternating dense/sparse execution order over five warmups and 21 measurements.
Both paths produced bit-identical features on every iteration. These timings
include the complete frontend, including sparse-filter construction.

| Audio duration | Dense frontend (ms) | Sparse frontend (ms) | Speedup |
|---|---:|---:|---:|
| 0.16 s | 0.284 | 0.084 | 3.37x |
| 2.24 s | 3.364 | 0.302 | 11.14x |
| 14.07 s | 20.695 | 1.220 | 16.96x |
| 120 s | 176.512 | 9.459 | 18.66x |

Dense and sparse timing ranges did not overlap for any of these inputs.

## Accuracy and Coverage After Cleanup

- Both FP32 variants pass official Python parity on the 14.07-second input with
  CPU at one and eight threads, and Metal. Timestamps match exactly. Maximum
  probability error is below 0.00054 across these runs.
- Both variants also pass parity on repeated 120-second audio with eight CPU
  threads. Timestamps match exactly; maximum probability error is below 0.000003.
- Causal tests compare full-input and streaming probabilities, retained history,
  reset and finalization. Packet sizes include 1, 399, 160, 640, 2560 and 7777
  samples. The irregular-packet timestamps match full-input detection.
- Added feature-chunk checks at 31, 32, 33, 63, 64, 65, 255, 256 and 257 frames to
  exercise convolution block boundaries and partial tails. These pass on CPU at
  one and eight threads in the final 32-frame-tile runs. The earlier
  256-frame boundary checks also passed on Metal. The CPU graph-only fallback
  was checked against Python parity during development.
- Parity thresholds remain unchanged: normalized-feature maximum error 0.001,
  RMSE 0.00001, probability maximum error 0.002 and exact timestamps.
- Both unit tests pass: upstream postprocessing fixtures and sparse/dense Kaldi
  equivalence across short inputs, Povey/Hamming windows, LFR and CMVN.
- The probe rejects unsupported backend names instead of silently timing CPU.

The extra F16 VAD CPU test passes Python parity on the official English example.
For Q8 Stream-VAD, timestamps match, but 16-frame CPU streaming chunks retain a
pre-existing 0.006149 maximum probability difference from the dequantized Python
reference (above the 0.002 threshold). Chunked probabilities are bit-identical to
the baseline CPU implementation. Full-input error improves from 0.006148 to
0.000456 with BLAS. The Q8 streaming check is therefore not reported as a Python
parity pass, and its tolerance is unchanged.

See [the model guide](../../docs/models/firered_vad.md) for conversion, CLI and
reference/probe commands. CLI path cases are registered as `firered_vad_offline`
and `firered_stream_vad_streaming`.

## Normalized Options

Regenerated both variants in F32, F16 and Q8_0 with the normalized embedded spec.
All tensor names, shapes, types and bytes match the previous six packages exactly.
The public durations use milliseconds or seconds; conversion to 10 ms frames
happens in the session parser. No old-name aliases are provided.

Manual CLI comparisons with the previous binary and packages verified:

- Default F32 Metal segments match exactly.
- Q8_0 Metal offline segments match with equivalent non-default options, including
  rounding 35/105/95 ms to 4/11/10 frames and 0.405/0.645 seconds to 41/65 frames.
- Q8_0 Metal streaming events and final segments match with equivalent smoothing,
  speech/silence confirmation, start padding and forced duration splits.
- The rebuilt Stream-VAD F16 package runs through CPU streaming.
- Old option names and out-of-range chunk durations are rejected.
- Both postprocessing and sparse-frontend unit tests pass.
