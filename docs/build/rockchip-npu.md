# Rockchip NPU (RKNN) Decoder Path

Opt-in, per-stage acceleration of one model stage on Rockchip RK3588/RK3576/RK356x NPUs through
`librknnrt`. This is not a ggml backend: ggml has no RKNN backend, and a full backend port is a
different project. Here a single model stage that already computes through its own graph and
`ExecutionContext` is routed to a pre-compiled RKNN network instead of ggml.

Currently implemented for three families: the `qwen3_tts` speech tokenizer decoder (the vocoder,
codes to PCM), the `piper_tts` VITS decoder (latent to PCM) and the `kokoro_tts` iSTFTNet
generator (latent to spectrogram). Everything below is measured on an
Orange Pi 5 Plus (RK3588, 4x A76 + 4x A55).

**Conditions on the board numbers.** Every RTF, ms and RSS figure here was taken with the vendor
`rknpu_ondemand` governor, which ramps the NPU to its 1000 MHz OPP: check
`cat /sys/class/devfreq/fdab0000.npu/cur_freq` (1000000000 under load) and
`cat /sys/class/devfreq/fdab0000.npu/governor`. The OPP table on that overlay starts at 300 MHz,
so a board running the mainline `rocket` driver's `performance`-less default can sit at a fraction
of that clock and every figure here scales with it. The CPU side of the same runs is
`--threads 4` on the A76 cluster, and the native build (`scripts/build_linux.sh --backend cpu`, which
turns on the host-ISA ggml kernels): a plain `cmake -DCMAKE_BUILD_TYPE=Release` build of the same
tree measures the `qwen3_tts` CPU path at RTF 4.287 and 4510 MB against the 3.010 and 4200 MB below,
while the `kokoro_tts` CPU path reproduces to 1.409 against 1.415. Absolute CPU-path figures are
therefore build-dependent, ratios are not. The board's DRAM read ceiling is 26.4-29.4 GB/s, measured with a
cache-bypassing STREAM triad (4 A76 threads, 804 MB working set) and `sysbench memory` with a 64 MB
buffer; `sysbench` with its default 1 MB buffer reports 61 GB/s because it never leaves L3, and
`stress-ng --stream` is cache-sized too, so neither is a DRAM measurement.

## Requirements

Board side:

- `librknnrt.so` (RKNPU2 runtime, 2.x) on the loader path, and `rknn_api.h` for the headers.
  The Rockchip SDK's `install.sh` puts the library in `/usr/lib` and the headers in
  `/usr/local/include`.
- The RKNPU kernel driver. Check with `cat /sys/kernel/debug/rknpu/version` or `dmesg | grep -i rknpu`.

Host side (x86_64, for building the `.rknn`):

- `rknn-toolkit2` in a dedicated venv. Its pins are strict (`numpy<=1.26.4`, `onnx==1.16.1`,
  upper-bounded `torch`, Python 3.6-3.12); do not resolve them against a system Python.
- On Python 3.12 the 2.3.2 converter needs two patches the SDK does not ship: `setuptools<81`
  (`rknn.api.base_utils` imports `pkg_resources`, removed in setuptools 81), and an
  `onnx/mapping.py` shim inside the venv's `onnx` package (the module is gone in onnx 1.16, and
  onnx 1.15 has no cp312 wheel). RKNN does `import onnx` then `onnx.mapping.X`, so the shim must
  also be bound by appending `from onnx import mapping` to `onnx/__init__.py`.

## Build

Native on the board (the same flags work in a cross build):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DENGINE_ENABLE_RKNN=ON \
  -DENGINE_RKNN_INCLUDE_DIR=/usr/local/include \
  -DENGINE_RKNN_LIBRARY=/usr/lib/librknnrt.so \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=qwen3_tts
cmake --build build --parallel --target audiocpp_cli
```

`ENGINE_ENABLE_RKNN` is `OFF` by default and nothing in the default build links `librknnrt`. With
the flag off, passing `qwen3_tts.rknn_decoder` fails with
`Qwen3 speech decoder RKNN: built without ENGINE_ENABLE_RKNN` rather than falling back silently.

## Model preparation

The `.rknn` is not downloadable and nothing in the runtime produces it. Build it on the host from
the `Qwen/Qwen3-TTS-12Hz-0.6B-Base` checkpoint (the `speech_tokenizer` subdirectory, the same
weights the GGUF package was converted from):

```bash
python3 tools/prepare_qwen3_tts_rknn_vocoder.py --model-dir <snapshot-dir> --codes-length 64
python3 tools/prepare_qwen3_tts_rknn_vocoder.py --model-dir <snapshot-dir> --codes-length 16
```

Each run re-traces the decoder as one fixed-shape streaming step, checks the stepwise decode
against a full decode (SNR must stay above 40 dB), exports ONNX, and compiles it for
`--target` (default `rk3588`). The graph is stateful: the causal-conv tails and the sliding-window
KV history travel through the graph I/O as one flat fp32 cache, so a step decodes from continuous
state and there are no chunk seams. `--codes-length` is compiled in, so the two-tier first chunk
needs one file per batch size; the cache layout does not depend on it, so a small-batch step's
cache feeds the large-batch model directly.

The two stages want different toolchains (the export needs the model package, the conversion needs
the toolkit's pinned `torch`/`onnx`), so run them in the two venvs and hand the ONNX over:

```bash
<export-venv>/bin/python tools/prepare_qwen3_tts_rknn_vocoder.py \
  --model-dir <snapshot-dir> --codes-length 64 --skip-rknn --output vocoder_stateful_64.onnx
<converter-venv>/bin/python tools/prepare_qwen3_tts_rknn_vocoder.py \
  --onnx vocoder_stateful_64.onnx --target rk3588
```

`--onnx` reads the step contract (batch size, cache size, head dim) back out of the graph, so the
conversion stage restates nothing.

The export stage is version-sensitive: a recent `torch`/`onnx` decomposes LayerNorm and Softmax into
patterns (`Pad`, `Exp`, `Shape`, `ConstantOfShape`) that `rknn.build` aborts on inside
`fold_constant`, with a core dump and no Python traceback. The script counts those ops after
exporting and says so there, where it is cheap to act on. Measured with this graph: the
toolkit-era pair exports 1064 nodes and compiles; a current pair exports 2653 and aborts.

A `.rknn` is compiled per target SoC and is not portable between chips. The graph is built fp16
(w16a16) and deliberately not quantized: the Snake activation makes Q8/INT8 audibly noisy
(9.5-13.4 dB SNR) and fp16 measures lossless (56.8 dB).

`--target` accepts the RKNPU2 platforms the toolkit knows. Only `rk3588` has been built and
measured on a board here; `rk3566` conversions of all three families have since been built
successfully on the host (the engines are unmeasured on a board), the others are untested, not
unsupported.

### Piper (VITS) decoder

`tools/prepare_piper_tts_rknn_vocoder.py` slices a Piper voice ONNX at the seam the runtime already
has — the inverse-flow input `z [1,192,N]` and its frame mask `y_mask [1,1,N]` (plus `g` for a
multi-speaker voice) — checks the fixed-window decode against a full-length decode, and compiles
the slice:

```bash
<export-venv>/bin/python tools/prepare_piper_tts_rknn_vocoder.py \
  --voice-onnx en_US-lessac-medium.onnx --skip-rknn --output decoder_z.onnx
<converter-venv>/bin/python tools/prepare_piper_tts_rknn_vocoder.py \
  --onnx decoder_z.onnx --target rk3588
```

The conversion stage re-runs the self-check on white-noise latents (the voice's own encoder is not
involved once the slice exists), and its global gate applies only to decodes that contain seams: a
latent shorter than the window is one tail window with no seam, and its global SNR is the documented
tail deficit, not a stitching failure.

The seam is the flow input and not the input of `dec.conv_pre`: the flow belongs to the decoder
half, and the runtime's latent is the pre-flow one. Slicing at `conv_pre` compiles a decoder that
expects post-flow samples and feeds it pre-flow ones, which is quiet mush (measured: output RMS
4x low, no intelligible speech).

RKNN has no dynamic shapes, so the engine is compiled for one window of 93 latent slices and the
runtime walks the latent over it: 45 slices emitted per step, 24 slices of real context on each
side, a short window zero-padded and mask-zeroed at the end that has no data, and the window that
reaches the end of the latent extends its left context to fill the window. The 24 is measured, not
inherited: at the 45+5+5 window paroli ships, interior windows score 8-60 dB against a
full-length decode (93: 98+ dB, 77: 51+ dB). The mask is 1 over the real slices and 0 over the
padding — the flow multiplies its activations by the mask after every convolution, so a padded
frame marked 1 propagates its conv biases into the real frames and costs 10-20 dB there.

The last slices of the latent are the one thing a fixed-shape window cannot reproduce: a
full-length decode has the end of the tensor there, where a window has padding, and a conv layer
two levels deep reads the pad's own conv output (nonzero, from the biases) where the full decode
reads the implicit zeros past the tensor end. No mask value **in the padding** fixes it, because
the mask reaches only the flow: the VITS decoder (`conv_pre`, 18 resblock convs, 3 upsamples,
`conv_post`) is unmasked, so the pad's conv output is what the last real slices read. Measured: the
last ~7 slices of a short latent land 20-40 dB off, the interior windows are exact. paroli ships
the same trade.

`--dec-mask` does fix it, by masking inside the decoder: `Mul(y_mask)` after every `/dec/`
Conv/ConvTranspose, with the mask upsampled by all-ones `ConvTranspose1d` at each `dec.ups` stride
(8/8/4) so it is the right length at each level. Measured on the tail window: 28.6 → 117.8 dB
(47-frame latent) and 43.8 → 118.6 dB (30-frame) against a full-length decode, interiors unchanged
byte for byte, and 4.8 → 45.8 dB for a 47-frame tail window run through `librknnrt` directly. It
costs +21% window time (RTF 0.0688 → 0.0832 on the 443-character utterance) and changes nothing
there, because every chunk of that utterance exceeds 69 frames, so no tail window is short: the
defect is a short-utterance property. It stays opt-in because the board does not hear it: over 8
realistic short utterances (one word to a phone-number string, three of them short enough that the
tail window carries 28-35 pad slices) the two engines score the same against the CPU decode within
0.1 dB on every one (25.82/25.82, 24.62/24.62, 33.27/33.24 on the padded three), because the fp16
engine's own noise floor against the CPU decode (22.4-36.2 dB) sits at or below the fp32 tail
deficit (28.6-43.8 dB) — the defect is buried under the engine's noise, so the +21% buys nothing
audible. The two engines' outputs do differ by ~60 dB, so the masked engine is live and distinct.
A `Resize` node for the mask upsample measures 50-53 dB in the PC simulator and -7.9 dB on the
board — `Resize` does not lower to a target the board runtime accepts, so the upsampler is a
`ConvTranspose1d` of all ones (exact nearest, and the same operator the decoder already runs on the
NPU).

### Kokoro (iSTFTNet) generator

`tools/prepare_kokoro_tts_rknn_generator.py` re-traces the generator the GGUF weights describe
(`decoder.module.generator.*` in `kokoro-v1_0.pth`) at one fixed window and compiles it. The seam
is the generator alone — `x [1,512,W]`, the harmonic conditioning `har [1,22,60W+1]`, the style
vector, and the frame mask at the two rates the graph normalizes at, in and out as
`spec_phase [1,22,60W+1]`. Source-signal synthesis, the forward STFT and the iSTFT head stay on
CPU:

```bash
<export-venv>/bin/python tools/prepare_kokoro_tts_rknn_generator.py \
  --checkpoint kokoro-v1_0.pth --config config.json --dump-dir <runtime-dump> \
  --output-dir artifacts --frames 128 --skip-rknn
<converter-venv>/bin/python tools/prepare_kokoro_tts_rknn_generator.py \
  --onnx artifacts/generator_w128.onnx --target rk3588
```

The window is capped by the NPU, not by the model: an op-encoding field on the time axis is 13
bits, so the longest axis in the graph, the fine rate `60W + 1`, has to stay under 8192. At W=512
the compiler emits ~800k `REGTASK: The bit width of field value exceeds the limit, ... limit:
0x1fff, value: 0x7800` lines and the engine voices clipped noise (rms 32764 of 32767, SNR -27 dB
against the CPU decode). 128 is the widest power of two inside the limit, and it emits
`128 - 2*28 = 72` latent frames per step, where 28 is the measured receptive field. The line itself
is not the failure: the RK3566 builds of the three shipping engines log a few hundred to a few
thousand of them (measured: 352 for the Kokoro w128 engine, 355 for the Piper w93 engine, 6570 for
the Qwen3 B=64 vocoder) and measure clean on the board; the RK3588 build of the Piper w93 engine
logs none. What kills the engine is the magnitude — `value: 0x7800` is 30720, four times past the
field, repeated ~800k times. `tools/rknn_build_log.py` is the gate that separates the two regimes:
it captures fds 1/2 around `rknn.build`, because those lines come from the C++ compiler and not from
Python logging, and aborts past 100 000 lines.

The field is per axis and per op, and the ceiling is escapable: slicing the fine-rate axis inside
the graph takes W=256 from 403 587 REGTASK lines to 0, and the re-traced module keeps every gate —
60.5 dB against the runtime's generator, 135.6 dB padded-window exactness, 124.4 dB export against
torch. The engine then voiced fp16-saturated output (117 682 non-finite values out of a 14.85 s
utterance, contiguous in columns 8329-13872, peak 65440 = fp16 max), and the cause is the AdaIN
global pooling: its statistics are raw sums over the fine axis, so the same site that peaks at 7681
at W=128 peaks at 1.1e6 at W=256 (per-node maxima from an onnxruntime fp32 run: first node past
fp16 max a Mul at 612 487, largest node an Add at 1 106 990). The diagnosis is a probe, not a
shipped change: recomputing those statistics as count fractions holds the same graph's fp32 range to
15 361 with first-over = None. The cure in the producer is `--max-axis`,
which moves the runtime's window walk *inside* the graph — windows of `(max_axis-1)//scale` frames
with `--padding` frames of context per side, each window through the full unsliced module, emitted
columns concatenated the way `decode_windowed` overwrites the seam column. Then every in-graph
reduction axis is at most `max_axis`, so the 13-bit field and the fp16 range are satisfied by
construction: W=256 (3 windows of 136 at padding 28) builds with 1 REGTASK line and voices 0
non-finite values, peak 56.25, 46.8 dB against the chunked torch reference, and W=160 (2 windows)
builds with 0 lines. The 9.5 dB against the *monolithic* reference is the per-window AdaIN gap, not
the engine: the chunked-vs-monolithic gate measures 9.9 dB and the fp32 PC simulator of the same
graph gives the identical 9.5 dB. It does not ship: the in-graph escape replicates the module
inside the engine (142.7 MB for 3 windows against 74.7 MB for one W=128 window), so widening the
window that way costs the weight copies the runtime walk exists to avoid, and the stitched result is
a per-window synthesis, not the monolithic one.

The cheap wide window — the widest mono window the field allows, W=136 (emitted 80, 15 windows for
the 14.85 s utterance, 76.0 MB) — was built and measured on the board, and it is not a win: RTF
1.057 against 1.114 at W=128, RSS 2345 against 2340 MB, 0 non-finite values, and the pumping spread
**7.67 dB against 7.23 dB**. Fewer seams and statistics over more frames do not help, the same
verdict overlap-blending gave: the pumping is per-window loudness, not seam geometry. The same run
settles where the normalized-statistics cure applies at all: at W=136 the raw sums peak at 8161
(nothing over 65504) and at W=256 at 1.1e6, so the band of widths where the sums saturate *and* the
field passes is empty. Normalized statistics matter only in a graph whose span exceeds 8191, which is
the sliced wide window the field rejects, so they are not implemented.

**The per-window statistics are the audible artifact; feeding them is not the fix.** Listening
compared the NPU render with the CPU render of the same text and seed found the loudness of
successive 0.9 s windows swinging -2.8 to +4.5 dB against the CPU render (spread 7.2 dB) with no
clicks at the seams - the per-window renormalization, heard as compression. `--global-stats` makes
the statistics graph I/O (`stats_in`/`stats_out`/`use_global`, 7 inputs / 2 outputs) and the runtime
computes them on CPU from the full latent, one full-length generator pass. In fp32 that is exact:
the stitched per-window decode against the full-length decode goes from 9.7 dB to 135.4 dB on the
486-frame dump, with the stitch, the tail and the padding exact once the statistics are global. That
confirms the diagnosis and rules out this engine as the cure: on the board the fed statistics scale
the window activations past the engine's fp16 range and every window after the first renders as a
constant-energy tone. Measured on a 14.85 s utterance: windows 1..15 carry rms flat at 51.5 dB
(the CPU render swings 57.1 to 67.4), spectral flatness 1.7e-17, and all the energy at DC, 4800 Hz
and 9600 Hz - Nyquist/2 and 0.8 * Nyquist, i.e. a saturated output, not speech. Only window 0 is
intelligible, because that one is taken from the CPU pass. The render also costs 1.8x the wall
clock (RTF 2.18 against 1.08) and +1.5 GB RSS. The path stays opt-in and off by default, as a
PC-simulator diagnostic for the AdaIN hypothesis; the fix has to be in the window walk, not in the
statistics. The capture path needs one runtime detail: `ggml_build_forward_expand` in the vendored ggml
does not set `GGML_TENSOR_FLAG_OUTPUT`, so a captured statistics node with no other consumer is
freed by the allocator and its buffer is handed to a later node - the statistics nodes must be
marked as graph outputs. Default off.

Three rewrites are compiled in, all of them the reference Kokoro RKNN port's:

- **Weight-norm folded at load.** Left in the graph, `vector_norm` traces to 51 `ReduceL2` nodes
  and `rknn.build` aborts.
- **Masked instance norm through `GlobalAveragePool`.** A `ReduceSum` over the time axis lowers to
  a register field capped at 8191 (the same limit as above); GAP is the pooling engine's own
  reduction. The mask has to zero the padding at *every* stage boundary — after the upsampling
  reflect-pad, on the resblock sum, on the noise branch output — not only inside AdaIN's
  statistics, because a conv after a masked AdaIN writes real-frame sums into the pad and the next
  `conv_transpose` reads them where a full decode reads implicit zeros. With that, a padded window
  reproduces the same frames decoded unpadded to 135 dB.
- **Snake's `sin` to a range-reduced minimax polynomial, and its `1/alpha` folded into the
  following convolution.** `librknnrt` has no NPU `Sin`: the 48 sine nodes of a 128-frame window
  ran on CPU and were 41% of the window's time (350 ms of 846 ms, `RKNN_LOG_LEVEL=5` operator
  ranking). The polynomial is the fold (below) plus 3 Mul + 2 Add at degree 5, and degree 5, 7 and
  9 measure the same 50.0 dB against the exact sine once the argument is folded, so the cheap fit
  wins. The alpha-fold is exact algebra and removes one multiply and one divide per Snake;
  elementwise ops are 84% of this graph's NPU time and the convolutions 15%, so removing
  full-tensor passes is where the wall clock is.
- **`--no-sin-reduce` clips the Snake sine argument at ±pi instead of folding it (the pre-fold
  export, kept for reproducibility).** The clip is what sets a 30.4 dB floor: 1.6% of the 49M
  Snake arguments of a 200-frame decode are past pi (max 20.5), and the clip evaluates the
  polynomial on a clamped argument, so degrees 5, 7 and 9 all plateau at 17.6 dB sine-level SNR.
  The fold `k = Floor(x/(2*pi) + 0.5); xr = x - k*2*pi` (Mul/Add/Floor/Mul/Sub, all NPU-native)
  puts the polynomial on `xr` in [-pi, pi], where the fit is valid: 41.7 dB sine-level at degree 5
  (70.5 at 7, 103.1 at 9) and 50.0 dB at the generator output against the exact sine (clipped
  form: 30.4). The fold is not free: measured on the board with both engines built from the same
  producer revision and interleaved three times per utterance, it costs **+36% generator time**
  (RU 33.6 s: 28400 ms median against 20849, RTF 1.074 against 0.851; EN 5.5 s: 4482 against 3239 ms,
  +38%), and the NPU load says why — the clipped engine holds Core0 at 86-94% during a run, the
  folded one at 70-72%, so the `Floor` chain does not stay on the NPU end to end. Both engines log 0
  REGTASK lines on RK3588 (the 352-line count belongs to the RK3566 build, above). The two NPU
  renders differ by **17.3 dB** (RU; 17.0 dB on EN, max|d| 8020 of 32767), i.e. the change is
  audible, and in a listening test the folded engine ranks between the CPU render and the clipped
  one. The default is the fold on quality, not on speed. The CPU-vs-NPU SNR of both (0.46/0.48 dB)
  is the documented per-window AdaIN
  renormalization, not the sine.

The exported graph is checked against the torch graph it was traced from, on the last window of
the runtime's walk, with the torch reference evaluated in the same sine form (124.6 dB), so the
gate measures the export and not the approximation.

## Verified

- Export stage: stepwise-vs-full-decode self-check 111.4 dB at B=64 and 103.7 dB at B=16,
  `cache_numel` 1303424, `head_dim` 64, `kv_len` 71.
- Conversion stage: converting a reference ONNX reproduces the reference `.rknn` byte for byte
  (identical MD5, 259688011 bytes).
- Kokoro export stage: the re-traced generator reproduces the runtime's own generator at 60.5 dB
  interior SNR against a dumped decode, a padded 128-frame window reproduces the same frames
  decoded unpadded at 135.1 dB, and the exported ONNX reproduces the torch graph it was traced
  from at 124.6 dB.


## Run

```bash
./build/bin/audiocpp_cli --task tts --family qwen3_tts --mode streaming \
  --model models/Qwen3-TTS-12Hz-0.6B-Base-GGUF --text "..." --voice-ref ref.wav \
  --out out.wav \
  --session-option qwen3_tts.conv_weight_type=f16 \
  --session-option qwen3_tts.rknn_decoder=/root/tts-rknn/vocoder/vocoder_stateful_64b_fp16.rknn \
  --session-option qwen3_tts.rknn_decoder_first=/root/tts-rknn/vocoder/vocoder_stateful_16_fp16.rknn
```

`rknn_decoder_first` is optional and only affects streaming: the first token group runs on the
small-batch model, which is what cuts time-to-first-audio. When the RKNN decoder is active the
ggml decoder weights are not loaded at all, which is where most of the memory saving comes from.

Piper takes one option and is offline only:

```bash
./build/bin/audiocpp_cli --task tts --family piper_tts \
  --model models/Piper-TTS-GGUF/piper-en-us-lessac-medium-orig.gguf \
  --text "..." --text-chunk-size 60 --out out.wav \
  --session-option piper_tts.rknn_decoder=/root/tts-rknn/piper-lessac-w93.rknn
```

The engine is per voice: a `.rknn` compiled from lessac's latent is validated against the loaded
voice's channel count at init and fails with a message naming the producer if it does not match.

Kokoro takes one option and is offline only:

```bash
./build/bin/audiocpp_cli --task tts --family kokoro_tts \
  --model models/Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf \
  --text "..." --seed 1234 --out out.wav \
  --session-option kokoro_tts.rknn_generator=/root/tts-rknn/gen_w128.rknn
```

The engine is per model, not per voice: the generator weights are in the GGUF, and the engine is
validated against the loaded model's latent channel count, its spectrogram row count and its audio
samples per latent frame at init.

## Measured on RK3588

300 Russian characters, greedy decoding, `conv_weight_type=f16`, all other stages on CPU:

| mode | RTF / wall | peak RSS | decoder ms per token |
|---|---|---|---|
| offline, ggml decoder | 3.01 | 4200 MB | 130.1 |
| offline, RKNN B=64 | 2.17 | 2560 MB | 64.1 |
| streaming, ggml decoder | 151.2 s | - | 365 |
| streaming, RKNN 16+64 | 57.7 s, first audio 0.88 s | - | 62.4 |

Streaming gains more than offline because the ggml streaming decoder re-voices the left-context
tokens in every window, while the stateful model voices each token once.

Quality: offline ggml-vs-RKNN output SNR 27 dB, with the per-second dips confined to
near-silence (the fp16 noise floor), and no discontinuity at the group boundaries. The two
batch sizes agree to 57 dB.

### Piper, measured on the same board

`piper-en-us-lessac-medium-orig.gguf`, 443 characters at `--text-chunk-size 60` (10 chunks,
25.18 s of audio), seed 1234:

| path | RTF | peak RSS | decoder ms |
|---|---|---|---|
| ggml decoder | 0.198 | 299 MB | 2070 |
| RKNN decoder (93-slice window) | 0.0696 | 175 MB | 1030 |

2.85x wall, 2.0x on the decoder itself (0.95 → 0.46 ms per emitted slice), -42% RSS. The
decoder-side headroom is smaller than the projection from paroli's numbers because the window
carries 24+24 slices of context for 45 emitted slices, and this board's ggml VITS decoder is
already fast (RTF 0.20).

Quality: ggml-vs-RKNN output SNR 24.1 dB global, 40 dB median per 256-sample slice, no alignment
drift (best lag 0 samples in every window), and the max sample-to-sample step is unchanged
(13384 vs 13412), so no seam clicks. The dips are near-silence (LSB noise) and one loud plosive
onset, i.e. the fp16 noise floor, not the stitching.

### Kokoro, measured on the same board

`kokoro-82m-q8_0.gguf`, 185 Russian characters, seed 1234, `--threads 4` (14.85 s of audio,
1188 latent frames, 17 windows of 72 emitted frames). The Kokoro voice set has no Russian voice
(`frontend.cpp` accepts a, b, e, f, h, i, j, p, z; `--language russian` throws
`unsupported Kokoro language: russian`), so this utterance is Russian text voiced through
`--voice-id af_heart`, i. e. the built-in eSpeak-ng G2P applied to Cyrillic — a language transfer,
not a supported Russian path, and a reproduction has to use the same voice and text:

| path | RTF | peak RSS | generator ms |
|---|---|---|---|
| ggml generator | 1.415 | 2777 MB | 17748 |
| RKNN generator (128-frame window, clipped sine) | 0.867 | 2334 MB | 9609 |

1.63x wall, 1.85x on the generator itself (14.9 → 8.1 ms per latent frame processed), -16% RSS.
**The shipping default is the folded engine, and it is slower than this row**: on a 224-character
(33.6 s) Russian utterance, engines built from one producer revision and interleaved three times,
the folded engine measures RTF 1.074 / 28400 ms against the clipped engine's 0.851 / 20849 ms
(+36%, see the sine bullet above). Quote the clipped row as the clipped engine's figure, not as the
default's.
The generator-side gain is smaller than the 5.1x measured in the reference Kokoro engine because
this window carries 28+28 frames of context for 72 emitted frames (1.78x of compute spent on
context, and the 28 is the exact receptive field, so it cannot shrink), and because the window
width is capped at 136 frames by the register-field limit, which is what forces the 17 windows.

Quality: ggml-vs-RKNN output SNR 0.49 dB. That number is *not* a defect measurement and has to be
read against what it compares. The generator normalizes per window (AdaIN), so a multi-window
decode is a different valid synthesis and not an approximation of the full-length one: dropping 2
of 486 latent frames from the statistics moves the generator output by 25% RMS. What is measured
is exactness per window — a padded window reproduces the same frames decoded unpadded to 135.1 dB
— and the module reproduces the runtime's own generator to 60.5 dB interior (the floor is the
q8_0 style affine). The reference Kokoro engine, with the same per-window normalization at
`T_fix=50`, measures 17.11 dB against its own CPU decode on a 7.1 s utterance (2 windows); the
gap between its 17 dB and our 0.5 dB is window count, not stitching. Utterances of 72 latent
frames or fewer (about 0.9 s) fit one window and are exact.

### The talker is the only DRAM-bound candidate, and it is not at the ceiling

The sidecars and the code predictor read 0.08-0.72 GB/s of weights, 0.3-3% of the board's
26.4-29.4 GB/s ceiling, so shrinking their weights cannot pay. The talker decode is the one
component that streams its whole weight set per token: 196 Q8_0 matmuls plus `codec_head` =
471.3 MB per token, at 41.0 ms/token measured (36.5 ms/token documented) = **11.5-12.9 GB/s**,
i.e. 44-49% of the ceiling. Below it, and bytes convert to time sublinearly. Same conditions as
above (307 Russian characters, greedy `seed=42`, `--threads 4`, `conv_weight_type=f16`, vocoder on
the NPU), talker matmuls re-quantized with `audiocpp_gguf --type orig --keep-type <name>=q6_k`
(`q4_k_m` is not a supported output type in this converter, so `q4_k`):

| talker weights | MB/token | talker ms/token | RTF | peak RSS |
|---|---|---|---|---|
| Q8_0 (shipped) | 471.3 | 41.0 | 2.820 | 3144 |
| Q6_K | 364.6 | 41.6 | 2.740 | 3044 |
| Q4_K | 251.0 | **31.4** | **2.519** | 2934 |

Q4_K is −47% bytes for −23% talker time and −10.7% RTF; Q6_K is −23% bytes for **nothing** — the
dequant of a 6-bit K-quant eats the saving. No gate applies to these renders: a re-quantized talker
changes the greedy argmax trajectory, so each variant is a separate valid synthesis (inter-run
correlation 0.003/−0.012, per-sample SNR −2.5/−3.3 dB is trajectory divergence, not noise; the runs
are 286/300/272 tokens, so RTF compares audio-seconds across syntheses). Listening verdict: Q4_K
passes against the shipped render, Q6_K shows a slight split on the opening breath (not judged
critical) — which is the weaker result of the two, and Q6_K is also the one that buys no time. So
this is a measured lever, not a shipped default.

## NPU core use

`rknn.config` in 2.3.2 has no `core_num` (it has `single_core_mode`), so an engine is built
multi-core and the runtime core mask decides. `RKNN_NPU_CORE_AUTO` lets the driver pick, and it
picks one core for every one of these engines: `/sys/kernel/debug/rknpu/load` stays at
`Core0: 55%, Core1: 0%, Core2: 0%` for the whole `qwen3_tts` vocoder run and at `Core0: 100%` for
the smaller ones. All three sidecars therefore ask for `RKNN_NPU_CORE_0_1_2` explicitly and fall
back to `AUTO` if the driver rejects it (single-core SoCs do). What the mask buys is per engine:
`piper_tts` and `kokoro_tts` get real spread (kokoro Core0 95%, Core1 18%, Core2 19%), and the
`qwen3_tts` vocoder gets spread too (41/15/16) for 1.8% on the stage (21766 → 21383 ms for 300
Russian characters, same binary with both masks, greedy). The reason the extra cores buy almost
nothing is not memory bandwidth: measured on this board, the engines stream their weights at
0.08-0.10 GB/s (Kokoro 74.68 MB per window over 763 ms of window time, Qwen3 vocoder 259.69 MB
per 64-token step over 3090 ms), against a DRAM read ceiling of 26.4-29.4 GB/s measured with a
cache-bypassing STREAM triad on the A76 cluster (4 threads, 804 MB working set; `sysbench` with a
1 MB buffer reports a fake 61 GB/s because it never leaves L3). They sit at 0.3% of the ceiling,
so they are compute- and scheduler-bound, and the headroom is in op scheduling, weight reuse
across steps and tiling — not in shrinking the weights.

One context per sidecar, and one job per context: the mask spreads a *single* job over the cores
(kokoro measures Core0 95%, Core1 18%, Core2 19% during a run). Running the windows concurrently
on three `rknn_dup_context` contexts, which is what the reference Kokoro engine does, was built
and measured here at 1.15x on the window loop (9650 → 8375 ms) with corrupted output (-7 dB SNR,
rms 2x the reference): duplicated contexts share the runtime's tensor staging, and the gain is
small because the windows are compute-bound (0.098 GB/s of weight traffic against a 26.4-29.4 GB/s
ceiling, above), so three cores sharing one job beat three jobs on three cores.

The core mask is therefore also a numerics knob, and it is why every gate in this document is a
tolerance gate. The reverse-engineered RK3588 notes record (erratum E7) that the DPU's fp16
results differ per core on lane 0 of each 16-channel group, so which cores an op is scheduled on
can change the fp16 value it returns; the mask is not a pure scheduler decision. Consequence for
this path: "bit-identical" is a claim that holds for the CPU greedy paths (the export gates, the
gated-cast change on the f16 package), never for an NPU render. Every NPU number here is SNR and
max|d| against the CPU render, and the observed NPU-to-NPU spread is of the same order as the
NPU-to-CPU one (the two Kokoro sine engines differ by 28.2 dB from each other). Not characterized
here: the per-core fp16 difference itself, which would need the same engine run with a
single-core mask and the outputs compared.

## What this path is not for

The talker and the code predictor stay on the CPU. Moving the code predictor to the same NPU was
built and measured at 5.4x slower than the CPU path: the step is roughly 75 tiny S=1 GEMV
operations, and the NPU matrix units only pay off on real GEMMs like the vocoder's 64-token
chunk. The Mali GPU is 2.1x slower than the CPU for this workload.
