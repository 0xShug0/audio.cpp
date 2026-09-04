# VibeASR VAE encoder in audio.cpp

[VibeASR.cpp](https://github.com/microsoft/VibeASR.cpp) is Microsoft's CPU-first
port of the VibeVoice ASR stack, quantized end to end for edge inference: the
audio VAE encoder runs on INT8 weights *and* INT8 activations, and the Qwen2
decoder runs on BitNet-style ternary weights. This page covers the first half of
that port — the VAE encoder — which is what this entry currently adds.

## Relation to the existing `vibevoice_asr` family

audio.cpp already ships [VibeVoice ASR](../asr.md#vibevoice-asr) in the core model
tree, and it is the same model: the same acoustic/semantic causal ConvNeXt
tokenizers, the same connectors, the same Qwen2 decoder. That family runs F32 /
Q8_0 weights through the generic ggml ops and supports streaming.

What VibeASR.cpp adds is a different *numeric pipeline* for that architecture,
not a different architecture:

| | `vibevoice_asr` (core) | this entry |
|---|---|---|
| Encoder weights | F32 / Q8_0 | `GGML_TYPE_I8_S`, one F32 scale per tensor |
| Encoder activations | F32 | INT8 throughout; every stage requantizes |
| Ops | generic ggml | the five fused I8_S ops (`ggml_mul_mat_add`, `ggml_mul_mat_add_relu`, `ggml_add_scaled`, `ggml_rms_norm_scaled`, `ggml_im2col_asym`) |
| Decoder | Q8_0 Qwen2 | ternary `GGML_TYPE_I2_S` (not ported yet) |
| Backends | CPU, CUDA, Metal | CPU only — the I8_S ops have no GPU kernels |
| Streaming | yes | no |

So this is an alternative execution path for weights that were quantized
upstream, useful where the INT8/ternary package is the point (no F32
activations anywhere, integer dot products, and a much smaller decoder once the
I2_S kernel lands).

It stays a separate community entry rather than becoming a weight path inside
`vibevoice_asr`, because the two share no graph code: every activation here is
I8_S and every node is one of the fused CPU-only ops, so folding it in would put
a second, mutually exclusive graph builder and a second backend policy behind one
family's loader. The reuse that is worth having — tokenizer vocabulary, prompt
layout, feature-injection order — is data and conventions, and this entry follows
`vibevoice_asr` on all of it.

## Architecture

Both branches are identical in shape and differ only in latent width:

- **Input**: mono 16 kHz waveform in `[-1, 1]`, quantized to a single I8_S
  tensor (one scale for the whole waveform, `amax` floored at 1e-5 to match
  upstream).
- **7 stages**, strides `{1, 2, 2, 4, 5, 5, 8}` (upstream `encoder_ratios`
  `[8, 5, 5, 4, 2, 2]` reversed, with a stride-1 stem), so **3200 samples per
  frame** — 5 frames per second. Channels `32 → 64 → 128 → 256 → 512 → 1024 →
  2048`, depths `3-3-3-3-3-3-8`.
- Each stage starts with a **strided causal conv** (left pad `K - stride`, right
  pad 0) and then runs its ConvNeXt-style blocks: RMSNorm → depthwise conv →
  layer scale → residual → RMSNorm → FC1 → ReLU → FC2 → layer scale → residual.
- **Latent head**: causal conv to `vae_dim` — 64 acoustic, 128 semantic.
- **Connector**: `FC1 → RMSNorm → FC2`, both 1536 wide, i.e. the decoder hidden
  size. Output is `[frames][1536]` for each branch.

Two details the port copies rather than corrects:

- RMSNorm epsilon is **1e-5 everywhere**, including the norms the checkpoint
  metadata labels 1e-6. Upstream hardcodes it and the published weights were
  validated that way.
- The converter left-pads the 7-tap depthwise kernels with leading zeros up to a
  SIMD-friendly width. Convolving with the padded width and a matching causal
  left pad is bit-exact with convolving the unpadded kernel, so the geometry is
  read back from the weight shapes rather than from metadata.

The encoder geometry is derived from the tensor table (which block tensors
exist, what shape each weight has), not from GGUF KV metadata — the same
approach upstream takes, and it keeps the loader working for any checkpoint with
this topology.

## Usage

VibeASR.cpp already ships the encoder quantized, so there is nothing to
re-quantize. The two forks only disagree on the numeric type *ids* — the VibeASR
fork put I2_S/I8_S at 36/37, which upstream ggml had already spent on the retired
`IQ4_NL_4_4` / `IQ4_NL_4_8` slots, so audio.cpp registers them at 42/43. The
converter rewrites the 4-byte type field in each tensor info and copies
everything else through byte for byte:

```bash
# inspect first
python3 tools/community_models/convert_vibeasr_vae.py \
    --input vibeasr-vae-encoder-i8_s.gguf --list

# produce the audio.cpp package (~703 MB, both branches)
python3 tools/community_models/convert_vibeasr_vae.py \
    --input vibeasr-vae-encoder-i8_s.gguf \
    --output models/vibeasr/vae_encoder-i8_s.gguf

# confirm an already-converted package needs no further remapping
python3 tools/community_models/convert_vibeasr_vae.py \
    --input models/vibeasr/vae_encoder-i8_s.gguf --check
```

There is no CLI family yet (see [Status](#status)); the encoder is reached
through `VibeASRVaeEncoderRuntime` or the parity probe:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DENGINE_BUILD_MODEL_TESTS=ON
cmake --build build -j --target test_vibeasr_vae_encoder

./build/bin/test_vibeasr_vae_encoder \
    --model models/vibeasr/vae_encoder-i8_s.gguf \
    --audio assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0000.wav \
    --threads 8
```

Without `--reference-*` the probe checks shape, finiteness, and frame count only.
It exits 125 (SKIP) when the model or audio is missing, so it is safe in ctest.

## Parity

The reference dump is raw F32, `frames * dim`, row-major, produced by calling
`vae_encode_acoustic` / `vae_encode_semantic` from VibeASR.cpp's own `vae.h` on
the same WAV:

```bash
./build/bin/test_vibeasr_vae_encoder \
    --model models/vibeasr/vae_encoder-i8_s.gguf \
    --audio assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0000.wav \
    --reference-acoustic ref_acoustic.f32 \
    --reference-semantic ref_semantic.f32 \
    --threads 8
```

3.505 s LibriSpeech clip, 17 frames × 1536 per branch:

| Branch | max abs | mean abs | cosine |
|---|---|---|---|
| acoustic | 1.478 (12.1% of range) | 0.0930 (0.76% of range) | 0.99238739 |
| semantic | 2.526 (9.5% of range) | 0.1804 (0.68% of range) | 0.98475210 |

**Layer by layer, stage 0 is bit-exact** — every int8 byte and every scale
matches, which is what pins the layouts, the causal padding, the kernel padding,
and the weight mapping. The first divergence is 5 of 1,794,560 elements one int8
step apart at an identical scale, entering stage 1, and it grows from there
because each of the remaining stages requantizes.

Bit-exactness is not reachable and the tolerances say so. audio.cpp stores each
per-tensor scale as a multiplier (`amax/127`, dequantize by multiplying) while
VibeASR.cpp stores its reciprocal (`127/amax`, dequantize by dividing) — the
same number to within the last float bit, which is enough to flip a value that
sits on a rounding boundary. Upstream also rounds ties to even in its vector
body but away from zero in its scalar tail, so no single convention reproduces it
exactly.

To calibrate what that is worth, nudging **one** input sample by one int8 step
and re-running VibeASR.cpp against *itself* moves its own output by cosine
0.99592 (acoustic) / 0.98700 (semantic) — the graph amplifies a single LSB about
as far as the two implementations differ from each other. The probe therefore
gates on mean-abs-relative ≤ 2% and cosine ≥ 0.98; anything tighter would be
testing rounding luck.

`i8_s_fused_ops_test` (under ctest) covers the op arithmetic itself against
plain-loop references, including the in-band scale surviving
`ggml_cont(ggml_permute(...))` — the encoder flips activations between
channel-major and length-major constantly, and a byte copy that drops the scale
leaves the values right and everything downstream off by an arbitrary factor.

## Measured performance

Release build, CPU backend, 24-core AMD EPYC 7V13, 3.505 s clip, both branches
(the encoder is run twice — once per branch — because that is what the decoder
consumes):

| Threads | acoustic | semantic | both | RTF |
|---|---|---|---|---|
| 8 | 276 ms | 270 ms | 546 ms | 0.156 |
| 1 | 1657 ms | 1600 ms | 3257 ms | 0.929 |

Peak RSS is 1.31 GB against 703 MB of weights: `BackendWeightStore` stages each
tensor before upload, so weight loading briefly holds roughly two copies. The
graph arena itself is 64 MB by default.

## Status

Ported:

- I8_S VAE encoder graph, both branches, CPU backend.
- GGUF type remapping tool and geometry-from-tensors asset loader.
- Parity probe against upstream, plus op-level unit tests.

Not ported yet:

- **The decoder.** VibeASR.cpp runs it on ternary `GGML_TYPE_I2_S` weights whose
  matmul kernel is not in tree yet, so there is no loader, no session, and no
  `--family vibeasr` — nothing can transcribe through this path today. The
  encoder output is the LM input, so the halves are independently reviewable but
  only useful together. Two follow-up PRs cover it: the I2_S matmul kernel, then
  the decoder graph plus loader and session.

Known limitations:

- **CPU only.** The fused I8_S ops have no CUDA or Metal kernels; the probe pins
  the backend to CPU.
- **Offline only.** No streaming; upstream's encoder is causal, so streaming is
  implementable, but the state machine is not ported.
- Bit-exact parity with upstream is out of reach by design; see
  [Parity](#parity).

## Upstream

- Model port: <https://github.com/microsoft/VibeASR.cpp> (`src/vae.cpp`)
- Base model: VibeVoice ASR, also in tree as [`vibevoice_asr`](../asr.md#vibevoice-asr)
