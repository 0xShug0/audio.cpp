# VibeASR in audio.cpp

[VibeASR.cpp](https://github.com/microsoft/VibeASR.cpp) is Microsoft's CPU-first
port of the VibeVoice ASR stack, quantized end to end for edge inference: the
audio VAE encoder runs on INT8 weights *and* INT8 activations, and the Qwen2
decoder runs on BitNet-style ternary weights. This entry ports both halves, so
`--family vibeasr` transcribes end to end on CPU.

## Relation to the existing `vibevoice_asr` family

audio.cpp already ships [VibeVoice ASR](../asr.md#vibevoice-asr) in the core model
tree, and it is the same model: the same acoustic/semantic causal ConvNeXt
tokenizers, the same connectors, the same Qwen2 decoder. That family runs F32 /
Q8_0 weights through the generic ggml ops.

What VibeASR.cpp adds is a different *numeric pipeline* for that architecture,
not a different architecture:

| | `vibevoice_asr` (core) | this entry |
|---|---|---|
| Encoder weights | F32 / Q8_0 | `GGML_TYPE_I8_S`, one F32 scale per tensor |
| Encoder activations | F32 | INT8 throughout; every stage requantizes |
| Ops | generic ggml | the five fused I8_S ops (`ggml_mul_mat_add`, `ggml_mul_mat_add_relu`, `ggml_add_scaled`, `ggml_rms_norm_scaled`, `ggml_im2col_asym`) |
| Decoder weights | Q8_0 Qwen2 | ternary `GGML_TYPE_I2_S`, 993 MB for a 1.5B decoder |
| Backends | CPU, CUDA, Metal | CPU only — the I8_S and I2_S kernels have no GPU variants |
| Decode | greedy, sampling, beam search | greedy |
| Output | text, segments, speaker turns | text |

Both are offline-only.

So this is an alternative execution path for weights that were quantized
upstream, useful where the INT8/ternary package is the point: no F32 activations
anywhere, integer dot products, and a decoder that fits in under 1 GB.

It stays a separate community entry rather than becoming a weight path inside
`vibevoice_asr`, because the two share no encoder graph code: every activation
there is I8_S and every node is one of the fused CPU-only ops, so folding it in
would put a second, mutually exclusive graph builder and a second backend policy
behind one family's loader. The reuse that is worth having — tokenizer
vocabulary, prompt layout, feature-injection order, audio normalization — is data
and conventions, and this entry follows `vibevoice_asr` on all of it. The decoder
half needs no new graph code at all: it is
`modules::QwenCausalDecoderModule` unchanged, because every projection goes
through `LinearModule`'s plain `ggml_mul_mat`, which dispatches on the weight
type.

## Architecture

### Encoder

Both branches are identical in shape and differ only in latent width:

- **Input**: mono 24 kHz waveform in `[-1, 1]`, quantized to a single I8_S
  tensor (one scale for the whole waveform, `amax` floored at 1e-5 to match
  upstream).
- **7 stages**, strides `{1, 2, 2, 4, 5, 5, 8}` (upstream `encoder_ratios`
  `[8, 5, 5, 4, 2, 2]` reversed, with a stride-1 stem), so **3200 samples per
  frame** — 7.5 frames per second at 24 kHz. Channels `32 → 64 → 128 → 256 →
  512 → 1024 → 2048`, depths `3-3-3-3-3-3-8`.
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

### Decoder

A stock Qwen2 causal decoder, geometry read from the LM GGUF's KV block: 28
layers, hidden 1536, intermediate 8960, 12 heads over 2 KV heads, head_dim 128,
RMSNorm eps 1e-6, RoPE theta 1e6, context 65536. The checkpoint has no
`qwen2.attention.key_length`, so `head_dim` comes from
`qwen2.rope.dimension_count`, which for this model equals
`embedding_length / head_count`; the loader cross-checks
`head_dim * head_count == embedding_length` and validates the declared geometry
against `token_embd.weight`'s shape.

Weight types are mixed on purpose, exactly as published:

| Tensors | Type |
|---|---|
| `blk.N.{attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down}.weight` | `I2_S` (ternary) |
| `token_embd.weight` | Q6_K |
| `output.weight` | F16 |
| norms and `blk.N.attn_{q,k,v}.bias` | F32 |

`I2_S` packs `{-1, 0, +1}` as codes `{0, 1, 2}`, 128 values per 32-byte group,
over the whole flat tensor, with one F32 absmax scale after the payload. The
kernel asserts `ne00 % 128 == 0`; hidden 1536 and intermediate 8960 both satisfy
it, and the weight is always 2-D by the time `LinearModule` calls
`ggml_mul_mat`.

### Prompt

Qwen2.5 ChatML, assembled to match `VibeASR.cpp/utils/prompt_builder.h` token for
token:

```
<|im_start|>system\nYou are a helpful assistant that transcribes audio input into text output in JSON format.<|im_end|>\n
<|im_start|>user\n<|speech_start|><|speech_pad|>×N<|speech_end|>\nThis is a 3.50 seconds audio, please transcribe it.<|im_end|>\n
```

- The special tokens are inserted by numeric id (151643–151648), not through the
  tokenizer, because the GGUF vocabulary still carries Qwen2.5's original text
  for those slots while the embedding rows are the ones VibeVoice trained. Every
  text segment is tokenized with `parse_special = false`.
- There is deliberately **no generation prompt**: the model emits its own
  `<|im_start|>assistant\n` header, and the session strips that leading triple
  before decoding, as upstream does.
- `N` is the encoder frame count. Upstream builds `ceil(samples / 3200)` pads but
  prefills only `min(pads, frames)` of them, so emitting exactly `frames` pads
  produces the same sequence.
- The `<|speech_pad|>` rows are replaced in-graph by a `ggml_set_rows` over the
  embedding lookup, with the speech features being the **element-wise sum** of
  the acoustic and semantic connector outputs — both are 1536 wide, which is what
  makes the sum well-defined.
- `output_format=json` swaps the instruction for `please transcribe it with these
  keys: Start, End, Speaker, Content`; `context=...` switches to the
  `with extra info:` suffix variant.

Decoding is greedy, stopping at `<|im_end|>` or `<|endoftext|>`. Upstream's
default is temperature 0.7 / top-p 0.9 sampling with `--greedy` as an opt-in;
this port only implements the deterministic path, which is what parity is
measured against.

### Audio front end

Mixdown to mono, resample to 24 kHz, RMS-normalize to −25 dBFS with `eps = 1e-6`,
then divide by `max_abs` if it exceeded 1.0. This is audio.cpp's own
`vibevoice_asr` front end, not upstream's: VibeASR.cpp resamples with a naive
linear kernel and omits the clamp. For a clip already at 24 kHz the two agree;
for anything else the resampler differs and so do the encoder features (see
[Parity](#parity)).

## Usage

VibeASR.cpp already ships both halves quantized, so there is nothing to
re-quantize. The two forks only disagree on the numeric type *ids* — the VibeASR
fork put I2_S/I8_S at 36/37, which upstream ggml had already spent on the retired
`IQ4_NL_4_4` / `IQ4_NL_4_8` slots, so audio.cpp registers them at 43/42. The
converter rewrites the 4-byte type field in each tensor info and copies
everything else through byte for byte:

```bash
# inspect first
python3 tools/community_models/convert_vibeasr_gguf.py \
    --input vibeasr-vae-encoder-i8_s.gguf --list

# fix both GGUFs in place (703 MB encoder, 993 MB decoder)
python3 tools/community_models/convert_vibeasr_gguf.py \
    --input models/vibeasr/vibeasr-vae-encoder-i8_s.gguf --in-place
python3 tools/community_models/convert_vibeasr_gguf.py \
    --input models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf --in-place

# confirm an already-converted package needs no further remapping
python3 tools/community_models/convert_vibeasr_gguf.py \
    --input models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf --check
```

Use `--output <path>` instead of `--in-place` to keep the original.

The package is two GGUFs plus the tokenizer, so `--model` points at the LM GGUF
and the spec is resolved from the repo — the same invocation shape as
[`minimax_h3`](minimax_h3.md):

```
models/vibeasr/
├── vibeasr-vae-encoder-i8_s.gguf
├── vibeasr-lm-i2_s-embed-q6_k.gguf
├── tokenizer.json
└── tokenizer_config.json
```

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target audiocpp_cli

./build/bin/audiocpp_cli \
    --task asr \
    --family vibeasr \
    --model models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf \
    --model-spec-override model_specs \
    --backend cpu \
    --threads 8 \
    --audio assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0000.wav \
    --metrics
```

```
text_output=Concord returned to its place amidst the tents.
metrics.wall_ms=1284.65
metrics.rtf=0.36652
```

Request options: `output_format` (`text` | `json`), `context` (a string folded
into the prompt to bias recognition), `max_new_tokens` (default 1024). Session
options: `vibeasr.encoder_graph_arena_mb` (64),
`vibeasr.prefill_graph_arena_mb` (256), `vibeasr.decode_graph_arena_mb` (256).

Note that `output_format=json` returns an empty transcript on short
single-speaker clips — the model emits an immediate end-of-turn. VibeASR.cpp
behaves identically on the same input; this port does not paper over it.

## Tests

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DENGINE_BUILD_MODEL_TESTS=ON
cmake --build build -j --target test_vibeasr_asr test_vibeasr_vae_encoder

# end to end: loader, session, prompt, both graphs, greedy decode
./build/bin/test_vibeasr_asr --threads 8

# encoder only: shape, finiteness, frame count, and optional upstream parity
./build/bin/test_vibeasr_vae_encoder \
    --model models/vibeasr/vibeasr-vae-encoder-i8_s.gguf \
    --audio assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0000.wav \
    --threads 8
```

Both exit 125 (SKIP) when the checkpoint is missing, so they are safe in ctest.
`i2_s_mul_mat_test` and `i8_s_fused_ops_test` cover the kernels themselves
against plain-loop references and need no checkpoint.

## Parity

### End to end

Four LibriSpeech clips, greedy on both sides, against VibeASR.cpp's own
`asr_infer --greedy` on the same two GGUFs:

| Clip | VibeASR.cpp | this port |
|---|---|---|
| test-clean 6930-75918-0000 | `Concord returned to its place amidst the tents.` | identical |
| test-clean 6930-75918-0001 | `The english forwarded to the french baskets of flowers, of which they had made a plentiful provision to greet the arrival of the young princess. The french, in return, invited the english to a supper, which was to be given the next day.` | identical |
| test-other 7902-96591-0001 | `Don't cry, he said. I was obliged to come.` | identical |
| test-other 7902-96591-0000 | `I'm from the cut or lying off the coast.` | `I'm from the cutter lying off the coast.` |

Three of four match token for token. The fourth diverges because these clips are
16 kHz and the two resamplers differ — this port uses soxr, upstream uses naive
linear interpolation — which perturbs the encoder features enough to flip one
greedy argmax. (Reference text: `I AM FROM THE CUTTER LYING OFF THE COAST`.) A
clip already at 24 kHz skips resampling entirely and does not have this failure
mode.

### Encoder

The reference dump is raw F32, `frames * dim`, row-major, produced by calling
`vae_encode_acoustic` / `vae_encode_semantic` from VibeASR.cpp's own `vae.h` on
the same WAV:

```bash
./build/bin/test_vibeasr_vae_encoder \
    --model models/vibeasr/vibeasr-vae-encoder-i8_s.gguf \
    --audio assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0000.wav \
    --reference-acoustic ref_acoustic.f32 \
    --reference-semantic ref_semantic.f32 \
    --threads 8
```

3.505 s LibriSpeech clip fed at its native 16 kHz, 17 frames × 1536 per branch:

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

`i8_s_fused_ops_test` covers the op arithmetic itself against plain-loop
references, including the in-band scale surviving `ggml_cont(ggml_permute(...))`
— the encoder flips activations between channel-major and length-major
constantly, and a byte copy that drops the scale leaves the values right and
everything downstream off by an arbitrary factor.

## Measured performance

Release build, CPU backend, 24 vCPU AMD EPYC 7V13, 3.505 s clip resampled to
24 kHz (26 speech frames, 72-token prompt, 13 generated tokens):

| Threads | encoder (both branches) | prefill | decode | wall | RTF |
|---|---|---|---|---|---|
| 8 | 758 ms | 214 ms | 239 ms | 1285 ms | 0.367 |
| 1 | 4652 ms | 1415 ms | 941 ms | 7081 ms | 2.020 |

The encoder dominates: it is run twice, once per branch, and it processes raw
samples rather than tokens. Decode is about 18 ms/token at 8 threads.

The encoder-only probe reports 546 ms for both branches at 8 threads because it
feeds the clip at its native 16 kHz (17 frames); the session resamples to 24 kHz
first (26 frames).

Peak RSS is 2.20 GB against 1.70 GB of weights: `BackendWeightStore` stages each
tensor before upload, so weight loading briefly holds roughly two copies of the
tensor being uploaded. Graph arenas are 64 MB (encoder) + 256 MB (prefill) +
256 MB (decode) by default.

## Status

Ported:

- I8_S VAE encoder graph, both branches, CPU backend.
- Ternary I2_S matmul kernel and the Qwen2 decoder graph on top of it, with
  prefill + static-cache single-step decode.
- Prompt assembly, speech-feature injection, greedy decode, tokenizer, loader,
  session, and `--family vibeasr`.
- GGUF type remapping tool and geometry-from-tensors asset loader.
- End-to-end and encoder parity probes against upstream, plus op-level unit
  tests.

Known limitations:

- **CPU only.** The fused I8_S ops and the I2_S matmul have no CUDA or Metal
  kernels; the session pins the backend to CPU.
- **Offline only**, like `vibevoice_asr` itself. Upstream's encoder is causal, so
  streaming is implementable, but the state machine is not ported.
- **Greedy only.** Upstream's sampling path (temperature, top-p) is not ported,
  and neither is `vibevoice_asr`'s beam search.
- **Text only.** No `--segments-out` / `--turns-out` equivalent; `output_format=json`
  is a prompt variant, not structured decoding.
- The package is two GGUFs, so it needs `--model <lm gguf>` plus
  `--model-spec-override model_specs` rather than a directory path.
- Bit-exact parity with upstream is out of reach by design; see
  [Parity](#parity).

## Upstream

- Model port: <https://github.com/microsoft/VibeASR.cpp> (`src/vae.cpp`,
  `src/lm.cpp`, `src/asr_server.cpp`, `utils/prompt_builder.h`)
- Base model: VibeVoice ASR, also in tree as [`vibevoice_asr`](../asr.md#vibevoice-asr)
- Weights: <https://huggingface.co/microsoft/VibeVoice-ASR-BitNet>
