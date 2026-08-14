# MiniMax-Music3

MiniMax-Music3 generates full songs (vocals plus arrangement, 44.1 kHz stereo, up to six
minutes) from a music description caption and lyrics. The port follows the diffusers
reference pipeline: a Qwen3-8B autoregressive stage emits one semantic code per 40 ms
frame with classifier-free guidance, a 4-layer RVQ depth decoder fills seven residual
codebooks per frame, the fused per-frame hidden states condition a 36-layer flow-matching
transformer over 200-frame windows, and a DAC-style Flow-VAE decoder renders the latents
to stereo audio.

Upstream weights: [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3).
Reference implementation: `MiniMaxMusic3ModularPipeline` in diffusers (0.40.0.dev0 or newer).

## Package Layout

The package directory must contain the files referenced by `model_specs/minimax_music3.json`:

```text
MiniMax-Music3-GGUF/
  lm_q8_0.gguf                  global Qwen3-8B, Q8_0, lm_head sliced to the sampleable rows
  lm_q4_k.gguf                  optional Q4_K variant of the global LM
  depth_decoder_f16.gguf
  dit_f16.gguf                  flow-matching transformer
  condition_encoder_f32.gguf
  vocoder_f16.gguf              Flow-VAE decoder, torch weight norm folded
  tokenizer/tokenizer.json
  tokenizer/tokenizer_config.json
```

The DiT and depth decoder are stored F16 rather than the checkpoint's BF16: the CUDA
BF16 matmul path is several times slower than F16 on Ampere, the weights fit F16's range
with a wide margin (max magnitude about 3.2), and component parity is equal or better.

`--model` takes the `lm_*.gguf` entry file; the runtime resolves the other component
files from its parent directory.

## Conversion

`scripts/minimax_music3/convert_gguf.py` converts the HF snapshot per component:

```bash
hf download MiniMaxAI/MiniMax-Music3 --local-dir models/MiniMax-Music3-hf \
  --exclude "qwen_7B/qwen_7B/*.safetensors"

python scripts/minimax_music3/convert_gguf.py --component lm \
  --snapshot models/MiniMax-Music3-hf \
  --output models/MiniMax-Music3-GGUF/lm_q8_0.gguf --type q8_0 \
  --override "lm_head_sliced.weight=bf16"
python scripts/minimax_music3/convert_gguf.py --component depth_decoder \
  --snapshot models/MiniMax-Music3-hf \
  --output models/MiniMax-Music3-GGUF/depth_decoder_f16.gguf --type f16 \
  --override "*norm*=f32" --override "pos_embedding*=f32"
python scripts/minimax_music3/convert_gguf.py --component dit \
  --snapshot models/MiniMax-Music3-hf \
  --output models/MiniMax-Music3-GGUF/dit_f16.gguf --type f16 \
  --override "*norm*=f32" --override "*bias*=f32" --override "time_proj.weight=f32"
python scripts/minimax_music3/convert_gguf.py --component condition_encoder \
  --snapshot models/MiniMax-Music3-hf \
  --output models/MiniMax-Music3-GGUF/condition_encoder_f32.gguf --type f32
python scripts/minimax_music3/convert_gguf.py --component vocoder \
  --snapshot models/MiniMax-Music3-hf \
  --output models/MiniMax-Music3-GGUF/vocoder_f16.gguf --type f16 \
  --override "*.alpha=f32" --override "*.bias=f32"
cp models/MiniMax-Music3-hf/tokenizer/tokenizer.json \
   models/MiniMax-Music3-hf/tokenizer/tokenizer_config.json \
   models/MiniMax-Music3-GGUF/tokenizer/
```

The LM conversion slices the 200k-row lm_head to the 16385 rows the sampler can ever
pick (the audio end token plus the 16384 semantic codes); the vocoder conversion folds
torch `weight_g`/`weight_v` weight-norm pairs into plain conv weights. The `qwen_7B/`
safetensors in the upstream repo are an alternative packaging of the same LM and are
not needed. The optional Q4_K LM variant needs the `ggml-quantize-raw` build target
(gguf-py cannot produce K-quants):

```bash
cmake --build build/linux-cuda-release --target ggml-quantize-raw
python scripts/minimax_music3/convert_gguf.py --component lm \
  --snapshot models/MiniMax-Music3-hf \
  --output models/MiniMax-Music3-GGUF/lm_q4_k.gguf --type q4_k \
  --override "lm_head_sliced.weight=bf16" --override "model.embed_tokens.weight=q8_0"
```

Pass `--model .../lm_q4_k.gguf` to select it; the other components resolve from the
package directory either way.

## Run

```bash
build/linux-cuda-release/bin/audiocpp_cli \
  --task gen \
  --family minimax_music3 \
  --model models/MiniMax-Music3-GGUF/lm_q8_0.gguf \
  --backend cuda \
  --threads 8 \
  --text "$CAPTION" \
  --request-option lyrics="$LYRICS" \
  --request-option duration_seconds=60 \
  --seed 42 \
  --metrics \
  --out song.wav
```

`--text` carries the music description caption (genre, mood, vocals, instrumentation,
arrangement). `lyrics` carries the lyrics; structure tags such as `[verse]` or `[chorus]`
must each be on their own line. Options: `duration_seconds` (upper bound in seconds, the
model may stop earlier, maximum 360), `num_inference_steps` (flow Euler steps per window,
default 30), `guidance_scale` (flow CFG, default 1.7), `seed`. The autoregressive stage's
sampling recipe (CFG 1.5, top-50) is fixed by the checkpoint contract.

## Validation

Component parity against the diffusers reference (`tests/minimax_music3/reference_dump.py`
generates fixtures, `tests/minimax_music3/minimax_music3_component_probe.cpp` runs the
same component in isolation; build the probe with `-DENGINE_BUILD_WARMBENCH=ON`):

| Component | Result |
|---|---|
| Tokenizer (prompt template, caption cleaning, lyrics normalization) | exact id match |
| LM prefill (both CFG branches) | corr 0.9998 Q8_0 / 0.993 Q4_K, argmax match |
| RVQ depth decoder (argmax rollout) | all 8 codes exact, hidden max diff 2.1e-3 (f16) |
| Condition encoder | max diff 5.7e-5 |
| Flow transformer forward | corr 0.99998, max diff 5.2e-2 (f16 flash attention) |
| Vocoder | about 48 dB SNR (f16) |

## Performance

RTX 3090, CUDA, 32 s of audio at 30 flow steps:

| Configuration | AR | Flow | Vocode | Wall | RTF |
|---|---|---|---|---|---|
| Q8_0 LM, BF16 DiT (first pass) | 39.1 s | 84.5 s | 2.0 s | 130.6 s | 4.08 |
| Q8_0 LM, F16 DiT, flash attention | 39.1 s | 36.9 s | 1.9 s | 82.1 s | 2.57 |
| + batched CFG decode, on-device depth sampling | 23.9 s | 37.0 s | 2.0 s | 66.6 s | 2.08 |
| same with Q4_K LM | 22.0 s | 37.3 s | 2.0 s | 64.5 s | 2.02 |

The flow transformer runs flash attention with F16 weights and activations (norms and
residuals stay F32); the BF16-to-F16 storage switch alone is a 3.9x DiT forward speedup
on Ampere. The autoregressive stage batches the conditional and unconditional CFG
branches into one decode graph (weights stream once per frame; the two sequences always
share positions, so one KV write slot and mask serve both), and the seven depth-decoder
codebook steps run as a single unrolled graph with on-device sampling: classifier-free
guidance, a top-k mask built from `ggml_top_k`, host-supplied Gumbel noise, and an
argmax, which draws exactly from the reference's renormalized top-k distribution. Both
stages now sit close to their weight-bandwidth floors (`minimax_music3.ar_lm_decode_ms`
and `minimax_music3.ar_depth_ms` timing logs give the split). Remaining headroom is
architectural: pipelining flow-matching windows onto a second GPU while the
autoregressive stage streams frames, and bucketed KV-cache views for very long songs.

VRAM peaks around 14 GB during the autoregressive phase (Q8_0 LM, two KV states, depth
decoder) and around 8 GB during the flow phase; `mem_saver` (default on) loads each
phase's weights on demand and frees them afterwards.
