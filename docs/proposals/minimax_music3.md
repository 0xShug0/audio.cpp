# MiniMax-Music3 port design

Status: implemented; see docs/community_models/minimax_music3.md for the user-facing
documentation and validation results.

MiniMax-Music3 ([MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3))
generates full songs (vocals plus arrangement, up to six minutes, 44.1 kHz stereo) from a
music description caption and lyrics. The reference implementation is the diffusers
modular pipeline `MiniMaxMusic3ModularPipeline` (diffusers >= 0.40.0.dev0).

## Architecture summary

Five checkpoint components run in four stages:

1. **Tokenize.** Qwen2 BPE tokenizer (`tokenizer/` subfolder). The prompt is a fixed
   special-token template over the cleaned caption and normalized lyrics:
   `<|im_start|><|caption_start|>C<|caption_end|><|lyrics_start|>[start]\nL<|lyrics_end|><|im_end|><|audio_start|>`.
   Maximum 5000 prompt tokens. Classifier-free guidance uses a token-level pair: the
   unconditional prompt is the conditional one with every id except the first and the
   trailing two replaced by the audio-CFG token (id 151654).
2. **Autoregressive stage, 25 frames/s.** The conditional and unconditional sequences run
   as a batch of two through a Qwen3-8B causal LM (36 layers, hidden 4096, 32 query and
   8 KV heads, head dim 128, ffn 12288, vocab 200000, untied lm_head). Per frame the LM
   samples one semantic code out of 16384 (logits masked to the code range at offset
   151675 plus the end token 151670, CFG scale 1.5 restricted to the conditional top-50,
   then top-50 sampling). A 4-layer depth decoder (hidden 4096, 16 heads, ffn 6144,
   SwiGLU, RMSNorm, learned positions, causal, no RoPE) then autoregressively samples the
   seven residual codebooks (1024 entries each) with the same CFG and top-50 recipe. The
   frame feedback embedding is `embed(semantic) + sum(residual embeds)` scaled by
   `8^-0.5`. The stage's real output is not the codes but the per-frame hidden states:
   `concat(LM last hidden, 7 depth-step hiddens)`, shape `[frames, 8 * 4096]`.
3. **Flow matching over 200-frame windows** (100-frame hop). A small condition encoder
   (softmax-weighted mix of the 8 hidden slots, 3-tap Conv1d 4096 to 2048, nearest
   resample by 3.4453125) puts the window's hiddens on the Flow-VAE latent timeline
   (44100 / 512 latents per second, 689 latents per full window). A 36-layer, 2048-wide
   DiT (32 heads by 64, partial RoPE over the first 32 of 64 dims, LayerNorm, gated ff of
   inner size 8192, Fourier time embedding prepended as one token; the input is
   `concat(latent 128, zeros 128, condition 2048)` through a residual 1x1 conv) predicts
   flow velocity. The scheduler reduces to plain uniform Euler: `t_k = k / N`,
   `x += (1 / N) * v`, default N = 30, CFG scale 1.7 with all-zero conditioning as the
   unconditional branch. Window overlap is handled by re-injecting
   `(1 - (1 - 1e-6) t) * noise_prompt + t * previous_latent` over the first 172 latent
   frames before every step, and by carrying latent frames `[L-344, L-172)` to the next
   window.
4. **Vocoder.** DAC-style decoder: the 128-channel latent folds to two 64-channel streams
   (stereo), each runs in_proj, conv_in, four upsample blocks (strides 8, 8, 4, 2, snake
   activations, weight-norm convs, dilated residual units), and a tanh output conv.
   Total upsampling 512, so 44.1 kHz stereo. Waveform windows are stitched by dropping
   86 leading latent frames (times 512 samples) on every window after the first and 258
   trailing latent frames on every window before the last.

All AR-stage constants (special token ids, code offset, CFG scales, top-k, frame rate,
chunk sizes, overlap lengths) are checkpoint contract, fixed in the reference code rather
than configs. We keep them as named constants in the family code.

## Port plan

Family `minimax_music3` under `src/community_models/minimax_music3/` and
`include/engine/community_models/minimax_music3/`, registered with
`audiocpp_add_model`, spec-backed loader, task `gen` (`tasks: ["music"]`), offline mode,
CUDA-first (`runtime.tags: ["cuda", "gguf"]`).

### Package layout (multi-file, minimax_h3 idiom)

```text
MiniMax-Music3-<precision>-GGUF/
  lm_<p>.gguf                  global Qwen3-8B (Q4_K default, Q8_0 variant)
  depth_decoder_<p>.gguf       RVQ depth decoder
  dit_<p>.gguf                 flow-matching transformer
  condition_encoder_f16.gguf   condition encoder
  vocoder_f16.gguf             Flow-VAE decoder, weight norm folded
  tokenizer/tokenizer.json
  tokenizer/tokenizer_config.json
```

`model_specs/minimax_music3.json` (schema v1) maps these as `sources[].tensors` /
`files` entries with `roots.model = "."`; `--model` takes the `lm_*.gguf` entry file and
the parent directory is the package root, like minimax_h3's `dit.gguf` convention.
Multi-file packages do not embed a spec, so `default_contract_spec_path` in
`src/framework/model_spec/package.cpp` needs `minimax_music3` added to the same
workspace/builtin fallback as `minimax_h3`.

Converter: `scripts/minimax_music3/convert_gguf.py` (one script, `--component` selector),
reading the diffusers-format safetensors. Component notes:

- **lm**: lm_head is sliced to the 16385 rows that can ever be sampled (row 0 = end
  token 151670, rows 1..16384 = semantic codes at offset 151675) and stored bf16; the
  full embedding table stays (prompt tokens and code feedback need it). Norms f32.
- **depth_decoder**: fused q/k/v kept separate as in the checkpoint; bf16 or Q8_0.
- **dit**: bf16 default; `ff_in` is stored fused (gate and value in one matrix) and kept
  that way. Attention out projections and time embedding stay bf16 in quantized variants.
- **vocoder**: fold `weight_g`/`weight_v` pairs into plain conv weights at conversion
  (torch `weight_norm` dim 0 convention; ConvTranspose1d normalizes over dims 1, 2),
  store f16, following `scripts/minimax_h3/convert_fold_audio_vae_gguf.py`.

Shapes are derived from GGUF tensor metadata at load (minimax_h3 house style); the only
sidecars are the two tokenizer files.

### Runtime components and reuse

| Component | Implementation |
|---|---|
| Text tokenizer and prompt build | `tokenizers::LlamaBpeTokenizer` (`Qwen2` pre-type, `tokenizer.json` path), caption cleaning and lyrics normalization ported from the reference, special ids resolved with `find_token_id` |
| Global LM | shared `modules::QwenCausalDecodeRuntime` (`use_qk_norm`, untied head, `output_mode` logits plus hidden), two KV states for the conditional and unconditional sequences; `decode_embedding` carries the frame feedback |
| AR sampler | host-side: CFG on the 16385 sliced logits, conditional top-50 restriction, top-50 softmax multinomial with a seeded RNG (`engine::sampling` helpers) |
| Depth decoder | hand-rolled small graph modeled on `qwen3_tts` `CodePredictorGraph` (4 layers, seq <= 9, batch 2, learned positions, no RoPE, no qk-norm) |
| Condition encoder | tiny graph (weighted mix plus Conv1d k3) with host-side nearest resample, modeled on `ace_step`'s condition encoder runtime |
| Flow DiT | dedicated graph (LayerNorm blocks, partial RoPE including the time token at position 0, fused gated ff); driven per window by `modules::FlowSamplerRuntime` (cond/uncond branches, CFG 1.7, Euler) or, if the per-step overlap injection does not fit its hooks, a host loop over `engine::sampling::diffusion_math` (`cfg_guidance`, `euler_step_in_place`) |
| Vocoder | dedicated DAC-decoder graph (snake, ConvTranspose1d, dilated residual units); stereo via the two folded channel groups, `interleave_planar_channels`, output 44.1 kHz stereo `AudioBuffer` |
| Session | `RuntimeSessionBase` + `IOfflineVoiceTaskSession`, spec-backed loader, `mem_saver` default on |

### Memory plan (single 24 GB GPU)

Sequential phases with `mem_saver`: the AR phase holds the LM (Q4_K about 5 GB) plus the
depth decoder and two KV states (about 4 GB at 60 s); frame hiddens accumulate on the
host (32 KB times frames times 4 bytes, about 200 MB per minute). The flow phase frees
the LM and holds the DiT (bf16 9.7 GB) plus per-window activations. The vocoder phase is
negligible. Peak stays under 16 GB, so bf16 DiT plus Q4_K LM fits one RTX 3090.

### Request surface

- `--text`: the music description caption (required).
- `--request-option lyrics=...` or `lyrics_file=...`: the lyrics (required).
- `audio_duration` (default 60 s, max 360 s), `num_inference_steps` (default 30),
  `seed`, and the standard `gen` options.

### Validation

Parity seams against the diffusers reference (bf16, `readback_round_type` BF16):

1. token ids of the assembled conditional and unconditional prompts,
2. prefill last hidden state,
3. first-frame guided logits with a pinned code sequence,
4. `frame_hiddens` for a short forced-code rollout,
5. condition encoder output for a fixed window,
6. one DiT forward at fixed t and latents,
7. vocoder waveform for a fixed latent,
8. end-to-end generation listening check plus RTF and VRAM numbers for the
   community-models table.

Python dump scripts live in `tests/minimax_music3/` next to a warm bench, following the
house pattern.
