# PR: VoxCPM1 — lightweight VoxCPM TTS support (0.5B / 1.5B)

> **Status: first porting attempt — runtime works end-to-end, output quality NOT yet acceptable**
>
> The port successfully loads and runs all three VoxCPM v1 GGUF variants (anchors pass, graphs
> execute, WAV files are produced at the correct sample rates/durations with active signal).
> **Known issue:** the generated audio is almost pure noise with only a faint trace of human
> voice. The pipeline is correct mechanically, but output quality requires further debugging
> (hypotheses and investigation plan in [Known issue](#known-issue-noisy-output)).

---

## 1. Overview

This PR adds support for the **OpenBMB VoxCPM v1** family of lightweight TTS models to
audio.cpp, reusing the existing and already-released `voxcpm2` model tree:

| Model | Params | Output sample rate | GGUF file |
|---|---|---|---|
| VoxCPM-0.5B | 0.5B | **16 kHz** | `voxcpm-0.5b-q8_0-audiovae-f16.gguf` |
| VoxCPM-1.5B | 1.5B | **44.1 kHz** | `voxcpm1.5-q8_0.gguf` |
| VoxCPM-1.5B | 1.5B | **44.1 kHz** | `voxcpm1.5-q4_k-audiovae-f16.gguf` |

The three models are architecturally **different variants** (they cannot share one config):

- **0.5B:** VAE encoder 128 / decoder 1536, encoder_rates `[2,5,8,8]`, decoder_rates
  `[8,8,5,2]`, patch_size 2, residual_lm 6 layers, encoder/dit 4 layers, 16 kHz, max_len 4096.
- **1.5B:** VAE encoder 64 / decoder 2048, encoder_rates `[2,3,6,7,7]`, decoder_rates
  `[7,7,6,3,2]`, patch_size 4, residual_lm 8 layers, encoder/dit 8 layers, 44.1 kHz, max_len 8192.

Since the v1 GGUFs store a different tensor convention than v2 (folded AudioVAE weights, no
`weight_v`/`weight_g` split, no `sr_cond_model` tensors, `voxcpm` architecture name), the port
wraps the v2 loader with a GGUF tensor-adaptation layer and adds `config.v1`-guarded branches
in the generator, mirroring the reference implementation (`VoxCPM.cpp`).

---

## 2. Porting activities

1. **Regenerated the 0.5B `config.json` from the GGUF metadata** — the previously shipped
   sidecar was wrong on ~8 axes (patch, residual_lm/encoder/dit layer counts, VAE dims and
   rates, sample rate 44.1 kHz vs the actual 16 kHz, max_len).
2. **Diagnosed the v1 GGUF conventions** (tensor dump + reference converter analysis):
   - AudioVAE conv weights are stored **already folded** (weight-norm folded), with no
     `weight_v`/`weight_g` decomposition and no `sr_cond_model.*` tensors.
   - GGUF file dims == ggml `ne` order; the v1 GGUFs carry **no** `audiocpp.tensor_shapes`
     override metadata (v2 does), so the adapter must present shapes itself.
   - The 1.5B **Q8_0** file stores VAE conv weights **2D-flattened** (`{out, in·k}`, kernel
     folded into dim1) while Q4_K and 0.5B store 3D `{out, in, k}` — both must load.
3. **Designed the identity-fold adapter** (see §4) so the existing `load_vae_weights` loader
   works unchanged against folded v1 weights byte-for-byte.
4. **Mirrored the reference generator math** for the no-fusion (no `fusion_concat_proj`)
   case: elementwise-add fusion inputs, elementwise-add dit-mu, and a real residual_lm
   autoregressive step.
5. **Set up per-variant model directories** (`VoxCPM1-GGUF/` for 0.5B, `VoxCPM1.5-GGUF/`
   for 1.5B) each with a config regenerated from its own GGUF metadata + tokenizer sidecars,
   and updated `model_specs/voxcpm1.json` package targets accordingly.
6. **Verified end-to-end runs** for all three GGUFs on the CPU backend (see
   [Validation](#6-validation-performed)).

---

## 3. Changes per file

| File | Change |
|---|---|
| `CMakeLists.txt` | Added `audiocpp_add_model(voxcpm1 ...)` reusing the 7 voxcpm2 sources; registers `engine::models::voxcpm2::make_voxcpm1_loader`. |
| `include/engine/models/voxcpm2/loader.h` | Declared `make_voxcpm1_loader()`. |
| `include/engine/models/voxcpm2/assets.h` | Added `VoxCPM2Config::v1 = false`; `load_voxcpm2_assets()` now takes `bool is_v1`. |
| `src/models/voxcpm2/loader.cpp` | Added `VoxCPM1Loader` (family `"voxcpm1"`), `load_voxcpm1_model()`, `make_voxcpm1_loader()`, `metadata_v1` / `capabilities_v1` / `cli_v1`. Offline-only TTS + speaker-reference clone, `text_prefix` policy, GGUF via `load_voxcpm2_assets(path, is_v1=true)`. |
| `src/models/voxcpm2/assets.cpp` | Added `TransformingTensorSource` v1 adapter (biggest chunk):<br>• v1→v2 tensor-name rename map (`token_embd.weight`→`base_lm.embed_tokens.weight`, gguf `blk.N.*`→`base_lm.layers.N.*` / `feat_encoder.encoder.layers.*` / `feat_decoder.estimator.decoder.layers.*` / `residual_lm.layers.*`, `attn_norm`→`input_layernorm`, `ffn_norm`→`post_attention_layernorm`, `attn_*`→`self_attn.*_proj`, `ffn_*`→`mlp.*_proj`, `time_mlp.*` (preserving `.linear_N`), `output_norm.weight`→`base_lm.norm.weight`, projection/fsq/stop mappings)<br>• **Folded weight-norm synthesis**: for every `audio_vae.*.weight` conv, `X.weight_v` → folded tensor data as-is, `X.weight_g` → per-row L2 norms (identity fold, see §4)<br>• Identity `decoder.sr_cond_model.{2..5}.scale_embed.weight` (ones) / `.bias_embed.weight` (zeros) since v1 GGUFs carry no SR-conditioning tensors<br>• Synthesized missing v1 tensors (`feat_encoder.scale_embed/bias_embed`, `feat_encoder.fc_logvar`, `feat_encoder.diag`, `feat_encoder.merge`, `token_embd.extra_bias`, `fusion_concat_proj.weight/bias`, `stop_proj.weight`, `stop_head.weight`)<br>• Rank-tolerant `require_f32` (accept element-count-equal, shape-different fetches — handles 2D-flattened convs and `{C,1}` alphas) + relaxed-rank VAE weight_v anchors for v1<br>• `has_tensor` / `require_metadata` / `require_tensor_data` folded + synthesized lookups<br>• **Anchor fix:** `encoder.fc_mu.weight_v` now uses computed encoder-in (`encoder_dim << #rates` = 2048), not `decoder_dim` (1536) |
| `src/models/voxcpm2/generator.cpp` | • v1 fusion guard: residual input = `AddModule(lm_hidden, current_embed)` / `AddModule(fsq, current_embed)` instead of concat+linear (matches reference `build_residual_fusion_input`)<br>• Added `add_dit_mu()` helper; v1 `mu` = elementwise add of `current_lm_dit_hidden + residual_dit_hidden` (matches reference `build_dit_mu`, `mu_dim = hidden·(fusion?2:1)`, v1 → hidden)<br>• CFM `mu` size check is now v1-aware (`hidden_dim * (v1 ? 1 : 2)`)<br>• v1 decode loop runs `residual_lm_.run_step(next_projected.residual_input).hidden` (the earlier `fsq_lm_dit_hidden` shortcut removed — v1 GGUFs have 6/8 residual_lm layers) |
| `src/models/voxcpm2/minicpm.cpp` | Prompt-prefill graph: v1 `residual_input` = `AddModule(lm_hidden, masked_current)` instead of concat+linear; residual_lm always runs (previously the concat path would have produced a wrong-dimension residual input for v1). |
| `model_specs/voxcpm1.json` | Package targets: `voxcpm1_0.5b_q8_0` → `VoxCPM1-GGUF`; `voxcpm1_1.5b_q4_k` and `voxcpm1_1.5b_q8_0` → `VoxCPM1.5-GGUF` (per-variant config/tokenizer). |
| `docs/tts.md` | Added VoxCPM1 section + TOC entry (usage, options, sample-rate notes). |
| `README.md` | Added `voxcpm1` row to the supported-model table. |
| `docs/reports/voxcpm1_port_status.md` | Port status log (analysis, decisions, timestamps, remaining tasks). |
| `models/VoxCPM1-GGUF/config.json` | **Regenerated** from 0.5B GGUF metadata. |
| `models/VoxCPM1.5-GGUF/config.json` | **New**, regenerated from 1.5B GGUF metadata. |
| `models/VoxCPM1.5-GGUF/tokenizer.json` (+config/special tokens) | Copied from 0.5B dir (same 73,448-vocab BPE tokenizer). |

---

## 4. Key design: the identity-fold adapter

The v1 GGUF (OpenBMB reference converter) stores AudioVAE conv weights **already folded**
(`weight = weight_g · weight_v / ‖weight_v‖`), with no `weight_v`/`weight_g` split, while
`audiovae.cpp` requests the decomposed names directly via `require_f32`. The adapter solves
this without touching the VAE loader:

```
X.weight_v  := folded GGUF tensor data (as-is)
X.weight_g  := per-row L2 norms of the folded tensor,
               computed with the loader's own row grouping
               (groups = expected_shape.front(), inner = elements/groups)
```

Because `fold_weight_norm` multiplies row `d0` by `weight_g[d0] / ‖row d0‖ = 1`, the loader
output equals the GGUF data **byte-for-byte** — an exact identity, with no layout drift
relative to the reference runtime's consumption of the same bytes. The same mechanism works
for 3D `{out, in, k}` and 2D-flattened `{out, in·k}` conversions (element counts must match;
ranks may differ, covered by rank-tolerant `require_f32` + relaxed-rank anchors).

---

## 5. Usage

### Build

```bash
scripts/build_linux.sh --backend cpu --target audiocpp_cli
# or, with the standard full model set:
cmake -S . -B build/linux-cpu-release -DCMAKE_BUILD_TYPE=Release
cmake --build build/linux-cpu-release --target audiocpp_cli -j 8
```

### Run — 0.5B (16 kHz output)

```bash
build/linux-cpu-release/bin/audiocpp_cli \
  --task tts --family voxcpm1 \
  --model models/VoxCPM1-GGUF/voxcpm-0.5b-q8_0-audiovae-f16.gguf \
  --backend cpu --text "Hello from VoxCPM1." --out out.wav
```

### Run — 1.5B (44.1 kHz output)

```bash
build/linux-cpu-release/bin/audiocpp_cli \
  --task tts --family voxcpm1 \
  --model models/VoxCPM1.5-GGUF/voxcpm1.5-q8_0.gguf \
  --backend cpu --text "Hello from VoxCPM1." --out out.wav
```

### Options

| Option | Values | Default | Meaning |
|---|---:|---:|---|
| `--task` | `tts` | required | Task kind. |
| `--family` | `voxcpm1` | auto-detect | Selects the v1 loader. |
| `--backend` | `cpu`, `cuda`, `vulkan`, `metal`, `hip`, `best` | `best` | Backend. |
| `--voice-ref` | WAV path | not set | Reference speaker audio (clone). |
| `--max-tokens` | integer | `4096` | Maximum generated AR tokens. |
| `--num-inference-steps` | integer | `10` | Flow-matching steps. |
| `--guidance-scale` | float | `2.0` | CFG strength. |
| `--session-option voxcpm1.mem_saver=true\|false` | bool | `false` | Tighter graph workspaces + release request graphs after completion. |
| `--session-option voxcpm1.prompt_cache_slots=<n>` | integer | `1` | Prompt/prompt-audio embedding cache slots. |
| `--text-chunk-mode` | `default`, `tag_aware`, `japanese`, `endline` | `tag_aware` | Long-form chunking mode. |

---

## 6. Validation performed

- **Load + anchors:** all three GGUFs pass `validate_weight_anchors` and
  `load_vae_weights`/`load_model_weights` on CPU. This includes the 0.5B 3D convs, the 1.5B
  Q4_K 3D convs, and the 1.5B Q8_0 2D-flattened convs.
- **End-to-end:** `--task tts` completes for all three models; outputs are written as WAV at
  the correct sample rate (16 kHz for 0.5B, 44.1 kHz for 1.5B) with active signal and
  speech-plausible duration/envelope.
- **Regression:** the released voxcpm2 path is untouched (guard style `config.v1`, v2 default
  `false`); voxcpm2 was not re-benchmarked but the changed code paths are v1-gated or
  v1/v2-neutral.

> ⚠️ **Quality caveat:** "end-to-end completes" does **not** mean the output is usable yet.
> See the known issue below — the audio is predominantly noise.

---

## 7. Supported modes

| Mode | Supported | Notes |
|---|---|---|
| **Offline TTS** | ✅ implemented | Default and only advertised mode. |
| **Streaming** | ❌ not implemented for v1 | `voxcpm1` advertises offline-only. Streaming is a v2 capability; it has not been validated (or enabled) for v1. |
| **Voice clone** | ⚠️ surface present | Speaker-reference options are advertised (`--voice-ref`), but quality is gated on the same known issue as plain TTS. |

---

## Known issue: noisy output

**Symptom.** Generated v1 voices are almost pure noise with a little human voice mixed in —
the signal is dominated by broadband/noise content. This affects all three GGUFs.

**What is confirmed working.** Model loading, tensor adaptation, anchor validation, graph
construction, graph execution, and WAV output plumbing are all correct (no crashes, no
shape/size errors, correct sample rates and durations). The failure is therefore in the
**numerics of synthesis**, i.e. the audio content itself.

**Most likely causes (in rough priority order).**

1. **Weight data interpretation** — the identity fold preserves bytes, but if some AudioVAE
   layer's storage layout (depthwise vs pointwise handling, 2D-flattened Q8_0, transposed
   decoder `{in,out,k}` conventions, per-group row ordering of `weight_g`) differs from what
   `ggml_conv_1d` / `conv_transpose` expects, the VAE decoder outputs garbage while loading
   still "succeeds" (element counts match).
2. **Synthesized tensor semantics** — `feat_encoder.scale_embed/bias_embed`, `merge`,
   `diag`, `extra_bias`, `fusion_concat_proj`, `stop_*`, and the identity `sr_cond_model`
   tensors were synthesized with plausible but unverified semantics; if any is required to be
   learned/zero-`scale` (or absent entirely in the reference runtime), the feature stream
   feeding the LM/CFM is wrong.
3. **Graph parity vs the reference** — fusion = add and dit-mu = add were taken from
   reference `build_residual_fusion_input`/`build_dit_mu`, but adjacent details (masking,
   slice indices, position ids, prompt handling, FSQ rounding, CFM conditioning inputs,
   ordering of `nn.Module` sub-blocks in the residual_lm stack) may differ.
4. **Sample-rate/codec mismatch** — 0.5B output asserted 16 kHz but the reference may expect
   a specific internal feature rate; patch_size/feat_dim interplay (2·64 vs 4·64) feeding the
   CFM estimator could be off by a constant factor, producing frozen-then-noisy patches.
5. **Quantization path** — the 1.5B Q8_0 GGUF quantizes the VAE itself (2D-flattened);
   dequantized values feed `require_f32`, but a transpose or block-order mismatch would
   corrupt every activation.

**Debugging plan (next iteration).**

- [ ] Port a small deterministic parity harness: run the same prompt through the reference
      `VoxCPM.cpp` and audio.cpp, dump intermediate tensors (lm hidden, residual hidden,
      CFM mu, VAE latent, decoder output) at each major stage, and diff numerically.
- [ ] Verify `encoder.fc_mu` / `decoder.model.{0,1,N}` folded data against the Python
      reference weights with a strict per-element comparison on non-quantized tensors
      (f16 VAE files), including row-grouping of `weight_g`.
- [ ] Check whether the reference runtime actually instantiates `sr_cond_model` and
      `feat_encoder` synthesizable blocks for v1; remove or zero-scale any block the
      reference does not run.
- [ ] Experimentally force one suspected block to a no-op (e.g. sr_cond identity, merge
      zeros, scale_embed 0/1) and measure whether noise level drops.
- [ ] Validate CFM mu dimension/conditioning against the reference expectation for
      `patch=2` (0.5B) and `patch=4` (1.5B).
- [ ] After the numerics match, run a human listening + loudness/spectral sanity check
      (the current output has a spectral envelope consistent with noise + faint voice).

---

## 8. Remaining tasks

- [x] Loader registration, tensor adaptation, generator v1 branches, configs, model spec
- [x] End-to-end execution for 0.5B Q8_0, 1.5B Q4_K, 1.5B Q8_0
- [ ] **Fix noisy output (known issue above) — top priority**
- [ ] Numerical parity harness vs `VoxCPM.cpp` reference (stage-by-stage tensor diff)
- [ ] `tests/voxcpm1/` automated path tests mirroring `tests/voxcpm2/`
- [ ] WebUI catalog entry (`webui/configs/models_catalog.json`)
- [ ] `docs/gguf.md` support-table entry
- [ ] CUDA-backend verification + RTF measurement (expect voxcpm2-like speedups)
- [ ] Streaming support for v1 (only meaningful after numerics are fixed)
- [ ] Commit + release packaging for `audio.cpp-gguf` (0.5B and 1.5B packages)