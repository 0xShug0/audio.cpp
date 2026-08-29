# Audio8 TTS 0.1B Falcon-H1 Native GGML Port — Implementation Plan

**Goal:** Replace `src/community_models/audio8_tts/ar.cpp:1098` Python delegate
(`/workspace/.torch_venv/bin/python /tmp/gen_01b_for_cpp.py` via `system()` + `/tmp/falcon_prompt_*.txt` → `/tmp/falcon_codes_*.bin` `[8+8+10*frames*4]`) and stub `falcon_forward_stateless:864` (`x*silu(gate)` + `scale(cur,0)` attn) with a clean native Falcon-H1 hybrid `ggml` implementation, reusing vendored `external/ggml` `ssm_conv/scan` (already present for `cpu/cuda/metal/vulkan/opencl/sycl/cann`).

**References**
- Golden HF: `/workspace/models/Audio8-TTS-Preview-0.1b/modeling_arktts.py:303` `FalconH1Model` / `FalconH1DecoderLayer` / `FalconH1Mixer`
- Llama reference: `../llama.cpp/src/models/falcon-h1.cpp:1` + `../llama.cpp/src/models/mamba-base.cpp:151` `build_mamba2_layer`
- Current stub: `audio.cpp/src/community_models/audio8_tts/ar.cpp:103` `FalconH1LayerWeights`, `:364 load_falcon_layer`, `:842 build_falcon_embeddings`, `:864 falcon_forward_stateless`, `:1091 generate` (`is_falcon` branch)
- GGML ops: `external/ggml/include/ggml.h:2512` `ggml_ssm_conv/scan`, `external/ggml/src/ggml.c:5768`, `external/ggml/src/ggml-cpu/ops.cpp:9564/9634`, `external/ggml/src/ggml-cuda/ssm-*.cu`, `external/ggml/src/ggml-metal/kernels/ssm.metal`
- STT gate: `ffmpeg -y -i out.wav -ar 16000 -ac 1 -c:a pcm_s16le /tmp/tmp16k.wav && curl -X POST http://192.168.1.2:11533/v1/audio/transcriptions -F file=@/tmp/tmp16k.wav -F model=sensevoice-small`
- Models: `models/Audio8-TTS-Preview-0.1B-GGUF/audio8-tts-preview-0.1b-q8_0.gguf` (812M), `models/Audio8-TTS-Preview-0.1b/model.safetensors` (`slow.embed_tokens 69633*512`, `in_proj 1688*512`)

---

## 1. Scope & Non-Goals

**In scope (0.1B only):**
- `hidden 512`, `d_inner 768 (=32*24)`, `d_state 64`, `d_conv 4`, `dt_rank/n_head 24`, `n_group 1`, `n_layer` from `Audio8TtsTextConfig` (20), `GQA 8/2` (`q 512/512, k/v 128/512`), `rope_theta 1e11`, `intermediate 768`, `vocab 4097` compact (`codebook 1024+EOS`) vs `69633` text, `embedding_multiplier 0.1088 / lm_head 0.0781` + `ssm/attn/mlp` multipliers from `assets.cpp:types.h`

**Out of scope:**
- Generic Falcon-H1 families (`36/66` layers `falcon-h1.cpp:17` mapping) — follow-up 1d
- `K>1` speculation (`cparams.n_rs_seq` rollback snapshots `mamba-base.cpp:170`) — init `K=1`
- `external/ggml` fork — vendored checkout already has `ssm_*` for all backends; bump only if `ssm_scan` bugfix needed

**Dirty hack removed:** hard-coded `/tmp` writes + `std::system` + `python` path + `full recompute` fallback loop `ar.cpp:1124`.

---

## 2. Gap — Why Stub Fails STT

| Aspect | Llama | Current `ar.cpp` | Fix |
|---|---|---|---|
| **Conv** | `concat(conv_state[3,896], transpose(xBC[896,seq]))` → `ggml_ssm_conv` → `+bias`→`silu` + ring `conv_states_all` | `view_2d gate 768 / xBC 896` no `transpose`, `bias_bcast=repeat` then `silu` but **no `ssm_conv`** | Call `ggml_ssm_conv` correctly |
| **SSM** | `x[32,24]/B[64,1]/C[64,1]/dt[24]` 4D → `ggml_ssm_scan(ssm_state,x,dt,A,B,C,ids,K)` → `y+D*x` → `swiglu(cont(z),y)` → grouped `ssm_norm` → `ssm_out` | `y_gated=x*silu(gate); ssm_out*y_gated` — no `B/C/dt/A/D/scan/state/swiglu/norm` | Full scan |
| **Attn** | `Q/K/V→rope→flash_attn→wo` parallel to Mamba `falcon-h1.cpp:139` | `scale(cur,0)` zero | Native `Q/K/V` + `rope 1e11` + `TransformerKVCache` |
| **State** | `llama-memory-recurrent` ring `conv[3,896,mem]` + `ssm[64,32,24,mem]` + KV + `K` | No `conv/ssm` state; fallback loop recomputes `seq` each step `O(N²)` → `还过没.` / identical `md5 7426eed2` | `RecurrentState` buffers |
| **Multipliers** | N/A | Only `ssm_out/lm_head` | Thread all `assets` multipliers |

---

## 3. Target Architecture

Keep `BackendWeightStore` + `Audio8TtsAssets` + `TransformerKVCache` scaffold. Add `FalconH1LayerModule::build` (raw `ggml_*` inside `ModuleBuildContext`, not via `QwenDecoder` abstraction) similar to `mamba-base.cpp:151` but using `audio.cpp` `ggml_context + gallocr` pattern (`ar.cpp:1283/1417`).

```
cur[512,seq] → input_layernorm → ┬→ Q/K/V proj → rope(1e11) → KV cache → attn_out[512,seq]
                                └→ zxBCdt[1688,seq]=ssm_in*cur → z[32,24]/xBC[896]/dt[24] → conv → silu → x/B/C → dt+=dt_b → ssm_scan(state)+D*x → swiglu(z,y) → ssm_norm? → ssm_out[512,seq]
                               add(attnOut, ssmOut) + cur → pre_ff_layernorm → ffn(gate/up/down+silu) → cur_next
```

State: `conv_state: [3,896,mem]` `ssm_state: [64,32,24,mem]` (ring `kv_head/mem_size` like `mamba-base.cpp:211`) + `KV cache [head_dim, n_kv, mem]` for attn. `PrefillGraph` writes `seq` tokens at once; `StepGraph` advances `1` token.

`ggml_ssm_*` auto-dispatches: `ggml-cpu/ops.cpp` always, `ggml-cuda/ssm-*.cu` fused `ssm_conv+bias+silu` (`ggml-cuda.cu:3983`), `ggml-metal/ssm.metal`, `ggml-vulkan`, etc. — no new kernels.

---

## 4. Milestones & Tasks

### M0 — Audit & Harness (0.5d, GATE: `logits@4 tok max|Δ|<1e-3` vs HF)

- [ ] Dump GGUF `python -c "import gguf; r=gguf.GGUFReader('models/...0.1b-q8_0.gguf'); [print(t.name,t.shape) for t in r.tensors]"` vs `safetensors` `slow.*` vs `falcon-h1.cpp:73`
- [ ] Write `scripts/compare_falcon_logits.py` (HF `AutoProcessor+Model bf16` `forward(embed * embedding_multiplier)` vs native `falcon_forward_stateless` on fixed 4-token prompt) — baseline currently fails `1e-3`
- [ ] Record STT baseline: `./build/.../bin/audiocpp_cli --model ...0.1b-q8_0.gguf --task tts --family audio8 ... --out /tmp/x.wav && ffmpeg ... && curl 11533` → expect `还过没.` before fix

### M1 — Config & Weights (0.5d)

- [ ] `include/.../types.h` + `src/.../assets.cpp` thread `embedding_multiplier, lm_head_multiplier, ssm_in/out, attention_in/out, key, ssm_D, mlp` already parsed — wire through `FalconH1LayerModule`
- [ ] `ar.cpp:103` add `FalconH1LayerWeights.ssm_norm` optional (`{768/1?}` grouped); `ar.cpp:419 load_falcon_layer` add `ssm_norm` load, normalize `ssm_in {512,1688}` transpose check (`meta.shape`), transform `ssm_A: A=-exp(A_log)` on load, keep `ssm_D {1,24}` broadcast. No `external/ggml` commit needed

### M2 — Hybrid Layer Module (1.0d, CORE)

- [ ] New helper `FalconH1LayerModule` in `ar.cpp` (or `src/.../falcon_h1.cpp` if >500 LOC): function `build_falcon_h1_layer(ggml_context*, ggml_cgraph*, cur, conv_state, ssm_state, kv_cache, layer, multipliers)`
- [ ] Fix splits: `zxBCdt[1688,seq]` → `z: view_4d 32*24` / `xBC 896` / `dt 24` (currently 2D), `conv_x = concat(conv_state, transpose(xBC))` → `ggml_ssm_conv` → `add(bias)` → `silu`, split `x[32,24]/B[64,1]/C[64,1]`
- [ ] `dt = add(cont(dt), dt_b)`, `y_packed = ggml_ssm_scan(ssm_state, x, dt, A, B, C, ids, 1)` wrap via `build_rs`-style ids (see `mamba-base.cpp:256`), `y = view_4d(y_packed) + D*x`, `y = swiglu(cont(z), y)`, optional `rms_norm` grouped `d_inner/n_group`, `cur_ssm = mul_mat(ssm_out, reshape_2d(y,768,seq)) * ssm_out_multiplier`
- [ ] Parallel attn: `Q= q_proj*cur (512)`, `K 128`, `V 128` → `rope 1e11` (`ggml_rope_ext`) → `flash_attn` via existing `TransformerKVCache` (reuse `runtime::TransformerKVCacheOptions allow_bf16` logic). Zero `wo_b` optional. `hybrid = add(attn_out * attn_out_mult, ssm_out)` (HF multipliers)
- [ ] `pre_ff_layernorm` → `ffn: gate/up→silu(gate)*up → down * mlp_mult`

### M3 — Prefill/Step State & Generate (1.0d, REMOVES `/tmp`)

- [ ] Clone `PrefillGraph:1283`/`StepGraph:1417` as `FalconPrefillGraph`/`FalconStepGraph` in `ar.cpp:1275`:
  - `state_ctx` ring `conv_states[(3*896)*n_layer*mem]` + `ssm_states[64*32*24*n_layer*mem]` + `KV` via `ggml_backend_alloc_ctx_tensors` (`:1422` pattern)
  - `prefill(seq)` builds full `seq` graph with `n_written=min(seq,1)` circular `conv_states` write (`mamba-base.cpp:211`)
  - `step(1)` updates `kv_head`/`ssm_state` ring + `ids` (simple `ids=[kv_head]` for `K=1`)
- [ ] `Audio8TtsARRuntime::Impl::generate:1091` delete `tmp_prompt/tmp_codes/system()` + fallback loop `:1124`; branch `is_falcon` now `falconPrefill(prompt.matrix)` → `step` loop with `sample_frame:1814` (keep `codebook 1024+EOS→4097 expand` `:1134`, `RAS window 10`). Keep `Qwen` branch untouched for `0.6B`
- [ ] Remove includes `<fstream>/<cstdio>/<cstdlib>` for `system()` and hard-coded `/workspace/.torch_venv/bin/python`, `/workspace/models/Audio8-TTS-Preview-0.1b`, `/tmp/gen_01b_for_cpp.py` paths

### M4 — Validation & Cleanup (0.5d, GATE: STT PASS)

- [ ] `cmake --build build/linux-cpu-release --target audiocpp_cli` (~30s) + `./bin/audiocpp_cli --model ...0.1b-q8_0.gguf --task tts --family audio8 --text "The quick brown fox..." --out out/t1.wav` etc. for 6 prompts (fox, `Artificial intelligence...`, `你好欢迎使用audio8...` + `ana/demo_01_man/demo_02_woman` clones). `ffmpeg 16k` → `curl 11533` must transcribe correctly (no `还过没.`).
- [ ] Bit-exact check `cpu` vs `cuda` (`cmake -DENGINE_ENABLE_CUDA=ON build/linux-cuda-release`) `logits` diff
- [ ] Keep Python `scripts/gen_01b_for_cpp.py` only as `AUDIO8_TTS_USE_PYTHON=1` opt-in for CI diff, not hard-coded. Remove `/tmp` hardcodes. Update `AGENTS.md` / this file
- [ ] Optional: `external/ggml` bump cherry-pick if `ssm_scan` `d_state64` SSD fix needed (not required for correctness)

**Effort:** 3.0–3.5d CPU STT-pass; `+0.5d` GPU enable. `K>1` speculation + generic `36/66` layers = `+1.5d` follow-up.

---

## 5. File Changes

- `src/community_models/audio8_tts/ar.cpp` (primary) — `FalconH1LayerWeights`, `load_falcon_layer`, new `FalconH1LayerModule` + `FalconPrefill/StepGraph` + `generate` is_falcon branch
- `include/engine/community_models/audio8_tts/types.h` + `src/community_models/audio8_tts/assets.cpp` — wiring multipliers (no API break)
- Optional `src/community_models/audio8_tts/falcon_h1.cpp/.h` split if `ar.cpp>2500` LOC
- No `external/ggml` fork; scripts `scripts/compare_falcon_logits.py` ( harness )

## 6. Risks & Mitigations

- `A_log→-exp` & `D` broadcast wrong → STT silence — verify via `M0` harness `max|Δ|`
- `ids/K` ring off-by-one → `ssm_scan` hang/crash — start `K=1`, simple `ids=[kv_head]`, test `seq=1,4,64`
- `rope 1e11` overflow — use `ggml_rope_ext` with `freq_base` from `Audio8TtsTextConfig.rope_base`
- `transpose` of `ssm_in` (`[out,in]` HF vs `{hidden,proj}`) — assert `meta.shape` on load
- Vulkan `ssm_scan` subgroup limit — CI `cpu` gate, GPU is best-effort

---

*Plan: 2026-08-29 · 0.1B Falcon-H1 only · `K=1` · reuse `external/ggml` ssm backends*
