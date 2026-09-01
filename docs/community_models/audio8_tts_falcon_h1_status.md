# Audio8 TTS 0.1B (Falcon-H1) — Port Status & Open Issues

> **Status 2026-09-01:** The Falcon-H1 slow-AR path is now a **stateful native
> implementation** (Mamba2 + hybrid GQA attention), replacing the previous
> stub. It builds and produces audio, and the prefill logits are in the right
> numeric range, **but the 0.1B path is NOT yet correct for synthesis**.
> Two open issues remain (see below). The 0.6B Qwen path is unaffected and
> fully functional.

## What has been done

Branch `feat/audio8-tts-falcon-h1-mamba2`, commits `af7bcd4` and `27ae3f4`.

- `src/community_models/audio8_tts/ar.cpp`
  - `FalconH1StepState` + `init_falcon_step_state`: per-layer conv/SSM states
    and attention KV cache.
  - `falcon_forward_step`: stateful single-token forward —
    `RMSNorm -> (Mamba2 || GQA attention) -> residual -> gated FFN -> RMSNorm
    -> semantic_output`, matching `transformers.models.falcon_h1`.
  - `build_falcon_embedding_step`: `(text_emb + codebook_sum) *
    embedding_multiplier` (multiplier applies to the whole sum).
  - `generate()` falcon branch: token-by-token prefill + generation.
- `external/ggml/src/ggml-metal/ggml-metal.metal`
  - Fixed `kernel_ssm_scan_f32` reduction: the old
    `simd_sum(shared_sums[sgitg*NW + tiisg])` read garbage columns when
    `sgptg < NW` (happens for `d_state=64` with `n_t=1`). Replaced with an
    explicit loop summing `shared_sums[(i2+sgitg)*NW + g]` over `g < sgptg`.

## Open Issue 1 — logits argmax does not match the transformers reference

**Symptom:** for the text "你好" the first generated semantic code is wrong:
synthesis says a different word (e.g. ASR round-trip gives "三星" instead of
"你好").

**Evidence:**
- transformers reference (local safetensors, `AutoModel` +
  `trust_remote_code=True`, `transformers==4.57.x`):
  first-frame `argmax code = 2732`, `eos logit ≈ 15.6`.
- audio.cpp: first-frame `argmax code = 3620` (with the lm_head fix), so the
  ranking differs even though the logit *scale* is comparable.

**Suspected cause:** a remaining per-layer math/layout mismatch in the slow
forward. The scale is right (lm_head_multiplier is correctly **not** applied
to the compact `semantic_output` head), but the relative logits still differ,
so some tensor is still transposed/ordered differently from HF.

**Debugging approach (recommended):**
1. Run the reference on the local snapshot
   (`~/.local/opt/audio.cpp/models/Audio8-TTS-Preview-0.1B-hf`) and dump
   per-layer hidden states / the first-frame logits.
2. Dump the same per-layer tensors from `falcon_forward_step` and diff.
3. Pay special attention to: `in_proj` output split `[z, xBC, dt]`, the
   `ssm_conv` kernel direction, `B/C` view offsets, RoPE type (should be
   `GGML_ROPE_TYPE_NEOX`, base `1e11`), and the FFN gate/up/down order.

## Open Issue 2 — recurrent SSM state blows up / dies on long sequences

**Symptom:** runs fine for ~180 tokens, then logits go to zero (or NaN).
Observed per-layer state values at `pos=180`: layers 6/14/19/20 already reach
`1e15..1e18` (not yet NaN), then overflow at the next step.

**Root cause:** `ggml_ssm_scan` is a **token-by-token recurrent** scan
(`state = state*dA + B*x*dt` in float32). Several layers have a weak decay
`A = -exp(A_log)` (e.g. `A ≈ -0.3`), so `dA = exp(softplus(dt)*A)` is close
to 1 and the state accumulates without decay, overflowing in float32.
The HF reference avoids this because generation uses
`mamba2_chunk_scan` (parallel associative scan) or the `mamba_ssm` CUDA
kernel; the pure-Python recurrent fallback in `modeling_falcon_h1.py` has the
same math but is only used when the kernel is missing.

**Known mitigations / fix directions:**
1. Implement a chunked parallel scan (Mamba-2 style) instead of the
   token-by-token recurrent path, or
2. Add numerical guards: float64 state accumulation, or clamp
   `dt_softplus` to `[time_step_min, time_step_max]` (0.1B config has
   `0.001 / 0.1`, but HF's recurrent fallback does **not** apply this clamp —
   only `mamba2_chunk_scan` does).

## Notes for the next agent

- `ar.cpp` currently contains a lot of temporary debug logging
  (`fopen("/tmp/falcon_dbg.txt", "a")`, `fprintf`, `dbg_ts`). These must be
  cleaned up before merging. The core fixes are: per-layer `scan_ts[li]` /
  `sx_ts[li]`, `ggml_set_input` on every graph constant, `ggml_set_output`
  on the `ssm_scan` result (keeps the state tail alive for host read-back),
  F32 storage for `A_log/D/dt_bias/conv1d`, and the conv1d kernel flip.
- `A_log` / `D` / `dt_bias` / `conv1d.weight` must load with
  `assets::TensorStorageType::F32` (GGUF stores them quantized; `Native`
  keeps the quantized type and `ggml_backend_tensor_get` then reads out of
  bounds).
- The GGUF layout for `conv1d.weight` is `[d_conv, 1, conv_dim]`, NOT the HF
  `[conv_dim, 1, d_conv]`. Use `require_f32_tensor` (which returns the GGUF
  shape) and flip along `dims[0]`.
- Reference environment: `uv venv` + `uv pip install torch "transformers>=4.57,<5"`;
  `transformers>=5` renames `FalconHybridMambaAttentionDynamicCache` and
  breaks the 0.1B remote code.
