# Audio8 TTS 0.1B (Falcon-H1) — Port Status

> **Status 2026-09-01 (resolved):** The Falcon-H1 slow-AR path is a **stateful
> native implementation** (Mamba2 + hybrid GQA attention) and now matches the
> transformers reference: first-frame semantic argmax = 2732, per-step argmax
> parity over the whole prompt, ASR round-trip of synthesized "你好" returns
> "你好。", and long generation (~600 positions) is numerically stable. The
> 0.6B Qwen path is unaffected and fully functional.

## What has been done

Branch `feat/audio8-tts-falcon-h1-mamba2`.

- `src/community_models/audio8_tts/ar.cpp`
  - `FalconH1StepState` + `init_falcon_step_state`: per-layer conv/SSM states
    and attention KV cache.
  - `falcon_forward_step`: stateful single-token forward —
    `RMSNorm -> (Mamba2 || GQA attention) -> residual -> gated FFN -> RMSNorm
    -> semantic_output`, matching `transformers.models.falcon_h1`.
  - `build_falcon_embedding_step`: `(text_emb + codebook_sum) *
    embedding_multiplier` (multiplier applies to the whole sum).
  - `generate()` falcon branch: token-by-token prefill + generation.
- `src/community_models/audio8_tts/falcon_kv_cache.h`
  - `append_falcon_kv_token`: host KV-cache append with per-head re-stride
    (see "Root causes" below). Covered by `audio8_tts_falcon_kv_cache_test`.
- `external/ggml/src/ggml-metal/ggml-metal.metal`
  - Fixed `kernel_ssm_scan_f32` reduction: the old
    `simd_sum(shared_sums[sgitg*NW + tiisg])` read garbage columns when
    `sgptg < NW` (happens for `d_state=64` with `n_t=1`). Replaced with an
    explicit loop summing `shared_sums[(i2+sgitg)*NW + g]` over `g < sgptg`.

## Resolved Issue 1 — logits argmax mismatch vs transformers

**Symptom (before):** first generated semantic code was wrong (argmax 3620
instead of 2732; ASR round-trip said "三星" instead of "你好").

**Root causes (three, all fixed):**

1. **conv1d kernel flip was wrong.** `ggml_ssm_conv` computes
   `y[c] = sum_k w[k,c]*window[k,c]` with `window[0]` the OLDEST frame —
   the same orientation as HF (`nn.Conv1d` prefill and the cached
   `torch.sum(conv_states * w, dim=-1)` decode are both cross-correlation).
   The GGUF tensor `[d_conv,1,conv_dim]` is the HF `[conv_dim,1,d_conv]`
   weight with unchanged flat bytes, i.e. already in the layout ssm_conv
   wants. An earlier "fix" that flipped the kernel taps corrupted the x/B/C
   split every step. Fix: feed the kernel unflipped (`load_falcon_layer`,
   `conv1d_kernel`).
2. **Unprotected host read-back of intermediate tensors.** `sx` (conv
   window) and `k_r`/`v` (fresh K/V) are graph intermediates whose buffers
   gallocr reuses; reading them back without `ggml_set_output` returned
   garbage and corrupted conv state / KV cache every step. Fix:
   `ggml_set_output` on exactly those three tensors per layer (pinning ~300
   tensors corrupts the whole graph — pin only what is read back).
3. **KV cache head-stride bug (the decisive one).** The host cache used the
   *current* sequence length as the per-head stride while appending only the
   new token: at step 1 the new head-0 token was written over token 0's
   head-1 block, so every head past the first read corrupted context from
   the second token on (head 0 was always correct, which masked the bug).
   Fix: `append_falcon_kv_token` re-lays existing entries into the new
   stride before appending. Regression test:
   `tests/unittests/test_audio8_tts_falcon_kv_cache.cpp` (fails with the old
   algorithm at the second append, passes after).

**Verification:** bf16 GGUF vs f32 HF reference (`transformers==4.57.6`,
recurrent path forced for every token): per-layer conv/SSM/K/V states match
within bf16 rounding over the full 23-token prompt; argmax matches at every
prompt position except one knife-edge tie (ref top-2 margin 0.03 vs bf16
logit noise 0.19). First frame: argmax 2732 (logit 26.524 vs ref 26.557).
End-to-end: synthesized "你好" transcribes back as "你好。" (Qwen3-ASR), on
both CPU and Metal backends.

## Resolved Issue 2 — recurrent SSM state blow-up on long sequences

**Symptom (before):** ~180 tokens in, per-layer states reached 1e15..1e18,
then logits went to zero / NaN.

**Root cause:** not the recurrent scan itself — the corrupted KV cache
(Issue 1, cause 3) fed garbage attention output into the residual stream,
which drove `x`/`dt` of the Mamba2 branch into regime where the state
exploded. The f32 HF reference running the same recurrent math stays bounded
(states ~270 over the prompt), which ruled out the "inherent weak-decay"
theory previously recorded here.

**Verification:** 140-character text → 525 generated frames (position 605):
max per-layer SSM state ≈ 1.1e3, zero NaN, clean EOS, and the audio
transcribes back to the input text verbatim. No chunked-scan or dt-clamp
mitigation was needed; HF's recurrent fallback does not apply the
`time_step_min/max` clamp either (`time_step_limit` is hardcoded
`(0.0, inf)` in `modeling_falcon_h1.py`).

## Notes for the next agent

- The parity harness used for the fix (an HF golden-dump script forcing the
  recurrent path token-by-token, plus a differ) was session tooling and is not
  committed. To rebuild it: run `modeling_arktts` under `transformers==4.57.6`
  with each `layer.mamba.forward` replaced by the `use_precomputed_states`
  recurrent branch, dump `cache.conv_states/ssm_states/key_cache` per step,
  and diff against `state.ssm_states/conv_states/k_cache/v_cache` read back in
  `falcon_forward_step` (same flat layouts: ssm `[s + 64d + 2048h]`, conv rows
  oldest→newest, kv `d + 64*(t + T*h)`).
- `A_log` / `D` / `dt_bias` / `conv1d.weight` must load with
  `assets::TensorStorageType::F32` (GGUF stores them quantized; `Native`
  keeps the quantized type and `ggml_backend_tensor_get` then reads out of
  bounds).
- The GGUF layout for `conv1d.weight` is `[d_conv, 1, conv_dim]`, which is
  the HF `[conv_dim, 1, d_conv]` weight with unchanged flat bytes. Feed it to
  `ggml_ssm_conv` **unflipped** (see Resolved Issue 1, cause 1).
- `ggml_set_output` on a graph intermediate pins its buffer so host
  read-back is safe — but mass-pinning hundreds of tensors corrupts the
  whole graph (all-zero logits). Pin only the tensors actually read back.
- Reference environment: `uv venv` + `uv pip install torch "transformers>=4.57,<5"`;
  `transformers>=5` renames `FalconHybridMambaAttentionDynamicCache` and
  breaks the 0.1B remote code.
- Known remaining gap: the fast-AR codebooks during generation mean the C++
  rollout cannot be compared token-by-token against a reference that feeds
  zero codebook rows; parity was established over the prompt + first frame.
