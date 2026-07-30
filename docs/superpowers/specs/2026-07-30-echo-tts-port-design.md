# Echo-TTS port to audio.cpp — design

Date: 2026-07-30
Status: approved, pre-implementation
Target: community-model PR to `0xShug0/audio.cpp`

---

## 1. Why this model

### 1.1 Benchmark provenance

This is not a model picked from a leaderboard screenshot. Echo-TTS has been independently
benchmarked in [tts-bench](https://github.com/5uck1ess/tts-bench) — a public benchmark tracking
**62 local TTS models** across three lenses (speed, objective scores, human preference) on three
rigs — and it was selected by comparing every tracked model against audio.cpp's existing support
table. The supporting data is already published and reproducible:

- **Installed and run locally.** `venvs/echo/` with upstream source; both weight sets cached
  (`jordand/echo-tts-base`, `jordand/fish-s1-dac-min`); a dedicated runner
  (`runners/echo_runner.py`) documenting the exact upstream API and its gotchas.
- **Speed benched** on RTX 3090 CUDA, warm: **1.35× RTFx**, 4 326 ms TTFA, 9 357 MB peak VRAM.
- **Objectively scored** over the bench prompt set: **16 rows** in `scoring/scores.csv` across
  default and cloning lenses, via seed-tts-eval-style ASR + speaker verification.
- **Publicly auditioned**: generated wavs published to gh-pages and playable in the Listen lens.
- **Voted on blind**, twice — a frozen 397-vote pairwise study and an ongoing public arena that has
  since collected 738 cloning votes and 1 415 default-voice votes.

That measurement history is what makes the recommendation trustworthy, and it should be cited in the
PR body: the port is proposed because Echo *measured* well against 61 alternatives, not because it
looked promising.

### 1.2 The result

Echo-TTS is the highest-value model absent from audio.cpp, on three independent signals:

| Signal | Value | Source |
|---|---|---|
| Human-preference Elo (cloning) | **1162, #3 of 40** on 35 games | tts-bench live arena, 738 cloning votes |
| Speaker similarity (SIM) | **0.836 — 2nd of 41** scored models | `tts-bench/scoring/scores.csv` |
| Frozen blind study | **21-1-6**, near-tied #1 | `tts-bench/docs/cloning.md`, 397 votes |
| UTMOS / WER | 4.21 / 7.45 % | same |
| Output rate | **44.1 kHz** | model card |

Two qualifications, stated up front for honesty: the cloning arena averages ~30 games per model, so
gaps under ~100 Elo points are noise (the 1 415-vote default lens is firmer), and the whole cloning
ranking rests on a single reference clip (`chris_hemsworth_15s.wav`). Echo's position is robust to
both — it is top-3 on votes *and* top-2 on objective SIM, which are independent measurements.

It is also **explicitly open for contribution**. Upstream issue #34 lists `~~echo-tts~~` struck
through under "Candidate models", with the legend: *"For models crossed out: I will not impl these
models myself, but contributions are welcome."* Struck-through entries carry **zero duplication
risk**; un-struck candidates (Magpie, LongCat, Soprano, MiraTTS) may still be maintainer work.

Verified absent: no `echo`/`echodit`/`jordand` match anywhere in `src/`, `include/`, `docs/`,
`model_specs/`, `tools/`, or `README.md`; no PR (open/closed/draft) in 200+; no branch; GitHub code
search returns 0.

Compute profile suits the framework. Echo is ~2.8 B at 1.35× RTFx and 9.4 GB VRAM in PyTorch —
heavy enough that GGUF and session amortisation pay off. (Contrast Kokoro, whose `preview/kokoro`
branch measures **0.20×** on the long-lived-session chart — 5× *slower* than Python — because an
82 M model has nothing to amortise.)

---

## 2. Verified architecture

All facts below were read from source at `tts-bench/venvs/echo/src/` and from safetensors headers.
Anything not established by those files is marked OPEN in §9 rather than guessed.

### 2.1 Pipeline

```
reference wav
  → decode ≤300 s → mono → resample 44 100 Hz → divide by max(|peak|, 1)
  → truncate ≤ 6400×2048 samples; chunk at 640×2048; zero-pad final chunk
  → fish_ae.encode_zq  → PCA project 1024→80 → × latent_scale
  → speaker_latent [1, Ls, 80],  speaker_mask [1, Ls],  Ls mod 4 == 0

text
  → WhisperD normalisation: prepend "[S1] "; colons/semicolons/emdashes → commas
  → UTF-8 *byte* tokens (256-entry vocab)
  → text_encoder

EchoDiT: 40 Euler steps in 80-D PCA space, latents [1, 640, 80]
  → PCA⁻¹ → quantizer.post_module → quantizer.upsample → decoder
  → waveform 44 100 Hz
  → crop at flattening point (20-frame std/mean scan, cut at frame × 2048)
```

`640 × 2048 / 44100 = 29.7215 s` — the fixed generation window.

### 2.2 EchoDiT

| Property | Value |
|---|---|
| Trunk depth | 24 blocks |
| Hidden dim | 2048 |
| Attention | joint: self + text KV + speaker KV (+ latent-prefix KV, blockwise only) |
| MLP | SwiGLU |
| Conditioning | adaLN on both attention and MLP, driven by timestep |
| Positional | RoPE, **rotating only half the heads** |
| Norm | RMSNorm, FP32 accumulation |

Text frontend is **byte-level** — no phonemizer, no G2P, no external pronunciation dependency.
This is a significant scope win and removes the class of dependency problem that sank Kokoro.

### 2.3 Parameter inventory

| Component | Params | Needed for inference |
|---|---:|---|
| EchoDiT total | 2 800 742 736 | yes |
| — trunk joint attention (24) | 880 902 144 | yes |
| — trunk MLP (24) | 868 220 928 | yes |
| — attention adaLN (24) | 75 644 928 | yes |
| — MLP adaLN (24) | 75 644 928 | yes |
| — text_encoder | 294 000 640 | yes |
| — speaker_encoder | 294 083 840 | yes (when cloning) |
| — **latent_encoder** | 294 083 840 | **blockwise/long-form only** |
| — misc (timestep MLP, projections, norms) | 18 161 488 | yes |
| PCA state | 82 945 elements | yes |
| Fish S1-DAC checkpoint | 694 993 282 elements | — |
| — **trainable weights only** | **391 430 530** | — |
| — `freqs_cis` + `causal_mask` buffers | 303 562 752 | **regenerate at runtime, do not ship** |

PCA: `pca_components [80,1024]`, `pca_mean [1024]`, `latent_scale [1] = 0.0555555559694767` (= 1/18).

### 2.4 The decode/encode asymmetry

Decode and encode need nearly disjoint Fish submodules:

| Path | Modules | Approx weights |
|---|---|---:|
| **Decode** (generation) | PCA⁻¹, `quantizer.post_module`, `quantizer.upsample`, `decoder` | ~184 M |
| **Encode** (speaker ref) | `encoder`, `quantizer.downsample`, `quantizer.pre_module`, semantic RVQ + 9× residual RVQ, PCA forward | ~207 M |

The decode path is entirely matmul/conv/transformer. The encode path needs RVQ nearest-neighbour
search, rated **Hard** to port. This asymmetry is the basis for the milestone split in §4.

Note: `encode_zq` as written runs the *full* quantizer forward, then discards the result and
re-derives from the selected codes. `post_module` and `upsample` inside that first call can be
skipped — numerically equivalent, since only `codes` are consumed.

### 2.5 Sampler

`sample_euler_cfg_independent_guidances`: 40 Euler steps, **dual independent CFG** — `cfg_scale_text`
3.0 and `cfg_scale_speaker` 8.0 (5.0 in the blockwise example) — gated to `t ∈ [cfg_min_t=0.5,
cfg_max_t=1.0]`, `truncation_factor` 0.8. Unconditioning is **mask-based**, not zeroed encoder
states. Optional `speaker_kv_scale` ("Force Speaker", default 1.5 when enabled) corrects speaker
drift on out-of-distribution text.

---

## 3. Long-form: rolling latent continuation

**Blockwise does not extend past 640.** Verified directly:

- `inference_blockwise.py:161` — `block_sizes=[128,128,64], # (sums to 320, ~15 seconds; supports up to 640)`
- `inference_blockwise.py:194-195` — `sum(block_sizes) + continuation_latent.shape[1] should be < 640`
- `README.md:122-124` — *"prefix and continuation are up to 30 seconds combined"*; *"Blockwise
  functionality hasn't been thoroughly tested"*

Blockwise **subdivides** one ≤30 s window; it does not extend it. Nor is there any text-compression
transform — long text fitting into 30 s is *learned* behaviour via global attention, and the
tokenizer hard-truncates past 768 UTF-8 bytes (`inference.py:146-149`).

**Design:** carry the tail latents of chunk N directly into chunk N+1 as the continuation prefix.
Because we generate latents natively, this needs **no decode→re-encode round trip**. Each call
resets the window; the constraint is `prefix + new < 640` per call.

This preserves prosody across joins, which crossfading cannot. Requirements and caveats:

- Requires `latent_encoder` (+294 M) plus `wk_latent`/`wv_latent` — exactly what
  `delete_blockwise_modules=True` strips.
- The prompt for chunk N+1 **must include the carried prefix's transcript**.
- Total prefix length must be divisible by 4 (speaker patch size, `model.py:458-459`).
- Upstream calls blockwise under-tested. **This must be disclosed in the PR, not discovered by the
  maintainer.**

Fallback if M3 fails validation: sentence-boundary chunking with `cross_fade_duration_sec` seams,
and drop the `long_form` capability claim.

---

## 4. Milestones

Each milestone has a gate. **No milestone is "done" on report — only on executed evidence.**

### M0 — spec + draft PR
- `model_specs/echo_tts.json`, `"schema_version": 1`, placed in `model_specs/` (not `model_specs_v1/`).
- `capabilities` **omits `long_form`** until M3 earns it.
- Draft PR opened, explicitly raising: the 29.72 s window, the blockwise-untested caveat, and the
  CC-BY-NC-SA output-licence constraint.
- Gate: spec validates against the framework schema; PR open and marked **draft**.

### M1 — decode path, parity-gated
- GGUF conversion script; EchoDiT minus `latent_encoder`; PCA⁻¹; Fish decode path.
- Speaker latent injected from a `.npy` dumped by PyTorch — validates the hard 2.5 B without RVQ.
- Gate: per-tensor cosine ≥ 0.999 vs reference on fixed seed; generated wav audibly correct.

### M2 — native speaker encoding
- Fish encoder + downsample + pre_module + semantic/residual RVQ + PCA forward.
- Gate: speaker latent from C++ matches PyTorch `encode_zq` → PCA output, cosine ≥ 0.999;
  end-to-end clone from a raw wav with no Python in the loop.

### M3 — long-form
- Rolling latent continuation per §3.
- Gate: `tools/audiocpp_cli/audiocpp_cli_longform_tts_clone_cases.json` renders correctly and is
  listened to end-to-end for seam artefacts. Only on success does `long_form` enter `capabilities`.

### M4 — quantisation, performance, docs
- Q8_0 and F16 GGUF; `docs/community_models/echo_tts.md`; warm-bench test.
- Gate: **RTF < 1.0** (the explicit community bar); VRAM stable across repeated requests.

---

## 5. Definition of Ready — the PR does not leave draft until all of these pass

This is a hard gate, mirroring audio.cpp's stated review bar (issue #54 and README §36: *"exact
build/run commands, model paths or package ids, generated outputs, parity or path-test results, and
relevant performance or memory notes"*).

1. **Builds clean** on Linux CUDA release; no new warnings in our files.
2. **Parity**: per-tensor cosine ≥ 0.999 against PyTorch on a fixed seed, for DiT output, PCA⁻¹,
   Fish decode, and (M2+) speaker encode. Numbers recorded in the PR.
3. **Path tests**: the family passes the CLI path-test matrix for safetensors, F16 GGUF, Q8_0 GGUF.
4. **Long-form**: the shared long-form clone case renders and is auditioned for seam artefacts —
   or `long_form` is not claimed and the limit is documented.
5. **RTF < 1.0** measured on the RTX 3090, warm, with the command line included.
6. **VRAM stable** across ≥5 consecutive requests (no growth); `mem_saver` used if tuning is needed,
   never to mask a leak.
7. **Generated wavs attached** for both default-reference and custom-reference cloning.
8. **Licence disclosed**: CC-BY-NC-SA-4.0 on weights *and outputs*.
9. **Independent review**: Codex authored → Claude reviews. Reviewer ≠ author, always.

Only when 1–9 are green does the PR move from draft to ready-for-review.

---

## 6. audio.cpp integration surface

Follows Confucius4-TTS, the spec-v1 exemplar named in issue #128.

```
model_specs/echo_tts.json                     # schema_version 1
src/community_models/echo_tts/*.cpp
include/engine/community_models/echo_tts/*.h
tests/echo_tts/echo_tts_warm_bench.cpp
docs/community_models/echo_tts.md
CMakeLists.txt                                # audiocpp_add_model(echo_tts SOURCES … INCLUDES … LOADERS …)
```

- **No `loader.cpp`.** Spec-v1 models use the generic spec-backed loader (issue #128).
- **Loader symbol is `engine::models::echo_tts::make_echo_tts_loader`** — namespace `models`, *not*
  `community_models`, matching `inflect_v2`. Getting this wrong is a link error.
- **GGUF preferred over safetensors**, self-contained with the spec embedded; safetensors optional.
- **Normalised option names** (framework-validated): reference audio is `target_voice`, durations are
  `*_sec`, chunking uses `audio_chunk_threshold_sec` / `audio_chunk_duration_sec` /
  `cross_fade_duration_sec`. Do not copy Python names into the spec.

Proposed options: `cfg_scale_text`, `cfg_scale_speaker`, `num_steps`, `truncation_factor`,
`speaker_kv_scale`, `seed`, `target_voice`.

---

## 7. Implementation traps

Each of these would cost days if hit blind.

1. **`autoencoder.py:943-965` — decoder transformer that never executes.** It exists only as an
   unregistered local variable. Porting the apparent configuration would be silently wrong.
2. **Weight normalisation**: most DAC convolutions store weight-norm parameters, not ready conv
   weights. Fold at conversion time.
3. **FP32 boundaries are load-bearing**: RMSNorm and adaLN accumulate in FP32; the sampler, PCA, and
   Fish weights are FP32 while Echo weights are BF16. Low-precision-only normalisation diverges.
4. **Do not serialise `freqs_cis` / `causal_mask`** into GGUF (303.6 M elements). Regenerate.
5. **Half-head RoPE**: the trunk rotates only half the heads — unusual, easy to get wrong.
6. **Snake activation** in the DAC likely needs a composed or custom kernel.
7. **Causal conv padding/cropping** computes right-padding from runtime length; transposed conv crops
   asymmetrically. Off-by-one here is silent audio corruption.
8. **Shape divisibility**: speaker and prefix latents reshape in groups of 4.
9. **Mask-based unconditioning**: CFG unconditions via masks, not zeroed encoder states.

---

## 8. Testing strategy

- **Parity harness**: dump reference intermediates from PyTorch (fixed seed) to `.npy`; C++ loads and
  compares per-stage with cosine + max-abs-error. Stage boundaries: text_encoder out, speaker_encoder
  out, per-block DiT out (first/middle/last), final latents, PCA⁻¹ out, decoder out.
- **Bit-exactness is not the goal.** Gaussian RNG is device-specific; aim for statistical equivalence
  on the noise and ≥0.999 cosine downstream.
- **Ear check is mandatory** at M1, M2, M3. Cosine can pass while audio is wrong (the flattening-point
  crop is a host-side loop, not covered by tensor parity).
- **Regression**: reuse the bench's `chris_hemsworth_15s.wav` reference so output is directly
  comparable to the 16 existing scored Echo rows in tts-bench.

---

## 9. Open questions

- `latent_scale` is resolved (1/18) but its *derivation* is unverified; confirm it is applied on both
  the forward and inverse PCA legs consistently.
- Exact RoPE theta for the Echo trunk — read from source at implementation time, do not assume.
- Whether `quantizer.post_module` + `upsample` can be skipped in the M2 encode call without drift, as
  §2.4 suggests. Verify numerically before optimising.
- Whether the maintainer will accept a `long_form` implementation built on an upstream path its own
  author calls under-tested. Raise in M0.

---

## 10. Licence

Echo-TTS weights **and generated outputs** are CC-BY-NC-SA-4.0 — the output constraint is forced by
the Fish S1-DAC dependency. This is stricter than a weights-only NC licence and must be stated
plainly in `docs/community_models/echo_tts.md` and in the PR body.

Precedent exists in-tree: `higgs_audio_tts` (Research NC) and `omnivoice` (Apache code /
CC-BY-NC weights). The *output* restriction appears to be new for audio.cpp — flag it explicitly
rather than letting it be inferred.
