### 1. Adaptive generation window

`parse_sampler_options` hardcoded `sequence_length = max_sequence_length`, so
every chunk paid the full 640-latent / 29.72 s window and `find_flattening_point`
discarded the silent tail. Now estimated per chunk from the tokenized byte count,
with one automatic retry at full length if no flattening point is found.

The rate is derived, not guessed: 640 frames span 29.7215 s → 21.53 frames/s;
`kDefaultTextChunkSize` is documented in-file as ~20 s at 300 codepoints →
~15 bytes/s; 21.53 / 15 = **1.435 frames per UTF-8 byte**, with a 1.30 margin.

| chunk bytes | ≈ speech | window | denoiser cost |
|---|---|---|---|
| 60 | 4.0 s | 128 | 5.00x cheaper |
| 120 | 8.0 s | 256 | 2.50x |
| 200 | 13.3 s | 384 | 1.67x |
| 300 | 20.0 s | 576 | 1.11x |
| 340+ | 22.7 s+ | 640 | 1.00x (unchanged) |

**Off by default.** The original claim here was that an under-estimate "costs
time, never fidelity", because the 640-frame retry draws bit-identical noise from
the sequential Philox stream. The noise claim is true and the fidelity conclusion
does not follow from it. Echo's generated self-attention is fully non-causal
(`self_mask = torch.ones((batch_size, seq_len))`, `model.py:249`), so every
latent position attends across the entire window. Shrinking 640 to 128 changes
the computation at every retained position, not just how many positions survive
-- and the retry fires only when no flattening point is found, so a short window
that yields a plausible flat tail is never corrected.

Enable with `AUDIOCPP_ECHO_TTS_ADAPTIVE_WINDOW=1` once it has been A/B'd against
the full window on a fixed seed. The cost saving below is real; it is the
default that was wrong.

Estimates snap to a 64-frame grid (`kWindowQuantum`) because denoiser graphs are
keyed on `sequence_length` and rebuilt when it changes.

- Pin explicitly: `max_duration_sec` request option (skips the estimate).
- Enable: `AUDIOCPP_ECHO_TTS_ADAPTIVE_WINDOW=1` (off by default).

Confirmed in your logs: 23 bytes → 128-frame window, `keys` 824 → 312.

### 2. KV cache pre-expanded across CFG lanes

`joint_attention` called `expand()` — a `RepeatModule` broadcasting the cached
text and speaker K/V across the 2 or 3 CFG lanes — every layer, every step. The cache
is now allocated at 2 (for Joint-CFG) or 3 (for Independent-CFG) lanes and broadcast once in the conditioning graph,
so `expand()` short-circuits. The single-lane graph reads lane 0 through
`kv_for_lanes()`, a zero-copy view (the lane axis is outermost).

`kv_for_lanes()` reconciles both directions — it narrows when the cache is wider
than the graph, and returns the cache untouched when it is narrower, leaving
`expand()` to broadcast as before. That second case is what
`AUDIOCPP_ECHO_TTS_NO_KV_EXPAND=1` produces:

| `NO_KV_EXPAND` | cache | graph | `kv_for_lanes` | `expand` |
|---|---|---|---|---|
| off | 3 | 1 | slice to 1 | no-op |
| off | 3 | 3 | passthrough | no-op |
| on | 1 | 1 | passthrough | no-op |
| on | 1 | 3 | passthrough | repeat to 3 |

**Tradeoff:** the cache is 3x larger. Confirmed active in your logs by
`kv_text.0.k n=141312` = 23 × 2048 × 3.

### 3. `reference_duration_sec` defaults to 15 s

Previously unbounded to the trained maximum, so an untrimmed clip charged up to
1600 speaker tokens to `keys` in all 24 blocks at every step, plus a linear
encode pass per chunk of reference. Confirmed in your logs: 30 s → 161 tokens
became 15 s → 80 tokens, taking `keys` from 824 to 744 at the same window.

### 4. `guidance_interval` (opt-in, default 1 = off)

Echo guides every step in the t >= 0.5 window with three forward passes (or
two forward passes in joint CFG mode):

```
v_pred = v_cond + 3.0*(v_cond - v_text_uncond) + 8.0*(v_cond - v_speaker_uncond)
```

`v_cond` moves quickly in t; the *correction* does not. `guidance_interval` measures
the correction every Nth guided step and reuses it in between, so skipped steps
cost one forward pass instead of three.

| steps | interval | guided | refreshes | lane-evals | vs 40/1 |
|---|---|---|---|---|---|
| 40 | 1 | 20 | 20 | 80 | 1.00x |
| 40 | 2 | 20 | 10 | 60 | 1.33x |
| 40 | 3 | 20 | 7 | 54 | 1.48x |
| 30 | 1 | 15 | 15 | 60 | 1.33x |
| 30 | 2 | 15 | 8 | 46 | 1.74x |
| 20 | 2 | 10 | 5 | 30 | 2.67x |
| 14 | 1 | 7 | 7 | 28 | 2.86x |
| 14 | 2 | 7 | 4 | 22 | 3.64x |

**The number that matters is `refreshes`, not the speedup.** Ten refreshes
across the guided phase means each reused correction is one small t-step stale.
Four means the correction was already coarsely sampled before you subsampled it,
and the speaker term's weight of 8.0 multiplies any staleness straight into
timbre and pronunciation -- a failure mode you hear rather than see on a
waveform.

So: **raise it to 2 at 30+ steps; leave it at 1 below ~20.** `interval=3` buys
1.48x against 1.33x at 40 steps -- most of the fidelity risk for a fraction of
the extra speed. The curve flattens because the unguided half of the schedule is
a floor: at 40 steps, 20 of the 80 lane-evals were never guided, so no interval
gets past 2.0x.

Note that **40 steps at interval 2 and 30 steps at interval 1
both cost 60 lane-evals.** Same compute, spent differently: fewer steps coarsens
the whole ODE trajectory, a longer interval leaves the trajectory intact and only
lets the guidance go stale. Neither dominates on paper. Compare them by
listening.

The implementation holds the correction in absolute units rather than as a
ratio, so a stale value cannot amplify a small `v_cond`, and the first guided
step always refreshes so a stale delta is never applied before one exists.
Verified against a mock denoiser at 0.011% max deviation for interval 2 and
0.020% for interval 3 -- but that mock had deliberately smooth uncond offsets,
so treat those as a lower bound, not a measurement on real audio.

### 5. `echo_tts.mem_saver` (opt-in, default false)
Mem-saver frees VRAM by free-ing the audio encoder allocations after reference
speaker encoding. For cached reference speakers, this has no performance hit,
for an uncached reference speaker this has a small performance hit. Peak VRAM
often occurs during speaker encoding, but depends on the length of prompts you
send (long prompts make the audio decode buffers larger than the encode buffers).
Actual savings will depend on model quantization and backend architecture. For
Q8_0 model weights, the following were measured on a GTX 1080 Ti and a RTX 4070:
- GTX 1080 Ti (false/true): 8122 MB / 7416 MB (8.7% savings)
- RTX 4070  (false/true): 6084 MB / 5356 MB (12.0% savings)

### 6. `echo_tts.cfg_mode` (opt-in, default independent)
Echo-TTS allows two classifier free guidance modes: the reference Independent
mode with a guidance scale for the reference speaker and a guidance scale for the
text, and a Joint mode with a single combined guidance scale. Joint-CFG allows
for a slightly smaller VRAM allocation, and fewer forwards passes. While actual
results will depend on the backend architecture, the following numbers are
provided as examples measured on a GTX 1080 Ti and an RTX 4080:
- GTX 1080 Ti (independent/joint): 16.9s / 11.9s (0.7x, or ~30% faster)
- RTX 4070 (independent/joint): 3.5s / 2.7s (0.77x, or ~23% faster)

The setting also has an effect on audio quality. Measured by @5uck1ess on an RTX
5090, 4 English prompts x 3 seeds run through Whisper-v3 WER and WavLM SIM:

| Mode & Scale | WER | UTMOS | SIM |
|---|---|---|---|
| independent, 3.0/8.0 | 0.069 | 4.20 | 0.831 |
| joint, 4.0 | 0.081 | 4.32 | 0.804 |
| joint, 6.0 | 0.060 | 4.31 | 0.821 |
| joint, 8.0 | 0.076 | 4.29 | 0.829 |
| joint, 10.0 | 0.069 | 4.26 | 0.830 |

Joint-CFG at scale 8.0 is at parity, below ~6.0 scale similarity drops. By ear:
near parity, independent is slightly better on a few, clone is almost identical.
@lojack5 also notes: Joint-CFG the most noticable (by ear) difference is at the
first ~0.2 seconds of the generated audio, where the speaker doesn't as closely
match the reference as Independent-CFG does. After the beginning of the
generation, the speaker quickly matches the reference audio.

### 7 `t_schedule_mode` (opt-in, default linear)
Like Irodori-TTS, Echo-TTS has two Euler solver timestep schedules available.
Upstream only implements the linear time schedule, so here Echo-TTS utilizes a shared
Sway timestep schedule (originally from Irodori-TTS), which changes the timestep grid
supplied to the ODE solver. Only active when `t_schedule_mode=sway`, and utilizes
`sway_coeff` to change the grid (default -1.0). Negative coefficients place
denser timesteps towards the beginning of the time schedule, closer to the
initial noise. Zero gives an equivalent to `t_schedule_mode=linear`. Coefficients
must be finite and produce strictly decreasing timesteps; invalid combinations
are rejected. When `t_schedule_mode=sway` is selected, the default for
`num_inference_steps` is changed to 12.

Sway scheduling can give similar quality at fewer inference steps than linear
scheduling. By-ear, a listener (@lojack5) was unable to distinguish between
linear 40-step and sway 12-step generations in a blind A/B/X test, both for
Joint-CFG and Independent-CFG; sway 11-step was distinguishable in 20% of the
tests.

Audio analysis backs this up: a reference audio generated using linear 40-step
time schedule was compared against a sweep of sway schedules (6-40), using the same
sampling parameters (seed, prompt, etc) and only changing the timestep schedule
(8-16 steps shown). "speaker cosine" is the cosine similarity after embedding with
speechbrain/spkrec-ecapa-voxceleb; "voiced disagreement" is a pitch and prosody
comparison.

| Sway steps | DTW logmel distance | Pooled logmel cosine | logmag mae | spectral convergence | speaker cosine | voiced disagreement |
|---|---|---|---|---|---|---|
| 8 | 0.003887 | 0.9949 | 1.1011 | 0.5997 | 0.9278 | 0.08445 |
| 9 | 0.001769 | 0.9978 | 0.8824 | 0.5998 | 0.9574 | 0.07693 |
| 10 | 0.002054 | 0.9998 | 0.4943 | 0.3411 | 0.9639 | 0.06218 |
| 11 | 0.003888 | 0.9990 | 0.6357 | 0.4087 | 0.9668 | 0.06218 |
| 12 | 0.001769 | 0.9998 | 0.4745 | 0.3150 | 0.9641 | 0.06741 |
| 13 | 0.001430 | 0.9999 | 0.4286 | 0.2482 | 0.9675 | 0.06093 |
| 14 | 0.001631 | 0.9998 | 0.4503 | 0.2594 | 0.9685 | 0.06729 |
| 15 | 0.001312 | 0.9999 | 0.4144 | 0.2281 | 0.9730 | 0.07787 |
| 16 | 0.001429 | 0.9999 | 0.4201 | 0.2481 | 0.9675 | 0.06449 |

Note: Sway timestep schedule does not purport to produce identical audio. It can
produce similar quality and cloning similarity at fewer inference steps.

At 12-step sway versus 40-step linear, the following speedups were observed:

| GPU | CFG Mode | timestep schedule | Audio Duration | Request Time | RTF | Speedup |
|---|---|---|---|---|---|---|
| GTX 1080 Ti | independent | linear-40 | 2.46s | 16.68s | x6.78 | - |
| GTX 1080 Ti | independent | sway-12 | 2.46s | 6.04s | x2.45 | x2.76 |
| GTX 1080 Ti | independent | linear-40 | 9.20s | 17.80s | x1.94 | - |
| GTX 1080 Ti | independent | sway-12 | 9.57s | 6.58s | x0.69 | x2.70 |
| GTX 1080 Ti | independent | linear-40 | 26.38s | 36.95s | x1.40 | - |
| GTX 1080 Ti | independent | sway-12 | 26.38s | 14.28s | x0.54 | x2.58 |
| GTX 1080 Ti | joint | linear-40 | 3.07s | 11.50s | x3.75 | - |
| GTX 1080 Ti | joint | sway-12 | 3.02s | 4.25s | x1.41 | x2.70 |
| GTX 1080 Ti | joint | linear-40 | 9.85s | 13.51s | x1.37 | - |
| GTX 1080 Ti | joint | sway-12 | 9.75s | 4.98s | x0.51 | x2.71 |
| GTX 1080 Ti | joint | linear-40 | 27.49s | 28.67s | x1.04 | - |
| GTX 1080 Ti | joint | sway-12 | 27.49s | 11.08s | x0.40 | x2.59 |
| RTX 4070 | independent | linear-40 | 2.46s | 3.91s | x1.59 | - |
| RTX 4070 | independent | sway-12 | 2.55s | 1.52s | x0.59 | x2.57 |
| RTX 4070 | independent | linear-40 | 9.20s | 4.34s | x0.47 | - |
| RTX 4070 | independent | sway-12 | 9.20s | 1.66s | x0.18 | x2.61 |
| RTX 4070 | independent | linear-40 | 26.38s | 8.91s | x0.38 | - |
| RTX 4070 | independent | sway-12 | 26.61s | 3.90s | x0.15 | x2.29 |
| RTX 4070 | joint | linear-40 | 2.83s | 2.92s | x1.03 | - |
| RTX 4070 | joint | sway-12 | 2.79s | 1.07s | x0.38 | x2.74 |
| RTX 4070 | joint | linear-40 | 9.85s | 3.18s | x0.32 | - |
| RTX 4070 | joint | sway-12 | 9.75s | 1.28s | x0.13 | x2.47 |
| RTX 4070 | joint | linear-40 | 27.49s | 6.80s | x0.25 | - |
| RTX 4070 | joint | sway-12 | 27.59s | 2.98s | x0.11 | x2.28 |

Measured with identical sampling parameters except where noted, and "warm" server: model weights loaded, reference speaker cached, GPUs up to temperature. Three prompt lengths, two GPUs, and both CFG modes tested. Again, Sway
scheduling does not purport to produce identical audio, merely similar quality at lower step counts. Using the
recommended step counts gives a ~2.3-2.7 times speedup.

---

## Suggested config

```json
"session_options": {
  "echo_tts.reference_cache_slots": "8",
  "echo_tts.reference_duration_sec": "15"
},
"default_request_options": {
  "num_inference_steps": 14,
  "guidance_interval": 1
}
```

Do **not** put `max_duration_sec` here — it pins the window and gives back the
whole of change 1.
