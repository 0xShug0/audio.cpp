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

An under-estimate costs time, never fidelity: `generate_torch_cuda_randn` is a
sequential Philox stream, so the 640-frame retry draws bit-identical noise to
the run that would have happened without this change.

Estimates snap to a 64-frame grid (`kWindowQuantum`) because denoiser graphs are
keyed on `sequence_length` and rebuilt when it changes.

- Pin explicitly: `sequence_length` request option (skips the estimate).
- Disable: `AUDIOCPP_ECHO_TTS_NO_ADAPTIVE_WINDOW=1`.

Confirmed in your logs: 23 bytes → 128-frame window, `keys` 824 → 312.

### 2. KV cache pre-expanded across CFG lanes

`joint_attention` called `expand()` — a `RepeatModule` broadcasting the cached
text and speaker K/V across the 3 CFG lanes — every layer, every step. The cache
is now allocated at `kMaxCfgLanes` and broadcast once in the conditioning graph,
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

### 3. `reference_max_seconds` defaults to 15 s

Previously unbounded to the trained maximum, so an untrimmed clip charged up to
1600 speaker tokens to `keys` in all 24 blocks at every step, plus a linear
encode pass per chunk of reference. Confirmed in your logs: 30 s → 161 tokens
became 15 s → 80 tokens, taking `keys` from 824 to 744 at the same window.

### 4. `cfg_interval` (opt-in, default 1 = off)

Echo guides every step in the t >= 0.5 window with three forward passes:

```
v_pred = v_cond + 3.0*(v_cond - v_text_uncond) + 8.0*(v_cond - v_speaker_uncond)
```

`v_cond` moves quickly in t; the *correction* does not. `cfg_interval` measures
the correction every Nth guided step and reuses it in between, so skipped steps
cost one forward pass instead of three.

| num_steps | interval | guided | refreshes | lane-evals | vs 40/1 |
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

---

## Suggested config

```json
"session_options": {
  "echo_tts.reference_cache_slots": "8",
  "echo_tts.reference_max_seconds": "15"
},
"default_request_options": {
  "num_steps": 14,
  "cfg_interval": 1
}
```

Do **not** put `sequence_length` here — it pins the window and gives back the
whole of change 1.
