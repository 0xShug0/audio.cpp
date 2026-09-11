# Recorded results

## `fleurs_weight_types.json`

Full output of one run of `compare_weight_types.py native f16 bf16 q8_0` —
every clip's reference transcript plus what each weight type actually
transcribed, so the aggregate numbers below can be re-derived or re-scored
with a different WER normalization without re-running anything (and without
re-downloading the corpus).

- Corpus: FLEURS test split, 5 clips per language
- 120 clips, 24 languages, 21.8 minutes of audio
- Run date: 2026-07-27

### Hardware

Intel i7-9750H (6C/12T, Coffee Lake). An ordinary 6-core desktop-class CPU —
nothing exotic and nothing especially old, so the CPU figures should
generalize reasonably to similar AVX2 machines.

**The GPU in this machine is not representative and CUDA numbers elsewhere in
these docs should be read with that in mind**: a GTX 1650 **with Max-Q
Design** — Turing, compute capability 7.5, 4 GB, and a **35 W** power limit.
That is a thermally- and power-constrained mobile part, and its compute
capability is below the 8.0 (Ampere) threshold at which ggml enables CUDA
graph capture at all. Null results on the CUDA side (flash attention, fused
QKV) are specifically *this card's* null results.

### Aggregate

| weight type | encode speed | transcript identical to f32 | WER vs f32 | absolute WER |
|---|---|---|---|---|
| `native` (f32) | 1.00x | 100% | — | 0.1338 |
| `f16` | ~1.00x | 99.2% | 0.0003 | 0.1336 |
| `bf16` | 1.14x | 99.2% | 0.0004 | 0.1334 |
| `q8_0` | 1.79x | 91.7% | 0.0062 | 0.1331 |

Speed figures are encoder graph compute on CPU at 6 threads, measured
separately with the drift-cancelling A/B harness — not from this run, which
does not control for thermal drift and is not a timing benchmark.

### Per-language WER vs the FLEURS human reference

| lang | native | f16 | bf16 | q8_0 |
|---|---|---|---|---|
| bg | 0.137 | 0.137 | 0.137 | 0.137 |
| cs | 0.122 | 0.122 | 0.122 | 0.122 |
| da | 0.257 | 0.257 | 0.257 | 0.263 |
| de | 0.029 | 0.029 | 0.029 | 0.029 |
| el | 0.419 | 0.413 | 0.419 | 0.413 |
| en | 0.050 | 0.050 | 0.050 | 0.050 |
| es | 0.045 | 0.045 | 0.045 | 0.045 |
| et | 0.164 | 0.164 | 0.164 | 0.184 |
| fi | 0.104 | 0.104 | 0.104 | 0.104 |
| fr | 0.042 | 0.042 | 0.042 | 0.042 |
| hr | 0.128 | 0.128 | 0.128 | 0.128 |
| hu | 0.140 | 0.140 | 0.140 | 0.117 |
| it | 0.000 | 0.000 | 0.000 | 0.000 |
| lt | 0.152 | 0.152 | 0.152 | 0.152 |
| lv | 0.326 | 0.326 | 0.326 | 0.326 |
| nl | 0.037 | 0.037 | 0.037 | 0.037 |
| pl | 0.147 | 0.147 | 0.137 | 0.147 |
| pt | 0.077 | 0.077 | 0.077 | 0.077 |
| ro | 0.142 | 0.142 | 0.142 | 0.142 |
| ru | 0.000 | 0.000 | 0.000 | 0.000 |
| sk | 0.078 | 0.078 | 0.078 | 0.078 |
| sl | 0.299 | 0.299 | 0.299 | 0.284 |
| sv | 0.296 | 0.296 | 0.296 | 0.296 |
| uk | 0.021 | 0.021 | 0.021 | 0.021 |

Five clips per language, so a single clip moves a language's number a lot;
these are a check that no language falls off a cliff under quantization, not
publishable per-language WER. The absolute values also reflect this harness's
deliberately crude normalization (NFKC, lowercase, strip punctuation) — no
number/ITN handling, which is why languages whose references spell out
numerals score worse. They are comparable *across columns*, not against other
papers.

### Where q8_0 diverges

`q8_0` differs from `native` on 10 of 120 clips. Sampling those diffs, most
are formatting rather than content:

```
[de] native  ... wie der T Rex war ihm nicht gewachsen
[de] q8_0    ... wie der T-Rex war ihm nicht gewachsen

[cs] native  ... táhne 80 km/50 mil do vnitrozemí
[cs] q8_0    ... táhne 80 km 50 mil do vnitrozemí
```

and where real words change, it goes both directions — this one is q8_0
correcting f32, not degrading it:

```
[hu] native  ... ez a helys turista számára ...
[hu] q8_0    ... ez a hely sok turista számára ...     <- correct reading
```

which is why the aggregate absolute WER does not move despite an 8.3%
transcript churn rate.
