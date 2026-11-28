# Drift-cancelling A/B benchmark harness

Historical harness documentation: the removed runner and original commands
are preserved in the [pre-cleanup backup](https://github.com/0xShug0/audio.cpp/blob/checkpoint/before-tests-cleanup/tests/parakeet_tdt/bench/README.md).
Reported measurements, hashes and limitations are unchanged. These recipes
are not instructions for the current tree.

The historical scripts compared parakeet_tdt builds and thread counts when
the difference was a few percent and the machine was not thermally stable.

## Why not just time both and compare

Because that silently produces wrong answers. On the laptop these were written
on, the *same binary* measures ~980 ms cold and ~1265 ms once thermally
saturated — a ~29% swing, far larger than any optimization being evaluated.
Running "config A for a while, then config B" therefore measures mostly
whichever one ran while the machine was cooler.

Two defences:

1. **Burn-in.** One run of each config is discarded up front, so measurement
   starts from steady state rather than mid-ramp.
2. **ABBA ordering, scored by the mean.** Each pass runs A, B, B, A and scores
   each config as the mean of its two slots. Under any linear drift both means
   land on the same midpoint (slots 1&4 vs 2&3), so the drift cancels exactly.

Within a single run the *median* iteration is taken rather than the minimum, so
one scheduler spike cannot decide the outcome.

Scoring by `min()` across the pass — which an earlier version of this harness
did — does **not** cancel drift: it hands the win to whichever config happened
to occupy the coldest slot. That version reported a confident 5-8%
"regression" for a change that is provably bit-exact and, under the corrected
estimator, is a small improvement. If you modify these scripts, preserve the
mean-of-both-slots property.

## Usage

[Archived command](https://github.com/0xShug0/audio.cpp/blob/checkpoint/before-tests-cleanup/tests/parakeet_tdt/bench/README.md) (pre-cleanup backup).

Both print a per-pass ratio plus the median across passes. A ratio below 1.0
means the second config is faster. Treat a result as real only if the sign is
consistent across passes — the per-pass spread tells you whether the machine
was quiet enough to trust the number at all.

The pinned guide above preserves the original environment overrides.

These compare *encoder graph compute* specifically, which is ~93-96% of wall
time and the part almost every optimization targets. For end-to-end numbers,
the old direct runner is preserved in the
[pinned harness source](https://github.com/0xShug0/audio.cpp/blob/checkpoint/before-tests-cleanup/tests/parakeet_tdt/parakeet_warm_bench.cpp).
