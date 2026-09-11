#!/bin/bash
# Interleaved A/B of two parakeet_warm_bench binaries, drift-cancelling.
#
# This box ramps thermally: the same binary measures ~980 ms cold and ~1265 ms
# at steady state. Two defences against that:
#   1. a burn-in run (discarded) so measurement starts at steady state;
#   2. ABBA ordering per pass, scored by the MEAN of each binary's two slots.
#      Under any linear drift both means land on the same midpoint (slots
#      1&4 vs 2&3), so the drift cancels exactly. Scoring by min() instead
#      would hand the win to whichever binary occupied the coldest slot.
# Within a single run we take the median iteration, not the min, so one
# scheduler spike cannot decide the result.
# usage: ab.sh <binA> <binB> [passes] [threads]
A=$1; B=$2; P=${3:-5}; T=${4:-6}
one() {
  "$1" --model "${PARAKEET_MODEL:-models/parakeet-tdt-0.6b-v3}" \
    --audio tests/parakeet_tdt/assets/2086-149220-0033.wav \
    --backend cpu --threads "$T" --warmup 1 --iterations 4 \
    --timing-file "${TIMING_LOG:-/tmp/parakeet_ab_timing.log}" 2>/dev/null \
  | python3 -c '
import sys, statistics
v=[]
for l in sys.stdin:
    l=l.strip()
    if l=="average": break
    if l.startswith("parakeet_tdt.encoder.graph.compute_ms="): v.append(float(l.split("=")[1]))
print(f"{statistics.median(v):.1f}")'
}
echo "burn-in (discarded)..." >&2
one "$A" > /dev/null; one "$B" > /dev/null
RESULTS=""
for p in $(seq 1 "$P"); do
  a1=$(one "$A"); b1=$(one "$B"); b2=$(one "$B"); a2=$(one "$A")
  line=$(python3 -c "
a=($a1+$a2)/2; b=($b1+$b2)/2
print(f'pass$p  A={a:8.1f}  B={b:8.1f}  B/A={b/a:.4f}')")
  echo "$line"
  RESULTS="$RESULTS $(echo "$line" | sed 's/.*B\/A=//')"
done
python3 -c "
import statistics
r=[float(x) for x in '''$RESULTS'''.split()]
print(f'--> median B/A = {statistics.median(r):.4f}   (min {min(r):.4f}, max {max(r):.4f})')"
