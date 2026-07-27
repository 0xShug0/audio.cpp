#!/bin/bash
# Drift-cancelling ABBA comparison of two thread counts for one binary.
# See ab.sh for why ABBA + mean (and not min) is required on this machine.
# usage: abt.sh <bin> <threadsA> <threadsB> [passes] [audio]
BIN=$1; TA=$2; TB=$3; P=${4:-4}
AUDIO=${5:-tests/parakeet_tdt/assets/2086-149220-0033.wav}
one() {
  "$BIN" --model "${PARAKEET_MODEL:-models/parakeet-tdt-0.6b-v3}" --audio "$AUDIO" --warmup-audio "$AUDIO" \
    --backend cpu --threads "$1" --warmup 1 --iterations 4 \
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
echo "burn-in..." >&2; one "$TA" >/dev/null; one "$TB" >/dev/null
R=""
for p in $(seq 1 "$P"); do
  a1=$(one "$TA"); b1=$(one "$TB"); b2=$(one "$TB"); a2=$(one "$TA")
  line=$(python3 -c "
a=($a1+$a2)/2; b=($b1+$b2)/2
print(f'pass$p  t$TA={a:8.1f}  t$TB={b:8.1f}  ratio={b/a:.4f}')")
  echo "$line"; R="$R $(echo "$line"|sed 's/.*ratio=//')"
done
python3 -c "
import statistics
r=[float(x) for x in '''$R'''.split()]
print(f'--> median t$TB/t$TA = {statistics.median(r):.4f}  (<1 means t$TB is faster)')"
