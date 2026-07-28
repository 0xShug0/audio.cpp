#!/usr/bin/env python3
"""Compare parakeet_tdt weight-storage types across a multilingual corpus.

For each weight type this transcribes every clip in the FLEURS manifest and
reports, against the f32 (native) run as the reference:
  - exact-match rate   : fraction of clips whose text is byte-identical to f32
  - WER vs f32         : how much the decoded text actually moves
  - WER vs FLEURS ref  : absolute quality, so a type that merely matches f32
                         while being bad at the task cannot hide

All clips go through a single process per weight type (--audio-sequence), so
the model and its quantised weights are built once rather than per clip.

Usage: run_accuracy.py [weight_type ...]
"""
import json
import os
import re
import subprocess
import sys
import unicodedata

BIN = os.environ.get("PARAKEET_BENCH", "build/linux-cpu-release/bin/parakeet_warm_bench")
MODEL = "models/parakeet-tdt-0.6b-v3"
CORPUS = os.environ.get("FLEURS_DIR", "tmp/fleurs")
MANIFEST = f"{CORPUS}/manifest.json"
TYPES = sys.argv[1:] or ["native", "f16", "bf16", "q8_0"]
THREADS = os.environ.get("PARAKEET_THREADS", "6")


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def wer(ref, hyp):
    r, h = normalize(ref).split(), normalize(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[len(h)] / len(r)


def transcribe_all(files, wtype):
    """One process, all clips, model loaded once."""
    out = subprocess.run(
        [BIN, "--model", MODEL,
         "--audio-sequence", ",".join(files),
         "--warmup-audio", files[0],
         "--backend", "cpu", "--threads", THREADS, "--warmup", "0",
         "--iterations", "1", "--timing-file", f"{CORPUS}/t_acc.log",
         "--session-option", f"parakeet_tdt.matmul_weight_type={wtype}"],
        capture_output=True, text=True, timeout=14400)
    for line in out.stdout.splitlines():
        if line.startswith("summary_json="):
            data = json.loads(line[len("summary_json="):])
            return [s["text_output"] for s in data["sequence_steps"]]
    sys.stderr.write(out.stdout[-2000:] + out.stderr[-2000:])
    raise RuntimeError(f"no summary_json for {wtype}")


manifest = json.load(open(MANIFEST))
files = [f"{CORPUS}/{m['file']}" for m in manifest]
print(f"{len(manifest)} clips, {len({m['lang'] for m in manifest})} languages, "
      f"types={TYPES}\n", flush=True)

results = {}
for wtype in TYPES:
    results[wtype] = transcribe_all(files, wtype)
    print(f"  {wtype}: {len(results[wtype])} transcripts", flush=True)
    json.dump(results, open(f"{CORPUS}/hyps.json", "w"), ensure_ascii=False)

ref_type = TYPES[0]
ref_hyps = results[ref_type]
print(f"\n{'type':<8} {'exact vs '+ref_type:>16} {'WER vs '+ref_type:>14} {'WER vs FLEURS':>14}")
for wtype in TYPES:
    hyps = results[wtype]
    n = min(len(ref_hyps), len(hyps))
    exact = sum(ref_hyps[i] == hyps[i] for i in range(n)) / max(1, n)
    w_self = sum(wer(ref_hyps[i], hyps[i]) for i in range(n)) / max(1, n)
    w_abs = sum(wer(manifest[i]["reference"], hyps[i]) for i in range(n)) / max(1, n)
    print(f"{wtype:<8} {exact:>15.1%} {w_self:>14.4f} {w_abs:>14.4f}")

print("\nper-language WER vs FLEURS reference:")
for lang in sorted({m["lang"] for m in manifest}):
    idxs = [i for i, m in enumerate(manifest) if m["lang"] == lang]
    cells = []
    for wtype in TYPES:
        vals = [wer(manifest[i]["reference"], results[wtype][i])
                for i in idxs if i < len(results[wtype])]
        cells.append(f"{wtype}={sum(vals)/max(1,len(vals)):.3f}")
    print(f"  {lang}: " + "  ".join(cells))

print("\nclips where q8_0 differs from native (first 10):")
if "q8_0" in results and "native" in results:
    shown = 0
    for i, m in enumerate(manifest):
        if i < len(results["q8_0"]) and results["native"][i] != results["q8_0"][i]:
            print(f"  [{m['lang']}] native: {results['native'][i][:110]}")
            print(f"  [{m['lang']}] q8_0  : {results['q8_0'][i][:110]}")
            shown += 1
            if shown >= 10:
                break
    if shown == 0:
        print("  (none)")
