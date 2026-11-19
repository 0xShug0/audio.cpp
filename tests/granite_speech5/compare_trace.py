#!/usr/bin/env python3
"""Compare a candidate ``granite_speech5`` trace against the pinned reference.

The candidate's ``trace.json`` declares its ``through`` stage; that stage is
the cutoff. Every reference entry through the cutoff (in pipeline order) must
exist in the candidate directory with an exactly matching shape. Comparison
stops at the first mismatch, reporting the stage, shapes, error metrics, and
worst coordinate.

Metrics per stage (thresholds come from the reference ``trace.json``)::

    max_abs = max |cand - ref|
    rel_rms = rms(cand - ref) / max(rms(ref), 1e-12)
    worst   = unravelled index of max |cand - ref|

Non-finite candidate data is always a failure. Exit 0 on full pass, 1 on any
failure (missing file, shape mismatch, threshold breach, non-finite data).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_f32(path: Path, shape: list[int]):
    import numpy as np

    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, f"unreadable file {path}: {exc}"
    expect = 1
    for d in shape:
        expect *= d
    if len(raw) != expect * 4:
        return None, (f"size mismatch: file has {len(raw)} bytes, "
                       f"shape {shape} needs {expect * 4}")
    arr = np.frombuffer(raw, dtype="<f4").reshape(shape).astype(np.float64)
    return arr, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np

    ref = json.loads((args.reference / "trace.json").read_text(encoding="utf-8"))
    cand_meta = json.loads((args.candidate / "trace.json").read_text(encoding="utf-8"))
    through = cand_meta.get("through")
    stages = ref["stages"]
    names = [s["name"] for s in stages]
    # `through` may be a pipeline phase (frontend|encoder|projector|prefill|
    # generate), resolved via the reference `cutoffs` map, or a literal
    # stage name.
    cutoffs = ref.get("cutoffs", {})
    if through in cutoffs:
        through = cutoffs[through]
    if through not in names:
        print(f"FAIL: candidate through={through!r} not in reference pipeline {names}")
        return 1
    cutoff = names.index(through)
    wanted = stages[:cutoff + 1]

    print(f"reference: {args.reference}  candidate: {args.candidate}  through: {through}")
    for entry in wanted:
        name, fname = entry["name"], entry["file"]
        shape = list(entry["shape"])
        rpath = args.reference / fname
        cpath = args.candidate / fname
        if not rpath.exists():
            print(f"FAIL [{name}]: reference file missing: {rpath}")
            return 1
        if not cpath.exists():
            print(f"FAIL [{name}]: candidate file missing: {cpath}")
            return 1
        rarr, err = load_f32(rpath, shape)
        if err is not None:
            print(f"FAIL [{name}]: reference file unreadable: {err}")
            return 1
        carr, err = load_f32(cpath, shape)
        if err is not None:
            print(f"FAIL [{name}]: {err}")
            return 1
        if not np.all(np.isfinite(carr)):
            bad = int(np.count_nonzero(~np.isfinite(carr)))
            print(f"FAIL [{name}]: {bad} non-finite values in candidate")
            return 1
        diff = carr - rarr
        max_abs = float(np.max(np.abs(diff)))
        rms_ref = float(np.sqrt(np.mean(rarr ** 2)))
        rms_diff = float(np.sqrt(np.mean(diff ** 2)))
        rel_rms = rms_diff / max(rms_ref, 1e-12)
        worst = tuple(int(i) for i in np.unravel_index(int(np.argmax(np.abs(diff))), diff.shape))
        ok = True
        reasons = []
        if entry.get("max_abs") is not None and not max_abs <= entry["max_abs"]:
            ok = False
            reasons.append(f"max_abs {max_abs:.3e} > {entry['max_abs']:.1e}")
        if entry.get("rel_rms") is not None and not rel_rms <= entry["rel_rms"]:
            ok = False
            reasons.append(f"rel_rms {rel_rms:.3e} > {entry['rel_rms']:.1e}")
        status = "PASS" if ok else "FAIL"
        print(f"{status} [{name}]: shape={shape} max_abs={max_abs:.3e} "
              f"rel_rms={rel_rms:.3e} worst={worst}")
        if not ok:
            print(f"FAIL [{name}]: " + "; ".join(reasons))
            return 1
    print(f"OK: all {len(wanted)} stages through '{through}' match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
